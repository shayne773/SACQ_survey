"""Closed-loop LIBERO Spatial rollouts for pi0.5 (original or simulated-quantized weights); grouped runner adapted
from the SmolVLA port's rollout2.py.

Protocol (identical for every condition):
- Official LIBERO initial states through the reset journal of the environment seed (reset_journal.py): each reset
  restores the journal's RNG state, forces the journal's fixture (cabinet/stove) poses and must reproduce the
  journal's first-frame fingerprint, otherwise the run aborts.
- openpi LIBERO example semantics (pisim.py): 256x256 agentview + wrist cameras, 10 no-op settle steps, policy input
  = 180-degree rotated, resize_with_pad 224 images + eef pos/axis-angle/gripper state + task language; episode ends
  when LIBERO reports done (success) or after 220 policy steps (openpi's libero_spatial cap).
- pi0.5 call: openpi input transforms, 10 Euler flow-matching steps, output transforms -> (10, 7) chunk; the first 5
  actions are executed before the next call (openpi replan_steps = 5).
- Flow-matching noise is a fixed function of (policy seed, env seed, task, episode, policy-call index).
- Every policy call has batch size 1 (eager), so an episode's actions never depend on other episodes.

Throughput layout (RAM-bounded under WSL): G simulator processes x K environments each; a group's K environments
step sequentially inside its process while the policy (main process, GPU) serves other groups. A group is drained
and replaced by a fresh process when its resident memory exceeds --group-rss-cap-mb (hard resets leak renderer memory).
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import os
import sys
import time
from multiprocessing.connection import wait
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import picore as P  # noqa: E402
import reset_journal as RJ  # noqa: E402


def group_entry(conn, gid, nslots):
    os.environ["LP_NUM_THREADS"] = os.environ.get("SIM_LP_THREADS", "4")
    import pisim
    pisim.group_main(conn, gid, nslots)


class Slot:
    def __init__(self, idx):
        self.idx = idx
        self.phase = "free"  # free | wait (command in flight) | ready (obs awaiting decision) | act (action pending)
        self.item = None
        self.obs = None
        self.action = None
        self.video = None


class Group:
    def __init__(self, ctx, gid, nslots):
        self.ctx, self.gid, self.nslots = ctx, gid, nslots
        self.generation = -1
        self.slots = [Slot(i) for i in range(nslots)]
        self._spawn()

    def _spawn(self):
        self.conn, child = self.ctx.Pipe()
        self.proc = self.ctx.Process(target=group_entry, args=(child, self.gid, self.nslots), daemon=True)
        self.proc.start()
        self.generation += 1
        self.inflight = None
        self.draining = False
        self.rss = 0

    def recycle(self):
        try:
            self.conn.send(("close",))
        except (BrokenPipeError, OSError):
            pass
        self.proc.join(120)
        if self.proc.is_alive():
            self.proc.kill()
            self.proc.join(10)
        self.conn.close()
        self._spawn()


def apply_overlay(model, path):
    import torch

    if path in (None, "", "original"):
        return 0, None
    if path.startswith("cond:"):  # a named condition: background + stage files (sacqpi.condition_files)
        import sacqpi
        files = sacqpi.apply_condition(model, path[5:])
        return len(files), files
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    weights = payload["weights"] if "weights" in payload else payload
    params = dict(model.named_parameters())
    n = 0
    with torch.no_grad():
        for name, w in weights.items():
            p = params[name]
            if tuple(p.shape) != tuple(w.shape):
                raise ValueError(f"shape mismatch {name}")
            p.copy_(w.to(p.device, p.dtype))
            n += 1
    return n, P.sha256_file(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True)
    ap.add_argument("--overlay", default="original")
    ap.add_argument("--env-seed", type=int, default=17)
    ap.add_argument("--tasks", type=int, nargs="*", default=list(range(10)))
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(50)))
    ap.add_argument("--pairs", default="", help="comma list of task:episode; overrides --tasks/--episodes")
    ap.add_argument("--groups", type=int, default=2)
    ap.add_argument("--envs-per-group", type=int, default=3)
    ap.add_argument("--group-rss-cap-mb", type=int, default=5000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--capture", action="store_true", help="save every policy-call input (calibration collection)")
    ap.add_argument("--video", default="none", help="none | all | comma list of task:episode")
    ap.add_argument("--max-steps", type=int, default=P.MAX_STEPS)
    a = ap.parse_args()
    if not (1 <= a.groups <= 6 and 1 <= a.envs_per_group <= 6):
        raise SystemExit(f"refusing groups={a.groups} x envs={a.envs_per_group}: outside the RAM-checked range")

    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = a.out.resolve()
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    if a.capture:
        (out / "captures").mkdir(exist_ok=True)
    pairs = [tuple(map(int, s.split(":"))) for s in a.pairs.split(",")] if a.pairs else \
        [(t, e) for t in a.tasks for e in a.episodes]
    journals = {t: RJ.load(a.env_seed, t) for t in sorted({t for t, _ in pairs})}
    items = [(t, e) for t, e in pairs if not (out / "episodes" / f"task{t}_ep{e:02d}.json").exists()]
    print(f"[run] {a.condition}: {len(items)} episodes to run", flush=True)
    if not items:
        return
    video_set = None if a.video in ("none", "all") else {tuple(map(int, s.split(":"))) for s in a.video.split(",")}
    K = a.envs_per_group

    ctx = mp.get_context("spawn")
    groups = [Group(ctx, g, K) for g in range(a.groups)]

    t_load = time.time()
    pol = P.Policy("cuda")
    n_overlay, overlay_sha = apply_overlay(pol.model, a.overlay)
    load_s = time.time() - t_load
    torch.cuda.reset_peak_memory_stats()
    print(f"[run] policy loaded in {load_s:.1f}s, overlay tensors {n_overlay}", flush=True)

    meta = dict(condition=a.condition, overlay=a.overlay, overlay_sha256=overlay_sha, overlay_tensors=n_overlay,
                env_seed=a.env_seed, policy_seed=P.POLICY_SEED, batch=1, groups=a.groups, envs_per_group=K,
                max_steps=a.max_steps, replan=P.REPLAN, num_denoise_steps=P.NUM_STEPS, capture=a.capture,
                lp_threads=os.environ.get("SIM_LP_THREADS", "4"), group_rss_cap_mb=a.group_rss_cap_mb,
                gpu=torch.cuda.get_device_name(), torch=torch.__version__, ckpt=P.CKPT)
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1))

    queue = list(items)
    ep = {}
    policy_times = []
    started = time.time()
    done_count = 0
    steps_total = 0
    recycles = 0
    max_rss = {}

    def assign(g, s):
        t, e = queue.pop(0)
        jr = journals[t]
        rec = jr["records"][e]
        record = a.video == "all" or (video_set is not None and (t, e) in video_set)
        s.item = (t, e)
        ep[(t, e)] = dict(steps=0, queries=0, plan=collections.deque(), success=False, record=record, t0=time.time(),
                          captures=[], group=g.gid, generation=g.generation, slot=s.idx)
        return [("task", s.idx, jr["meta"]["bddl"], jr["meta"]["language"]),
                ("reset", s.idx, rec["state"], rec["rng"], rec["reset_seed"], rec["fixtures"], rec["fingerprint"],
                 record)]

    def finish(g, s, reason):
        nonlocal done_count
        t, e = s.item
        st = ep.pop((t, e))
        res = dict(condition=a.condition, task=t, episode=e, env_seed=a.env_seed, policy_seed=P.POLICY_SEED,
                   success=st["success"], steps=st["steps"], queries=st["queries"], end=reason,
                   first_fingerprint=st["fp"], fingerprint_match=st["fp_ok"], fixtures=st["fixtures"],
                   language=journals[t]["meta"]["language"], seconds=round(time.time() - st["t0"], 2),
                   group=st["group"], generation=st["generation"], slot=st["slot"])
        if st["captures"]:
            caps = st["captures"]
            np.savez_compressed(out / "captures" / f"task{t}_ep{e:02d}.npz",
                                steps=np.array([c[0] for c in caps]), queries=np.array([c[1] for c in caps]),
                                image=np.stack([c[2]["observation/image"] for c in caps]),
                                wrist_image=np.stack([c[2]["observation/wrist_image"] for c in caps]),
                                state=np.stack([c[2]["observation/state"] for c in caps]))
            res["captures"] = len(caps)
        tmp = out / "episodes" / f".task{t}_ep{e:02d}.json.tmp"
        with open(tmp, "w") as f:  # durable: survives a hard VM restart (no 0-byte "finished" episodes)
            f.write(json.dumps(res, indent=1))
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(out / "episodes" / f"task{t}_ep{e:02d}.json")
        _dfd = os.open(str(out / "episodes"), os.O_RDONLY)
        os.fsync(_dfd)
        os.close(_dfd)
        done_count += 1
        s.item = None
        s.obs = None
        if st["record"]:
            tag = "success" if st["success"] else "fail"
            s.video = str(out / "frames" / f"task{t}_ep{e:02d}_{a.condition}_{tag}.npz")
        s.phase = "free"

    def decide(s):
        """Slot has a fresh observation: take the next planned action, calling the policy when the plan is empty."""
        st = ep[s.item]
        if not st["plan"]:
            t, e = s.item
            nz = torch.from_numpy(P.noise_array((P.POLICY_SEED, a.env_seed, t, e, st["queries"]))[None]).cuda()
            t0 = time.time()
            chunk = pol.actions([s.obs], nz)[0].cpu().numpy()  # (10, 7)
            policy_times.append(time.time() - t0)
            if a.capture:
                st["captures"].append((st["steps"], st["queries"], s.obs))
            st["queries"] += 1
            st["plan"].extend(chunk[:P.REPLAN])
        s.action = st["plan"].popleft()
        s.phase = "act"

    def dispatch(g):
        nonlocal recycles
        if g.inflight is not None:
            return False
        live = [s for s in g.slots if s.item is not None]
        if live and all(s.phase == "ready" for s in live):
            for s in live:
                decide(s)
        cmds, owners = [], []
        if g.draining and all(s.item is None and s.video is None for s in g.slots):
            g.recycle()
            recycles += 1
        for s in g.slots:
            if s.phase == "act":
                cmds.append(("step", s.idx, s.action)); owners.append((s, "step"))
                s.action = None
                s.phase = "wait"
            elif s.phase == "free" and s.video is not None:
                cmds.append(("video", s.idx, s.video, 20)); owners.append((s, "video"))
                s.video = None
                s.phase = "wait"
            elif s.phase == "free" and s.item is None and queue and not g.draining:
                for c in assign(g, s):
                    cmds.append(c); owners.append((s, c[0]))
                s.phase = "wait"
        if not cmds:
            return False
        g.conn.send(("run", cmds))
        g.inflight = owners
        return True

    def receive(g):
        nonlocal steps_total
        results, rss = g.conn.recv()
        g.rss = rss
        max_rss[g.gid] = max(max_rss.get(g.gid, 0), rss)
        owners, g.inflight = g.inflight, None
        for (s, op), msg in zip(owners, results):
            if op == "task":
                continue
            if op == "video":
                s.phase = "free"
                continue
            if op == "reset":
                _, obs, fp, ok, fixtures, secs = msg
                st = ep[s.item]
                st.update(fp=fp, fp_ok=ok, fixtures=fixtures)
                if not ok:
                    raise RuntimeError(f"first-frame fingerprint mismatch task{s.item[0]} ep{s.item[1]}")
                s.obs = obs
                s.phase = "ready"
            elif op == "step":
                _, obs, done = msg
                st = ep[s.item]
                st["steps"] += 1
                steps_total += 1
                if done:
                    st["success"] = True
                    finish(g, s, "success")
                elif st["steps"] >= a.max_steps:
                    finish(g, s, "max_steps")
                else:
                    s.obs = obs
                    s.phase = "ready"
        if rss > a.group_rss_cap_mb:
            g.draining = True

    last_log = time.time()
    while True:
        for g in groups:
            dispatch(g)
        busy = [g for g in groups if g.inflight is not None]
        if not busy:
            if not queue and not ep and all(s.video is None for g in groups for s in g.slots):
                break
            continue
        for conn in wait([g.conn for g in busy], timeout=5.0):
            g = next(x for x in groups if x.conn is conn)
            receive(g)
        if time.time() - last_log > 60:
            last_log = time.time()
            el = time.time() - started
            pt = np.array(policy_times[-300:]) if policy_times else np.zeros(1)
            print(f"[run] {a.condition} {done_count}/{len(items)} eps, {steps_total} steps, {steps_total / el:.1f} "
                  f"steps/s, policy {pt.mean() * 1000:.0f} ms/call, rss {[g.rss for g in groups]} MB, "
                  f"recycles {recycles}, queue {len(queue)}", flush=True)

    for g in groups:
        try:
            g.conn.send(("close",))
        except (BrokenPipeError, OSError):
            pass
    for g in groups:
        g.proc.join(60)
    el = time.time() - started
    summary = dict(meta, episodes=done_count, wall_seconds=el, policy_steps=steps_total,
                   policy_calls=len(policy_times),
                   policy_ms_mean=float(np.mean(policy_times) * 1000) if policy_times else None,
                   recycles=recycles, max_group_rss_mb=max_rss,
                   main_rss_mb=int(open("/proc/self/status").read().split("VmRSS:")[1].split()[0]) // 1024,
                   max_memory_allocated=torch.cuda.max_memory_allocated(),
                   max_memory_reserved=torch.cuda.max_memory_reserved(), load_seconds=load_s)
    (out / f"summary_{int(time.time())}.json").write_text(json.dumps(summary, indent=1))
    print("[run] finished", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
