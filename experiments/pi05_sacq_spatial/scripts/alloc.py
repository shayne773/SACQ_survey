"""Stage 2: mixed-bit allocation by greedy coordinate descent, following the author's final allocator
reference_sources/allocation/coord_descent_final_multisuite_2026_08_24.py (helpers: ..._setting1_improved_2026_08_23.py,
..._floor_scope_persistent_2026_08_23.py), single suite.

Faithful parts:
- ladder (sentinel k): 4-bit scalar (90000), 3-bit scalar (80000), then vector codebooks K = 2048, 1024, 512, 256,
  128, 64, 32 (2.75 ... 1.25 index bits); K = 16 / 8 lie below the default floor min_bit = 1.25 and are never legal;
- every candidate starts at the 4-bit rung (start_k 90000; no candidate is ever BF16);
- per round: install the current state over the fixed background, measure the base proxy and, for every candidate
  whose next rung is legal under the active floor, the proxy after demoting it one rung (then restore);
  proxy = mean over requests of ||a - r|| / ||r|| on the decoded (unnormalized) action chunk of the request, float64,
  r = the original BF16 model's chunk for the same request and noise (7000 + request index);
- selection (scope_policy "hard"): among the scope ("vlm" = language model, "dit" = action expert incl. adaRMS) with
  the higher actual serialized bits per weight, take the legal move with the smallest delta / serialized bytes saved
  (ties by name); one move per step; the moved candidate's delta is dropped until the next refresh;
- refresh all deltas after 100 moves, when the floor changes, or when no measured delta remains;
- moving floor and checkpoints exactly as the author: nominal checkpoints every 0.10 bits from the start down to the
  target_nominal_bits (1.50); active floor = the largest ladder bit >= min_bit below the next checkpoint target;
  a checkpoint CSV is written when the nominal average crosses it;
- serialized bytes per layer exactly payload_bytes(): scalar 3/4-bit with the historical HQQ packing convention
  (4-bit: params/2 + 4 bytes per 64-group; 3-bit: ceil(groups/10)*64*4 + 4 per group), vector codebook:
  ceil(n_vectors*log2(K)/8) + K*4*2 (BF16 book) + 2 per 64-group (BF16 scale) + 2 (BF16 shift);
- stop when the candidates' total serialized bytes <= target_payload_bytes.
Deviations (see the report): single suite (Spatial) instead of four suite checkpoints; scalar rungs are GPTQ
candidates (upstream d3270b8 "New searches evaluate GPTQ candidates at both 3 and 4 bits"), prepared by gptq_prep.py;
no hand-picked hotspot list (the author's 11 "hot" VLM layers were pre-protected in an earlier ablation; the final
script treats them as ordinary candidates, which is what every pi0.5 candidate is here); the proxy requests are
proxy_snapshots (disjoint from the 640 calibration snapshots, never the 64 validation snapshots); the byte target is
the serialized size of the same candidates in the uniform K = 64 (1.5 index bits) model, so the allocated and the
uniform model have the same candidate storage.

Exact speed-ups (verified bitwise at startup against the plain full forward): the vision/language embeddings are
computed once (the background never changes); action-expert/adaRMS trials reuse the round's base prefix KV cache
(the author's cached-VLM trick); a language-model trial in layer i restarts the prefix pass at layer i from the
round's base hidden state, reusing the base K/V of layers < i.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers.cache_utils import DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sacqpi as S  # noqa: E402

BIT_OF_K = {90000: 4.0, 80000: 3.0, 2048: 2.75, 1024: 2.50, 512: 2.25, 256: 2.00, 128: 1.75, 64: 1.50, 32: 1.25,
            16: 1.00, 8: 0.75}
LADDER = sorted(BIT_OF_K, reverse=True)
SCALAR = {90000: 4, 80000: 3}
GROUP_SIZE, VECTOR_DIM = 64, 4


def payload_bytes(params: int, k: int) -> int:
    assert params % GROUP_SIZE == 0
    if k in SCALAR:
        groups = params // GROUP_SIZE
        metadata = groups * 4
        packed = params // 2 if SCALAR[k] == 4 else math.ceil(groups / 10) * GROUP_SIZE * 4
        return packed + metadata
    bits = math.ceil(math.log2(k))
    indices = ((params // VECTOR_DIM) * bits + 7) // 8
    return indices + k * VECTOR_DIM * 2 + (params // GROUP_SIZE) * 2 + 2


class Bank:
    def __init__(self, dev):
        self.dev = dev

    def weight(self, name, k):
        if k in SCALAR:
            w = torch.load(S.WORK / "bank" / f"gptq{SCALAR[k]}" / f"{name}.pt", map_location="cpu",
                           weights_only=False)["weight"]
            return w.to(self.dev, torch.bfloat16)
        p = torch.load(S.WORK / "bank" / f"k{k}" / f"{name}.pt", map_location="cpu", weights_only=False)["packed"]
        assert int(p["codebook"].shape[0]) == k
        return S.decode_packed(p, self.dev).to(torch.bfloat16)  # BF16 gather/mul/add, CUDA


@contextlib.contextmanager
def pinned_tensor_factory(keep, replay=False):
    """Route torch.tensor(..., device=cuda) through pinned host buffers kept alive, so a captured CUDA graph re-reads
    the same constants on every replay (as the SmolVLA port's graphpolicy.py). Pinned allocation is not allowed while
    a stream is capturing, so the buffers are created on a warmup pass (replay=False) and handed out again, in the
    same order and checked equal, during the capture (replay=True)."""
    orig = torch.tensor
    it = iter(list(keep)) if replay else None

    def tensor(data, *args, device=None, **kwargs):
        if device is not None and torch.device(device).type == "cuda":
            if replay:
                host = next(it)
                ref = orig(data, *args, **kwargs)
                assert ref.dtype == host.dtype and torch.equal(ref, host), "constant differs from warmup"
            else:
                host = orig(data, *args, **kwargs).pin_memory()
                keep.append(host)
            return host.to(device, non_blocking=True)
        return orig(data, *args, device=device, **kwargs)

    torch.tensor = tensor
    try:
        yield
    finally:
        torch.tensor = orig


class GraphDenoiser:
    """CUDA-graph replay of the 10 Euler denoising steps (PI0Pytorch.sample_actions after the prefix pass) for one
    request shape. Inputs (prefix K/V cache, state, prefix pad mask, noise) are copied into static buffers before each
    replay; weights are read from the live parameters (candidates are swapped in place with .copy_). The loop runs
    exactly NUM_STEPS iterations, which is what sample_actions' `while time >= -dt / 2` does for dt = -1/10.
    Bitwise equality with the eager loop is asserted at construction and on demand."""

    def __init__(self, model, cache, state, pad, noise):
        self.model = model
        self.kc = [k.clone() for k in cache.key_cache]
        self.vc = [v.clone() for v in cache.value_cache]
        self.cache = DynamicCache()
        self.cache.key_cache, self.cache.value_cache = self.kc, self.vc
        self.cache._seen_tokens = cache._seen_tokens
        self.state, self.pad, self.noise = state.clone(), pad.clone(), noise.clone()
        self.keep = []
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            self._loop()
            with pinned_tensor_factory(self.keep):
                self._loop()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with pinned_tensor_factory(self.keep, replay=True), torch.cuda.graph(self.graph):
            self.out = self._loop()

    def _loop(self):
        m = self.model
        dt = torch.tensor(-1.0 / S.P.NUM_STEPS, dtype=torch.float32, device="cuda")
        t = torch.tensor(1.0, dtype=torch.float32, device="cuda")
        x_t = self.noise
        b = x_t.shape[0]
        for _ in range(S.P.NUM_STEPS):
            v_t = m.denoise_step(self.state, self.pad, self.cache, x_t, t.expand(b))
            x_t = x_t + dt * v_t
            t += dt
        return x_t

    def run(self, cache, state, pad, noise):
        for dst, src in zip(self.kc, cache.key_cache):
            dst.copy_(src)
        for dst, src in zip(self.vc, cache.value_cache):
            dst.copy_(src)
        self.state.copy_(state)
        self.pad.copy_(pad)
        self.noise.copy_(noise)
        self.graph.replay()
        return self.out.clone()


class Proxy:
    """Cached, exact proxy evaluation of the pi0.5 policy on the proxy requests."""

    def __init__(self, pol, snaps, refs, positions):
        self.pol, self.model = pol, pol.model
        self.lm = self.model.paligemma_with_expert.paligemma.model.language_model
        self.snaps, self.positions = snaps, positions
        self.refs = {p: refs[p].double() for p in positions}
        self.req = {}
        cap = {}

        def grab(mod, args, kwargs):
            cap.update(kwargs)
        h = self.lm.layers[0].register_forward_pre_hook(grab, with_kwargs=True)
        with torch.no_grad():
            for p in positions:
                inputs = pol.model_inputs(snaps.elements(p))
                obs = pol.observation(inputs)
                images, img_masks, lang_tokens, lang_masks, state = self.model._preprocess_observation(obs, train=False)
                embs, pad, att = self.model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
                cap.clear()
                S.prefix_forward(pol, inputs)
                kw = {k: v for k, v in cap.items() if k not in ("past_key_value", "use_cache")}
                self.req[p] = dict(inputs=inputs, state=state, pad=pad, embs=embs.to(torch.bfloat16), kw=kw,
                                   noise=snaps.noise(p))
        h.remove()

    # ---------------------------------------------------------------- pieces
    def prefix_from(self, p, start, base_cache=None, h=None):
        """Run LM layers start..17 from hidden h (layer `start` input); K/V of layers < start come from base_cache."""
        r = self.req[p]
        cache = DynamicCache()
        if start:
            cache.key_cache = list(base_cache.key_cache[:start])
            cache.value_cache = list(base_cache.value_cache[:start])
            cache._seen_tokens = base_cache._seen_tokens
        x = r["embs"] if start == 0 else h
        for j in range(start, S.NLAYERS):
            x = self.lm.layers[j](x, past_key_value=cache, use_cache=True, **r["kw"])[0]
        return cache

    def enable_graph(self):
        p0 = self.positions[0]
        r = self.req[p0]
        cache = self.prefix_from(p0, 0)
        self.graph = None
        eager = self.denoise(p0, cache)
        self.graph = GraphDenoiser(self.model, cache, r["state"], r["pad"], r["noise"])
        for p in self.positions[:3]:
            c = self.prefix_from(p, 0)
            self.graph_off = True
            e = self.denoise(p, c)
            self.graph_off = False
            g = self.denoise(p, c)
            assert torch.equal(e, g), f"graph replay differs from eager on request {p}"
        assert torch.equal(eager, self.denoise(p0, cache))

    def denoise(self, p, cache):
        r = self.req[p]
        if getattr(self, "graph", None) is not None and not getattr(self, "graph_off", False):
            return self.graph.run(cache, r["state"], r["pad"], r["noise"])
        m = self.model
        dt = torch.tensor(-1.0 / S.P.NUM_STEPS, dtype=torch.float32, device="cuda")
        x_t = r["noise"]
        t = torch.tensor(1.0, dtype=torch.float32, device="cuda")
        bsize = x_t.shape[0]
        while t >= -dt / 2:
            v_t = m.denoise_step(r["state"], r["pad"], cache, x_t, t.expand(bsize))
            x_t = x_t + dt * v_t
            t += dt
        return x_t

    def err(self, p, chunk):
        a = self.pol.unnormalize(chunk).double().cpu()
        return float((a - self.refs[p]).norm() / self.refs[p].norm().clamp_min(1e-9))

    # ---------------------------------------------------------------- round API
    @torch.no_grad()
    def base(self):
        """Base caches for the current state; resets the layer sweep; returns the base proxy."""
        self.base_cache = {}
        errs = []
        for p in self.positions:
            cache = self.prefix_from(p, 0)
            self.base_cache[p] = cache
            errs.append(self.err(p, self.denoise(p, cache)))
        self.h = {p: self.req[p]["embs"] for p in self.positions}
        self.h_layer = 0
        return sum(errs) / len(errs)

    @torch.no_grad()
    def advance_to(self, layer):
        """Move the sweep's base hidden states to the input of `layer` (current weights must be the base state)."""
        assert layer >= self.h_layer
        while self.h_layer < layer:
            j = self.h_layer
            for p in self.positions:
                self.h[p] = self.lm.layers[j](self.h[p], past_key_value=DynamicCache(), use_cache=True,
                                              **self.req[p]["kw"])[0]
            self.h_layer += 1

    @torch.no_grad()
    def trial_expert(self):
        return sum(self.err(p, self.denoise(p, self.base_cache[p])) for p in self.positions) / len(self.positions)

    @torch.no_grad()
    def trial_lm(self, layer):
        assert self.h_layer == layer
        errs = []
        for p in self.positions:
            cache = self.prefix_from(p, layer, self.base_cache[p], self.h[p])
            errs.append(self.err(p, self.denoise(p, cache)))
        return sum(errs) / len(errs)

    @torch.no_grad()
    def full(self):
        """Plain full forward through PI0Pytorch.sample_actions (reference for the exactness check)."""
        errs = [self.err(p, self.pol.sample_normalized(self.req[p]["inputs"], self.req[p]["noise"]))
                for p in self.positions]
        return sum(errs) / len(errs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=S.WORK / "alloc")
    ap.add_argument("--requests-per-task", type=int, default=2)
    ap.add_argument("--target-nominal-bits", type=float, default=1.50)
    ap.add_argument("--target-payload-bytes", type=int, default=0, help="0: the uniform K=64 model's candidate bytes")
    ap.add_argument("--min-bit", type=float, default=1.25)
    ap.add_argument("--refresh-every", type=int, default=100)
    ap.add_argument("--checkpoint-step", type=float, default=0.10)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--subset", type=int, default=0, help="smoke test: first N candidates of each scope")
    ap.add_argument("--no-resume", action="store_true", help="ignore an existing out/progress.json")
    ap.add_argument("--graph", type=int, default=1, help="CUDA-graph replay of the denoising loop (bitwise-checked)")
    a = ap.parse_args()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    pol = S.load_policy()
    model = pol.model
    psnaps = S.Snapshots(S.WORK / "snapshots" / "proxy_snapshots.pt")
    prefs = torch.load(S.WORK / "snapshots" / "proxy_targets.pt", map_location="cpu", weights_only=False)["actions"]
    positions = [p for p in range(len(psnaps)) if p % 4 < a.requests_per_task]  # task-major, 4 per task
    cands = S.candidates()
    if a.subset:
        lm, ex = [c for c in cands if c[1] == "lm"], [c for c in cands if c[1] != "lm"]
        cands = lm[::len(lm) // a.subset][:a.subset] + ex[::len(ex) // a.subset][:a.subset]  # spread over layers
    names = [c[0] for c in cands]
    layer_of = {c[0]: c[2] for c in cands}
    owner = {c[0]: c[1] for c in cands}
    params = {n: model.get_submodule(n).weight.numel() for n in names}
    scopes = {"vlm": [n for n in names if S.scope_of(n) == "vlm"], "dit": [n for n in names if S.scope_of(n) == "dit"]}
    scope_params = {s: sum(params[n] for n in ns) for s, ns in scopes.items()}
    target_bytes = a.target_payload_bytes or sum(payload_bytes(params[n], 64) for n in names)
    S.apply_background(model)
    bank = Bank(dev)
    pmap = dict(model.named_parameters())

    def put(n, w):
        pmap[n + ".weight"].data.copy_(w.to(pmap[n + ".weight"].dtype))

    state = {n: 90000 for n in names}

    def install():
        for n in names:
            put(n, bank.weight(n, state[n]))

    def nominal(ns=names):
        return sum(params[n] * BIT_OF_K[state[n]] for n in ns) / sum(params[n] for n in ns)

    def actual(ns=names):
        return sum(payload_bytes(params[n], state[n]) * 8 for n in ns) / sum(params[n] for n in ns)

    def total_bytes():
        return sum(payload_bytes(params[n], state[n]) for n in names)

    def save(path):
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["layer", "params", "bit", "k", "scope", "payload_bytes", "scalar_method"])
            for n in sorted(names):
                w.writerow([n, params[n], f"{BIT_OF_K[state[n]]:.2f}", state[n], S.scope_of(n),
                            payload_bytes(params[n], state[n]), "GPTQ" if state[n] in SCALAR else ""])
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        S.fsync_dir(path.parent)

    checkpoints = []
    point = (math.ceil((nominal() - 1e-9) / a.checkpoint_step) - 1) * a.checkpoint_step
    while point > a.target_nominal_bits + 1e-9:
        checkpoints.append(round(point, 2))
        point -= a.checkpoint_step

    def floor_for(target):
        legal = [b for b in set(BIT_OF_K.values()) if a.min_bit - 1e-9 <= b < target - 1e-9]
        return max(legal) if legal else a.min_bit

    def next_target():
        return checkpoints[0] if checkpoints else a.target_nominal_bits

    floor = floor_for(next_target())
    install()
    t_setup = time.time()
    proxy = Proxy(pol, psnaps, prefs, positions)
    print(f"[alloc] candidates={len(names)} (vlm {len(scopes['vlm'])}, dit {len(scopes['dit'])}) params="
          f"{sum(params.values())} requests={len(positions)} target_bytes={target_bytes} "
          f"({target_bytes * 8 / sum(params.values()):.4f} bpw) start_bytes={total_bytes()} floor={floor:.2f} "
          f"checkpoints={checkpoints}", flush=True)

    # exactness check of the cached paths against the plain full forward
    b0 = proxy.base()
    f0 = proxy.full()
    assert b0 == f0, (b0, f0)
    tl = next(n for n in names if owner[n] == "lm" and layer_of[n] > 0)
    te = next(n for n in names if owner[n] != "lm")
    proxy.advance_to(layer_of[tl])  # base weights still installed
    trial_vals = {}
    for n, fn in ((tl, lambda: proxy.trial_lm(layer_of[tl])), (te, proxy.trial_expert)):
        cur = pmap[n + ".weight"].detach().clone()
        put(n, bank.weight(n, LADDER[LADDER.index(state[n]) + 1]))
        c, f = fn(), proxy.full()
        put(n, cur)
        assert c == f, (n, c, f)
        trial_vals[n] = c
    print(f"[alloc] exactness OK (base proxy {b0:.6f}); setup {time.time() - t_setup:.0f}s", flush=True)
    if a.graph:
        with torch.no_grad():
            proxy.enable_graph()
            assert proxy.base() == b0
            proxy.advance_to(layer_of[tl])
            for n, fn in ((tl, lambda: proxy.trial_lm(layer_of[tl])), (te, proxy.trial_expert)):
                cur = pmap[n + ".weight"].detach().clone()
                put(n, bank.weight(n, LADDER[LADDER.index(state[n]) + 1]))
                c = fn()
                put(n, cur)
                assert c == trial_vals[n], (n, c, trial_vals[n])
            tg = time.time()
            proxy.trial_expert()
            print(f"[alloc] CUDA-graph denoising enabled (base and trials bitwise equal to eager); expert trial "
                  f"{time.time() - tg:.2f}s", flush=True)

    def measure():
        t0 = time.time()
        base = proxy.base()
        deltas = {}
        order = sorted(names, key=lambda n: (owner[n] != "lm", layer_of[n]))  # LM layer-ascending, then expert
        for n in order:
            i = LADDER.index(state[n])
            if i + 1 >= len(LADDER) or BIT_OF_K[LADDER[i + 1]] < floor - 1e-12:
                continue
            if owner[n] == "lm":
                proxy.advance_to(layer_of[n])
            cur = pmap[n + ".weight"].detach().clone()
            put(n, bank.weight(n, LADDER[i + 1]))
            err = proxy.trial_lm(layer_of[n]) if owner[n] == "lm" else proxy.trial_expert()
            put(n, cur)
            deltas[n] = err - base
        return base, deltas, time.time() - t0

    history = []
    started = time.time()
    step, since = 0, 0
    moves = []
    prog = out / "progress.json"
    if prog.exists() and not a.no_resume:
        # Resume at the last refresh (progress.json is written right after each refresh measurement): restore the
        # state, re-cross the checkpoints already passed, and re-measure (deterministic: the same deltas).
        d = json.loads(prog.read_text())
        state.update({n: int(k) for n, k in d["state"].items()})
        history, moves = d["history"], d["moves"]
        step = len(moves)
        started -= d.get("elapsed", 0.0)
        install()
        while checkpoints and nominal() <= checkpoints[0] + 1e-12:
            checkpoints.pop(0)
        floor = floor_for(next_target())
        print(f"[alloc] resumed at step {step}: nominal={nominal():.5f} bytes={total_bytes()} floor={floor:.2f}",
              flush=True)
    base, deltas, secs = measure()
    if not history:
        history.append(dict(step=0, proxy=base, bytes=total_bytes(), nominal=nominal(), actual=actual(), floor=floor,
                            measured=len(deltas), seconds=secs))
    print(f"[alloc] round at step {step}: proxy={base:.6f} deltas={len(deltas)} ({secs:.0f}s)", flush=True)
    while (step < a.max_steps) if a.max_steps else (total_bytes() > target_bytes):
        scope_now = {s: actual(ns) for s, ns in scopes.items() if ns}
        available = {}
        for s, ns in scopes.items():
            rows = []
            for n in ns:
                if n not in deltas:
                    continue
                i = LADDER.index(state[n])
                if i + 1 >= len(LADDER) or BIT_OF_K[LADDER[i + 1]] < floor - 1e-12:
                    continue
                saved = payload_bytes(params[n], state[n]) - payload_bytes(params[n], LADDER[i + 1])
                assert saved > 0
                rows.append((deltas[n] / saved, n, LADDER[i + 1]))
            if rows:
                available[s] = rows
        if not available:
            raise RuntimeError(f"no legal move at bytes={total_bytes()} target={target_bytes}")
        sel = max(available, key=lambda s: (scope_now[s], s))
        cost, n, nk = min(available[sel], key=lambda r: (r[0], r[1]))
        old = state[n]
        state[n] = nk
        put(n, bank.weight(n, nk))
        del deltas[n]
        step += 1
        since += 1
        moves.append(dict(step=step, layer=n, k_from=old, k_to=nk, cost=cost, scope=sel))
        while checkpoints and nominal() <= checkpoints[0] + 1e-12:
            reached = checkpoints.pop(0)
            save(out / f"allocation_ckpt{reached:.2f}.csv")
            old_floor = floor
            floor = floor_for(next_target())
            print(f"[alloc] checkpoint={reached:.2f} step={step} nominal={nominal():.5f} actual={actual():.5f} "
                  f"bytes={total_bytes()} floor={old_floor:.2f}->{floor:.2f}", flush=True)
            if floor != old_floor:
                since = a.refresh_every
        if step % 50 == 0:
            print(f"[alloc] step={step} nominal={nominal():.5f} actual={actual():.5f} bytes={total_bytes()} "
                  f"scope=VLM:{actual(scopes['vlm']):.5f}/DiT:{actual(scopes['dit']):.5f} last={n}->"
                  f"{BIT_OF_K[nk]:.2f} cost={cost:.4e} elapsed={time.time() - started:.0f}s", flush=True)
        if since >= a.refresh_every or not deltas:
            base, deltas, secs = measure()
            since = 0
            history.append(dict(step=step, proxy=base, bytes=total_bytes(), nominal=nominal(), actual=actual(),
                                floor=floor, measured=len(deltas), seconds=secs))
            print(f"[alloc] refresh step={step} proxy={base:.6f} deltas={len(deltas)} ({secs:.0f}s) "
                  f"elapsed={time.time() - started:.0f}s", flush=True)
            S.write_json(dict(history=history, moves=moves, state=state, elapsed=time.time() - started), prog)
    final_proxy = proxy.base()
    save(out / "allocation.csv")
    counts = {}
    for k in state.values():
        counts[str(k)] = counts.get(str(k), 0) + 1
    S.write_json(dict(
        created_utc=datetime.now(timezone.utc).isoformat(), suites=["libero_spatial"], requests=positions,
        candidate_counts={s: len(ns) for s, ns in scopes.items()}, candidate_params=sum(params.values()),
        target_payload_bytes=target_bytes, final_payload_bytes=total_bytes(), final_nominal_bpw=nominal(),
        final_actual_bpw=actual(), scope_actual_bpw={s: actual(ns) for s, ns in scopes.items()},
        scope_nominal_bpw={s: nominal(ns) for s, ns in scopes.items()}, final_proxy=final_proxy,
        state_counts_by_k=counts, min_bit=a.min_bit, refresh_every=a.refresh_every,
        checkpoint_step=a.checkpoint_step, target_nominal_bits=a.target_nominal_bits, scope_policy="hard",
        scalar_method="GPTQ", history=history, moves=moves, steps=step, elapsed_seconds=time.time() - started),
        out / "allocation.manifest.json")
    print(f"[alloc] FINAL proxy={final_proxy:.6f} nominal={nominal():.5f} actual={actual():.5f} bytes={total_bytes()} "
          f"steps={step} counts={counts}", flush=True)


if __name__ == "__main__":
    main()
