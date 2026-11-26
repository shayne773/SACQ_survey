"""Select the calibration snapshots (and the stage-2 proxy requests) from the seed-1234 original-model rollout captures,
and record the original model's action targets. Adapted from the SmolVLA port's prep_snapshots.py.

Captures: the full pi0.5 policy input (both 224x224 camera images, the 8-D state, the task instruction) at every
policy call of every seed-1234 episode, i.e. every 5 environment steps (openpi replans every 5 actions; the author's
GR00T capture recorded policy requests, one per 8-step replanning call).

Selection (author: 640 requests, selection seed 0, five observations per request, every tenth position validation):
per task, shuffle that task's captures with seed (0, task); the first 320 = 64 snapshots x 5 observations go to the
calibration set (balanced: 64 per task), the next 20 = 4 snapshots x 5 to the stage-2 allocation proxy set (disjoint
from every calibration snapshot). The 640 calibration snapshots are shuffled together with seed 0 -> positions
0..639; position % 10 == 0 -> validation (64), otherwise training (576); fitting subset = first 64 training
positions. The 40 proxy snapshots are ordered task-major (task t -> positions 4t..4t+3).

Targets: original pi0.5 (all BF16, no quantized background), each snapshot as one batch of 5 with noise seeded by
(7000 + position, observation index); the full unnormalized (10, 7) chunk per observation is stored.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sacqpi as S  # noqa: E402
import reset_journal as RJ  # noqa: E402

PER = 5
PER_TASK = 64
PROXY_PER_TASK = 4
KEYS = ("image", "wrist_image", "state")


def select(run_dir: Path, out: Path):
    files = {}
    for f in sorted(glob.glob(str(run_dir / "captures" / "task*_ep*.npz"))):
        name = Path(f).stem
        files[(int(name[4:name.index("_")]), int(name.split("_ep")[1]))] = f
    pools = {t: [] for t in range(10)}
    for (t, e), f in sorted(files.items()):
        with np.load(f) as z:
            steps = z["steps"]
        for i, s in enumerate(steps):
            pools[t].append((e, int(s), i))
    cal, proxy = [], []
    for t in range(10):
        pool = pools[t]
        need = (PER_TASK + PROXY_PER_TASK) * PER
        if len(pool) < need:
            raise RuntimeError(f"task {t}: {len(pool)} captures < {need}")
        order = np.random.default_rng([0, t]).permutation(len(pool))
        chosen = [pool[k] for k in order[:need]]
        for j in range(PER_TASK):
            cal.append((t, chosen[j * PER:(j + 1) * PER]))
        for j in range(PER_TASK, PER_TASK + PROXY_PER_TASK):
            proxy.append((t, chosen[j * PER:(j + 1) * PER]))
    perm = np.random.default_rng(0).permutation(len(cal))
    cal = [cal[k] for k in perm]
    langs = {t: RJ.load(1234, t)["meta"]["language"] for t in range(10)}
    meta_common = dict(env_seed=1234, capture="every policy call (every 5 env steps)", observations_per_snapshot=PER,
                       selection_seed=0, pool_sizes={t: len(pools[t]) for t in range(10)}, episodes_used=len(files))
    for tag, snaps, extra in (
            ("", cal, dict(per_task=PER_TASK, split_rule="position % 10 == 0: validation; otherwise train",
                           fit_subset="first 64 training positions")),
            ("proxy_", proxy, dict(per_task=PROXY_PER_TASK, role="stage-2 allocation proxy requests; disjoint from "
                                   "all 640 calibration snapshots; ordered task-major"))):
        want, records = {}, []
        for pos, (t, members) in enumerate(snaps):
            for j, (e, s, i) in enumerate(members):
                want.setdefault((t, e), []).append((pos * PER + j, i))
            split = ("validation" if pos % 10 == 0 else "train") if not tag else "proxy"
            records.append(dict(position=pos, task=t, language=langs[t], split=split,
                                members=[dict(task=t, episode=e, step=s) for (e, s, i) in members]))
        nrows = len(snaps) * PER
        obs = None
        for key, rows in want.items():
            with np.load(files[key]) as z:
                arrs = {k: z[k] for k in KEYS}
            if obs is None:
                obs = {k: np.empty((nrows,) + v.shape[1:], v.dtype) for k, v in arrs.items()}
            for row, i in rows:
                for k in KEYS:
                    obs[k][row] = arrs[k][i]
            del arrs
        meta = dict(meta_common, snapshots=len(records), **extra)
        out.mkdir(parents=True, exist_ok=True)
        for k, v in obs.items():
            np.save(out / f"{tag}obs_{k}.npy", v)
        S.save_atomic(dict(meta=meta, records=records, obs_keys=list(obs), prefix=tag), out / f"{tag}snapshots.pt")
        S.write_json(dict(meta=meta, records=records), out / f"{tag}snapshots.json")
        print(tag or "calibration", json.dumps(meta), flush=True)


@torch.no_grad()
def targets(out: Path, tag: str):
    snaps = S.Snapshots(out / f"{tag}snapshots.pt")
    pol = S.load_policy()
    t0 = time.time()
    chunks = torch.stack([S.actions(pol, snaps, pos).cpu() for pos in range(len(snaps))])
    for pos in sorted({0, 1, len(snaps) // 2}):  # determinism check
        assert torch.equal(S.actions(pol, snaps, pos).cpu(), chunks[pos]), pos
    S.save_atomic(dict(actions=chunks, model="original BF16 pi05_libero (openpi PyTorch conversion)",
                       noise="fixed_noise((7000 + position, obs_index))", seconds=time.time() - t0),
                  out / f"{tag}targets.pt")
    print(f"{tag}targets {tuple(chunks.shape)} in {time.time() - t0:.0f}s; recompute bitwise OK", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=S.LAB / "runs" / "calib1234")
    ap.add_argument("--out", type=Path, default=S.WORK / "snapshots")
    ap.add_argument("--step", choices=["select", "targets", "all"], default="all")
    a = ap.parse_args()
    if a.step in ("select", "all"):
        select(a.run, a.out)
    if a.step in ("targets", "all"):
        targets(a.out, "")
        targets(a.out, "proxy_")
