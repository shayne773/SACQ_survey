"""Reset-only reference journal per (environment seed, task), mirroring gr00t_libero/reset_inputs.py and the SmolVLA
port's reset_journal.py, with openpi's LIBERO environment construction (pisim.make_env, no reset at creation).

Fresh environment; Python and NumPy RNGs seeded with the environment seed (libero_eval.py seeds both at task start);
for each official episode 0..49 in order: record the RNG state, env.seed(seed) on episode 0 only (openpi seeds the
env once after creation), reset, record the natural fixture poses, set the official initial state, run the 10 no-op
settle steps, and record the first-frame fingerprint. No policy actions are executed. Output: reset/seed{S}/task{T}.pkl
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import picore as P  # noqa: E402
import pisim as W  # noqa: E402


def build(args):
    seed, task_id, out = args
    meta, states = P.task_meta(task_id)
    t0 = time.time()
    env = W.make_env(meta["bddl"])
    random.seed(seed)
    np.random.seed(seed)
    records = []
    for e in range(50):
        rng = (random.getstate(), np.random.get_state())
        raw, natural = W.reset_episode(env, states[e], rng, seed if e == 0 else None, poses=None)
        records.append(dict(episode=e, rng=rng, reset_seed=seed if e == 0 else None, state=states[e].copy(),
                            state_sha256=hashlib.sha256(np.ascontiguousarray(states[e]).tobytes()).hexdigest(),
                            fixtures=natural, fingerprint=W.fingerprint(raw)))
    env.close()
    payload = dict(meta=meta, seed=seed, records=records)
    path = Path(out) / f"seed{seed}" / f"task{task_id}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(payload, protocol=5))
    tmp.replace(path)
    return task_id, time.time() - t0, len(set(r["fingerprint"] for r in records)), W.rss_mb()


def load(seed, task_id):
    return pickle.loads((P.LAB / "reset" / f"seed{seed}" / f"task{task_id}.pkl").read_bytes())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--tasks", type=int, nargs="*", default=list(range(10)))
    p.add_argument("--procs", type=int, default=4)
    a = p.parse_args()
    out = P.LAB / "reset"
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(a.procs, len(a.tasks)), maxtasksperchild=1) as pool:
        for task_id, secs, distinct, rss in pool.imap_unordered(build, [(a.seed, t, str(out)) for t in a.tasks]):
            print(json.dumps(dict(task=task_id, seconds=round(secs, 1), distinct_fingerprints=distinct, rss_mb=rss)),
                  flush=True)


if __name__ == "__main__":
    main()
