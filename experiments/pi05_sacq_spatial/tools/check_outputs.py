"""Integrity check of outputs written after a given UTC time (for use after a hard VM restart).

Loads every file under sacq/, runs/ and reset/ modified after --since: .pt via torch.load (mmap), .json, .npy/.npz,
.pkl. Flags: zero-byte files, load failures, floating tensors that are entirely zero or contain NaN/Inf.
Leftover *.tmp files (from interrupted atomic saves) are listed separately; they are never read by the pipeline.
"""
import argparse, datetime, glob, json, math, os, pickle, sys

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--since", required=True, help="UTC, e.g. 2026-11-24T20:45")
ap.add_argument("--roots", default="sacq,runs,reset")
a = ap.parse_args()
since = datetime.datetime.fromisoformat(a.since).replace(tzinfo=datetime.timezone.utc).timestamp()

bad, tmps, n = [], [], 0


def tensors(obj, path=""):
    if isinstance(obj, torch.Tensor):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from tensors(v, f"{path}/{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from tensors(v, f"{path}[{i}]")


for root in a.roots.split(","):
    for f in sorted(glob.glob(f"{root}/**/*", recursive=True)):
        if not os.path.isfile(f) or os.path.getmtime(f) < since:
            continue
        if ".tmp" in os.path.basename(f):
            tmps.append(f)
            continue
        n += 1
        if os.path.getsize(f) == 0:
            bad.append((f, "zero bytes"))
            continue
        try:
            if f.endswith(".pt"):
                obj = torch.load(f, map_location="cpu", mmap=True, weights_only=False)
                for p, t in tensors(obj):
                    if t.is_floating_point() and t.numel() > 0:
                        tf = t.float()
                        if not torch.isfinite(tf).all():
                            bad.append((f, f"non-finite in {p}"))
                        elif t.numel() >= 16 and not tf.abs().amax() > 0:
                            bad.append((f, f"all-zero tensor {p} {tuple(t.shape)}"))
                del obj
            elif f.endswith(".json"):
                json.load(open(f))
            elif f.endswith(".npy"):
                np.load(f, allow_pickle=False)
            elif f.endswith(".npz"):
                z = np.load(f, allow_pickle=False)
                [z[k] for k in z.files]
            elif f.endswith(".pkl"):
                pickle.load(open(f, "rb"))
        except Exception as e:  # noqa: BLE001
            bad.append((f, f"load failed: {type(e).__name__}: {e}"))

print(f"checked {n} files modified after {a.since}Z")
for f in tmps:
    print(f"TMP  {f} ({os.path.getsize(f)} bytes)")
for f, why in bad:
    print(f"BAD  {f}: {why}")
print("OK" if not bad else f"{len(bad)} bad files")
sys.exit(1 if bad else 0)
