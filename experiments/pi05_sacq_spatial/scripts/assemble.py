"""Assemble a stage-1 model from the fitted bank.

  uniform : every one of the 284 candidates at K = 64 (6-bit index per 4-weight vector = 1.5 index bits/weight)
            -> <out>/packed.pt (packed_vector_codebook_v1 per layer) + <out>/weights.pt (dequantized BF16)
  alloc   : the stage-2 allocation CSV: vector-codebook layers from bank/k{K} -> <out>/packed.pt; scalar GPTQ 3/4-bit
            layers from bank/gptq{3,4} -> <out>/fixed.pt (reused byte-for-byte in stages 3 and 4, as upstream);
            <out>/weights.pt holds every candidate's dequantized weight (the stage-1+2 model).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sacqpi as S  # noqa: E402

SCALAR = {90000: 4, 80000: 3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["uniform", "alloc"])
    ap.add_argument("--csv", type=Path, default=S.WORK / "alloc" / "allocation.csv")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    names = S.candidate_names()
    if a.mode == "uniform":
        ks = {n: S.UNIFORM_K for n in names}
    else:
        with open(a.csv) as f:
            ks = {r["layer"]: int(r["k"]) for r in csv.DictReader(f)}
        assert sorted(ks) == sorted(names), "allocation CSV does not cover exactly the 284 candidates"
    packed, fixed, weights = {}, {}, {}
    for n in names:
        k = ks[n]
        if k in SCALAR:
            w = torch.load(S.WORK / "bank" / f"gptq{SCALAR[k]}" / f"{n}.pt", map_location="cpu",
                           weights_only=False)["weight"].to(torch.bfloat16)
            fixed[n + ".weight"] = w
            weights[n + ".weight"] = w
        else:
            p = torch.load(S.WORK / "bank" / f"k{k}" / f"{n}.pt", map_location="cpu", weights_only=False)["packed"]
            assert int(p["codebook"].shape[0]) == k, (n, k)
            packed[n] = p
            weights[n + ".weight"] = S.decode_packed(p, "cuda").to(torch.bfloat16).cpu()
    a.out.mkdir(parents=True, exist_ok=True)
    S.save_atomic(dict(packed=packed, ks=ks), a.out / "packed.pt")
    S.save_atomic(dict(weights=weights), a.out / "weights.pt")
    if fixed:
        S.save_atomic(dict(weights=fixed), a.out / "fixed.pt")
    counts = {}
    for k in ks.values():
        counts[k] = counts.get(k, 0) + 1
    print(f"[assemble] {a.mode}: {len(packed)} vector-codebook layers, {len(fixed)} scalar layers; counts by k "
          f"{dict(sorted(counts.items(), reverse=True))}", flush=True)


if __name__ == "__main__":
    main()
