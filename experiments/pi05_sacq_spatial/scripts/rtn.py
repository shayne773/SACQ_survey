"""Round-to-nearest references for the 284 candidates (same layers and the same 64-column scale groups as SACQ):
asymmetric min/max per (row, 64-group), the arithmetic of gr00t_libero/gptq_core.py's RowGroupQuantizer without
GPTQ updates (as in the SmolVLA port). Output: rtn/rtn{2,3}.pt (dequantized BF16 weights)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sacqpi as S  # noqa: E402
from fit import load_original_weights  # noqa: E402


@torch.no_grad()
def rtn(weight, bits, g=64):
    w = weight.detach().to("cuda", torch.float32)
    o, n = w.shape
    x = w.reshape(o, n // g, g)
    maxq = (1 << bits) - 1
    zeros = torch.zeros_like(x[..., 0])
    xmin = torch.minimum(x.amin(dim=2), zeros)
    xmax = torch.maximum(x.amax(dim=2), zeros)
    dead = (xmin == 0) & (xmax == 0)
    xmin = torch.where(dead, -torch.ones_like(xmin), xmin)
    xmax = torch.where(dead, torch.ones_like(xmax), xmax)
    scale = ((xmax - xmin) / maxq).unsqueeze(-1)
    zero = torch.round(-xmin.unsqueeze(-1) / scale)
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return (scale * (q - zero)).reshape(o, n).to(torch.bfloat16).cpu()


def main():
    names = S.candidate_names()
    orig = load_original_weights([n + ".weight" for n in names])
    for bits in (2, 3):
        weights = {k: rtn(v, bits) for k, v in orig.items()}
        rel = sum(float((weights[k].float() - v.float()).norm() ** 2) for k, v in orig.items()) ** 0.5 / \
            sum(float(v.float().norm() ** 2) for v in orig.values()) ** 0.5
        out = S.WORK / "rtn" / f"rtn{bits}.pt"
        if out.exists():  # deterministic; skip the 4.6 GB rewrite after a restart (saves are fsynced + atomic)
            print(f"[rtn] {bits}-bit: {out} exists, not rewritten", flush=True)
        else:
            S.save_atomic(dict(weights=weights, recipe=f"RTN {bits}-bit asymmetric min/max, group 64"), out)
        print(f"[rtn] {bits}-bit: {len(weights)} layers, overall weight rel err {rel:.4f}", flush=True)


if __name__ == "__main__":
    main()
