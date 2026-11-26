"""Stage 1 (structure-aware codebook fitting) for the 284 pi0.5 candidates, at every vector-codebook rung of the
allocation ladder (the "bank"): K = 2048, 1024, 512, 256, 128, 64, 32 (2.75 ... 1.25 index bits/weight; the author's
allocator floor is 1.25). K = 64 alone is the uniform 1.5-bit stage-1 model. Adapted from the SmolVLA port's fit.py.

sacq.vector_quantization.quantize_vector_codebook (this repository, unchanged) with the role metrics of stats.py.
Settings follow gr00t_libero/presets/gr00t_libero_sacq2_seed17.json:
  language model (preset "vlm"): sample 50,000 vectors, 8 k-means iterations, seed 20260713 + the per-role offsets of
      reference_sources/metric_v2/vlm_fitter.py (q/k: layer*17 + {0,1009}; gate/up/down: layer*53 + {0,1009,2017};
      v/o: layer*67 + {3011,4021}); the same seeds at every K, as the author's bank (preset_stages.py: --seed 20260713
      for every vlm_k{k}).
  action expert incl. adaRMS dense (preset "dit"): sample 200,000 vectors, 12 iterations, seed
      20260601 + crc32(name) % 9973 (dit_fitter.py), at every K.
  group size 64, vector_dim 4, no rotation, no scale^2 weighting, return_packed=True.
Output: bank/k{K}/<name>.pt = {"packed": packed_vector_codebook_v1}, and bank/k{K}/fit.json (layer-output errors).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sacqpi as S  # noqa: E402
import sacq.vector_quantization as VQ  # noqa: E402
from sacq.vector_quantization import quantize_vector_codebook  # noqa: E402
from kmeans_fast import weighted_kmeans_full_fast  # noqa: E402

REPO_KMEANS_FULL = VQ.weighted_kmeans_full

LADDER = (64, 2048, 1024, 512, 256, 128, 32)  # K=64 first: the uniform pipeline needs it earliest


def expand_blocks(blocks):
    """(n/4, 4, 4) diagonal blocks -> (n, n) matrix with zeros elsewhere (exactly what qk_full reads)."""
    nb = blocks.shape[0]
    n = nb * 4
    M = torch.zeros(n, n, device=blocks.device, dtype=torch.float32)
    idx = torch.arange(n, device=blocks.device).reshape(nb, 4)
    M[idx[:, :, None], idx[:, None, :]] = blocks
    return M


class Metrics:
    def __init__(self, stats, weights, dev):
        self.s, self.w, self.dev = stats, weights, dev

    def g(self, key):
        v = self.s[key].to(self.dev, torch.float32)
        return expand_blocks(v) if v.ndim == 3 else v

    def W(self, name):
        key = name + ".weight"
        w = self.w[key] if isinstance(self.w, dict) else self.w.get_tensor(key)  # lazy: safetensors handle
        return w.to(self.dev, torch.float32)

    def role(self, name, owner, i, role):
        g = self.g
        if owner == "ada":
            key = "final" if role == "final.dense" else f"exp{i}.{role.split('.')[0]}"
            G = g(("final.dense" if role == "final.dense" else f"exp{i}.{role}") + ".G")
            n2 = g(f"exp{i}.{'final' if role == 'final.dense' else role.split('.')[0]}.n2")
            if role == "final.dense":
                wout = self.W("action_out_proj").square().sum(dim=0)
                oi = torch.cat([wout * n2, wout, torch.full_like(n2, 1e-12)])
            else:
                y2 = g(f"exp{i}.{'attn_y2' if role.startswith('input') else 'mlp_y2'}")
                oi = torch.cat([n2, torch.ones_like(n2), y2])
            del key
            return dict(metric="qk_full", input_metric=G, output_importance=oi.clamp_min(1e-12))
        pre = f"{owner}{i}"
        prefix = S.LM if owner == "lm" else S.EXP
        gate_attn = g(f"exp{i}.input_layernorm.gate2") if owner == "exp" else None
        gate_mlp = g(f"exp{i}.post_attention_layernorm.gate2") if owner == "exp" else None
        if role == "q_proj":
            return dict(metric="qk_full", input_metric=g(f"{pre}.q_proj.G"), output_importance=g(f"{pre}.q_imp"))
        if role == "k_proj":
            return dict(metric="qk_full", input_metric=g(f"{pre}.k_proj.G"), output_importance=g(f"{pre}.k_imp"))
        if role == "v_proj":
            wo = self.W(S.module_name(prefix, i, "o_proj")).square()       # (out, 8*256)
            if gate_attn is not None:
                wo = wo * gate_attn[:, None]
            v_out = wo.sum(dim=0).reshape(8, 256).sum(dim=0)
            return dict(metric="qk_full", input_metric=g(f"{pre}.Vmet"), output_importance=v_out)
        if role == "o_proj":
            n = self.W(name).shape[0]
            oi = gate_attn if gate_attn is not None else torch.ones(n, device=self.dev)
            return dict(metric="qk_full", input_metric=g(f"{pre}.o_proj.G"), output_importance=oi)
        wd = self.W(S.module_name(prefix, i, "down_proj"))
        wd2 = wd.square() * (gate_mlp[:, None] if gate_mlp is not None else 1.0)
        down_energy = wd2.sum(dim=0).unsqueeze(1)   # apply_swiglu_down_diag_curvature (vlm_metric_v2 = diag)
        if role == "gate_proj":
            return dict(metric="qk_diag", scalar_importance=g(f"{pre}.gate_imp") * down_energy)
        if role == "up_proj":
            return dict(metric="qk_diag", scalar_importance=g(f"{pre}.up_imp") * down_energy)
        if role == "down_proj":
            oi = gate_mlp if gate_mlp is not None else torch.ones(wd.shape[0], device=self.dev)
            return dict(metric="qk_full", input_metric=g(f"{pre}.ztz"), output_importance=oi)
        raise ValueError(role)


def seed_for(owner, i, role, name):
    if owner == "lm":
        off = {"q_proj": i * 17, "k_proj": i * 17 + 1009, "gate_proj": i * 53, "up_proj": i * 53 + 1009,
               "down_proj": i * 53 + 2017, "v_proj": i * 67 + 3011, "o_proj": i * 67 + 4021}[role]
        return 20260713 + off
    return 20260601 + zlib.crc32(name.encode()) % 9973


def load_original_weights(names):
    from safetensors import safe_open
    out = {}
    with safe_open(os.path.join(S.P.CKPT, "model.safetensors"), framework="pt", device="cpu") as f:
        for n in names:
            out[n] = f.get_tensor(n)
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default=",".join(map(str, LADDER)))
    ap.add_argument("--layers", type=int, default=0, help="debug: first N candidates only")
    ap.add_argument("--alloc-subset", type=int, default=0, help="smoke: the candidates of alloc.py --subset N")
    ap.add_argument("--fast-kmeans-above", type=int, default=64,
                    help="use the vectorized (mathematically identical) qk_full k-means update for K above this")
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    stats = torch.load(S.WORK / "stats" / "stats.pt", map_location="cpu", weights_only=False, mmap=True)
    cands = S.candidates()[: a.layers or None]
    if a.alloc_subset:  # same selection as alloc.py --subset
        lm, ex = [c for c in cands if c[1] == "lm"], [c for c in cands if c[1] != "lm"]
        cands = lm[::len(lm) // a.alloc_subset][:a.alloc_subset] + ex[::len(ex) // a.alloc_subset][:a.alloc_subset]
    from safetensors import safe_open
    weights = safe_open(os.path.join(S.P.CKPT, "model.safetensors"), framework="pt", device="cpu")  # lazy reads
    M = Metrics(stats, weights, dev)
    for K in [int(k) for k in a.ks.split(",")]:
        out = S.WORK / "bank" / f"k{K}"
        out.mkdir(parents=True, exist_ok=True)
        fast = K > a.fast_kmeans_above
        VQ.weighted_kmeans_full = weighted_kmeans_full_fast if fast else REPO_KMEANS_FULL
        print(f"[fit] K={K}: qk_full k-means update = {'vectorized' if fast else 'repository loop'}", flush=True)
        rep_path = out / "fit.json"
        report = json.loads(rep_path.read_text()) if rep_path.exists() else {}
        t0 = time.time()
        for (name, owner, i, role) in cands:
            dst = out / f"{name}.pt"
            if dst.exists() and name in report:
                continue
            w = M.W(name)
            kw = M.role(name, owner, i, role)
            sample, iters = (50000, 8) if owner == "lm" else (200000, 12)
            t = time.time()
            wq, packed = quantize_vector_codebook(
                w, group_size=S.GROUP, vector_dim=S.VECTOR_DIM, codebook_size=K, sample_size=sample, iters=iters,
                seed=seed_for(owner, i, role, name), return_packed=True, scalar_importance=kw.get("scalar_importance"),
                input_metric=kw.get("input_metric"), output_importance=kw.get("output_importance"), metric=kw["metric"])
            dec = S.decode_packed(packed, dev)
            assert float((dec - wq).norm() / wq.norm()) < 0.02  # packed BF16 book/scales vs FP32 fit output
            assert int(packed["group_size"]) == S.GROUP
            S.save_atomic(dict(packed=packed), dst)
            report[name] = dict(metric=kw["metric"], rel_err=float((dec - w).norm() / w.norm()), fast_kmeans=fast,
                                seconds=round(time.time() - t, 2), shape=list(w.shape), owner=owner)
            del kw, wq, dec
            if len(report) % 20 == 0:
                S.write_json(report, rep_path)
                print(f"[fit] K={K} {len(report)}/{len(cands)} {time.time() - t0:.0f}s (last {name}: "
                      f"{report[name]['seconds']}s, rel {report[name]['rel_err']:.4f})", flush=True)
        S.write_json(report, rep_path)
        print(f"[fit] K={K} done: {len(report)} layers, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
