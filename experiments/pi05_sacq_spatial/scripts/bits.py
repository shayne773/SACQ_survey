"""Storage accounting: effective bits per weight of the 284 candidates and whole-model size for every condition.

Serialized formats (the author's allocator payload_bytes() convention for the candidates):
  vector codebook (SACQ) : ceil(n_vectors * log2 K / 8) index bytes + K x 4 BF16 codebook + one BF16 scale per
                           (row, 64-group) + one BF16 shift
  scalar GPTQ/HQQ 4-bit  : params / 2 + 4 bytes per 64-group (BF16 scale + BF16 zero)
  scalar 3-bit           : the author's packing, ceil(groups / 10) * 64 * 4 bytes + 4 bytes per 64-group
  RTN b-bit              : b bits per weight densely packed + 4 bytes per 64-group (BF16 scale + BF16 zero)
  background             : the 163 vision/projector linears as scalar 4-bit above; the token embedding affine 4-bit
                           group 64 (4 bits + BF16 scale + BF16 minimum per group)
  everything else        : its run-time precision (BF16, or F32 for the set openpi keeps in F32).
The 5 dead layer-17 language-model linears are counted as removed (pi0.5 never consumes their outputs) in every
quantized condition; the original model is counted as loaded (expert lm_head dropped, as at inference).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sacqpi as S  # noqa: E402
from alloc import payload_bytes  # noqa: E402


def meta_model():
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    tc = S.P.train_config()
    with torch.device("meta"):
        model = PI0Pytorch(config=tc.model)
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    for n in S.P.FLOAT32_OUTER:
        getattr(model, n).float()
    model.paligemma_with_expert.gemma_expert.lm_head = None
    return model


def main():
    model = meta_model()
    params = dict(model.named_parameters())
    cands = S.candidate_names()
    cw = {n: params[n + ".weight"].numel() for n in cands}
    ncw = sum(cw.values())
    bg = S.background_linear_names(model)
    bgw = {n: params[n + ".weight"].numel() for n in bg}
    emb = params[S.EMBED + ".weight"]
    dead = {n + ".weight" for n in S.DEAD_LM}
    bg_bytes = sum(payload_bytes(w, 90000) for w in bgw.values())
    emb_bytes = emb.numel() // 2 + emb.shape[0] * (emb.shape[1] // 64) * 4
    quantized = {n + ".weight" for n in cands} | {n + ".weight" for n in bg} | {S.EMBED + ".weight"}
    rest_bytes = sum(p.numel() * p.element_size() for k, p in params.items() if k not in quantized | dead)
    orig_bytes = sum(p.numel() * p.element_size() for p in params.values())
    cand_orig_bytes = sum(params[n + ".weight"].numel() * params[n + ".weight"].element_size() for n in cands)

    def rtn_bytes(b):
        return sum(w * b // 8 + (w // 64) * 4 for w in cw.values())

    conds = {"uniform_k64": {n: 64 for n in cands}}
    alloc_csv = S.WORK / "alloc" / "allocation.csv"
    if alloc_csv.exists():
        with open(alloc_csv) as f:
            conds["allocated"] = {r["layer"]: int(r["k"]) for r in csv.DictReader(f)}
    cand_bytes = {name: sum(payload_bytes(cw[n], ks[n]) for n in cands) for name, ks in conds.items()}
    cand_bytes["rtn2"] = rtn_bytes(2)
    cand_bytes["rtn3"] = rtn_bytes(3)
    cand_bytes["bf16_original"] = cand_orig_bytes
    nominal = {"uniform_k64": 1.5, "rtn2": 2.0, "rtn3": 3.0, "bf16_original": cand_orig_bytes * 8 / ncw}
    if "allocated" in conds:
        bit_of = {90000: 4.0, 80000: 3.0, 2048: 2.75, 1024: 2.5, 512: 2.25, 256: 2.0, 128: 1.75, 64: 1.5, 32: 1.25}
        nominal["allocated"] = sum(cw[n] * bit_of[conds["allocated"][n]] for n in cands) / ncw
    gb = lambda b: b / 1e9  # noqa: E731
    out = dict(
        candidate_layers=len(cands), candidate_weights=ncw,
        candidate_bpw_effective={k: v * 8 / ncw for k, v in cand_bytes.items()},
        candidate_bpw_nominal=nominal,
        candidate_bytes=cand_bytes,
        background=dict(vision_projector_weights=sum(bgw.values()),
                        vision_projector_bpw=bg_bytes * 8 / sum(bgw.values()),
                        embedding_weights=emb.numel(), embedding_bpw=emb_bytes * 8 / emb.numel()),
        rest_bytes=rest_bytes, model_params=sum(p.numel() for p in params.values()),
        size_gb={"original": gb(orig_bytes), "bg_only": gb(cand_orig_bytes + bg_bytes + emb_bytes + rest_bytes),
                 **{k: gb(v + bg_bytes + emb_bytes + rest_bytes) for k, v in cand_bytes.items()
                    if k != "bf16_original"}},
        by_part_gb=dict(candidates_original=gb(cand_orig_bytes), background_quantized=gb(bg_bytes + emb_bytes),
                        rest=gb(rest_bytes)),
    )
    S.write_json(out, S.WORK / "bits.json")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
