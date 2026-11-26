"""Stage 3: index reassignment (second-order compensation with fixed codebooks and scales).

Follows reference_sources/adapters/gr00t_index_worker.py and the SmolVLA port's index.py: install the stage-1
quantized model (fixed 4-bit background + every candidate at its stage-1 state: vector codebooks, and for an
allocated model the fixed scalar GPTQ layers), hook every vector-codebook candidate with sacq.statistics.FullMoment
(FP32 uncentered input second moment, chunk rows 2048), run the policy on the 576 ordered training snapshots with
noise 7000 + position, and set H = total / rows. Then sacq.index_correction.compensate_packed(packed, original BF16
weight, H, block_columns=128, damping=0.1) per layer, unchanged repository code (every pi0.5 candidate has
64-column groups; only the indices change, asserted). Preset index settings: damping 0.1, block 128, noise base
7000, moment chunk rows 2048, TF32 allowed for matmuls (applied to the compensation math; the policy forwards keep the
evaluation numerics, as in the SmolVLA port).
pi0.5 specifics: padded prefix tokens are excluded from language-model moments; linears reading the same input share
one moment; the 1 GB language-model down_proj moments are collected in extra passes (prefix-only forwards, which
compute exactly the same language-model inputs).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sacqpi as S  # noqa: E402
import gptq_prep as GP  # noqa: E402
from gptq_prep import add_rows  # noqa: E402
import sacq.index_correction as IC  # noqa: E402
from sacq.statistics import FullMoment  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", type=Path, required=True, help="stage-1 packed.pt (vector-codebook candidates)")
    ap.add_argument("--fixed", type=Path, default=None, help="weights.pt of fixed scalar layers (allocated model)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--downs-per-pass", type=int, default=3)
    a = ap.parse_args()
    dev = torch.device("cuda")
    snaps = S.Snapshots()
    pol = S.load_policy()
    model = pol.model
    packed = torch.load(a.packed, map_location="cpu", weights_only=False)["packed"]
    names = list(packed)
    originals = {n: model.get_submodule(n).weight.detach().to("cpu").clone() for n in names}
    S.apply_background(model)
    if a.fixed is not None:
        S.apply_weights(model, a.fixed)
    S.load_overlay(model, {n + ".weight": S.decode_packed(packed[n], dev).to(torch.bfloat16) for n in names})
    ctx = GP.Ctx(model)
    mods = dict(model.named_modules())
    groups = [(k, [n for n in ns if n in packed], kind) for k, ns, kind in GP.input_groups(model) if kind != "vision"]
    groups = [g for g in groups if g[1]]
    passes, _ = GP.plan_passes(groups, model, a.downs_per_pass)
    print(f"[index] {len(names)} vector-codebook layers, {len(groups)} input groups, {len(passes)} passes", flush=True)
    new_packed, weights, report = {}, {}, {}
    t_all = time.time()
    train = snaps.train_positions()
    t_moments = 0.0
    for pi, pas in enumerate(passes):
        moments, hooks = {}, []
        for key, ns, kind in pas:
            moments[key] = FullMoment(mods[ns[0]].in_features, "cuda", 2048)
            hooks.append(mods[ns[0]].register_forward_pre_hook(
                lambda mod, args, key=key, kind=kind: add_rows(moments[key], ctx.rows(args[0], kind))))
        torch.backends.cuda.matmul.allow_tf32 = False  # policy forwards with the evaluation numerics
        torch.backends.cudnn.allow_tf32 = False
        prefix_only = all(kind == "lm" for _, _, kind in pas)
        t0 = time.time()
        with torch.no_grad():
            for k, pos in enumerate(train):
                for fm in moments.values():
                    fm.begin_request()
                inputs = pol.model_inputs(snaps.elements(pos))
                if prefix_only:
                    S.prefix_forward(pol, inputs)
                else:
                    pol.sample_normalized(inputs, snaps.noise(pos))
                if (k + 1) % 96 == 0:
                    print(f"[index] pass {pi} moments {k + 1}/{len(train)} {time.time() - t0:.0f}s "
                          f"max alloc {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
        for h in hooks:
            h.remove()
        t_moments += time.time() - t0
        torch.backends.cuda.matmul.allow_tf32 = True   # preset index_settings for the compensation math
        torch.backends.cudnn.allow_tf32 = True
        for key, ns, kind in pas:
            fm = moments.pop(key)
            Hm = fm.total.div_(fm.rows)
            for n in ns:
                q, rep = IC.compensate_packed(packed[n], originals[n].to(dev, torch.float32), Hm, checkpoint_dir=None,
                                              provenance=dict(layer=n), block_columns=128, damping=0.1)
                IC.assert_only_indices_changed(packed[n], q)
                new_packed[n] = q
                weights[n + ".weight"] = S.decode_packed(q, dev).to(torch.bfloat16).cpu()
                r = rep["roles"]["whole"]
                report[n] = dict(rows=fm.rows, calls=fm.calls, changed=r["changed_indices"], total=r["total_indices"],
                                 mse_before=rep["layer_output_mse_before"], mse_after=rep["layer_output_mse_after"])
            del fm, Hm
            torch.cuda.empty_cache()
        print(f"[index] pass {pi} done ({len(report)}/{len(names)} layers) {time.time() - t_all:.0f}s", flush=True)
    assert set(new_packed) == set(names)
    S.save_atomic(dict(packed=new_packed), a.out / "packed.pt")
    S.save_atomic(dict(weights=weights), a.out / "weights.pt")
    S.write_json(dict(report=report, seconds=dict(moments=t_moments, total=time.time() - t_all)), a.out / "index.json")
    ch = sum(r["changed"] for r in report.values()) / sum(r["total"] for r in report.values())
    imp = sum(r["mse_after"] < r["mse_before"] for r in report.values())
    print(f"[index] done in {time.time() - t_all:.0f}s; {ch:.1%} of indices changed; layer-output MSE improved in "
          f"{imp}/{len(report)} layers", flush=True)


if __name__ == "__main__":
    main()
