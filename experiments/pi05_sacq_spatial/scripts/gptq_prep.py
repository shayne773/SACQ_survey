"""BF16-model input moments and scalar GPTQ: the fixed 4-bit background and the 3/4-bit allocation candidates.

Recipe (upstream gr00t_libero/GPTQ_ALLOCATION.md, d3270b8; also the SmolVLA port's background.py): untouched BF16
model, the 576 ordered training snapshots (every tenth position excluded), noise seed 7000 + position, group size
64, block size 128, damping 0.01, asymmetric, no act-order, TF32 enabled; FP32 uncentered input moments
(sacq.statistics.FullMoment, chunk rows 2048) collected in one pass per layer group, then
gptq_core.GPTQAccumulator.fasterquant on H = 2 * total / rows (the accumulator's convention).
  background : the 162 SigLIP encoder Linears + the multi-modal projector at 4 bits, the token embedding affine
               4-bit group 64 -> background/background.pt
  scalar bank: every one of the 284 SACQ candidates at 3 and 4 bits (upstream: "New searches evaluate GPTQ
               candidates at both 3 and 4 bits. The selected scalar weights are reused byte-for-byte in index
               correction and action calibration.") -> bank/gptq{3,4}/<name>.pt (dequantized BF16)
Rows: padded prefix tokens (masked everywhere in pi0.5) are excluded from language-model moments; the unused third
camera slot (pi05_libero feeds an all-zero right-wrist image with mask False) is excluded from vision moments.
Linears that read the same input tensor (q/k/v, gate/up, the two adaRMS dense of a layer) share one moment.
Memory: the 17 language-model down_proj moments are 1 GB each, so they are collected in extra passes.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sacqpi as S  # noqa: E402
import gptq_core  # noqa: E402
from sacq.statistics import FullMoment  # noqa: E402


def input_groups(model):
    """[(input_key, [linear names], kind)] with kind in {vision, lm, exp, cond}."""
    groups = []
    for n in S.background_linear_names(model):
        groups.append((n, [n], "vision"))
    # merge q/k/v of vision layers would also be exact, but keep per-linear keys for the background (small)
    cands = {c[0]: c for c in S.candidates()}
    for i in range(S.NLAYERS):
        for prefix, owner in ((S.LM, "lm"), (S.EXP, "exp")):
            qkv = [S.module_name(prefix, i, r) for r in ("q_proj", "k_proj", "v_proj")]
            qkv = [n for n in qkv if n in cands]
            groups.append((f"{owner}{i}.attn_in", qkv, owner))
            for key, roles in ((f"{owner}{i}.o_in", ["o_proj"]), (f"{owner}{i}.mlp_in", ["gate_proj", "up_proj"]),
                               (f"{owner}{i}.down_in", ["down_proj"])):
                names = [S.module_name(prefix, i, r) for r in roles if S.module_name(prefix, i, r) in cands]
                if names:
                    groups.append((key, names, owner))
        groups.append((f"exp{i}.cond", [S.module_name(S.EXP, i, r) for r in S.ADA], "cond"))
    groups.append(("final.cond", [S.EXP_FINAL_DENSE], "cond"))
    covered = [n for _, ns, k in groups if k != "vision" for n in ns]
    assert sorted(covered) == sorted(cands), (len(covered), len(cands))
    return groups


def plan_passes(groups, model, downs_per_pass=3):
    """Pass 0: every group except the 1 GB language-model down_proj moments (~5 GB); then those in chunks."""
    mods = dict(model.named_modules())
    size = {key: mods[ns[0]].in_features ** 2 * 4 / 1e9 for key, ns, _ in groups}
    big = [g for g in groups if g[2] == "lm" and g[0].endswith("down_in")]
    small = [g for g in groups if g not in big]
    passes = [small] + [big[i:i + downs_per_pass] for i in range(0, len(big), downs_per_pass)]
    return passes, size


def add_rows(fm, rows):
    """FullMoment.add, skipping an empty row set (the unused third camera slot contributes no valid image rows)."""
    if rows.shape[0]:
        fm.add(rows)


class Ctx:
    def __init__(self, model):
        self.prefix_valid = None
        self.img_masks = None
        self.img_idx = 0
        self.img_valid = None
        pwe = model.paligemma_with_expert
        orig_fwd, orig_embed, orig_prefix = pwe.forward, pwe.embed_image, model.embed_prefix
        ctx = self

        def fwd(*args, **kwargs):
            if kwargs["inputs_embeds"][0] is not None:
                ctx.prefix_valid = kwargs["attention_mask"][:, 0].diagonal(dim1=1, dim2=2) == 0
            return orig_fwd(*args, **kwargs)

        def embed_prefix(images, img_masks, *a, **k):
            ctx.img_masks, ctx.img_idx = img_masks, 0
            return orig_prefix(images, img_masks, *a, **k)

        def embed_image(img):
            ctx.img_valid = ctx.img_masks[ctx.img_idx]
            ctx.img_idx += 1
            return orig_embed(img)

        pwe.forward, pwe.embed_image, model.embed_prefix = fwd, embed_image, embed_prefix

    def rows(self, x, kind):
        if kind == "vision":
            x = x[self.img_valid]
            return x.reshape(-1, x.shape[-1])
        if kind == "lm":
            return x[self.prefix_valid]
        return x.reshape(-1, x.shape[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--downs-per-pass", type=int, default=3)
    ap.add_argument("--only", default="", help="comma list of pass indices to run (resume)")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True   # recipe: TF32 enabled
    torch.backends.cudnn.allow_tf32 = True
    snaps = S.Snapshots()
    pol = S.P.Policy("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    model = pol.model
    ctx = Ctx(model)
    mods = dict(model.named_modules())
    groups = input_groups(model)
    passes, size = plan_passes(groups, model, a.downs_per_pass)
    print(f"[gptq] {len(groups)} input groups in {len(passes)} passes: "
          f"{[round(sum(size[g[0]] for g in p), 2) for p in passes]} GB", flush=True)
    out_bank = S.WORK / "bank"
    bg_weights, report = {}, {}
    bg_path = S.WORK / "background" / "background.pt"
    t_all = time.time()
    train = snaps.train_positions()
    only = {int(x) for x in a.only.split(",")} if a.only else None
    for pi, pas in enumerate(passes):
        if only is not None and pi not in only:
            continue
        todo = [g for g in pas if not all((out_bank / "gptq4" / f"{n}.pt").exists() for n in g[1])
                or g[2] == "vision"]
        if not todo:
            print(f"[gptq] pass {pi}: all outputs exist", flush=True)
            continue
        moments, hooks = {}, []
        for key, names, kind in todo:
            m = mods[names[0]]
            moments[key] = FullMoment(m.in_features, "cuda", 2048)
            hooks.append(m.register_forward_pre_hook(
                lambda mod, args, key=key, kind=kind: add_rows(moments[key], ctx.rows(args[0], kind))))
        t0 = time.time()
        prefix_only = all(kind in ("lm", "vision") for _, _, kind in todo)
        with torch.no_grad():
            for k, pos in enumerate(train):
                for fm in moments.values():
                    fm.begin_request()
                if prefix_only:  # language-model / vision moments only need the prefix pass
                    S.prefix_forward(pol, pol.model_inputs(snaps.elements(pos)))
                else:
                    S.sample_norm(pol, snaps, pos)
                if (k + 1) % 96 == 0:
                    print(f"[gptq] pass {pi} moments {k + 1}/{len(train)} {time.time() - t0:.0f}s "
                          f"max alloc {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
        for h in hooks:
            h.remove()
        t_m = time.time() - t0
        for key, names, kind in todo:
            fm = moments.pop(key)
            Hm = fm.total.div_(fm.rows).mul_(2.0)  # in place; fasterquant only resets dead diagonal entries
            for n in names:
                m = mods[n]
                for bits in ((4,) if kind == "vision" else (4, 3)):
                    acc = gptq_core.GPTQAccumulator(m)
                    acc.H = Hm
                    acc.nsamples = fm.rows
                    t = time.time()
                    q, loss, damp = acc.fasterquant(bits=bits, groupsize=64, blocksize=128, percdamp=0.01,
                                                    symmetric=False)
                    w0 = m.weight.detach().float()
                    rel = float((q.float() - w0).norm() / w0.norm())
                    report[f"{n}@{bits}"] = dict(rows=fm.rows, calls=fm.calls, gptq_loss=loss, damp=damp, rel_err=rel,
                                                 seconds=round(time.time() - t, 2))
                    if kind == "vision":
                        bg_weights[n + ".weight"] = q.cpu()
                    else:
                        S.save_atomic(dict(weight=q.cpu(), bits=bits, rows=fm.rows, rel_err=rel,
                                           recipe="GPTQ group64 block128 damp0.01 asym, BF16 model, 576 train snapshots"),
                                      out_bank / f"gptq{bits}" / f"{n}.pt")
                    del acc
            del fm, Hm
            torch.cuda.empty_cache()
        print(f"[gptq] pass {pi} done: moments {t_m:.0f}s, total {time.time() - t0:.0f}s", flush=True)
        S.write_json(report, S.WORK / "bank" / f"gptq_report_pass{pi}.json")
    if bg_weights:
        emb = mods[S.EMBED]
        eq = S.affine_embedding_dequant(emb.weight, 4, 64)
        bg_weights[S.EMBED + ".weight"] = eq.cpu()
        report[S.EMBED] = dict(rel_err=float((eq.float() - emb.weight.float()).norm() / emb.weight.float().norm()))
        S.save_atomic(dict(weights=bg_weights, recipe=dict(
            vision="GPTQ 4-bit group 64 block 128 damping 0.01 asymmetric, 576 training snapshots, BF16 model",
            embedding="affine 4-bit group 64")), bg_path)
        print(f"[gptq] background saved: {len(bg_weights)} tensors", flush=True)
    print(f"[gptq] all done in {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
