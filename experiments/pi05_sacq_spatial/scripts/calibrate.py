"""Stage 4: action-level calibration of codebooks and row-group scales, indices frozen (adapted from the SmolVLA port).

Follows reference_sources/calibration/train_centroid_ft_2026_08_05.py with the preset "training" block: 576
training snapshots, 12 epochs, global batch 6 (96 updates/epoch, 1152 updates), AdamW (weight decay 0) with codebook
lr 1e-4 and scale lr 1e-5, FP32 codebook gradient accumulation, gradient-norm clip 1.0, epoch order shuffled by
random.Random(1); loss = MSE between the quantized policy's decoded (unnormalized) action chunk and the original
BF16 policy's chunk for the same snapshot and fixed noise (7000 + position); validation = mean relative error
||a - r|| / ||r|| over the 64 validation snapshots at step 0, every 100 updates and at the endpoint; the best
validation state is kept. Every vector-codebook candidate becomes a sacq.calibration_layers.QuantLinear (BF16
runtime reconstruction, FP32 master parameters) initialised from the stage-3 packed payloads; scalar GPTQ layers of
an allocated model stay fixed (upstream: "selected scalar weights are reused byte-for-byte in index correction and
action calibration").

Laptop memory adaptations (numerically neutral): frozen indices held as int32 instead of int64 (2.3 GB instead of
4.6 GB for pi0.5's 2.3B candidate weights); every language-model decoder layer is activation-checkpointed during
the prefix pass (its reconstructions are rebuilt in backward instead of being kept for all 18 layers), with a
KV cache that overwrites instead of appending when a checkpointed layer is recomputed; the language-model
QuantLinears therefore reconstruct per call (they run once per request), while the action expert keeps the
source's per-request reconstruction cache across its 10 denoising steps.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers.cache_utils import DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sacqpi as S  # noqa: E402
import sacq.calibration_layers as CL  # noqa: E402
from sacq.cache import clear_reconstruction_cache  # noqa: E402

CL.FP32_CODEBOOK_GRAD = True  # preset training.fp32_codebook_grad


class PiQuantLinear(CL.QuantLinear):
    """QuantLinear with int32 frozen indices, a `.weight` dtype probe (pi0.5 code reads layer.weight.dtype), and an
    optional no-cache mode for checkpointed language-model layers. F32 modules (adaRMS dense) receive the BF16-exact
    reconstruction cast to F32, as the dequantized overlay does."""

    def __init__(self, packed, bias, device, dtype, cache):
        super().__init__(packed, bias, device)
        self.idx = self.idx.to(torch.int32)
        self._dtype = dtype
        self._cache = cache
        if bias is not None:
            self.bias = bias.detach().to(device, dtype)
        self.register_buffer("_probe", torch.empty(0, dtype=dtype, device=device), persistent=False)

    @property
    def weight(self):
        return self._probe

    def _reconstruct(self):
        return self.weight_().to(torch.bfloat16).to(self._dtype)

    def prime(self):
        """Build this request's cached reconstruction (outside any checkpointed region)."""
        if self._cache and self._wc is None:
            self._wc = self._reconstruct()

    def forward(self, x):
        W = self._wc if self._cache else None
        if W is None:
            W = self._reconstruct()
            if self._cache:
                self._wc = W
        return nn.functional.linear(x, W, self.bias)


class OverwriteCache(DynamicCache):
    """DynamicCache whose update() replaces a layer's entry when it already exists (checkpoint recomputation)."""

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if len(self.key_cache) > layer_idx and self.key_cache[layer_idx].numel():
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
            return key_states, value_states
        return super().update(key_states, value_states, layer_idx, cache_kwargs)


def install(model, packed):
    mods = []
    for name, p in packed.items():
        parent_name, child = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        old = getattr(parent, child)
        q = PiQuantLinear(p, old.bias, "cuda", old.weight.dtype, cache=not name.startswith(S.LM))
        setattr(parent, child, q)
        del old
        mods.append((name, q))
    torch.cuda.empty_cache()
    return mods


def enable_prefix_checkpointing(model):
    """Checkpoint every language-model decoder layer when grad is enabled, and give the prefix pass an
    OverwriteCache (PaliGemmaWithExpertModel.forward passes past_key_values through to GemmaModel)."""
    lm = model.paligemma_with_expert.paligemma.model.language_model
    for layer in lm.layers:
        orig = layer.forward

        def fwd(*args, _orig=orig, **kwargs):
            if torch.is_grad_enabled():
                return checkpoint(_orig, *args, use_reentrant=False, **kwargs)
            return _orig(*args, **kwargs)
        layer.forward = fwd
    pwe = model.paligemma_with_expert
    orig_fwd = pwe.forward

    def pwe_fwd(*args, **kwargs):
        if kwargs.get("inputs_embeds", [None])[0] is not None and kwargs.get("past_key_values") is None:
            kwargs["past_key_values"] = OverwriteCache()
        return orig_fwd(*args, **kwargs)
    pwe.forward = pwe_fwd


def enable_expert_step_checkpointing(model, mods):
    """Checkpoint each denoising step (PI0Pytorch.denoise_step) when grad is enabled.

    Without it the eager expert attention keeps, for every layer and step, the prefix K/V repeated to 8 heads
    (2 x 5 x 8 x 978 x 256 BF16 = 40 MB per layer-step, 7.2 GB for 18 layers x 10 steps). The expert's per-request
    reconstruction cache is primed *before* the checkpointed call, so the recomputation reuses the same
    reconstructed-W graph nodes (gradients still accumulate across the 10 steps exactly as in the source).
    Non-grad tensor arguments are cloned: sample_actions passes time.expand(b) and then updates `time` in place,
    which a non-reentrant checkpoint would otherwise see, mutated, at recomputation."""
    exp = [m for n, m in mods if not n.startswith(S.LM)]
    orig = model.denoise_step

    def ds(*args, **kwargs):
        if not torch.is_grad_enabled():
            return orig(*args, **kwargs)
        for m in exp:
            m.prime()
        args = [a.clone() if torch.is_tensor(a) and not a.requires_grad else a for a in args]
        kwargs = {k: (v.clone() if torch.is_tensor(v) and not v.requires_grad else v) for k, v in kwargs.items()}
        return checkpoint(orig, *args, use_reentrant=False, **kwargs)
    model.denoise_step = ds


def snapshot_state(mods):
    """The trainable state (FP32 master codebooks and scales) on the CPU: small (36M values for pi0.5)."""
    return {name: (m.codebook.detach().cpu().clone(), m.scales.detach().cpu().clone()) for name, m in mods}


def load_state(mods, state):
    with torch.no_grad():
        for name, m in mods:
            cb, sc = state[name]
            m.codebook.copy_(cb.to(m.codebook.device))
            m.scales.copy_(sc.to(m.scales.device))


def export(mods):
    packed, weights = {}, {}
    with torch.no_grad():
        for name, m in mods:
            w = m.weight_().to(torch.bfloat16)
            weights[name + ".weight"] = w.cpu()
            packed[name] = dict(codebook=m.codebook.detach().cpu().clone(), scales=m.scales.detach().cpu().clone())
    return packed, weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", type=Path, required=True, help="stage-3 packed.pt (vector-codebook layers)")
    ap.add_argument("--fixed", type=Path, default=None, help="weights.pt of fixed scalar layers (allocated model)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--global-batch", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-scale", type=float, default=1e-5)
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=0, help="smoke test: stop after N updates")
    ap.add_argument("--no-resume", action="store_true", help="ignore an existing out/state.pt")
    a = ap.parse_args()
    out = a.out
    snaps = S.Snapshots()
    refs = torch.load(S.WORK / "snapshots" / "targets.pt", map_location="cpu", weights_only=False)["actions"]
    pol = S.load_policy()
    model = pol.model
    S.apply_background(model)
    if a.fixed is not None:
        S.apply_weights(model, a.fixed)
    packed = torch.load(a.packed, map_location="cpu", weights_only=False)["packed"]
    mods = install(model, packed)
    enable_prefix_checkpointing(model)
    enable_expert_step_checkpointing(model, mods)
    for p in model.parameters():
        p.requires_grad_(False)
    cb = [m.codebook for _, m in mods]
    sc = [m.scales for _, m in mods]
    for p in cb + sc:
        p.requires_grad_(True)
    train_params = cb + sc
    opt = torch.optim.AdamW([{"params": cb, "lr": a.lr}, {"params": sc, "lr": a.lr_scale}], weight_decay=0.0)
    print(f"[cal] {len(mods)} quantized layers; trainable {sum(p.numel() for p in train_params) / 1e6:.2f}M "
          f"(codebooks {sum(p.numel() for p in cb)}, scales {sum(p.numel() for p in sc)}); "
          f"allocated {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)

    def run(pos, grad):
        with torch.set_grad_enabled(grad):
            act = S.actions(pol, snaps, pos)
            r = refs[pos].to(act.device)
            loss = (act - r).square().mean()
            rel = (act - r).norm() / r.norm().clamp_min(1e-9)
            rel0 = (act[:, 0] - r[:, 0]).norm() / r[:, 0].norm().clamp_min(1e-9)
        return loss, float(rel), float(rel0)

    def validate():
        clear_reconstruction_cache(model)
        rels, rel0s = [], []
        with torch.no_grad():
            for pos in snaps.val_positions():
                clear_reconstruction_cache(model)
                _, r, r0 = run(pos, False)
                rels.append(r); rel0s.append(r0)
        clear_reconstruction_cache(model)
        return sum(rels) / len(rels), sum(rel0s) / len(rel0s)

    train_items = snaps.train_positions()
    spe = len(train_items) // a.global_batch
    budget = spe * a.epochs
    rng = random.Random(1)
    state_path = out / "state.pt"
    t0 = time.time()
    resumed_from = 0
    if state_path.exists() and not a.no_resume:
        # Resume (as the source's `resume`: latest state every 100 updates): parameters, AdamW moments, best state,
        # log; the epoch shuffles are replayed from random.Random(1) and finished updates skipped.
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        load_state(mods, st["current"])
        opt.load_state_dict(st["opt"])
        step, best, best_step, best_state, log, v0 = st["step"], st["best"], st["best_step"], st["best_state"], \
            st["log"], st["init"]
        t0 -= st["elapsed"]
        resumed_from = step
        print(f"[cal] resumed at step {step} (best {best:.6f} at step {best_step})", flush=True)
    else:
        log = []
        v0, v0_first = validate()
        best, best_step = v0, 0
        best_state = snapshot_state(mods)
        log.append(dict(step=0, val=v0, val_first_action=v0_first, seconds=time.time() - t0))
        print(f"[cal] step 0 VAL rel {v0:.6f} (first action {v0_first:.6f}) {time.time() - t0:.0f}s", flush=True)
        step = 0
    stop = False
    t_run, steps_run = time.time(), 0

    def save_state():
        S.save_atomic(dict(step=step, current=snapshot_state(mods), opt=opt.state_dict(), best=best,
                           best_step=best_step, best_state=best_state, log=log, init=v0,
                           elapsed=time.time() - t0), state_path)
    for ep in range(a.epochs):
        order = train_items[:]
        rng.shuffle(order)
        for bi in range(spe):
            if ep * spe + bi < resumed_from:
                continue  # finished before the resume
            chunk = order[bi * a.global_batch:(bi + 1) * a.global_batch]
            opt.zero_grad(set_to_none=True)
            losses, rels = [], []
            for pos in chunk:
                clear_reconstruction_cache(model)  # fresh reconstructed W graph per request
                loss, rel, _ = run(pos, True)
                (loss / len(chunk)).backward()     # == summed gradients divided by the global batch
                losses.append(float(loss)); rels.append(rel)
            clear_reconstruction_cache(model)
            torch.nn.utils.clip_grad_norm_(train_params, 1.0)
            opt.step()
            step += 1
            steps_run += 1
            if step % 10 == 0 or step <= 3:
                print(f"[cal] ep{ep} step{step} loss={sum(losses) / len(losses):.6f} rel={sum(rels) / len(rels):.4f} "
                      f"({(time.time() - t_run) / steps_run:.2f}s/step, max alloc "
                      f"{torch.cuda.max_memory_allocated() / 1e9:.1f} GB)", flush=True)
            if step % a.val_every == 0 or step == budget or step == a.max_steps:
                v, vf = validate()
                log.append(dict(step=step, val=v, val_first_action=vf, train_loss=sum(losses) / len(losses),
                                seconds=time.time() - t0))
                tag = ""
                if v < best:
                    best, best_step = v, step
                    best_state = snapshot_state(mods)
                    tag = " (best)"
                print(f"[cal] step {step} VAL rel {v:.6f} first-action {vf:.6f}{tag}", flush=True)
                S.write_json(dict(log=log, best=best, best_step=best_step), out / "log.json")
                save_state()
            if a.max_steps and step >= a.max_steps:
                stop = True
                break
        if stop:
            break
    load_state(mods, best_state)
    packed_best, weights_best = export(mods)
    final = {}
    for name, p in packed.items():
        q = dict(p)
        q["codebook"] = packed_best[name]["codebook"].to(torch.bfloat16)
        q["scales"] = packed_best[name]["scales"].to(torch.bfloat16)
        final[name] = q
    S.save_atomic(dict(packed=final), out / "packed.pt")
    S.save_atomic(dict(weights=weights_best), out / "weights.pt")
    S.write_json(dict(log=log, best=best, best_step=best_step, init=v0, updates=step, seconds=time.time() - t0,
                      max_memory_allocated=torch.cuda.max_memory_allocated(),
                      settings={k: str(v) for k, v in vars(a).items()}), out / "log.json")
    print(f"[cal] done: best val {best:.6f} at step {best_step} (init {v0:.6f}); {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
