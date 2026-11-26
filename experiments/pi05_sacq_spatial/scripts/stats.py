"""Stage-1 statistics: structure-aware fitting metrics for the 284 pi0.5 SACQ candidate linears.

Run on the 64-snapshot fitting subset (5 observations each, noise 7000 + position) with the fixed 4-bit background
installed and every candidate still BF16 (upstream 3cf0523: "BF16 language and action weights over the 4-bit
vision/embedding background"). Adapted from the SmolVLA port's stats.py.

Reference roles: reference_sources/metric_v2/vlm_fitter.py (language model) and dit_metrics.py / dit_fitter.py (DiT,
AdaLN). Both pi0.5 stacks are Gemma blocks (MQA: 8 query heads, 1 KV head of dim 256, RoPE over (i, i+128) pairs,
GeGLU with tanh-GELU), so both use the VLM fitter's role metrics evaluated on the attention each block performs:
  LM layer i      : bidirectional self-attention over the valid prefix (images, instruction); padded image/text
                    tokens excluded everywhere.
  expert layer i  : the 10 action-token queries attend over [LM layer i's cached prefix keys ; their own keys]
                    (every denoising step). The LM's K/V therefore serve two query sets: LM layer i's k_proj/v_proj
                    statistics average over the LM queries AND the expert queries that attend to those keys
                    (LM layer 17: expert queries only, its own queries feed nothing).
Per role (FP32 accumulation over every valid token; expert statistics over all 10 denoising steps):
  q_proj : qk_full, input metric E[x x^T], output importance = mean squared key over attended keys (RoPE pair-avg)
  k_proj : qk_full, input metric E[x x^T], output importance = mean squared query summed over the 8 heads sharing
           the KV head (RoPE pair-avg)
  v_proj : qk_full, attention-weighted input metric sum_s (sum_{h,q} p_hqs^2) x_s x_s^T / #queries,
           output importance = o_proj column energy summed over heads (expert: o_proj rows weighted by E[gate^2])
  o_proj : qk_full, input metric E[ctx ctx^T], output importance 1 (LM) / E[gate_attn^2] (expert gated residual)
  gate/up: qk_diag, E[(gelu'(g) u)^2 x^2] / E[gelu(g)^2 x^2], times the down_proj column energy (vlm_metric_v2 diag)
  down   : qk_full, input metric E[z z^T] with z = gelu(g) * u, output importance 1 (LM) / E[gate_mlp^2] (expert)
  adaRMS dense (GR00T "adaln" role): qk_full, input metric E[c c^T] of the time condition, output importance
           [E[n^2] (scale), 1 (shift), E[y^2] (gate: y = the gated sublayer output)]; final norm dense:
           [w_out E[n^2], w_out, ~0 (its gate is unused)] with w_out = action_out_proj column energy (as the author's
           proj_out_1: downstream column energy x squared normalized activation).
Input metrics wider than 4096 (LM down_proj, 16384) are stored as their 4x4 diagonal blocks: quantize_vector_codebook
reads only the diagonal and the 4x4 blocks of the vectors' own columns, so fit.py rebuilds an exactly equivalent
matrix (zeros elsewhere).
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sacqpi as S  # noqa: E402

H, HKV, D = 8, 1, 256
G = H // HKV
BLOCK_ABOVE = 4096


def pair_avg(e):
    """Average per-dimension energies over RoPE pairs (i, i + D/2) within each head. e: (..., heads*D)."""
    shp = e.shape
    x = e.reshape(*shp[:-1], -1, 2, D // 2)
    x = x.mean(dim=-2, keepdim=True).expand_as(x)
    return x.reshape(shp)


def gelu_tanh_and_grad(x):
    c = math.sqrt(2.0 / math.pi)
    inner = c * (x + 0.044715 * x.pow(3))
    t = torch.tanh(inner)
    y = 0.5 * x * (1 + t)
    dy = 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * c * (1 + 3 * 0.044715 * x * x)
    return y, dy


class Acc:
    def __init__(self):
        self.d = {}
        self.n = {}

    def add(self, key, value, count=None, cpu=False):
        if cpu:
            value = value.cpu()
        if key in self.d:
            self.d[key] += value
        else:
            self.d[key] = value.clone() if not cpu else value
        if count is not None:
            self.n[key] = self.n.get(key, 0.0) + float(count)


def gram_or_blocks(x):
    x = x.float()
    if x.shape[1] > BLOCK_ABOVE:
        xb = x.reshape(x.shape[0], -1, 4)
        return torch.einsum("nbi,nbj->bij", xb, xb)
    return x.t() @ x


class Collector:
    def __init__(self, model):
        self.model = model
        pwe = model.paligemma_with_expert
        self.pwe = pwe
        self.lm_layers = pwe.paligemma.model.language_model.layers
        self.exp_layers = pwe.gemma_expert.model.layers
        cfg = pwe.paligemma.config.text_config
        assert cfg.num_attention_heads == H and cfg.num_key_value_heads == HKV and cfg.head_dim == D
        assert len(self.lm_layers) == S.NLAYERS and len(self.exp_layers) == S.NLAYERS
        self.acc = Acc()
        self.ctx = dict(phase=None, prefix_valid=None)
        self.lm_vin = {}
        self.handles = []
        self.cands = {c[0] for c in S.candidates()}
        S.tag_attention(model)
        self._install()

    # ------------------------------------------------------------ hooks
    def _install(self):
        import transformers.models.gemma.modeling_gemma as mg
        col = self
        orig_fwd = self.pwe.forward
        self._orig_fwd = orig_fwd
        self._mg = mg
        self._orig_eager = mg.eager_attention_forward

        def fwd(*args, **kwargs):
            embeds = kwargs["inputs_embeds"]
            mask = kwargs["attention_mask"]
            if embeds[0] is not None:
                col.ctx["phase"] = "prefix"
                col.ctx["prefix_valid"] = mask[:, 0].diagonal(dim1=1, dim2=2) == 0
            else:
                col.ctx["phase"] = "denoise"
            return orig_fwd(*args, **kwargs)

        def eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
            out = col._orig_eager(module, query, key, value, attention_mask, scaling, dropout=dropout, **kw)
            col.attention_stats(module._sacq, query, key, attention_mask)
            return out

        self.pwe.forward = fwd
        mg.eager_attention_forward = eager
        for owner, layers in (("lm", self.lm_layers), ("exp", self.exp_layers)):
            for i, layer in enumerate(layers):
                for role in S.ATTN:
                    name = S.module_name(S.LM if owner == "lm" else S.EXP, i, role)
                    m = getattr(layer.self_attn, role)
                    if name in self.cands or (owner == "lm" and role == "v_proj"):
                        self.handles.append(m.register_forward_pre_hook(self._lin_hook(owner, i, role, name)))
                if owner == "lm" and i == S.NLAYERS - 1:
                    continue  # dead MLP
                self.handles.append(layer.mlp.register_forward_pre_hook(self._mlp_hook(owner, i, layer.mlp)))
                if owner == "exp":
                    self.handles.append(layer.input_layernorm.register_forward_hook(self._norm_hook(i, "input_layernorm")))
                    self.handles.append(layer.post_attention_layernorm.register_forward_hook(
                        self._norm_hook(i, "post_attention_layernorm")))
                    self.handles.append(layer.input_layernorm.dense.register_forward_pre_hook(
                        self._cond_hook(f"exp{i}.input_layernorm.dense")))
                    self.handles.append(layer.post_attention_layernorm.dense.register_forward_pre_hook(
                        self._cond_hook(f"exp{i}.post_attention_layernorm.dense")))
                    self.handles.append(layer.self_attn.o_proj.register_forward_hook(self._ysq_hook(f"exp{i}.attn_y2")))
                    self.handles.append(layer.mlp.down_proj.register_forward_hook(self._ysq_hook(f"exp{i}.mlp_y2")))
        fin = self.pwe.gemma_expert.model.norm
        self.handles.append(fin.register_forward_hook(self._norm_hook(S.NLAYERS, "final")))
        self.handles.append(fin.dense.register_forward_pre_hook(self._cond_hook("final.dense")))

    def rows(self, x, owner):
        if owner == "lm":
            return x[self.ctx["prefix_valid"]]
        return x.reshape(-1, x.shape[-1])

    def _lin_hook(self, owner, i, role, name):
        def hook(mod, args):
            x = args[0]
            if owner == "lm" and role == "v_proj":
                self.lm_vin[i] = x.detach()  # (B, Lp, 2048): for the expert queries' attention onto LM values
            if name in self.cands:
                r = self.rows(x, owner)
                self.acc.add(f"{owner}{i}.{role}.G", gram_or_blocks(r), r.shape[0])
                if role == "v_proj" and owner == "exp":
                    self.ctx[("exp_vin", i)] = x.detach()
        return hook

    def _mlp_hook(self, owner, i, mlp):
        def hook(mod, args):
            h = self.rows(args[0], owner).float()
            wg = mlp.gate_proj.weight.float(); wu = mlp.up_proj.weight.float()
            gi = ui = zz = None
            for s in range(0, h.shape[0], 2048):
                hc = h[s:s + 2048]
                g = torch.nn.functional.linear(hc, wg); u = torch.nn.functional.linear(hc, wu)
                act, dact = gelu_tanh_and_grad(g)
                h2 = hc.square()
                a = (dact * u).square().t() @ h2
                b = act.square().t() @ h2
                z = gram_or_blocks(act * u)
                gi = a if gi is None else gi + a
                ui = b if ui is None else ui + b
                zz = z if zz is None else zz + z
            big = owner == "lm"  # (16384, 2048) FP32 importances: accumulate on the host
            self.acc.add(f"{owner}{i}.gate_imp", gi, h.shape[0], cpu=big)
            self.acc.add(f"{owner}{i}.up_imp", ui, h.shape[0], cpu=big)
            self.acc.add(f"{owner}{i}.ztz", zz, h.shape[0])
        return hook

    def _norm_hook(self, i, which):
        def hook(mod, args, output):
            x = args[0].float()
            n = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + mod.eps)
            n2 = n.square().reshape(-1, n.shape[-1])
            self.acc.add(f"exp{i}.{which}.n2", n2.sum(0), n2.shape[0])
            gate = output[1]
            if gate is not None and which != "final":
                gg = gate.float().square().expand(x.shape).reshape(-1, x.shape[-1])
                self.acc.add(f"exp{i}.{which}.gate2", gg.sum(0), gg.shape[0])
        return hook

    def _cond_hook(self, key):
        def hook(mod, args):
            c = args[0].float().reshape(-1, args[0].shape[-1])
            self.acc.add(f"{key}.G", c.t() @ c, c.shape[0])
        return hook

    def _ysq_hook(self, key):
        def hook(mod, args, output):
            y = output.float().reshape(-1, output.shape[-1])
            self.acc.add(key, y.square().sum(0), y.shape[0])
        return hook

    def attention_stats(self, tag, q, k, mask):
        owner, i = tag
        B, _, Lq, _ = q.shape
        Lk = k.shape[2]
        pv = self.ctx["prefix_valid"]
        Lp = pv.shape[1]
        qf = q.float()                             # (B, 8, Lq, 256) post-RoPE
        kf = k.float()[:, 0]                       # (B, Lk, 256) single KV head
        allowed = mask[:, 0] == 0                  # (B, Lq, Lk)
        if owner == "lm":
            qvalid, kvalid = pv, pv
        else:
            qvalid = torch.ones(B, Lq, dtype=torch.bool, device=q.device)
            kvalid = allowed[:, 0, :]              # prefix pad mask + all suffix keys
        # q importance: mean squared key over valid keys (pair-averaged), identical for the 8 heads
        k2 = (kf.square() * kvalid[..., None]).sum(dim=(0, 1))
        if not (owner == "lm" and i == S.NLAYERS - 1):
            self.acc.add(f"{owner}{i}.q_imp", pair_avg(k2.repeat(H)), kvalid.sum().item())
        # k importance: mean squared query over valid queries, summed over the heads sharing the KV head
        q2 = (qf.square() * qvalid[:, None, :, None]).sum(dim=(0, 2)).sum(0)   # (256,)
        nq = qvalid.sum().item()
        if not (owner == "lm" and i == S.NLAYERS - 1):
            self.acc.add(f"{owner}{i}.k_imp", pair_avg(q2), nq)
        if owner == "exp":
            self.acc.add(f"lm{i}.k_imp", pair_avg(q2), nq)  # expert queries also read LM layer i's keys
        # attention probabilities exactly as eager (fp32 logits with the additive mask)
        logits = torch.einsum("bhqd,bkd->bhqk", qf, kf) * D ** -0.5 + mask.float()
        p = torch.softmax(logits, dim=-1) * qvalid[:, None, :, None]
        w = p.square().sum(dim=(1, 2))             # (B, Lk) source weights
        if owner == "lm":
            if i != S.NLAYERS - 1:
                self._vmet(f"lm{i}.Vmet", self.lm_vin[i][:, :Lp], w, nq)
        else:
            self._vmet(f"lm{i}.Vmet", self.lm_vin[i], w[:, :Lp], nq)
            self._vmet(f"exp{i}.Vmet", self.ctx.pop(("exp_vin", i)), w[:, Lp:], nq)

    def _vmet(self, key, x, w, nq):
        assert w.shape == x.shape[:2], (key, w.shape, x.shape)
        wx = (x.float() * w.clamp_min(0).sqrt()[..., None]).reshape(-1, x.shape[-1])
        self.acc.add(key, wx.t() @ wx, nq)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.pwe.forward = self._orig_fwd
        self._mg.eager_attention_forward = self._orig_eager

    def finalize(self):
        out = {}
        for key, val in self.acc.d.items():
            v = val.float() / self.acc.n[key]
            if v.ndim == 2 and v.shape[0] == v.shape[1] and not key.endswith(("gate_imp", "up_imp")):
                v = 0.5 * (v + v.t())
            out[key] = v.cpu()
        out["__counts__"] = dict(self.acc.n)
        return out


@torch.no_grad()
def main():
    snaps = S.Snapshots()
    pol = S.load_policy()
    S.apply_background(pol.model)
    col = Collector(pol.model)
    t0 = time.time()
    fit = snaps.fit_positions()
    for n, pos in enumerate(fit):
        S.sample_norm(pol, snaps, pos)
        col.lm_vin.clear()
        if (n + 1) % 8 == 0:
            print(f"[stats] {n + 1}/{len(fit)} snapshots, {time.time() - t0:.0f}s, "
                  f"max alloc {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
    col.remove()
    res = col.finalize()
    c = res["__counts__"]
    assert c["exp0.k_imp"] == 10 * 10 * 5 * len(fit), c["exp0.k_imp"]
    res["__meta__"] = dict(snapshots=fit, seconds=time.time() - t0, lm_valid_prefix_tokens=c["lm0.q_proj.G"],
                           expert_query_rows=c["exp0.q_proj.G"])
    S.save_atomic(res, S.WORK / "stats" / "stats.pt")
    print(f"[stats] done in {time.time() - t0:.0f}s; LM valid prefix tokens {c['lm0.q_proj.G']:.0f}, "
          f"expert query rows {c['exp0.q_proj.G']:.0f}", flush=True)


if __name__ == "__main__":
    main()
