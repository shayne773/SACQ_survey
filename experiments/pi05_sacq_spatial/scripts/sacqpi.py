"""Shared pieces of the SACQ-on-pi0.5 pipeline (adapted from the SmolVLA port's sacqsmol.py).

Layer scope (mirrors the GR00T SACQ recipe, gr00t_libero/presets/gr00t_libero_sacq2_seed17.json, whose 338
candidates are the language-model linears and every DiT linear incl. the AdaLN `norm1.linear` and `proj_out_1/2`):
- SACQ candidates (vector codebooks over 4-weight vectors, row-group scales, group size 64; every pi0.5 candidate
  shape divides into 64-column groups, so the narrow-layer guard never triggers):
    language model (PaliGemma Gemma-2B, 18 layers): q/k/v/o/gate/up/down = 126 linears, minus the 5 linears of
      layer 17 whose outputs pi0.5 never consumes (q, o, gate, up, down: the prefix pass's final hidden states are
      discarded; only every layer's K/V cache reaches the action expert) -> 121;
    action expert (Gemma-300M, 18 layers): q/k/v/o/gate/up/down = 126;
    action expert adaRMS modulation linears (time-conditioned scale/shift/gate, GR00T's AdaLN analogue):
      input_layernorm.dense + post_attention_layernorm.dense per layer (36) + final norm.dense (1) = 37.
  Total 284 candidate linears.
- Fixed 4-bit background (GR00T: vision Linears incl. the vision->LM merger at GPTQ 4-bit; embedding affine 4-bit):
  the 162 SigLIP encoder Linears + the multi-modal projector at scalar GPTQ 4-bit (group 64, asymmetric, block 128,
  damping 0.01, upstream gr00t_libero/gptq_core.py); the token embedding (tied with the unused PaliGemma lm_head)
  at affine 4-bit group 64.
- Original precision: SigLIP patch-embedding conv and position embedding, all norms' own weights, action_in_proj,
  action_out_proj, time_mlp_in/out (GR00T's action encoder/decoder and timestep encoder analogues, excluded there
  too). Not stored at all: the expert's lm_head (unused, dropped at load) and the 5 dead layer-17 LM linears (kept
  BF16 in the simulated model, where they cannot affect any output, and counted as removed in the size accounting).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import picore as P  # noqa: E402

REPO_ROOT = Path(os.environ.get("SACQ_REPO", Path(__file__).resolve().parents[3]))  # SACQ_survey checkout
sys.path.insert(0, str(REPO_ROOT))
LAB = P.LAB
WORK = Path(os.environ.get("SACQ_WORK", LAB / "sacq"))

PWE = "paligemma_with_expert"
LM = f"{PWE}.paligemma.model.language_model.layers"
EXP = f"{PWE}.gemma_expert.model.layers"
EXP_FINAL_DENSE = f"{PWE}.gemma_expert.model.norm.dense"
VIS = f"{PWE}.paligemma.model.vision_tower.vision_model"
PROJECTOR = f"{PWE}.paligemma.model.multi_modal_projector.linear"
EMBED = f"{PWE}.paligemma.model.language_model.embed_tokens"
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")
ADA = ("input_layernorm.dense", "post_attention_layernorm.dense")
NLAYERS = 18
DEAD_LM = {f"{LM}.17.self_attn.q_proj", f"{LM}.17.self_attn.o_proj", f"{LM}.17.mlp.gate_proj",
           f"{LM}.17.mlp.up_proj", f"{LM}.17.mlp.down_proj"}
NOISE_BASE = 7000  # author: noise seed 7000 + original request position
VECTOR_DIM = 4
GROUP = 64
UNIFORM_K = 64  # 6-bit index per 4-weight vector = 1.5 index bits / weight


def module_name(prefix, layer, role):
    if role in ADA:
        return f"{prefix}.{layer}.{role}"
    return f"{prefix}.{layer}.{'self_attn' if role in ATTN else 'mlp'}.{role}"


def candidates():
    """Ordered list of (name, owner, layer, role); owner in {lm, exp, ada}."""
    out = []
    for i in range(NLAYERS):
        for r in ATTN + MLP:
            n = module_name(LM, i, r)
            if n not in DEAD_LM:
                out.append((n, "lm", i, r))
    for i in range(NLAYERS):
        for r in ATTN + MLP:
            out.append((module_name(EXP, i, r), "exp", i, r))
    for i in range(NLAYERS):
        for r in ADA:
            out.append((module_name(EXP, i, r), "ada", i, r))
    out.append((EXP_FINAL_DENSE, "ada", NLAYERS, "final.dense"))
    assert len(out) == 284, len(out)
    return out


def candidate_names():
    return [c[0] for c in candidates()]


def scope_of(name):
    """Allocation scope: 'vlm' (language model) or 'dit' (action expert incl. adaRMS), as in the author's allocator."""
    return "vlm" if name.startswith(LM) else "dit"


def background_linear_names(model):
    names = [n for n, m in model.named_modules() if n.startswith(VIS + ".encoder.") and isinstance(m, torch.nn.Linear)]
    assert len(names) == 162, len(names)
    return names + [PROJECTOR]


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------- snapshots
class Snapshots:
    """N snapshots x 5 policy-input elements, in the selected (shuffled) order; position i % 10 == 0 is validation."""

    def __init__(self, path=None):
        path = Path(path or WORK / "snapshots" / "snapshots.pt")
        d = torch.load(path, map_location="cpu", weights_only=False)
        self.meta = d["meta"]
        self.records = d["records"]
        self.per = d["meta"]["observations_per_snapshot"]
        self.obs = {k: np.load(path.parent / f"{d['prefix']}obs_{k}.npy", mmap_mode="r") for k in d["obs_keys"]}

    def __len__(self):
        return len(self.records)

    def elements(self, pos):
        rec = self.records[pos]
        out = []
        for j in range(self.per):
            k = pos * self.per + j
            out.append({"observation/image": np.array(self.obs["image"][k]),
                        "observation/wrist_image": np.array(self.obs["wrist_image"][k]),
                        "observation/state": np.array(self.obs["state"][k]), "prompt": rec["language"]})
        return out

    def noise(self, pos, device="cuda"):
        return P.fixed_noise([(NOISE_BASE + pos, j) for j in range(self.per)], device)

    def train_positions(self):
        return [i for i in range(len(self)) if i % 10 != 0]

    def val_positions(self):
        return [i for i in range(len(self)) if i % 10 == 0]

    def fit_positions(self):
        return self.train_positions()[:64]


def sample_norm(pol, snaps, pos):
    """Normalized (5, 10, 32) chunk for snapshot pos (grad mode chosen by the caller)."""
    return pol.sample_normalized(pol.model_inputs(snaps.elements(pos)), snaps.noise(pos))


def actions(pol, snaps, pos):
    """Unnormalized (5, 10, 7) float32 chunk (the stored teacher target format)."""
    return pol.unnormalize(sample_norm(pol, snaps, pos))


# ---------------------------------------------------------------- weights / overlays
def decode_packed(p, device="cuda"):
    """Dequantize a packed_vector_codebook_v1 payload with the runtime arithmetic (BF16 gather, BF16 scale, shift),
    exactly sacq.index_correction.decode_fixed / sacq.calibration_layers.QuantLinear.weight_()."""
    from sacq.index_correction import decode_fixed
    return decode_fixed(p, device)


def load_overlay(model, weights):
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, w in weights.items():
            p = params[name]
            assert tuple(p.shape) == tuple(w.shape), name
            p.copy_(w.to(p.device, p.dtype))


def fsync_dir(d):
    fd = os.open(str(d), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save_atomic(obj, path):
    # Durable: data fsynced before the rename and the directory after it. Without this a hard VM restart left
    # 0-byte files under their final names (2026-11-24 14:58Z), which resume logic took for finished outputs.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        f.write(json.dumps(obj, indent=1))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


@torch.no_grad()
def affine_embedding_dequant(weight, nbits=4, group_size=64):
    """GR00T PackedAffineEmbedding arithmetic (BF16 scale/minimum per (row, 64-group), dequantized in FP32)."""
    w = weight.detach().to(torch.float32)
    vocab, dim = w.shape
    out = torch.empty(vocab, dim, dtype=torch.bfloat16, device=w.device)
    qmax = (1 << nbits) - 1
    for s in range(0, vocab, 65536):  # chunked: the 257k x 2048 embedding is 2 GB in FP32
        g = w[s:s + 65536].reshape(-1, dim // group_size, group_size)
        minv, maxv = g.amin(dim=2), g.amax(dim=2)
        scales = ((maxv - minv) / qmax).clamp_min(1e-8)
        q = torch.round((g - minv.unsqueeze(-1)) / scales.unsqueeze(-1)).clamp(0, qmax)
        s16 = scales.to(torch.bfloat16).to(torch.float32)
        z16 = minv.to(torch.bfloat16).to(torch.float32)
        out[s:s + 65536] = (q * s16.unsqueeze(-1) + z16.unsqueeze(-1)).reshape(-1, dim).to(torch.bfloat16)
    return out


def load_policy(device="cuda"):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return P.Policy(device)


def apply_background(model, path=None):
    bg = torch.load(path or WORK / "background" / "background.pt", map_location="cpu", weights_only=False, mmap=True)
    load_overlay(model, bg["weights"])
    return bg


def apply_weights(model, path):
    d = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    load_overlay(model, d["weights"])
    return d


def tag_attention(model):
    """Tag every Gemma attention module with (owner, layer) for the statistics hooks."""
    pwe = model.paligemma_with_expert
    for i, l in enumerate(pwe.paligemma.model.language_model.layers):
        l.self_attn._sacq = ("lm", i)
    for i, l in enumerate(pwe.gemma_expert.model.layers):
        l.self_attn._sacq = ("exp", i)


def prefix_forward(pol, inputs):
    """The prefix half of PI0Pytorch.sample_actions (SigLIP + Gemma-2B KV-cache pass), for statistics that only
    touch the vision tower and the language model; identical computation to the full call's first part."""
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

    model = pol.model
    images, img_masks, lang_tokens, lang_masks, _state = model._preprocess_observation(
        pol.observation(inputs), train=False)
    embs, pad, att = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    att2d = make_att_2d_masks(pad, att)
    pos = torch.cumsum(pad, dim=1) - 1
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
    _, kv = model.paligemma_with_expert.forward(attention_mask=model._prepare_attention_masks_4d(att2d),
                                                position_ids=pos, past_key_values=None,
                                                inputs_embeds=[embs, None], use_cache=True)
    return kv


# ---------------------------------------------------------------- evaluation conditions
def condition_files(cond):
    """Weight files applied in order over the original model for each evaluated condition (simulated quantization:
    dequantized weights loaded into the BF16/F32 model). Every quantized condition starts from the fixed 4-bit
    background (vision + projector GPTQ4, embedding affine4)."""
    W = WORK
    bg = W / "background" / "background.pt"
    table = {
        "original": [],
        "bg": [bg],                                                  # 4-bit background alone, candidates original
        "rtn2": [bg, W / "rtn" / "rtn2.pt"],
        "rtn3": [bg, W / "rtn" / "rtn3.pt"],
        "u1": [bg, W / "uniform" / "stage1" / "weights.pt"],        # uniform 1.5-bit, stage 1
        "u13": [bg, W / "uniform" / "stage3" / "weights.pt"],       # + stage 3
        "u134": [bg, W / "uniform" / "stage4" / "weights.pt"],      # + stage 4
        "a12": [bg, W / "alloc" / "stage1" / "weights.pt"],         # allocated, stages 1+2
        "a123": [bg, W / "alloc" / "stage1" / "fixed.pt", W / "alloc" / "stage3" / "weights.pt"],
        "a1234": [bg, W / "alloc" / "stage1" / "fixed.pt", W / "alloc" / "stage4" / "weights.pt"],
    }
    return table[cond]


def apply_condition(model, cond):
    """Load a condition's weight files; checks that a quantized condition covers every candidate and the background.
    Returns [(path, sha256)]."""
    files = condition_files(cond)
    covered = set()
    out = []
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False, mmap=True)
        load_overlay(model, d["weights"])
        covered |= set(d["weights"])
        out.append((str(f), P.sha256_file(f)))
        del d
    if cond not in ("original", "bg"):
        need = {n + ".weight" for n in candidate_names()} | {n + ".weight" for n in background_linear_names(model)} \
            | {EMBED + ".weight"}
        missing = need - covered
        assert not missing, (cond, len(missing), sorted(missing)[:3])
    return out
