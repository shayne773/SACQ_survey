"""Shared pi0.5 (openpi PyTorch, pi05_libero) helpers for the SACQ-on-pi0.5 experiment.

Everything that touches the model goes through these functions, so the original model and every quantized
variant see the same inputs, transforms and noise.

Model: the official pi05_libero checkpoint converted to PyTorch by openpi (CKPT, read-only). The weights are
loaded exactly as openpi's create_trained_policy leaves them (verified bitwise by verify_policy.py):
  - paligemma_with_expert: BF16, except the float32 set of PaliGemmaWithExpertModel.to_bfloat16_for_selected_params
    (SigLIP patch/position embeddings, every input/post-attention layernorm incl. the expert's adaRMS `dense`
    Linears, final norms);
  - action_in_proj, action_out_proj, time_mlp_in, time_mlp_out: float32 (openpi builds the model in float32 and
    only converts paligemma_with_expert).
Construction happens directly on the GPU in BF16 (openpi's own path stages a 13 GB float32 copy in host RAM).

Inference protocol (openpi examples/libero/main.py, as used by the vla_distill harness):
  agentview and wrist images rotated 180 deg ([::-1, ::-1]), resize_with_pad to 224x224, uint8;
  state = eef_pos (3) + axis-angle(eef_quat) (3) + gripper_qpos (2);
  prompt = task language; openpi input transforms (LiberoInputs, Normalize with quantiles, ResizeImages,
  TokenizePrompt max_token_len 200, PadStatesAndActions 32); 10 Euler denoising steps;
  output transforms (Unnormalize, LiberoOutputs first 7 dims) -> (10, 7) chunk, first 5 actions executed.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import numpy as np

CKPT = os.environ["CKPT"]  # openpi PyTorch conversion of pi05_libero; set by env.sh
CONFIG = "pi05_libero"
LAB = Path(__file__).resolve().parents[1]
SUITE = "libero_spatial"
ACTION_DIM = 7
HORIZON = 10
MODEL_ACTION_DIM = 32
NUM_STEPS = 10
REPLAN = 5
RESIZE = 224
ENV_RES = 256
NUM_STEPS_WAIT = 10
MAX_STEPS = 220  # openpi examples/libero/main.py: libero_spatial
POLICY_SEED = 20260824
FLOAT32_OUTER = ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out")


# ---------------------------------------------------------------- client-side observation (openpi example)
def quat2axisangle(quat):
    quat = np.array(quat, dtype=np.float64)
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def client_element(raw, prompt):
    """openpi examples/libero/main.py element from a raw LIBERO observation (before the policy transforms)."""
    from openpi_client import image_tools

    img = np.ascontiguousarray(raw["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(raw["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, RESIZE, RESIZE))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, RESIZE, RESIZE))
    state = np.concatenate((raw["robot0_eef_pos"], quat2axisangle(raw["robot0_eef_quat"]), raw["robot0_gripper_qpos"]))
    return {"observation/image": img, "observation/wrist_image": wrist, "observation/state": state, "prompt": str(prompt)}


# ---------------------------------------------------------------- model + transforms
def train_config():
    from openpi.training import config as _config

    return _config.get_config(CONFIG)


def transforms():
    """(input_transform, output_transform, norm_stats) exactly as create_trained_policy composes them."""
    import openpi.transforms as T
    from openpi.training import checkpoints as _checkpoints

    tc = train_config()
    data_config = tc.data.create(tc.assets_dirs, tc.model)
    norm_stats = _checkpoints.load_norm_stats(Path(CKPT) / "assets", data_config.asset_id)
    inp = T.compose([T.InjectDefaultPrompt(None), *data_config.data_transforms.inputs,
                     T.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
                     *data_config.model_transforms.inputs])
    out = T.compose([*data_config.model_transforms.outputs,
                     T.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
                     *data_config.data_transforms.outputs])
    assert data_config.use_quantile_norm
    return inp, out, norm_stats


def load_model(device="cuda", drop_expert_lm_head=True):
    import torch
    from safetensors import safe_open
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    tc = train_config()
    with torch.device("meta"):  # PaliGemmaWithExpertModel.__init__ already applies to_bfloat16_for_selected_params
        model = PI0Pytorch(config=tc.model)
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")  # create_trained_policy repeats it
    for n in FLOAT32_OUTER:
        getattr(model, n).float()
    model.to_empty(device=device)
    # non-persistent buffers (not in the checkpoint): recompute exactly as the constructors do, then apply the same
    # BF16 cast that PaliGemmaWithExpertModel.to(bfloat16) applied to them in openpi's own construction.
    emb = model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings
    emb.position_ids.copy_(torch.arange(emb.num_positions, device=device).expand((1, -1)))
    for rot in (model.paligemma_with_expert.paligemma.model.language_model.rotary_emb,
                model.paligemma_with_expert.gemma_expert.model.rotary_emb):
        inv_freq, scaling = rot.rope_init_fn(rot.config, device)
        rot.inv_freq.copy_(inv_freq.to(rot.inv_freq.dtype))
        rot.original_inv_freq = rot.inv_freq
        rot.attention_scaling = scaling
    params = dict(model.named_parameters(remove_duplicate=False))
    seen = set()
    with safe_open(os.path.join(CKPT, "model.safetensors"), framework="pt", device=device) as f:
        for k in f.keys():
            p = params[k]
            with torch.no_grad():
                p.copy_(f.get_tensor(k).to(p.dtype))
            seen.add(k)
    missing = [k for k in params if k not in seen]
    # tied: language_model.embed_tokens.weight is paligemma.lm_head.weight (the checkpoint stores only lm_head;
    # to_empty() on the meta model breaks the tie that post_init made, so restore it explicitly)
    pg = model.paligemma_with_expert.paligemma
    pg.language_model.embed_tokens.weight = pg.lm_head.weight
    emb = model.paligemma_with_expert.paligemma.language_model.embed_tokens.weight
    assert emb is model.paligemma_with_expert.paligemma.lm_head.weight
    assert all("embed_tokens" in k for k in missing), missing
    if drop_expert_lm_head:  # never used by pi0.5 (the expert only consumes/produces embeddings): 0.5 GB
        model.paligemma_with_expert.gemma_expert.lm_head = None
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def eager_sample_fn():
    """PI0Pytorch.sample_actions without torch.compile and without its no_grad decorator."""
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    return PI0Pytorch.sample_actions.__wrapped__


class Policy:
    """Model + openpi transforms; batched, eager, fixed noise."""

    def __init__(self, device="cuda", model=None):
        import torch

        self.device = device
        self.model = model if model is not None else load_model(device)
        self.inp, self.out, self.norm_stats = transforms()
        self._sample = eager_sample_fn()
        a = self.norm_stats["actions"]
        q01 = np.asarray(a.q01, np.float32)[:ACTION_DIM]
        q99 = np.asarray(a.q99, np.float32)[:ACTION_DIM]
        self.q01 = torch.as_tensor(q01, device=device)
        self.q99 = torch.as_tensor(q99, device=device)

    def model_inputs(self, elements):
        """Apply the openpi input transforms per element and stack -> dict of device tensors."""
        import torch

        items = [self.inp(dict(e)) for e in elements]

        def stack(xs):
            if isinstance(xs[0], dict):
                return {k: stack([x[k] for x in xs]) for k in xs[0]}
            return torch.from_numpy(np.stack([np.asarray(x) for x in xs])).to(self.device)

        return stack(items)

    def observation(self, inputs):
        from openpi.models import model as _model

        return _model.Observation.from_dict(inputs)

    def sample_normalized(self, inputs, noise):
        """(B, 10, 32) normalized action chunk; grad mode is the caller's choice."""
        return self._sample(self.model, self.device, self.observation(inputs), noise=noise, num_steps=NUM_STEPS)

    def unnormalize(self, chunk):
        """torch version of openpi Unnormalize(use_quantiles) + LiberoOutputs on (B, 10, 32) -> (B, 10, 7) float32."""
        x = chunk[..., :ACTION_DIM].float()
        return (x + 1.0) / 2.0 * (self.q99 - self.q01 + 1e-6) + self.q01

    def actions(self, elements, noise):
        import torch

        with torch.inference_mode():
            return self.unnormalize(self.sample_normalized(self.model_inputs(elements), noise))


# ---------------------------------------------------------------- noise
def noise_seed(*parts: int) -> int:
    ss = np.random.SeedSequence([int(p) for p in parts])
    return int(ss.generate_state(1, dtype=np.uint64)[0] % (2**63 - 1))


def noise_array(parts) -> np.ndarray:
    """One (10, 32) standard-normal float32 noise per identity tuple (vla_distill: default_rng(seed).standard_normal)."""
    return np.random.default_rng(noise_seed(*parts)).standard_normal((HORIZON, MODEL_ACTION_DIM)).astype(np.float32)


def fixed_noise(parts_list, device="cuda"):
    import torch

    return torch.from_numpy(np.stack([noise_array(p) for p in parts_list])).to(device)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def task_meta(task_id):
    import torch
    from libero.libero import benchmark, get_libero_path

    suite = benchmark.get_benchmark_dict()[SUITE]()
    task = suite.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    p = os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file)
    states = np.asarray(torch.load(p, weights_only=False))  # official LIBERO file (numpy array pickle)
    return dict(task_id=task_id, name=task.name, language=task.language, bddl=bddl), states
