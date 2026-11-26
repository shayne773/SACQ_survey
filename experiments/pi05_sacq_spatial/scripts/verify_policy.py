"""Check picore.Policy against openpi's own create_trained_policy path (same elements, same noise).

phase obs      : 8 real LIBERO Spatial observations (tasks 0-7, init state 0, after the 10 settle steps), saved.
phase mine     : picore.Policy actions, batch 1 and batch 8; parameter dtypes; timing; VRAM.
phase official : openpi create_trained_policy(pi05_libero, CKPT, num_steps=10).infer(element, noise=...) per element,
                 torch.compile disabled (TORCHDYNAMO_DISABLE=1) so both run eager.
phase compare  : max abs differences.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import picore as P  # noqa: E402

OUT = P.LAB / "verify"
OUT.mkdir(exist_ok=True)


def phase_obs():
    from libero.libero.envs import OffScreenRenderEnv
    els = []
    for t in range(8):
        meta, states = P.task_meta(t)
        env = OffScreenRenderEnv(bddl_file_name=meta["bddl"], camera_heights=P.ENV_RES, camera_widths=P.ENV_RES)
        env.seed(7)
        env.reset()
        raw = env.set_init_state(states[0])
        for _ in range(P.NUM_STEPS_WAIT):
            raw, _, _, _ = env.step([0.0] * 6 + [-1.0])
        els.append(P.client_element(raw, meta["language"]))
        env.close()
    np.save(OUT / "elements.npy", np.array(els, dtype=object), allow_pickle=True)
    print("saved", len(els))


def load_els():
    return list(np.load(OUT / "elements.npy", allow_pickle=True))


def phase_mine():
    import torch
    els = load_els()
    noise = np.stack([P.noise_array((1, t)) for t in range(len(els))])
    t0 = time.time()
    pol = P.Policy("cuda")
    load_s = time.time() - t0
    dtypes = {}
    for n, p in pol.model.named_parameters():
        dtypes[str(p.dtype)] = dtypes.get(str(p.dtype), 0) + p.numel()
    single = []
    for i, e in enumerate(els):
        a = pol.actions([e], torch.from_numpy(noise[i:i + 1]).cuda())
        single.append(a[0].cpu().numpy())
    torch.cuda.synchronize()
    ts = []
    for _ in range(3):
        t = time.time()
        a8 = pol.actions(els, torch.from_numpy(noise).cuda())
        torch.cuda.synchronize()
        ts.append(time.time() - t)
    ts1 = []
    for _ in range(3):
        t = time.time()
        pol.actions(els[:1], torch.from_numpy(noise[:1]).cuda())
        torch.cuda.synchronize()
        ts1.append(time.time() - t)
    # numpy output-transform path for comparison with the torch unnormalize
    with torch.inference_mode():
        raw = pol.sample_normalized(pol.model_inputs(els[:1]), torch.from_numpy(noise[:1]).cuda())
    np_out = pol.out({"state": np.zeros(32, np.float32), "actions": raw[0].float().cpu().numpy()})["actions"]
    np.savez(OUT / "mine.npz", single=np.stack(single), batch=a8.cpu().numpy(), np_out=np_out)
    info = dict(load_s=load_s, dtypes=dtypes, batch8_s=ts, batch1_s=ts1,
                max_alloc_gb=torch.cuda.max_memory_allocated() / 1e9,
                unnorm_torch_vs_numpy=float(np.abs(np_out - single[0]).max()),
                batch_vs_single=float(np.abs(a8.cpu().numpy() - np.stack(single)).max()))
    (OUT / "mine.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


def phase_official():
    import torch
    from openpi.policies import policy_config as _policy_config
    els = load_els()
    noise = np.stack([P.noise_array((1, t)) for t in range(len(els))])
    torch.manual_seed(0)
    t0 = time.time()
    policy = _policy_config.create_trained_policy(P.train_config(), P.CKPT, sample_kwargs={"num_steps": 10})
    load_s = time.time() - t0
    acts = [policy.infer(dict(e), noise=noise[i])["actions"] for i, e in enumerate(els)]
    dtypes = {}
    for n, p in policy._model.named_parameters():
        dtypes[str(p.dtype)] = dtypes.get(str(p.dtype), 0) + p.numel()
    np.savez(OUT / "official.npz", single=np.stack(acts))
    (OUT / "official.json").write_text(json.dumps(dict(load_s=load_s, dtypes=dtypes), indent=1))
    print(load_s, dtypes)


def phase_compare():
    m = np.load(OUT / "mine.npz")
    o = np.load(OUT / "official.npz")
    d = np.abs(m["single"] - o["single"])
    print(json.dumps(dict(max_abs=float(d.max()), mean_abs=float(d.mean()), bitwise=bool((d == 0).all()),
                          per_elem=[float(x) for x in d.reshape(len(d), -1).max(1)]), indent=1))


if __name__ == "__main__":
    {"obs": phase_obs, "mine": phase_mine, "official": phase_official, "compare": phase_compare}[sys.argv[1]]()
