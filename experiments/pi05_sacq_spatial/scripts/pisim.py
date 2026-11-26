"""Torch-free LIBERO simulator worker for pi0.5 rollouts (adapted from the SmolVLA port's simworker.py).

Environment semantics follow openpi's examples/libero/main.py (as used by the vla_distill harness libero_eval.py):
OffScreenRenderEnv(bddl, camera_heights=256, camera_widths=256) with LIBERO's defaults otherwise (control_freq 20,
hard resets, delta OSC control); per episode env.reset() -> set_init_state(official state) -> 10 no-op settle steps
([0]*6 + [-1]); termination when env.step returns done (LIBERO's success) or after 220 policy steps (Spatial).
The policy input element is built here exactly as the openpi client does (picore.client_element: 180-degree
rotation, resize_with_pad to 224, uint8, eef pos + axis-angle + gripper qpos, task language).

Episodes are reset through an explicit reset journal (reset_journal.py):
  restore Python/NumPy RNG state recorded before this episode's reset in the reference pass
  -> env.seed(seed) on journal episode 0 only -> env.reset() -> force the recorded fixture poses
  -> set_init_state(official state) -> 10 settle steps -> first-frame fingerprint.
The fixture (cabinet/stove) pose lives in model.body_pos, outside the simulator state, and LIBERO's hard reset makes
it depend on the environment's reset history even with restored RNG; forcing the reference pose makes every episode
start from the reference scene regardless of which worker runs it, and the fingerprint check proves it.
"""
from __future__ import annotations

import hashlib
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import picore as P  # noqa: E402  (torch-free at import time)

DUMMY = [0.0] * 6 + [-1.0]
FP_KEYS = ("agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos", "robot0_eef_quat",
           "robot0_gripper_qpos", "robot0_gripper_qvel", "robot0_joint_pos", "robot0_joint_vel")


def make_env(bddl_file):
    from libero.libero.envs import OffScreenRenderEnv

    return OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=P.ENV_RES, camera_widths=P.ENV_RES)


def fingerprint(raw) -> str:
    h = hashlib.sha256()
    for k in FP_KEYS:
        a = np.ascontiguousarray(raw[k])
        h.update(k.encode()); h.update(str((a.shape, a.dtype)).encode()); h.update(a.tobytes())
    return h.hexdigest()


def fixture_poses(env):
    inner = env.env
    m = inner.sim.model
    out = {}
    for name, f in inner.fixtures_dict.items():
        b = m.body_name2id(f.root_body)
        out[name] = (m.body_pos[b].copy(), m.body_quat[b].copy())
    return out


def force_fixture_poses(env, poses):
    inner = env.env
    m = inner.sim.model
    for name, (p, q) in poses.items():
        b = m.body_name2id(inner.fixtures_dict[name].root_body)
        m.body_pos[b] = p
        m.body_quat[b] = q
    # No sim.forward() here: set_init_state() runs forward() right after with the forced poses, and an extra
    # forward() changes MuJoCo's solver warm start (qacc_warmstart, not part of the flattened state), which made
    # e.g. Spatial task 0 episode 2 settle differently from its journal reference (fingerprint mismatch).


def reset_episode(env, state, rng, seed, poses=None):
    """One journal reset. Returns (raw_obs after the settle steps, natural fixture poses before forcing)."""
    random.setstate(rng[0])
    np.random.set_state(rng[1])
    if seed is not None:
        env.seed(seed)
    env.reset()
    natural = fixture_poses(env)
    if poses is not None:
        force_fixture_poses(env, poses)
    raw = env.set_init_state(state)
    for _ in range(P.NUM_STEPS_WAIT):
        raw, _, _, _ = env.step(DUMMY)
    return raw, natural


def rss_mb() -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) // 1024
    return 0


def group_main(conn, group_id, nslots):
    """Several environment slots in one process. One message = a list of per-slot commands executed in order; the
    reply is the list of results plus this process's resident memory. Commands:
      ("task", slot, bddl, language) -> ("ok",)    create/replace the slot's env if the task changed
      ("reset", slot, state, rng, seed, poses, expected_fp, record) -> ("obs", element, fp, ok, fixtures, secs)
      ("step", slot, action) -> ("obs", element, done)
      ("video", slot, path, fps) -> ("ok",)
    """
    os.environ.setdefault("LP_NUM_THREADS", "4")
    envs = [None] * nslots
    bddls = [None] * nslots
    langs = [None] * nslots
    frames = [None] * nslots
    try:
        while True:
            msg = conn.recv()
            if msg[0] == "close":
                break
            out = []
            for cmd in msg[1]:
                op, slot = cmd[0], cmd[1]
                if op == "task":
                    if cmd[2] != bddls[slot]:
                        if envs[slot] is not None:
                            envs[slot].close()
                        bddls[slot] = cmd[2]
                        envs[slot] = make_env(cmd[2])
                    langs[slot] = cmd[3]
                    out.append(("ok",))
                elif op == "reset":
                    _, _, state, rng, seed, poses, expected, record = cmd
                    env = envs[slot]
                    t = time.time()
                    raw, _natural = reset_episode(env, state, rng, seed, poses)
                    fp = fingerprint(raw)
                    applied = fixture_poses(env)
                    el = P.client_element(raw, langs[slot])
                    frames[slot] = [(raw["agentview_image"][::-1, ::-1].copy(),
                                     raw["robot0_eye_in_hand_image"][::-1, ::-1].copy())] if record else None
                    out.append(("obs", el, fp, fp == expected,
                                {k: (v[0].tolist(), v[1].tolist()) for k, v in applied.items()}, time.time() - t))
                elif op == "step":
                    env = envs[slot]
                    raw, reward, done, info = env.step(np.asarray(cmd[2]).tolist())
                    if frames[slot] is not None:
                        frames[slot].append((raw["agentview_image"][::-1, ::-1].copy(),
                                             raw["robot0_eye_in_hand_image"][::-1, ::-1].copy()))
                    out.append(("obs", P.client_element(raw, langs[slot]), bool(done)))
                elif op == "video":
                    path, fps = cmd[2], cmd[3]
                    if frames[slot]:
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        np.savez_compressed(path, agentview=np.stack([f[0] for f in frames[slot]]),
                                            wrist=np.stack([f[1] for f in frames[slot]]))
                    frames[slot] = None
                    out.append(("ok",))
                else:
                    raise ValueError(op)
            conn.send((out, rss_mb()))
    except EOFError:
        pass
    finally:
        for env in envs:
            if env is not None:
                env.close()
