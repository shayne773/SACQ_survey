"""Simulator-side wrapper: run the repo's simulator_runtime unchanged, record videos.

Invoked by rollout_runtime as: sim_wrapper.py sim --source <simulator.py> --plan P --socket S
Physics, resets, RNG journals and success criteria are untouched: the wrapper only
copies the already-rendered `video.image` observation of every inner env step.
Env vars: EVAL_CONDITION (label), EVAL_VIDEO_DIR (mp4 output dir), EVAL_LIVE=1 (cv2 window).
"""
import json
import os
from pathlib import Path
import sys
import time

WS = Path(os.environ['SACQ_LAB']) / 'ws'
sys.path[:0] = [str(WS / 'gr00t_final_seed17_20260922'), str(WS / 'libero_pro_eval')]

import numpy as np
import seed17_adapter
from rollout_exact_acceleration_20260920 import Profile

COND = os.environ.get('EVAL_CONDITION', 'unknown')
VIDEO_DIR = Path(os.environ['EVAL_VIDEO_DIR']) if os.environ.get('EVAL_VIDEO_DIR') else None
LIVE = os.environ.get('EVAL_LIVE') == '1'
LABEL = {'bf16': 'ORIGINAL BF16', 'sacq': 'SACQ W1.5A16'}.get(COND, COND)
state = dict(task=None, episode=None, frames=[], success=False, steps=0)
tasks = None


def annotate(img, text, ok=None):
    import cv2
    big = np.ascontiguousarray(np.repeat(np.repeat(img, 2, 0), 2, 1))
    bar = np.zeros((40, big.shape[1], 3), np.uint8)
    cv2.putText(bar, text, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    if ok is not None:
        color = (60, 200, 60) if ok else (220, 60, 60)
        cv2.putText(bar, 'SUCCESS' if ok else 'FAIL', (big.shape[1] - 110, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return np.concatenate([bar, big], 0)


viewer = dict(handle=None, env=None)


def live(env):
    """MuJoCo passive viewer (GLFW, shown through WSLg) on the live simulation."""
    import mujoco, mujoco.viewer
    if viewer['env'] is not env:
        if viewer['handle'] is not None:
            viewer['handle'].close()
        m, d = env._env.sim.model._model, env._env.sim.data._data
        h = mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False)
        h.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        h.cam.fixedcamid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, 'agentview')
        viewer.update(handle=h, env=env)
    h = viewer['handle']
    if h.is_running():
        t = state['task']
        status = 'SUCCESS' if state['success'] else '...'
        h.set_texts((None, None, f'{LABEL}  |  task {t}  episode {state["episode"]}',
                     f'step {len(state["frames"]) - 1}   {status}'))
        h.sync()


def frame(obs, env=None):
    img = np.asarray(obs['video.image'])
    state['frames'].append(img.copy())
    if LIVE and env is not None:
        live(env)


def flush():
    if VIDEO_DIR is None or state['episode'] is None or not state['frames']:
        state['frames'] = []
        return
    import imageio
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    ok = state['success']
    name = f'task{state["task"]}_ep{state["episode"]:02d}_{COND}_{"success" if ok else "fail"}.mp4'
    lang = tasks[state['task']]['language'] if tasks else ''
    with imageio.get_writer(str(VIDEO_DIR / name), fps=20, codec='libx264', quality=7,
                            macro_block_size=8) as w:
        for i, img in enumerate(state['frames']):
            w.append_data(annotate(img, f'{LABEL} | T{state["task"]} ep{state["episode"]} | {i:3d}',
                                   ok if i == len(state['frames']) - 1 or i > len(state['frames']) - 30 else None))
    meta = dict(task=state['task'], episode=state['episode'], condition=COND, success=ok,
                env_steps=len(state['frames']) - 1, language=lang)
    (VIDEO_DIR / (name[:-4] + '.json')).write_text(json.dumps(meta))
    state['frames'] = []


def recording_class(libero_class):
    base = _original(libero_class)

    class Recording(base):
        def reset(self, seed=None, options=None):
            obs, info = super().reset(seed=seed, options=options)
            state['frames'] = []
            frame(obs, self)
            return obs, info

        def step(self, action):
            obs, reward, done, truncated, info = super().step(action)
            state['success'] |= bool(np.any(info['success']))
            frame(obs, self)
            return obs, reward, done, truncated, info
    return Recording


_original = seed17_adapter.official_env_class
seed17_adapter.official_env_class = recording_class


class RecordingProfile(Profile):
    def timed(self, name, fn, *a, **kw):
        global tasks
        if name == 'environment_reset':
            flush()
            state.update(episode=a[1], success=False)
        result = super().timed(name, fn, *a, **kw)
        if name == 'waiting_for_policy' and isinstance(result, dict):
            if result.get('op') == 'task':
                flush()
                state['task'] = result['task_index']
            elif result.get('op') == 'close':
                flush()
        if name == 'environment_step' and (result[2] or result[3]):
            flush()
            state['episode'] = None
        return result


def main():
    global tasks
    import argparse
    if len(sys.argv) < 2 or sys.argv[1] != 'sim':
        raise ValueError('Expected sim --source ... --plan ... --socket ...')
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    args, rest = parser.parse_known_args(sys.argv[2:])
    plan = json.loads(Path(rest[rest.index('--plan') + 1]).read_text())
    tasks = json.loads((Path(plan['root']) / 'tasks.json').read_text())
    import simulator_runtime
    profile = RecordingProfile(Path(plan['root']) / 'performance_sim', f'sim_{os.getpid()}')
    sys.argv = [args.source, *rest]
    try:
        simulator_runtime.main(PROFILE=profile, egl_device='0')
    finally:
        flush()
        profile.save(True)


if __name__ == '__main__':
    main()
