"""Minimal driver around the repo's seed17 rollout path (rollout_runtime.main).

Reuses unchanged: rollout_runtime (policy loop, NoiseInjection, saved-noise reading),
model_install -> gr00t_rollout_model.install (SACQ overlays + GPTQ background),
DefaultAcceleration (exactness-checked CUDA graph), simulator_runtime + seed17_adapter
(official initial states, seed17 reset journal, 8-step chunks, 720-step cap).
Replaced: the receipt/authorization bookkeeping of rollout_worker.py (EXECUTION_RECEIPT,
protocol.json, sources list), because those receipts are not public; the plan below is
written by this driver and its inputs are hashed into plan_inputs.json for the report.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

L = Path(os.environ['SACQ_LAB'])
WS = L / 'ws'
R = WS / 'gr00t_final_seed17_20260922'
B = WS / 'libero_pro_eval'
REPO = B / 'Isaac-GR00T-N1.7'
E = L / 'eval'
BUNDLE = L / 'bundles' / 'sacq'
SUITE = 'libero_spatial'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def build_plan(condition, task_index, root, processor):
    tasks = json.loads((E / 'tasks.json').read_text())
    root.mkdir(parents=True, exist_ok=True)
    if not (root / 'tasks.json').exists():
        shutil.copyfile(E / 'tasks.json', root / 'tasks.json')
    assert json.loads((root / 'tasks.json').read_text()) == tasks
    contract_sha = sha(E / 'reset_contract.json')
    plan = dict(
        root=str(root), method='final_seed17', rollout_authorized=True, reset_identity_status='PASS',
        diagnostic_only=True, held_out=False,
        condition=condition, selected_suites=[SUITE], suites=[SUITE],
        cosmos=str(processor), models={SUITE: str(BUNDLE / 'model')},
        language_action_nominal_bpw=16 if condition == 'bf16' else 1.5,
        worker_task_index=task_index, socket_root=str(E / 'sock'),
        sim_python=str(L / 'sim/bin/python'), sim_source=str(R / 'simulator.py'),
        repo=str(REPO), worker_source=str(B / 'quantization_analysis/workshop_suitealloc_long500_s7_env1_20260919/source/worker.py'),
        rollout=dict(environment_seed=17, policy_seed=20260824, episodes_per_task=50, n_envs=1, action_steps=8,
                     max_episode_steps=720, termination='official_done_or_720', success='success_any', settling_steps=0),
        noise_root=str(E / 'noise'), reset_audit_root=str(E / 'reset'),
        reset_task_contract_sha256={f'{SUITE}/{t["task"]}': contract_sha for t in tasks},
        render_reference_mode='same_process_saved_rng_exact', paired_initial_observation_sha256={},
        denoising_steps=4)
    if condition == 'sacq':
        spec = json.loads((BUNDLE / 'DOWNLOAD_SPEC.json').read_text())['manifest']['inference']
        model = json.loads(json.dumps(spec['workshop_model']))
        for ref in [model[k] for k in ('vlm', 'dit', 'vlm_spec', 'dit_spec')] + model['gptq']:
            assert spec['files'][ref['path']] == ref['sha256']
            ref['path'] = str(BUNDLE / ref['path'])
        plan['workshop_model'] = model
    path = root / 'plans' / f'task{task_index}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2))
    return path, plan


def environment():
    os.environ.update(MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                      OPENBLAS_NUM_THREADS='2', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      TOKENIZERS_PARALLELISM='false', NO_ALBUMENTATIONS_UPDATE='1',
                      LIBERO_CONFIG_PATH=str(L / 'libero_config'))
    for key in list(os.environ):
        if key.startswith(('VLM_', 'DIT_', 'SACQ_', 'ACTION_HEAD_', 'ACT_QUANT', 'TAC_')):
            os.environ.pop(key)
    sys.path[:0] = [str(R), str(B), str(REPO), str(WS / 'simplerenv_n17_feasibility_2026_08_27'),
                    str(WS / 'libero_transfer_quantstats_20260922')]


def sim_smoke(plan_path, plan, calls):
    """Drive the simulator wrapper with zero actions (no policy) to test plumbing."""
    import socket, subprocess
    import numpy as np
    from vast12_hqq_sim_2026_09_09 import send, receive
    sock_path = str(Path(plan['socket_root']) / f'smoke_{os.getpid()}.sock')
    Path(plan['socket_root']).mkdir(parents=True, exist_ok=True)
    sim = subprocess.Popen([plan['sim_python'], str(E / 'scripts/sim_wrapper.py'), 'sim', '--source', plan['sim_source'],
                            '--plan', str(plan_path), '--socket', sock_path],
                           env={**os.environ, 'PYTHONPATH': plan['repo'] + ':' + plan['repo'] + '/external_dependencies/LIBERO'})
    while not Path(sock_path).exists():
        assert sim.poll() is None, 'sim died'
        time.sleep(.2)
    s = socket.socket(socket.AF_UNIX); s.connect(sock_path)
    rpc = lambda v: (send(s, v), receive(s))[1]
    t = time.time(); print(rpc({'op': 'task', 'task_index': plan['worker_task_index']}), time.time() - t)
    for ep in (0, 1):
        t = time.time(); (slot, obs, h), = rpc({'op': 'reset', 'items': [(0, ep)]}); print('reset', ep, h[:12], time.time() - t)
        zero = {k: np.zeros((8, 1), np.float32) for k in ('action.x', 'action.y', 'action.z', 'action.roll', 'action.pitch', 'action.yaw')}
        zero['action.gripper'] = np.zeros((8, 1), np.float32)
        t = time.time()
        for c in range(calls):
            (slot, obs, done, success, steps), = rpc({'op': 'step', 'items': [(0, zero)]})
            if done: break
        print('steps', steps, 'done', done, 'success', success, 'sec/call', (time.time() - t) / (c + 1))
    send(s, {'op': 'close'}); s.close(); sim.wait(60)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--condition', choices=('bf16', 'sacq'), required=True)
    p.add_argument('--task-index', type=int, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--processor', type=Path, default=BUNDLE / 'processor')
    p.add_argument('--episodes', type=int, default=50, help='Run only the first N episodes (pilot); 50 = full task')
    p.add_argument('--only-episode', type=int, help='Run just this one episode (live viewing)')
    p.add_argument('--sim-smoke', type=int, default=0, help='Zero-action simulator test with this many calls')
    a = p.parse_args()
    root = a.root.resolve()
    plan_path, plan = build_plan(a.condition, a.task_index, root, a.processor)
    environment()
    os.environ['EVAL_CONDITION'] = a.condition
    os.environ.setdefault('EVAL_VIDEO_DIR', str(root / 'videos'))
    if a.sim_smoke:
        return sim_smoke(plan_path, plan, a.sim_smoke)
    import torch
    from rollout_acceleration_default_20260920 import AsyncProfile, DefaultAcceleration
    import rollout_runtime as worker
    if a.episodes < 50 or a.only_episode is not None:
        # Pilot: mark later episodes as already complete placeholders in a scratch dir is not allowed;
        # instead pre-create result files outside the counted set and delete them afterwards.
        task = json.loads((root / 'tasks.json').read_text())[a.task_index]
        rd = root / 'episodes/final_seed17' / SUITE / task['task']
        rd.mkdir(parents=True, exist_ok=True)
        skip = [e for e in range(50) if e != a.only_episode] if a.only_episode is not None else range(a.episodes, 50)
        for e in skip:
            f = rd / f'{e:03d}.json'
            if not f.exists():
                f.write_text(json.dumps(dict(status='PILOT_SKIPPED', episode_id=e)))
    profile = AsyncProfile(root / 'performance', f'task{a.task_index}')
    accel = DefaultAcceleration(profile)
    Path(plan['socket_root']).mkdir(parents=True, exist_ok=True)
    sys.argv = [plan['worker_source'], '--plan', str(plan_path), '--gpu', str(a.task_index)]
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    try:
        worker.main(PROFILE=profile, ACCEL=accel, PROFILE_WRAPPER=str(E / 'scripts/sim_wrapper.py'))
    finally:
        summary = dict(condition=a.condition, task_index=a.task_index, wall_seconds=time.time() - started,
                       max_memory_allocated=torch.cuda.max_memory_allocated(),
                       max_memory_reserved=torch.cuda.max_memory_reserved(),
                       gpu=torch.cuda.get_device_name(), torch=torch.__version__, acceleration=accel.audit(),
                       plan_sha256=sha(plan_path))
        (root / 'summary').mkdir(exist_ok=True)
        (root / 'summary' / f'task{a.task_index}.json').write_text(json.dumps(summary, indent=2, default=str))
        profile.save(True)


if __name__ == '__main__':
    main()
