"""Diagnostic (run only when the evaluation is idle): replay one task's official reset sequence in three fresh
environments in one process and compare observation hashes, fixture poses and pixel differences per episode.
Run with GALLIUM_DRIVER=d3d12 (optionally under a concurrent CUDA load) or llvmpipe. Output JSON to argv[3]."""
import json, os, pickle, random, sys, importlib.util
from pathlib import Path
L = Path(os.environ['SACQ_LAB']); WS = L / 'ws'; E = L / 'eval'
REPO = WS / 'libero_pro_eval/Isaac-GR00T-N1.7'
sys.path[:0] = [str(REPO), str(REPO / 'external_dependencies/LIBERO'), str(WS / 'gr00t_final_seed17_20260922'), str(WS / 'libero_pro_eval')]
os.environ.update(MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', MUJOCO_EGL_DEVICE_ID='0', LIBERO_CONFIG_PATH=str(L / 'libero_config'))
import numpy as np
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); v = importlib.util.module_from_spec(spec); sys.modules[name] = v; spec.loader.exec_module(v); return v
from seed17_adapter import observation_sha, official_env_class
lib = module('lib', REPO / 'gr00t/eval/sim/LIBERO/libero_env.py')
ti, N, outp = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
task = json.loads((E / 'tasks.json').read_text())[ti]
states = lib.benchmark.get_benchmark_dict()[task['suite']]().get_task_init_states(task['task_id'])
records = pickle.loads((E / 'reset/tasks' / task['suite'] / task['task'] / 'reference.pkl').read_bytes())
Adapter = official_env_class(lib.LiberoEnv)
def run():
    env = Adapter(task['bddl'], task['language']); rows = []
    for e in range(N):
        s = records[e]; random.setstate(s['rng'][0]); np.random.set_state(s['rng'][1])
        obs, _ = env.reset(seed=s['reset_seed'], options=dict(episode_id=e, initial_state=np.asarray(states[e]), initial_state_sha256=s['state_sha256']))
        inner = env._env.env; m = inner.sim.model
        fx = {n: m.body_pos[m.body_name2id(f.root_body)].copy() for n, f in inner.fixtures_dict.items()}
        rows.append((observation_sha(obs), fx, {k: np.array(obs[k]).squeeze().copy() for k in ('video.image', 'video.wrist_image')}))
    env.close(); return rows
runs = [run() for _ in range(3)]
report = dict(task=task['task'], renderer=os.environ.get('GALLIUM_DRIVER'), episodes=N, rows=[])
for e in range(N):
    base = runs[0][e]
    row = dict(episode=e, hashes=[r[e][0][:12] for r in runs], same_hash=len({r[e][0] for r in runs}) == 1,
               fixture_pose_identical=all(all(np.array_equal(base[1][n], r[e][1][n]) for n in base[1]) for r in runs[1:]),
               fixtures={n: [round(float(x), 6) for x in v] for n, v in base[1].items()})
    if not row['same_hash']:
        diffs = {}
        for i, r in enumerate(runs[1:], 1):
            for k in base[2]:
                d = np.abs(base[2][k].astype(int) - r[e][2][k].astype(int)).max(-1)
                ys, xs = np.nonzero(d)
                if len(ys): diffs[f'env0_vs_env{i}_{k}'] = dict(pixels=int(len(ys)), max_abs=int(d.max()), bbox_yx=[int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())])
        row['pixel_diffs'] = diffs
    report['rows'].append(row)
report['summary'] = dict(identical_hash=sum(r['same_hash'] for r in report['rows']), differing=[r['episode'] for r in report['rows'] if not r['same_hash']],
                         fixture_pose_identical_all=all(r['fixture_pose_identical'] for r in report['rows']))
outp.write_text(json.dumps(report, indent=1)); print(json.dumps(report['summary']), flush=True)
os._exit(0)
