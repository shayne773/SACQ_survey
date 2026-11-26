"""Replay each task's official reset sequence (llvmpipe, CPU) and log fixture poses + observation hash per episode.

Fixture poses (e.g. the wooden cabinet) live in model.body_pos, outside get_sim_state(), so the rollout's
exact-pixel reset check covers them only implicitly. This replay records them explicitly; because the replay's
observation hashes equal the rollout's recorded initial_observation_sha256, each recorded pose is the pose both
models started from. Output: eval/pose_replay.json
"""
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
tasks = json.loads((E / 'tasks.json').read_text())
Adapter = official_env_class(lib.LiberoEnv)
out = {}
for ti in [int(x) for x in sys.argv[1:]] or range(10):
    task = tasks[ti]
    states = lib.benchmark.get_benchmark_dict()[task['suite']]().get_task_init_states(task['task_id'])
    records = pickle.loads((E / 'reset/tasks' / task['suite'] / task['task'] / 'reference.pkl').read_bytes())
    env = Adapter(task['bddl'], task['language'])
    rows = []
    for e in range(50):
        s = records[e]; random.setstate(s['rng'][0]); np.random.set_state(s['rng'][1])
        obs, _ = env.reset(seed=s['reset_seed'], options=dict(episode_id=e, initial_state=np.asarray(states[e]), initial_state_sha256=s['state_sha256']))
        inner = env._env.env; m = inner.sim.model
        fx = {name: [round(float(x), 7) for x in m.body_pos[m.body_name2id(f.root_body)]] for name, f in inner.fixtures_dict.items()}
        rows.append(dict(episode=e, observation_sha256=observation_sha(obs), fixtures=fx))
    env.close()
    out[task['task']] = rows
    print('task', ti, 'done', flush=True)
    path = E / 'pose_replay.json'
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update({task['task']: rows}); path.write_text(json.dumps(old, indent=1))
os._exit(0)
