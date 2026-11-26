"""Write the LIBERO Spatial task registry (same fields as vast12 --registry-only) and reset contracts."""
import json, os, sys, hashlib
from pathlib import Path
L = Path(os.environ['SACQ_LAB'])
repo = L/'ws/libero_pro_eval/Isaac-GR00T-N1.7'
adapter = L/'ws/gr00t_final_seed17_20260922'
E = L/'eval'
sys.path[:0] = [str(repo), str(repo/'external_dependencies/LIBERO')]
from libero.libero import benchmark, get_libero_path
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
b = benchmark.get_benchmark_dict()['libero_spatial']()
assert b.get_num_tasks() == 10
tasks = []
for i in range(10):
    t = b.get_task(i)
    tasks.append(dict(suite='libero_spatial', task=t.name, task_id=i, language=t.language,
        bddl=os.path.join(get_libero_path('bddl_files'), t.problem_folder, t.bddl_file)))
(E/'tasks.json').write_text(json.dumps(tasks, indent=2))
assets = repo/'external_dependencies/LIBERO/libero/libero'
srcs = [adapter/'seed17_adapter.py', repo/'gr00t/eval/sim/LIBERO/libero_env.py', repo/'gr00t/eval/sim/wrapper/multistep_wrapper.py']
inputs = []; ctasks = []
for i in range(10):
    t = b.get_task(i)
    init = assets/'init_files'/t.problem_folder/t.init_states_file
    bddl = assets/'bddl_files'/t.problem_folder/t.bddl_file
    inputs += [dict(path=str(init), sha256=sha(init)), dict(path=str(bddl), sha256=sha(bddl))]
    ctasks.append(dict(suite='libero_spatial', task=t.name, task_id=i, language=t.language,
                       initial_file=dict(path=str(init), sha256=sha(init))))
contract = dict(schema='sacq_reset_inputs_v1',
    settings=dict(environment_seed=17, python_seed=17, numpy_seed=17, n_envs=1, episode_ids=list(range(50)),
                  n_action_steps=8, max_episode_steps=720, terminate_on_success=False, policy_forward_calls=0, environment_steps=0),
    tasks=ctasks, repo=str(repo), adapter_directory=str(adapter), output=str(E/'reset'),
    sources=[dict(path=str(p), sha256=sha(p)) for p in srcs], inputs=inputs)
(E/'reset_contract.json').write_text(json.dumps(contract, indent=2))
print('tasks', len(tasks)); print('\n'.join(t['language'] for t in tasks))
