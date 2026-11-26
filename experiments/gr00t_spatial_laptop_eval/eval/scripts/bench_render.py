import os, sys, json, time, importlib.util
import numpy as np
repo=os.environ['SACQ_LAB'] + '/ws/libero_pro_eval/Isaac-GR00T-N1.7'
sys.path[:0]=[repo, repo+'/external_dependencies/LIBERO']
spec=importlib.util.spec_from_file_location('le', repo+'/gr00t/eval/sim/LIBERO/libero_env.py'); lib=importlib.util.module_from_spec(spec); spec.loader.exec_module(lib)
t=json.load(open('tasks.json'))[0]
env=lib.LiberoEnv(t['bddl'], t['language'])
from OpenGL import GL
obs,_=env.reset(seed=17)
a={k:np.zeros(1,np.float32) for k in ('action.x','action.y','action.z','action.roll','action.pitch','action.yaw','action.gripper')}
for _ in range(5): env.step(a)
t0=time.time()
for _ in range(40): o=env.step(a)[0]
print(os.environ.get('GALLIUM_DRIVER','llvmpipe'), 'sec/env-step', (time.time()-t0)/40, 'img mean', o['video.image'].mean(), flush=True)
os._exit(0)
