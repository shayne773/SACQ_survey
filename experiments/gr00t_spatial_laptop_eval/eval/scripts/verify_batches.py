"""Verify all 20 batches from per-episode result files (not completion markers).
Per condition/task: episodes 0-49 each present exactly once with status COMPLETE; each episode's
initial_observation_sha256 equals (a) the reset reference the simulator process rendered for that episode
(renderer_references), (b) the CPU pose replay (pose_replay.json, which also records fixture poses), and
(c) the other condition's value. Also counts reset reconstructions recorded in the simulator profiles.
Writes eval/verify_batches.json."""
import glob, json, os
from pathlib import Path
E = Path(os.environ['SACQ_LAB']) / 'eval'; F = E / 'full'
tasks = json.loads((E / 'tasks.json').read_text()); replay = json.loads((E / 'pose_replay.json').read_text())
out = dict(batches=[], ok=True)
hashes = {}
for c in ('bf16', 'sacq'):
    refs = {}
    for f in glob.glob(str(F / c / 'renderer_references/*/libero_spatial/*.json')):
        r = json.loads(Path(f).read_text())
        for e in r['episodes']: refs.setdefault((Path(f).stem, e['episode_id']), set()).add(e['renderer_observation_sha256'])
    recon = 0
    for f in glob.glob(str(F / c / 'performance_sim/*.json')):
        recon += json.loads(Path(f).read_text())['phases'].get('reset_reconstruction', {}).get('calls', 0)
    for ti, t in enumerate(tasks):
        files = sorted(glob.glob(str(F / c / 'episodes/final_seed17/libero_spatial' / t['task'] / '*.json')))
        rows = [json.loads(Path(f).read_text()) for f in files]
        ids = sorted(r.get('episode_id') for r in rows)
        complete = [r for r in rows if r.get('status') == 'COMPLETE']
        ref_ok = sum(r['initial_observation_sha256'] in refs.get((t['task'], r['episode_id']), set()) for r in complete)
        rep = {x['episode']: x['observation_sha256'] for x in replay[t['task']]}
        rep_ok = sum(r['initial_observation_sha256'] == rep[r['episode_id']] for r in complete)
        for r in complete: hashes.setdefault((ti, r['episode_id']), {})[c] = r['initial_observation_sha256']
        b = dict(condition=c, task_index=ti, files=len(files), episode_ids_0_49=ids == list(range(50)),
                 complete=len(complete), reset_hash_matches_process_reference=ref_ok, matches_cpu_pose_replay=rep_ok,
                 successes=sum(r['success'] for r in complete))
        b['ok'] = b['episode_ids_0_49'] and b['complete'] == 50 and ref_ok == 50 and rep_ok == 50
        out['ok'] &= b['ok']; out['batches'].append(b)
    out[f'{c}_reset_reconstructions'] = recon
paired = [k for k, v in hashes.items() if len(v) == 2]
out['paired_episodes'] = len(paired); out['paired_identical_initial_obs'] = sum(hashes[k]['bf16'] == hashes[k]['sacq'] for k in paired)
out['ok'] &= out['paired_episodes'] == 500 and out['paired_identical_initial_obs'] == 500
(E / 'verify_batches.json').write_text(json.dumps(out, indent=1))
for b in out['batches']:
    print(b['condition'], b['task_index'], b['files'], b['complete'], b['reset_hash_matches_process_reference'], b['matches_cpu_pose_replay'], b['successes'], 'OK' if b['ok'] else 'BAD')
print({k: v for k, v in out.items() if k != 'batches'})
