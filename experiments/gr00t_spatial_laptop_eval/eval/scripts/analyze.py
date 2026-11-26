"""Aggregate full-run episodes: per-task/overall success, Wilson CIs, exact McNemar, timings, VRAM."""
import glob
import json
import math
import os
from pathlib import Path
import statistics

from scipy.stats import binomtest

E = Path(os.environ['SACQ_LAB']) / 'eval'
F = E / 'full'
tasks = json.loads((E / 'tasks.json').read_text())


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (float('nan'),) * 2
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0., c - h), min(1., c + h)


def load(cond):
    rows = {}
    for f in glob.glob(str(F / cond / 'episodes/final_seed17/libero_spatial/*/*.json')):
        r = json.loads(Path(f).read_text())
        if r.get('status') == 'COMPLETE':
            rows[(r['task'], r['episode_id'])] = r
    return rows


res = {c: load(c) for c in ('bf16', 'sacq')}
out = dict(per_task=[], overall={}, paired={}, timing={}, vram={})
for i, t in enumerate(tasks):
    row = dict(task_index=i, language=t['language'])
    for c in ('bf16', 'sacq'):
        eps = [res[c][k] for k in res[c] if k[0] == t['task']]
        k = sum(r['success'] for r in eps); n = len(eps)
        row[c] = dict(successes=k, episodes=n, rate=k / n if n else None, wilson95=wilson(k, n))
    keys = [(t['task'], e) for e in range(50) if (t['task'], e) in res['bf16'] and (t['task'], e) in res['sacq']]
    row['bf16_only'] = sum(res['bf16'][k]['success'] and not res['sacq'][k]['success'] for k in keys)
    row['sacq_only'] = sum(res['sacq'][k]['success'] and not res['bf16'][k]['success'] for k in keys)
    row['identical_initial_obs'] = all(res['bf16'][k]['initial_observation_sha256'] == res['sacq'][k]['initial_observation_sha256'] for k in keys)
    out['per_task'].append(row)
for c in ('bf16', 'sacq'):
    k = sum(r['success'] for r in res[c].values()); n = len(res[c])
    out['overall'][c] = dict(successes=k, episodes=n, rate=k / n if n else None, wilson95=wilson(k, n))
common = sorted(set(res['bf16']) & set(res['sacq']))
b = sum(res['bf16'][k]['success'] and not res['sacq'][k]['success'] for k in common)
c_ = sum(res['sacq'][k]['success'] and not res['bf16'][k]['success'] for k in common)
both = sum(res['sacq'][k]['success'] and res['bf16'][k]['success'] for k in common)
neither = len(common) - b - c_ - both
p = binomtest(b, b + c_, 0.5).pvalue if b + c_ else 1.0
out['paired'] = dict(matched_episodes=len(common), both_success=both, neither=neither,
                     bf16_success_sacq_fail=b, sacq_success_bf16_fail=c_, mcnemar_exact_p=p,
                     difference_pp=100 * (out['overall']['sacq']['rate'] - out['overall']['bf16']['rate']) if common else None,
                     identical_initial_obs_all=all(res['bf16'][k]['initial_observation_sha256'] == res['sacq'][k]['initial_observation_sha256'] for k in common),
                     discordant=[dict(task_index=[t['task'] for t in tasks].index(k[0]), episode=k[1],
                                      bf16=res['bf16'][k]['success'], sacq=res['sacq'][k]['success']) for k in common
                                 if res['bf16'][k]['success'] != res['sacq'][k]['success']])
# timing. Batches ran solo (tasks 0-2 bf16 and most of task2 sacq) or two at once (two workers, from 22:08 local).
# Each per-episode wall delta (consecutive completion times within a batch; first episode excluded since it includes
# model load) is labelled solo/concurrent from the START/DONE windows of both worker logs, and deltas overlapping a
# host suspension (gap > 30 s in the 5-s GPU logger) are excluded.
import datetime
def windows():
    out = []
    for log in ('run_all.log', 'run_w2.log'):
        p = F / log
        if not p.exists(): continue
        open_ = {}
        for line in p.read_text().splitlines():
            parts = line.split()
            if len(parts) < 3 or parts[1] not in ('START', 'DONE', 'FAIL'): continue
            ts = datetime.datetime.fromisoformat(parts[0]).timestamp(); key = (parts[2], parts[3])
            if parts[1] == 'START': open_[key] = ts
            elif key in open_: out.append((open_.pop(key), ts, key))
        for key, ts in open_.items(): out.append((ts, float('inf'), key))
    return out
W = windows()
ticks = []
for name in ('vram.txt', 'vram_w2.txt'):
    if (F / name).exists():
        ticks += [int(l.split()[0]) for l in (F / name).read_text().splitlines() if l.split()]
ticks = sorted(set(ticks)); gaps = [(a, b) for a, b in zip(ticks, ticks[1:]) if b - a > 30]
def concurrent(a, b, key):
    return any(s < b and e > a and k != key for s, e, k in W)
def suspended(a, b):
    return any(s < b and e > a for s, e in gaps)
for c in ('bf16', 'sacq'):
    solo, conc, s_solo, s_conc, steps, susp = [], [], [], [], [], 0
    for t in tasks:
        ti = [x['task'] for x in tasks].index(t['task']); key = (f'task{ti}', c)
        eps = sorted((r for k, r in res[c].items() if k[0] == t['task']), key=lambda r: r['time'])
        for a, bb in zip(eps, eps[1:]):
            d = bb['time'] - a['time']
            if suspended(a['time'], bb['time']): susp += 1; continue
            steps.append(bb['steps'])
            (conc if concurrent(a['time'], bb['time'], key) else solo).append(d)
            (s_conc if concurrent(a['time'], bb['time'], key) else s_solo).append(d / max(bb['steps'], 1))
    summ = [json.loads(Path(f).read_text()) for f in glob.glob(str(F / c / 'summary/task*.json'))]
    pol_s = pol_n = sim_s = sim_n = 0
    for f in glob.glob(str(F / c / 'performance/*.json')):
        ph = json.loads(Path(f).read_text())['phases']
        for k, v in ph.items():
            if k.startswith('policy_') and k != 'policy_warmup': pol_s += v['seconds']; pol_n += v['calls']
        if 'rpc_step' in ph: sim_s += ph['rpc_step']['seconds']; sim_n += ph['rpc_step']['calls']
    m = lambda x: statistics.mean(x) if x else None
    md = lambda x: statistics.median(x) if x else None
    out['timing'][c] = dict(
        solo=dict(n=len(solo), mean_sec_per_episode=m(solo), median_sec_per_episode=md(solo), mean_sec_per_env_step=m(s_solo)),
        two_workers=dict(n=len(conc), mean_sec_per_episode=m(conc), median_sec_per_episode=md(conc), mean_sec_per_env_step=m(s_conc)),
        excluded_host_suspension=susp, mean_env_steps=m(steps),
        policy_sec_per_call=pol_s / pol_n if pol_n else None, policy_calls=pol_n,
        sim_sec_per_call_8_steps_incl_rpc=sim_s / sim_n if sim_n else None,
        total_task_wall_hours=sum(s['wall_seconds'] for s in summ) / 3600,
        graph_states=sorted({s['acceleration']['graph_state'] for s in summ}),
        cache_states=sorted({s['acceleration']['cache_state'] for s in summ}))
    out['vram'][c] = dict(max_memory_allocated_GB=max((s['max_memory_allocated'] for s in summ), default=0) / 1e9,
                          max_memory_reserved_GB=max((s['max_memory_reserved'] for s in summ), default=0) / 1e9)
out['host_suspension_gaps'] = [(datetime.datetime.fromtimestamp(a).isoformat(), b - a) for a, b in gaps]
used = []
for name in ('vram.txt', 'vram_w2.txt'):
    if (F / name).exists():
        used += [int(l.split()[1].rstrip(',')) for l in (F / name).read_text().splitlines() if len(l.split()) >= 2]
out['vram']['nvidia_smi_device_peak_MiB_incl_desktop_renderer_and_both_workers'] = max(used) if used else None
(E / 'analysis.json').write_text(json.dumps(out, indent=2))
print(json.dumps(dict(overall=out['overall'], paired={k: v for k, v in out['paired'].items() if k != 'discordant'},
                      timing=out['timing'], vram=out['vram']), indent=1))
for r in out['per_task']:
    print(r['task_index'], r['bf16']['successes'], r['bf16']['episodes'], r['sacq']['successes'], r['sacq']['episodes'],
          r['bf16_only'], r['sacq_only'], r['identical_initial_obs'])
