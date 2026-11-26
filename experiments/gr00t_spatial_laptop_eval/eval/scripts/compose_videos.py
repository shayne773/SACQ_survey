"""Copy required per-task clips and build original|quantized side-by-side clips."""
import glob
import json
import os
from pathlib import Path
import shutil
import subprocess

import imageio_ffmpeg

E = Path(os.environ['SACQ_LAB']) / 'eval'
OUT = Path(os.environ.get('SACQ_VIDEO_OUT', str(E / 'videos_selected')))
FF = imageio_ffmpeg.get_ffmpeg_exe()
tasks = json.loads((E / 'tasks.json').read_text())
an = json.loads((E / 'analysis.json').read_text())


def clip(cond, t, e):
    m = glob.glob(str(E / 'full' / cond / 'videos' / f'task{t}_ep{e:02d}_{cond}_*.mp4'))
    assert len(m) == 1, (cond, t, e, m)
    return Path(m[0])


per = OUT / 'per_task'; sbs = OUT / 'side_by_side'
per.mkdir(parents=True, exist_ok=True); sbs.mkdir(parents=True, exist_ok=True)
index = dict(per_task=[], side_by_side=[])
for t in range(10):
    for cond in ('bf16', 'sacq'):
        src = clip(cond, t, 0); dst = per / src.name
        shutil.copyfile(src, dst)
        index['per_task'].append(dict(file=f'per_task/{dst.name}', task=t, episode=0, model=cond,
                                      success='_success' in src.name, language=tasks[t]['language']))


def side(t, e):
    a, b = clip('bf16', t, e), clip('sacq', t, e)
    oa = a.stem.split('_')[-1]; ob = b.stem.split('_')[-1]
    name = f'task{t}_ep{e:02d}_original-{oa}_vs_sacq-{ob}.mp4'
    def duration(p):
        d = subprocess.run([FF, '-i', str(p)], capture_output=True, text=True).stderr.split('Duration: ')[1].split(',')[0]
        h, m, sec = d.split(':'); return 3600 * float(h) + 60 * float(m) + float(sec)
    dur = max(duration(a), duration(b)) + 1.0
    # Hold the shorter clip's last frame so both play to their own end.
    subprocess.run([FF, '-y', '-loglevel', 'error', '-i', str(a), '-i', str(b), '-filter_complex',
                    '[0:v]tpad=stop_mode=clone:stop_duration=60[a];[1:v]tpad=stop_mode=clone:stop_duration=60[b];'
                    '[a][b]hstack=inputs=2[v]', '-map', '[v]', '-t', f'{dur:.2f}', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-crf', '23', str(sbs / name)], check=True)
    index['side_by_side'].append(dict(file=f'side_by_side/{name}', task=t, episode=e, original=oa, sacq=ob,
                                      language=tasks[t]['language']))


chosen = []
disc = an['paired']['discordant']
# Spread over tasks and both directions: one original-succeeds/SACQ-fails pair per task (max 6), then up to 3 SACQ-only successes.
seen = set()
for d in disc:
    if d['bf16'] and not d['sacq'] and d['task_index'] not in seen and len(chosen) < 6:
        chosen.append((d['task_index'], d['episode'])); seen.add(d['task_index'])
for d in [d for d in disc if d['sacq'] and not d['bf16']][:3]:
    chosen.append((d['task_index'], d['episode']))
for t, e in [(0, 0), (4, 0), (9, 0)]:
    if (t, e) not in chosen:
        chosen.append((t, e))
for t, e in chosen:
    side(t, e)
(OUT / 'index.json').write_text(json.dumps(index, indent=2))
print(json.dumps(index['side_by_side'], indent=1)); print(len(index['per_task']), 'per-task clips')
