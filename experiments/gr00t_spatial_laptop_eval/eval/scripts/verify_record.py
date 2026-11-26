import sys, csv, hashlib, base64
from pathlib import Path
sp = Path(sys.argv[1]); bad = {}
for rec in sp.glob('*.dist-info/RECORD'):
    for row in csv.reader(rec.open()):
        if len(row) < 2 or not row[1].startswith('sha256='): continue
        p = (sp / row[0]).resolve()
        if not p.is_file(): bad.setdefault(rec.parent.name, []).append(('missing', row[0])); continue
        d = base64.urlsafe_b64encode(hashlib.sha256(p.read_bytes()).digest()).rstrip(b'=').decode()
        if d != row[1][7:]: bad.setdefault(rec.parent.name, []).append(('hash', row[0]))
for k, v in bad.items(): print(k, len(v), v[:2])
print('dists_bad', len(bad))
