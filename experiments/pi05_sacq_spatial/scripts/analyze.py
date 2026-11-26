"""Aggregate evaluation episodes: per-task / overall success with 95% Wilson intervals, exact McNemar tests vs the
original model on paired episodes, and the pairing checks (identical first-frame fingerprints and fixture poses
for every episode across conditions)."""
from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
EVAL = LAB / "runs" / "eval"


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b, c):
    """Two-sided exact McNemar (binomial on discordant pairs, p=0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def load(cond):
    out = {}
    for f in glob.glob(str(EVAL / cond / "episodes" / "task*_ep*.json")):
        r = json.loads(Path(f).read_text())
        out[(r["task"], r["episode"])] = r
    return out


def main():
    conds = [c for c in sys.argv[1:] if not c.startswith("--")] or ["original"]
    global EVAL
    for c in sys.argv[1:]:
        if c.startswith("--eval="):
            EVAL = Path(c.split("=", 1)[1])
    data = {c: load(c) for c in conds}
    data = {c: d for c, d in data.items() if d}
    base = data.get("original")
    res = dict(conditions={}, pairing={}, discordant={})
    for c, d in data.items():
        per = {}
        for t in range(10):
            eps = [r for (tt, e), r in d.items() if tt == t]
            k = sum(r["success"] for r in eps)
            per[t] = dict(n=len(eps), k=k, rate=k / len(eps) if eps else None, ci=wilson(k, len(eps)),
                          mean_steps=sum(r["steps"] for r in eps) / len(eps) if eps else None)
        n = len(d); k = sum(r["success"] for r in d.values())
        res["conditions"][c] = dict(n=n, k=k, rate=k / n, ci=wilson(k, n), per_task=per,
                                    fingerprint_all_match=all(r["fingerprint_match"] for r in d.values()),
                                    mean_steps=sum(r["steps"] for r in d.values()) / n,
                                    mean_episode_seconds=sum(r["seconds"] for r in d.values()) / n)
        if base is not None and c != "original":
            common = sorted(set(base) & set(d))
            b = sum(base[x]["success"] and not d[x]["success"] for x in common)
            cc = sum(d[x]["success"] and not base[x]["success"] for x in common)
            both = sum(base[x]["success"] and d[x]["success"] for x in common)
            neither = len(common) - b - cc - both
            same_fp = all(base[x]["first_fingerprint"] == d[x]["first_fingerprint"] for x in common)
            same_pose = all(base[x]["fixtures"] == d[x]["fixtures"] for x in common)
            per_task = {t: dict(orig_only=sum(base[x]["success"] and not d[x]["success"] for x in common if x[0] == t),
                                quant_only=sum(d[x]["success"] and not base[x]["success"] for x in common if x[0] == t))
                        for t in range(10)}
            res["pairing"][c] = dict(paired=len(common), both=both, neither=neither, orig_only=b, quant_only=cc,
                                     mcnemar_p=mcnemar_exact(b, cc), identical_first_fingerprints=same_fp,
                                     identical_fixture_poses=same_pose, per_task=per_task)
            res["discordant"][c] = [dict(task=x[0], episode=x[1], original=base[x]["success"], quantized=d[x]["success"])
                                    for x in common if base[x]["success"] != d[x]["success"]]
    (EVAL / "analysis.json").write_text(json.dumps(res, indent=1))
    for c, r in res["conditions"].items():
        p = res["pairing"].get(c)
        extra = (f" | vs original: orig-only {p['orig_only']}, quant-only {p['quant_only']}, McNemar p={p['mcnemar_p']:.3g}, "
                 f"same fp {p['identical_first_fingerprints']}, same poses {p['identical_fixture_poses']}") if p else ""
        print(f"{c:10s} {r['k']}/{r['n']} = {r['rate']:.1%} [{r['ci'][0]:.1%}, {r['ci'][1]:.1%}]{extra}")


if __name__ == "__main__":
    main()
