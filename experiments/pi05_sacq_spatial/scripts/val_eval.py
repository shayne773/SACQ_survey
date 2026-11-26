"""Validation action error of each condition: mean over the 64 calibration-validation snapshots of ||a - r|| / ||r||
on the full unnormalized (10, 7) chunk (the stage-4 validation metric) and on the first (executed) action, against
the original BF16 model's actions for the same snapshots and fixed noise (7000 + position). Also the same metric on
the 40 stage-2 proxy snapshots."""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sacqpi as S  # noqa: E402


@torch.no_grad()
def errors(pol, snaps, refs, positions):
    rels, rel0s = [], []
    for pos in positions:
        act = S.actions(pol, snaps, pos).cpu()
        r = refs[pos]
        rels.append(float((act - r).norm() / r.norm()))
        rel0s.append(float((act[:, 0] - r[:, 0]).norm() / r[:, 0].norm()))
    return sum(rels) / len(rels), sum(rel0s) / len(rel0s)


def main():
    conds = sys.argv[1:]
    snaps = S.Snapshots()
    refs = torch.load(S.WORK / "snapshots" / "targets.pt", map_location="cpu", weights_only=False)["actions"]
    psnaps = S.Snapshots(S.WORK / "snapshots" / "proxy_snapshots.pt")
    prefs = torch.load(S.WORK / "snapshots" / "proxy_targets.pt", map_location="cpu", weights_only=False)["actions"]
    out = S.WORK / "val_eval.json"
    results = json.loads(out.read_text()) if out.exists() else {}
    for cond in conds:
        t0 = time.time()
        pol = S.load_policy()
        files = S.apply_condition(pol.model, cond)
        v, v0 = errors(pol, snaps, refs, snaps.val_positions())
        p, p0 = errors(pol, psnaps, prefs, list(range(len(psnaps))))
        results[cond] = dict(val_rel=v, val_rel_first_action=v0, proxy_rel=p, proxy_rel_first_action=p0,
                             files=files, seconds=time.time() - t0)
        print(cond, json.dumps({k: v for k, v in results[cond].items() if k != "files"}), flush=True)
        del pol, files
        gc.collect()  # the model sits in reference cycles; without this every condition leaks a 7 GB model
        torch.cuda.empty_cache()
        assert torch.cuda.memory_allocated() < 1e9, torch.cuda.memory_allocated()
        S.write_json(results, out)


if __name__ == "__main__":
    main()
