# SACQ on pi0.5 (pi05_libero), LIBERO Spatial — unfinished

Applies the SACQ recipe from this repository to the official openpi **pi05_libero** checkpoint (PyTorch
conversion) and evaluates it on LIBERO Spatial. The pipeline calibrates on the original model's own rollouts,
builds a 4-bit background (SigLIP + projector + token embedding), fits stage-1 codebooks for 284 candidate
linears (language model, action expert and its adaRMS modulation layers, ≈ 2.30B weights), and then runs
stage 2 bit allocation, stage 3 index correction and stage 4 calibration at a 1.5-bit target. Comparisons
are uniform 1.5-bit (K=64) and RTN 2/3-bit.

## Status: paused partway through stage 2 (bit allocation); no success rate yet

The run was stopped deliberately on 2026-11-25 and is resumable. Done so far:

- original-model pilot 19/20 on Spatial; reset journals (seeds 17 and 1234); calibration collection (seed 1234,
  300 episodes);
- snapshots, GPTQ moments / 4-bit background / 3- and 4-bit GPTQ bank, stage-1 statistics, the codebook bank
  (K = 32 … 2048), the uniform K=64 stage-1 model, RTN 2/3-bit references, validation errors of those;
- stage 2 allocator: checkpointed at step 1779 (nominal 1.60, actual 1.851 bits per weight); the byte target is
  the uniform K=64 model's size (1.7505 serialized bits per weight).

Not done: the rest of allocation, stages 3 and 4, and **every simulator evaluation of a quantized model**. There
is no final SACQ success rate. The uniform-1.5-bit stage 3/4 side comparison is commented out at the end of
`tools/pipeline.sh`.

### Results so far (no rollouts)

Action error against the original model on 64 held-out calibration snapshots (mean ‖a − r‖ / ‖r‖ over the
full (10, 7) chunk, same fixed noise), from [`sacq/val_eval.json`](sacq/val_eval.json):

| Condition | val | val, first action | proxy (40 snapshots) |
|---|---:|---:|---:|
| original | 0 (bitwise) | 0 | 0 |
| bg: 4-bit SigLIP + projector + embedding only | 0.0240 | 0.0115 | 0.0158 |
| u1: uniform 1.5-bit (K=64), stage 1 only | 0.2430 | 0.2054 | 0.2409 |
| rtn3: RTN 3-bit, group 64 | 0.0873 | 0.0664 | 0.0715 |
| rtn2: RTN 2-bit, group 64 | 0.3660 | 0.2960 | 0.3272 |

Storage ([`sacq/bits.json`](sacq/bits.json)): uniform K=64 model 1.05 GB vs 6.95 GB original (candidate layers
at 1.7505 effective bits per weight).

## Hardware

RTX 5080 Laptop GPU (16 GB), WSL2 Ubuntu 24.04, CPU rendering (Mesa llvmpipe). Batch-1 policy calls take about
0.3 s; stage 4 runs about 20 s per update after expert-step checkpointing (peak 9.5 GB).

## What it expects (not in this repository)

| Variable / path | Contents |
|---|---|
| `OPENPI_DIR` (default `./openpi`) | An openpi checkout with its `.venv` (LIBERO in `third_party/libero`, robosuite 1.4.1, mujoco 3.2.3) |
| `CKPT` (required) | openpi's PyTorch conversion of the `pi05_libero` checkpoint (read-only) |
| `SACQ_REPO` (default: this repository) | Provides the `sacq` package used by the scripts |
| `MAMBA_ROOT_PREFIX` env `vla-tools` | User-local gcc/cmake toolchain sourced by `toolchain_env.sh` (no system compiler on the laptop) |
| `SACQ_STATUS_FILE` (optional) | File that receives one-line progress notices from `pipeline.sh` / `diskguard.sh`; default `/dev/null` |

Outputs are written inside this folder (`sacq/`, `runs/`, `reset/`, `verify/`, `logs/`) and are git-ignored;
they reach about 50 GB.

## How to run

```bash
cd experiments/pi05_sacq_spatial
export CKPT=/path/to/pi05_libero_pytorch OPENPI_DIR=/path/to/openpi
sed "s#@OPENPI_DIR@#$OPENPI_DIR#g" libero_config/config.yaml.template > libero_config/config.yaml
source env.sh
tools/start_guard.sh 5 22 && tools/start_diskguard.sh 30 20 25   # RAM guard; disk guard for WSL on C:
for ph in obs mine official compare; do $PY -u scripts/verify_policy.py $ph; done  # loader vs openpi policy
$PY -u scripts/reset_journal.py --seed 17 && $PY -u scripts/reset_journal.py --seed 1234
$PY -u scripts/rollout.py --condition calib1234 --env-seed 1234 --episodes $(seq 0 29) \
  --groups 3 --envs-per-group 3 --capture --out runs/calib1234   # calibration collection
tools/launch.sh pipeline tools/pipeline.sh                       # everything else; restartable
```

`tools/pipeline.sh` runs each step once (`logs/steps/NAME.done` markers) and resumes interrupted steps:
`prep_snapshots` → `gptq_prep` → `stats` → `fit` (K=64, then the bank) → `assemble` (uniform) → `rtn` →
`val_eval` → `alloc` → `assemble` (allocated) → `index` (stage 3) → `calibrate` (stage 4) → `rollout`
evaluations via `tools/eval.sh` (final SACQ 500 episodes, original 500, RTN 3-bit 500, others 100) → `bits`.
`scripts/analyze.py` computes success rates, Wilson intervals and McNemar tests from `runs/eval/`.
`tools/check_outputs.py --since <UTC>` checks that outputs written after a hard VM restart load cleanly.

| Script | Role |
|---|---|
| `scripts/picore.py`, `pisim.py`, `reset_journal.py`, `rollout.py` | pi0.5 loader and openpi transforms, simulator worker, reset journals, grouped rollout runner |
| `scripts/sacqpi.py` | Candidate scope, snapshots, overlays, named conditions |
| `scripts/prep_snapshots.py`, `gptq_prep.py`, `gptq_core.py`, `stats.py`, `fit.py`, `kmeans_fast.py` | Calibration data, GPTQ background/bank, stage-1 statistics and codebook fitting |
| `scripts/alloc.py`, `assemble.py`, `index.py`, `calibrate.py` | Stages 2, 1 (assembly), 3 and 4 |
| `scripts/rtn.py`, `val_eval.py`, `bits.py`, `analyze.py`, `verify_policy.py` | References, validation error, storage accounting, success-rate analysis, loader check |
| `tools/*.sh`, `tools/check_outputs.py` | Pipeline, eval wrapper, detached launcher, RAM and disk guards, output check |
