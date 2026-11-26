# GR00T LIBERO Spatial on a laptop: original BF16 vs SACQ W1.5A16

Success-rate evaluation of the original GR00T N1.7 Spatial checkpoint (BF16) against SACQ 1.5-bit weights with
16-bit activations (W1.5A16, the published `inference-spatial-v1` bundle, no A8 kernels) on LIBERO Spatial:
10 tasks × 50 official episodes = 500 episodes per model, run as paired episodes (same initial states, reset
journal, policy noise and seeds for both models).

The full write-up (method, per-task table, run checks, timing, VRAM, deviations, caveats) is
[REPORT.md](REPORT.md). The scripts and small records here are the ones used for that run; videos, episode dumps,
model bundles and environments are not included.

## Results

| Model | Successes | Success rate | 95% Wilson CI |
|---|---:|---:|---|
| Original BF16 | 476 / 500 | **95.2%** | 93.0 – 96.8% |
| SACQ W1.5A16 | 459 / 500 | **91.8%** | 89.1 – 93.9% |

Paired: 453 both succeed, 18 both fail, 23 original only, 6 SACQ only; exact McNemar p = **0.0023**.
All numbers are in [`eval/analysis.json`](eval/analysis.json); per-batch checks in
[`eval/verify_batches.json`](eval/verify_batches.json).

## Hardware

RTX 5080 Laptop GPU (16 GB, Blackwell SM120), WSL2 Ubuntu 24.04, CPU rendering with Mesa llvmpipe (no NVIDIA EGL
under WSL). Software versions are listed in [REPORT.md](REPORT.md#software-stack).

## What it expects (not in this repository)

All scripts work in one lab directory named by the environment variable `SACQ_LAB`:

| Path under `$SACQ_LAB` | Contents |
|---|---|
| `ws/` | The prepared GR00T/LIBERO workspace from this repository's tooling ([environment guide](../../gr00t_libero/ENVIRONMENT.md), [inference guide](../../gr00t_libero/INFERENCE.md)): `gr00t_final_seed17_20260922/`, `libero_pro_eval/` (with the Isaac-GR00T-N1.7 runtime and its bundled LIBERO), `libero_transfer_quantstats_20260922/`, `simplerenv_n17_feasibility_2026_08_27/` |
| `bundles/sacq/` | The SACQ inference bundle `inference-spatial-v1` (`model/`, `processor/`, `DOWNLOAD_SPEC.json`) from the [inference release](https://github.com/haesunglee116/SACQ/releases/tag/inference-spatial-v1) |
| `policy/` | Policy virtualenv (pinned `policy-linux-py310-20261006`: torch 2.7.1+cu128, flash-attn 2.7.4.post1, fast-hadamard-transform built for sm_120) |
| `sim/` | Simulator virtualenv (pinned `simulator-linux-py310-20261006`: mujoco 3.8.0, robosuite 1.4.0) |
| `toolchain/` | CUDA 12.8 toolchain used for source builds (only needed by `env.sh`) |
| `eval/` | A copy of this folder's [`eval/`](eval/) directory; runs write `full/`, `noise/`, `reset/`, `sock/` here |
| `libero_config/config.yaml` | LIBERO path config, from [`libero_config/config.yaml.template`](libero_config/config.yaml.template) |

The seed-17 reset journal (`eval/reset/`) and saved policy noise (`eval/noise/`) are regenerated with the
repository's public `reset_inputs.py` / `noise_inputs.py` (inputs recorded in
[`eval/reset_contract.json`](eval/reset_contract.json) and [`eval/noise_spec.json`](eval/noise_spec.json));
`$SACQ_LAB` in those records stands for the lab directory.

## How to run

```bash
export SACQ_LAB=/path/to/lab
mkdir -p "$SACQ_LAB/eval" "$SACQ_LAB/libero_config"
cp -r eval/. "$SACQ_LAB/eval/"
sed "s#@SACQ_LAB@#$SACQ_LAB#g" libero_config/config.yaml.template > "$SACQ_LAB/libero_config/config.yaml"
cd "$SACQ_LAB/eval"
../sim/bin/python scripts/make_tasks.py          # task registry + reset contract (tasks.json, reset_contract.json)
mkdir -p full/logs full/bf16 full/sacq
scripts/run_all.sh > full/run_all.log 2>&1      # 20 batches (10 tasks × bf16/sacq), resumable
# optional second worker on the same GPU, gated on worker 1's load window:
# scripts/run_w2b.sh > full/run_w2.log 2>&1
../sim/bin/python scripts/verify_batches.py     # per-episode coverage / reset-hash / pairing checks
../policy/bin/python scripts/analyze.py         # success rates, Wilson CIs, McNemar, timing, VRAM
```

`verify_batches.py` also needs `pose_replay.json` from `scripts/pose_replay.py` (CPU replay of the reset sequence).
The full run took 4 h 20 min with two workers for most batches.

| Script | Role |
|---|---|
| `eval/scripts/run_eval.py` | Driver around the repository's seed-17 rollout path (one condition × one task) |
| `eval/scripts/sim_wrapper.py` | Simulator-side wrapper: records mp4s / live viewer; physics untouched |
| `eval/scripts/run_all.sh`, `run_w2.sh`, `run_w2b.sh` | Batch runners (worker 1, worker 2, gated worker 2) |
| `eval/scripts/watch.sh [task] [episode] [bf16\|sacq\|both]` | Replay one evaluated episode live in a MuJoCo window (WSLg) |
| `eval/scripts/analyze.py`, `verify_batches.py`, `pose_replay.py` | Analysis and run checks |
| `eval/scripts/compose_videos.py` | Selected and side-by-side clips (output in `$SACQ_VIDEO_OUT`, default `$SACQ_LAB/eval/videos_selected`) |
| `eval/scripts/make_tasks.py`, `verify_record.py`, `bench_render.py` | Setup helpers: task registry, wheel RECORD check, renderer benchmark |
| `eval/scripts/d3d12_load_test.py` | GPU-renderer diagnostic for the discarded first run; written, never run |
| `env.sh` | Source to put `$SACQ_LAB/toolchain` on the build path |
