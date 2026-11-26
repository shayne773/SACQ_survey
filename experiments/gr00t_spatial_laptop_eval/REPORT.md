# LIBERO Spatial on the laptop: original BF16 vs SACQ W1.5A16

Run 2026-11-22/23 on the RTX 5080 Laptop GPU (16 GB), WSL2 Ubuntu 24.04.
Scope: one suite (LIBERO Spatial), 10 tasks × 50 official episodes = 500 episodes per model, run on
identical episodes. Models: the original GR00T N1.7 Spatial checkpoint in BF16, and SACQ W1.5A16 from the
published `inference-spatial-v1` bundle (no A8 kernels).

## Answer

| Model | Successes | Success rate | 95% Wilson CI |
|---|---:|---:|---|
| Original BF16 | 476 / 500 | **95.2%** | 93.0 – 96.8% |
| SACQ W1.5A16 | 459 / 500 | **91.8%** | 89.1 – 93.9% |

**SACQ is 3.4 percentage points worse, and the paired test says this is not chance.** On the same 500
episodes, 453 succeeded with both models and 18 failed with both. 23 succeeded only with the original;
6 succeeded only with SACQ. The exact McNemar test on 23 vs 6 gives **p = 0.0023**.

The largest single share of the 17-episode gap is task 5, "bowl on the ramekin": 35/50 → 28/50, which
accounts for 7 of the 17. Even excluding task 5, the original wins the disagreements 11 to 1.

Watching it: 20 first-episode clips and 11 side-by-side clips were saved locally (videos are not included in
this repository). You can also watch any episode live with one command (see "Watching the models"); this
needs the full lab setup described in [README.md](README.md).

## Per task

Each cell gives successes out of 50, with the 95% Wilson interval. "Orig only" and "SACQ only" count the
paired episodes where only that model succeeded.

| # | Task | Original BF16 | SACQ W1.5A16 | Orig only | SACQ only |
|---:|---|---|---|---:|---:|
| 0 | between the plate and the ramekin | 50 (92.9–100) | 50 (92.9–100) | 0 | 0 |
| 1 | next to the ramekin | 50 (92.9–100) | 50 (92.9–100) | 0 | 0 |
| 2 | from table center | 50 (92.9–100) | 47 (83.8–97.9) | 3 | 0 |
| 3 | on the cookie box | 49 (89.5–99.6) | 50 (92.9–100) | 0 | 1 |
| 4 | in the top drawer of the wooden cabinet | 48 (86.5–98.9) | 44 (76.2–94.4) | 4 | 0 |
| 5 | on the ramekin | 35 (56.2–80.9) | 28 (42.3–68.8) | 12 | 5 |
| 6 | next to the cookie box | 50 (92.9–100) | 50 (92.9–100) | 0 | 0 |
| 7 | on the stove | 47 (83.8–97.9) | 45 (78.6–95.7) | 2 | 0 |
| 8 | next to the plate | 47 (83.8–97.9) | 47 (83.8–97.9) | 0 | 0 |
| 9 | on the wooden cabinet | 50 (92.9–100) | 48 (86.5–98.9) | 2 | 0 |
| | **All** | **476 (93.0–96.8)** | **459 (89.1–93.9)** | **23** | **6** |

Every task name is "pick up the black bowl … and place it on the plate". Per-task intervals are wide at
n = 50, so no single task other than task 5 shows a clear difference on its own.

The overall paired test is the meaningful comparison. The discordant episodes are listed by task and
episode in `analysis.json` (`paired.discordant`).

## Watching the models

**Saved videos** (kept locally, 20 MB; not included in this repository; `eval/scripts/compose_videos.py`
rebuilds them from a run). `index.json` describes each clip. Each frame has a label bar showing the model, task, episode and step,
and the last frames are marked SUCCESS or FAIL.

- `per_task/`: episode 0 of every task, for both models (20 clips). In task 5, episode 0, the original
  succeeds at step 86, while SACQ knocks the ramekin off the table and times out at step 720.
- `side_by_side/`: original on the left, SACQ on the right, same episode, starting from the identical
  frame. The clip that finishes first holds its last frame. There are 11 clips:
  - five where the original succeeds and SACQ fails, the first such episode in each task where it
    happened: task 2 ep 4, task 4 ep 10, task 5 ep 0, task 7 ep 11, task 9 ep 16;
  - three where SACQ succeeds and the original fails: task 3 ep 8, and task 5 eps 4 and 11;
  - three where both succeed: tasks 0, 4 and 9, episode 0.

  Filenames state each outcome, for example `task5_ep00_original-success_vs_sacq-fail.mp4`.
- All 1,000 episodes were recorded. The full set is under the lab directory,
  `$SACQ_LAB/eval/full/{bf16,sacq}/videos/` (about 450 MB), and is not included in this repository.

**Live window (WSLg).** A real MuJoCo window opens on the Windows desktop and shows the policy driving the
simulation as it runs. It uses the fixed agent-view camera, with a label of model / task / episode / step
and the status. Command:

```bash
$SACQ_LAB/eval/scripts/watch.sh 5 0
```

- Arguments: task 0–9 and episode 0–49. A third argument of `bf16`, `sacq` or `both` selects the model;
  the default runs the original first, then SACQ.
- Each model takes about 1–2 minutes to load before the window opens. The window closes when the episode
  ends.
- It replays exactly the evaluated episode. I checked task 5, episode 0 with SACQ: same initial frame hash,
  same 720 steps, same 90 policy calls and the same failure as in the full run. It also saves the episode
  as an mp4 and prints where.
- Limits:
  - It needs the full lab directory (`$SACQ_LAB`: workspace, model bundles and environments; see
    [README.md](README.md)).
  - Don't run it alongside another evaluation, because of GPU memory.
  - It renders on the CPU (llvmpipe), the same renderer as the evaluation, so it plays at roughly 8
    sim-steps/s or slower, below real time.
  - I could not see the screen from here. The run completed with no viewer errors, and an earlier smoke
    test opened the window through WSLg, but you need to confirm it actually appears for you.

## Was the run sound?

I verified every batch from its per-episode result files, not from the runner's completion markers. Worker 2
writes those markers when it claims a batch, so they don't prove completion. `eval/verify_batches.json`
records the following for all 20 batches (10 tasks × 2 models):

- **Coverage:** each batch has exactly episodes 0–49, every one with status COMPLETE.
- **Reset check:**
  - Every episode's initial-observation fingerprint (SHA-256 over both camera images and the robot
    state) equals the reset reference that its simulator process rendered for that official episode.
  - The simulator only releases an episode after that check passes.
  - No reset reconstructions were needed in either model's run.
- **Explicit scene poses:**
  - Fixture poses, the wooden cabinet and the stove, live in the MuJoCo model (`model.body_pos`), outside
    the simulator state vector. The exact-pixel check therefore covers them only implicitly, through the
    agent-view image.
  - A separate CPU replay of every task's reset sequence (`scripts/pose_replay.py` →
    `$SACQ_LAB/eval/pose_replay.json`) records those poses explicitly for all 500 episodes.
  - Its observation fingerprint equals the run's recorded initial fingerprint for all 1,000 episodes. So
    each episode's cabinet and stove pose is the replayed, recorded pose, and it is the same for both
    models. Cabinet x ranged 0.020–0.040 m and y −0.280 to −0.260 m across episodes.
- **Pairing:** all 500 episode pairs started from bit-identical observations.

## Protocol (matched to the recorded protocol)

| Setting | Value used |
|---|---|
| Environment seed | 17 (not the calibration seed 1234) |
| Initial states | LIBERO official init states, episodes 0–49 per task, with the seed-17 reset journal |
| Policy noise | One fixed policy seed, 20260824. The saved noise table (50 episodes × 90 calls) is identical for both models |
| Action chunk | 8 steps per policy call |
| Denoising steps | 4 |
| Termination | Official done or 720-step cap; success = success at any step |
| Environments | 1 per worker, no settling steps |
| Paired | Same tasks, initial states, reset journal, noise and seed for both models |

The rollout used the repository's own seed-17 path unchanged:
- `rollout_runtime.main`, `simulator_runtime`, `seed17_adapter`, `NoiseInjection`;
- `DefaultAcceleration`, which uses an exactness-checked CUDA graph and weight cache;
- `MultiStepWrapper`;
- for SACQ, the bundle's 337 SACQ overlays plus the 105-layer GPTQ background.

## Open items at the time of the run

1. **Run the GPU-renderer diagnostic?** About 10 minutes on the laptop GPU. It would explain, or rule out
   render noise for, the first run's reset failures. If declined, they stay recorded as unexplained. Either
   way, the results above are unaffected.
2. **Keep the 56 GB lab directory?** The live-view command needs it. The saved videos remain either way.

## Speed and memory

**Seconds per episode.** Wall time between consecutive episode completions, with the first episode of each
batch excluded because it includes model loading.

| | Original BF16 | SACQ W1.5A16 |
|---|---:|---:|
| One worker on the GPU (tasks 0–2), mean / median | 13.5 / 12.9 s (n = 198) | 13.8 / 12.9 s (n = 115) |
| Two workers sharing the GPU, mean / median | 22.6 / 15.8 s | 26.9 / 18.2 s |
| Mean env steps per episode (failures run to 720) | 138 | 157 |

**Policy time per call.** This is one action chunk, measured by the runtime profiler. It includes input
preparation, the VLM and the action head.

| Batch type | Original BF16 | SACQ W1.5A16 |
|---|---:|---:|
| One worker, tasks 0–2 | 0.095–0.102 s | 0.101–0.105 s (weight cache on) |
| SACQ with the weight cache skipped (two workers) | — | ~0.40 s |

- The simulator step takes about 0.84 s per 8-step chunk, so on this laptop episode time is dominated by
  CPU rendering, not the model.
- **SACQ is not faster than BF16 here.** The runtime's speed path for SACQ is an exactness-checked cache of
  expanded weights (3.8 GB). With it, SACQ runs at BF16 speed. Without it, SACQ took 0.40 s per call
  under two workers, versus 0.12–0.17 s with the cache under the same two-worker conditions: about
  2.5–3× slower.
- The cache is skipped automatically when free VRAM is below what it needs plus 3 GB. That happened in 5
  SACQ batches (tasks 3, 6, 7, 8, 9) while two workers shared the GPU.
- The cache never changes actions: it is verified against the uncached action and disabled on any mismatch.
  So cache state affects speed only, not results.
- The author's 20.2 ms SACQ latency is a different metric: whole-model GPU time on an RTX 4090, with
  preprocessing excluded. It is not comparable to these numbers.

**Peak VRAM** (PyTorch, per process):

| Model | Peak allocated | Peak reserved | Notes |
|---|---:|---:|---|
| Original BF16 | 6.41 GB | 6.54 GB | |
| SACQ | 10.26 GB | 10.30 GB | The peak is the install step: BF16 load plus overlay install |

**Device total, one model alone** (from `nvidia-smi`, about 1.5 GB of it is the idle desktop):

| Model | Steady | Peak |
|---|---:|---:|
| Original BF16 | ~8.0 GB | 8.1 GB |
| SACQ | 7.5–7.6 GB with cache | 10.8 GB during install |

The highest device total with two workers was 15.8 of 16.3 GB, when one worker's SACQ install overlapped the
other worker's steady state.

**Wall clock.** The evaluation started at 20:43 local and ended at 01:04 (4 h 20 min), including:
- the two-worker phase from 22:08;
- a 21-minute suspension of the laptop/VM at 21:20–21:41. All logging stopped then; it is excluded from the
  timing above and did not affect results.

## The first attempt (discarded): reset-check failures on the GPU renderer, cause UNEXPLAINED

The first full run used Mesa's d3d12 driver to render on the RTX 5080 (`GALLIUM_DRIVER=d3d12`), because it
was 1.7× faster than CPU rendering in a short benchmark.

- In every batch, partway through, the simulator stopped with the repository's own check: "Official reset
  history mismatch after five-env reconstruction". That happened in 11 batches over tasks 0–5.
- I stopped that run after 203 completed episodes, set it aside (`$SACQ_LAB/eval/full_d3d12_aborted/`), and
  reran everything on CPU rendering (llvmpipe).
- Nothing from that run is in the results above.
- **The cause of those failures is not established.** Hypotheses and evidence:

1. **GPU render nondeterminism (d3d12 rendering under concurrent CUDA load).**
   - *For:*
     - Within the aborted run, the BF16 and SACQ simulator processes each rendered their own reset
       references, with the same code, the same journal and the same order. They disagreed on 4, 3, 7, 2, 3
       and 3 of 50 episodes for the six tasks reached. On llvmpipe the 20 processes agree on 50/50, and
       they match the regenerated journal exactly.
     - Twice, the reconstruction failed while replaying an earlier episode that had already passed in the
       same process: episode 33 failed while replaying episode 12, and in the other run while replaying
       episode 14.
     - A replay test on d3d12 with the GPU idle gave identical fingerprints across three fresh environments
       for 16/16 episodes. The same test under a synthetic CUDA matmul load matched on only 44 of 50.
   - *Weakness:* that load test recorded only the count 44. It did not record per-episode fingerprints,
     poses or pixel differences. No d3d12 run logged fixture poses, so a pose difference was never ruled
     out directly.
2. **Reset-history / cabinet-placement mismatch.** The cabinet pose is not in the simulator state, and it
   depends on reset history.
   - *For:*
     - Every failure was the reset-history check.
     - Repeatedly resetting the same episode on one environment moves the cabinet every time, even with
       explicit seed 17 and a restored RNG. Episode 0 went from x 0.0312 to 0.0279 to 0.0358 to 0.0284.
     - That is what produced the earlier `cmp_1.png` differences on the cabinet and the bowl on it. It
       happens on llvmpipe too, and re-rendering an unchanged pose gives identical pixels. So those
       particular differences were cabinet movement, not render noise.
   - *Against:* replaying the journal's reset sequence in fresh environments reproduces poses and
     fingerprints exactly on llvmpipe (12/12) and on idle-GPU d3d12 (16/16), with or without actions in
     between. Placement is CPU code driven by the restored RNG.
3. **Other:** for example, a WSL d3d12 driver issue unrelated to load, or an interaction between d3d12
   rendering and environment teardown. There is no specific evidence for or against.

The diagnostic that would settle this is ready, but **I did not run it, as the experiment owner instructed**. It
replays one task's reset sequence on d3d12 under CUDA load and logs per-episode fingerprints, cabinet poses
and pixel-difference boxes (`eval/scripts/d3d12_load_test.py`). It waits for the experiment
owner's approval. It does not affect the results above, which use only CPU rendering, verified as described.

## Deviations from the recorded protocol, and why

1. **Hardware.** RTX 5080 Laptop (Blackwell, SM120, 16 GB) under WSL2, not the recorded RTX 4090. That is
   the point of the test, but GPU arithmetic differs, so actions are not bit-identical to the author's
   runs. No author action values are published to compare against.
2. **Renderer.** Mesa llvmpipe (CPU) through EGL instead of NVIDIA EGL, because there is no NVIDIA EGL
   under WSL. The repository's `same_process_saved_rng_exact` mode recomputes reset references in-process,
   and those matched the reset journals exactly. Pixels may differ slightly from NVIDIA-rendered images,
   which can shift individual outcomes. Both models saw identical pixels.
3. **Regenerated private inputs.** The seed-17 reset journals and the saved noise table are not public. I
   regenerated them with the repository's public tools: `reset_inputs.py`, where all 10 tasks passed, and
   `noise_inputs.py` with seed 20260824. Whether they equal the author's archived files cannot be checked.
4. **Bookkeeping.** My driver (`eval/scripts/run_eval.py`) replaces `rollout_worker.py`'s receipt and
   authorization paperwork (EXECUTION_RECEIPT, protocol.json), because those receipts are not public. The
   scientific path is the repository's code, unchanged.
5. **Frame recording.** A simulator-side wrapper copies already-rendered frames to mp4 and optionally drives
   the live window. Physics, resets, RNG and success are untouched.
6. **Concurrency.** To finish sooner, at the experiment owner's request, 15 of 20 batches ran two at a time on the
   GPU, with atomic batch claims and no shared-file collisions. The acceleration is exactness-checked, so
   this changes speed only. The one consequence is the SACQ weight cache being skipped in 5 batches.
7. **Discarded first run.** The 203 episodes from the d3d12 run (above) were set aside and not mixed in.
8. **Build details.**
   - The pinned flash-attn 2.7.4.post1 wheel was used unchanged and works on SM120.
   - fast-hadamard-transform was built from source (1cc807e) for sm_120 with nvcc 12.8. Its max error vs a
     reference Hadamard is 0.0077 in bf16.
   - The GR00T workspace was prepared at the receipt commit 4b1dca9; the local clone's HEAD is 708617f.
9. **Pilot runs** (3 episodes per model on task 0) used a separate directory and are not counted.

## Caveats

- **One suite only.** This compares Spatial only. Object, Goal and LIBERO-10 were not run, and Spatial
  results do not transfer to them automatically.
- **The author's 93.55% is not comparable.** That figure is SACQ **W1.5A8** (INT8 activations and
  low-precision attention), over all four suites (Spatial 469/500 = 93.8%). The record is marked
  `diagnostic_only: true` and `held_out: false`, and it predates the final kernels. This run is **A16**, on
  different hardware and a different renderer.
- **No author BF16 or A16 Spatial success rate is published in the repo**, so there is no published number
  for these two configurations to compare with. The comparison here is internal: both models on the same
  machine and the same episodes.
- **Possible overlap between calibration scenes and evaluation episodes.**
  - SACQ was calibrated on observations captured during BF16 rollouts with environment seed 1234: 640
    Spatial requests of 5 observations each.
  - The calibration manifest identifies requests per task (e.g. `sp5_req0199`) but records no episode or
    initial-state identity, so overlap with these 500 evaluation episodes cannot be determined.
  - LIBERO draws initial scenes from the same 50 official initial states per task, so the calibration
    scenes very likely include these scenes, though not these exact frames. Treat this as in-distribution
    evaluation (consistent with `held_out: false`), not held-out.
- **Statistical scope.** The intervals and the McNemar test treat episodes as independent draws, for one
  policy seed and one environment seed. Another seed could move individual outcomes.

## Software stack

| Component | Version |
|---|---|
| Host | WSL2 kernel 6.6.87.2-microsoft-standard-WSL2, Ubuntu 24.04.3 |
| GPU | RTX 5080 Laptop GPU, 16,303 MiB; Windows driver 617.42, CUDA UMD 13.4 |
| Toolchain | nvcc 12.8.93 (conda toolchain), gcc 13.4.0 (conda-forge), uv 0.13.0, Python 3.10.19 |

**Policy environment** (pinned to `policy-linux-py310-20261006`):
- torch 2.7.1+cu128 (cuDNN 9.7.1), transformers 4.57.3, diffusers 0.35.1, numpy 1.26.4;
- flash-attn 2.7.4.post1, fast-hadamard-transform 1.1.0 (source 1cc807e, sm_120);
- hqq 0.2.8.post1, deepspeed 0.17.6;
- sacq-research 0.1.0 (wheel sha256 cc73b734…).

**Simulator environment** (pinned to `simulator-linux-py310-20261006`):
- mujoco 3.8.0, robosuite 1.4.0, bddl 1.0.1, gymnasium 0.29.1, gym 0.25.2;
- torch 2.5.1, opencv-python 4.6.0.66, imageio-ffmpeg 0.6.0, PyOpenGL 3.1.10;
- Mesa 25.2.8 (llvmpipe).

**Sources:** GR00T N1.7 runtime 4b1dca9, LIBERO 8f1084e, both verified against the SOURCE_RECEIPT.

**Models:**
- Original: the GR00T N1.7 Spatial checkpoint via the repository downloader.
- SACQ: the bundle `inference-spatial-v1`. Both bundles are BUNDLE_READY with INPUTS_VERIFIED.
- Cosmos-Reason2-2B (gated, accessed after the experiment owner accepted its license).

**Single-request sanity check** before the run: both models produced finite actions of shape [5,16,1] for
all 7 action keys. Max absolute differences between BF16 and SACQ:

| x | y | z | roll | pitch | yaw | gripper |
|---:|---:|---:|---:|---:|---:|---:|
| 0.154 | 0.137 | 0.243 | 0.009 | 0.077 | 0.024 | 0.557 |

## Disk used

| Location | Size |
|---|---:|
| Lab directory `$SACQ_LAB` | **56 GB** |
| `~/.cache/huggingface` (checkpoints, incl. Cosmos-Reason2-2B) | 12 GB |
| `~/.cache/uv` (package cache, shared by any uv user) | 27 GB |
| Report and saved videos (local) | 20 MB |

Breakdown of the lab directory:

| Part | Size |
|---|---:|
| Model bundles (BF16 and SACQ, 13 GB each) | 26 GB |
| Policy venv | 16 GB |
| Simulator venv | 6.3 GB |
| micromamba | 6.1 GB |
| CUDA toolchain | 4.5 GB |
| Sources | 1.3 GB |
| All rollout data and videos | 0.6 GB |

The disk had 829 GB free. Removing the lab directory frees the 56 GB, and with it the live-view command. The
caches stay until removed separately.

## Where things are

- This report: [REPORT.md](REPORT.md); how to reproduce: [README.md](README.md).
- In this repository, under [`eval/`](eval/): `analysis.json` (all numbers above, including the discordant
  list), `verify_batches.json`, `policy_timing_by_batch.json`, `tasks.json`, `reset_contract.json`,
  `noise_spec.json`, `notes_cabinet.md`.
- Only in the lab directory (`$SACQ_LAB/eval/`, not in the repository): `pose_replay.json`,
  `full/{bf16,sacq}/episodes/` (one JSON per episode), `full/*/videos/` (all 1,000 mp4s), logs in
  `full/logs/`, `full/run_all.log`, `full/run_w2.log`.
- Scripts, in [`eval/scripts/`](eval/scripts/):
  - `run_eval.py`, `sim_wrapper.py`, `run_all.sh`, `run_w2b.sh` (the run);
  - `analyze.py`, `verify_batches.py`, `pose_replay.py`, `compose_videos.py` (analysis);
  - `watch.sh` (live view);
  - `d3d12_load_test.py` (waiting for approval, not run).
