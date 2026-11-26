#!/bin/bash
# The whole remaining SACQ-on-pi0.5 pipeline, GPU-serial, restartable after a VM restart:
#   a step that finished leaves logs/steps/NAME.done and is skipped on the next start; an interrupted step is re-run
#   (rollout.py skips finished episodes, fit.py finished layers, gptq_prep.py finished passes, alloc.py and
#   calibrate.py resume from their last refresh / 100-update state).
# Order (main question first, uniform side comparison last):
#   A  calibration targets -> GPTQ moments/background/scalar bank -> stage-1 statistics -> K=64 fit -> uniform stage 1
#      -> RTN references -> validation errors of the cheap conditions
#   B  original model, 500 episodes  ||  the rest of the codebook bank (K = 32, 128 ... 2048)
#   C  stage 2 allocation -> allocated stage 1 -> stage 3 -> stage 4 -> validation errors
#   D  final SACQ (a1234) 500, RTN 3-bit 500; a123, a12, RTN 2-bit 100 each (+400 if > 10% success); bg 100
#   E  uniform stage 3 -> stage 4 -> validation errors; u134, u13, u1 100 each (+400 if > 10% success)
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh
mkdir -p logs/steps
STOP_AFTER=${STOP_AFTER:-}

step() {
  local name=$1; shift
  if [ -f "logs/steps/$name.done" ]; then echo "=== skip $name (done)"; return 0; fi
  echo "=== $(date -u +%FT%TZ) start $name: $*"
  local t=$(date +%s)
  "$@"
  echo "=== $(date -u +%FT%TZ) done $name in $(( $(date +%s) - t ))s"
  sync  # outputs durable on disk before the marker that makes a restart skip this step
  touch "logs/steps/$name.done"
  sync
  if [ "$STOP_AFTER" = "$name" ]; then echo "=== STOP_AFTER $name"; exit 0; fi
}

successes() {  # COND -> number of successful episodes recorded
  python3 -c "
import glob, json, sys
print(sum(json.load(open(f))['success'] for f in glob.glob('runs/eval/$1/episodes/task*_ep*.json')))"
}

eval_ext() {  # COND: 100 episodes (10 per task); extended to 500 only if it scores above 10% on those 100
  local c=$1
  step "eval_${c}_100" tools/eval.sh "$c" 10
  local s; s=$(successes "$c")
  if [ "$s" -gt 10 ]; then
    step "eval_${c}_500" tools/eval.sh "$c" 50
  else
    echo "=== $c: $s/100 successes (<= 10%), not extended"
  fi
}

# ---------------------------------------------------------------- A
step prep_snapshots $PY -u scripts/prep_snapshots.py
step gptq_prep $PY -u scripts/gptq_prep.py
step stats $PY -u scripts/stats.py
step fit_k64 $PY -u scripts/fit.py --ks 64
step assemble_uniform $PY -u scripts/assemble.py uniform --out sacq/uniform/stage1
step rtn $PY -u scripts/rtn.py
step val_cheap $PY -u scripts/val_eval.py original bg u1 rtn3 rtn2

# ---- Order changed 2026-11-24 ~21:20Z on the experiment owner's request; edited in place below the
# ---- line the running instance had reached, so it took effect without a restart:
# ----   * segment E (uniform 1.5-bit side comparison: stage 3, stage 4, their validation errors, u134/u13/u1 evals)
# ----     is dropped for now; kept commented out at the end so it can be re-added. `bits` moved to the end of D.
# ----   * the original-model 500-episode baseline leaves segment B and runs right after the final SACQ in D;
# ----     the SACQ success rate (unpaired) is appended to the status file as soon as its 500 episodes finish.
STATUS_FILE=${SACQ_STATUS_FILE:-/dev/null}  # optional: file that receives the one-line SACQ rate notice

post_rate() {  # COND LABEL: append the condition's success rate (with a 95% Wilson interval) to the status file
  local line
  line=$(python3 -c "
import glob, json, math
fs = glob.glob('runs/eval/$1/episodes/task*_ep*.json')
n = len(fs); s = sum(json.load(open(f))['success'] for f in fs)
z = 1.959964; p = s / n; d = 1 + z * z / n
c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
print(f'{s}/{n} = {100 * p:.1f}% (95% CI {100 * (c - h):.1f}-{100 * (c + h):.1f}%)')")
  echo "working: $2 on LIBERO Spatial (official initial states, env seed 17): $line; unpaired, the original-model 500-episode baseline starts now and the paired comparison follows in ~1.5-2 h" | tee -a "$STATUS_FILE"
}

# ---------------------------------------------------------------- B (codebook bank only)
step fit_bank $PY -u scripts/fit.py --ks 32,128,256,512,1024,2048

# ---------------------------------------------------------------- C
step alloc $PY -u scripts/alloc.py --requests-per-task 1  # 10 proxy requests: the author's --requests-per-suite default
step assemble_alloc $PY -u scripts/assemble.py alloc --out sacq/alloc/stage1
step val_a12 $PY -u scripts/val_eval.py a12
step index_alloc $PY -u scripts/index.py --packed sacq/alloc/stage1/packed.pt --fixed sacq/alloc/stage1/fixed.pt \
  --out sacq/alloc/stage3
step val_a123 $PY -u scripts/val_eval.py a123
step cal_alloc $PY -u scripts/calibrate.py --packed sacq/alloc/stage3/packed.pt --fixed sacq/alloc/stage1/fixed.pt \
  --out sacq/alloc/stage4
step val_a1234 $PY -u scripts/val_eval.py a1234

# ---------------------------------------------------------------- D
step eval_a1234_500 tools/eval.sh a1234 50
step post_a1234_rate post_rate a1234 "final SACQ (allocated 1.5-bit, stages 1-4)"
step eval_original_500 tools/eval.sh original 50
step eval_rtn3_500 tools/eval.sh rtn3 50
eval_ext a123
eval_ext a12
eval_ext rtn2
step eval_bg_100 tools/eval.sh bg 10
step bits $PY -u scripts/bits.py
echo "=== pipeline finished $(date -u +%FT%TZ)"
exit 0

# ---------------------------------------------------------------- E (dropped 2026-11-24; re-add by moving above `exit 0`)
# step index_uniform $PY -u scripts/index.py --packed sacq/uniform/stage1/packed.pt --out sacq/uniform/stage3
# step val_u13 $PY -u scripts/val_eval.py u13
# step cal_uniform $PY -u scripts/calibrate.py --packed sacq/uniform/stage3/packed.pt --out sacq/uniform/stage4
# step val_u134 $PY -u scripts/val_eval.py u134
# eval_ext u134
# eval_ext u13
# eval_ext u1
