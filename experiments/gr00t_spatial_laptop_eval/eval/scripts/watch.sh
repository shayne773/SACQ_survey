#!/bin/bash
# Watch one LIBERO Spatial episode live (MuJoCo window via WSLg): original BF16, then SACQ W1.5A16.
# Usage: watch.sh [task 0-9, default 0] [episode 0-49, default 0] [bf16|sacq|both, default both]
set -e
L=${SACQ_LAB:?set SACQ_LAB to the lab root (see README)}
T=${1:-0}; E=${2:-0}; WHICH=${3:-both}
ROOT=$L/eval/watch/$(date +%Y%m%d_%H%M%S)
export EVAL_LIVE=1 GALLIUM_DRIVER=llvmpipe  # same renderer as the evaluation
for c in bf16 sacq; do
  [ "$WHICH" != both ] && [ "$WHICH" != "$c" ] && continue
  echo "== $c: task $T episode $E (model load ~1-2 min, then the window opens)"
  $L/policy/bin/python $L/eval/scripts/run_eval.py --condition $c --task-index $T --only-episode $E --root $ROOT/$c 2>&1 \
    | grep --line-buffered -E "EPISODE_COMPLETE|Error" || true
done
echo "Videos: $ROOT/*/videos/"
