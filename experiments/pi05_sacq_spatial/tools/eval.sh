#!/bin/bash
# tools/eval.sh COND N_PER_TASK [GROUPS] [ENVS]  -> runs/eval/COND (seed 17, official initial states, episodes 0..N-1)
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh
COND=$1; N=$2; G=${3:-3}; E=${4:-3}
OV=$([ "$COND" = original ] && echo original || echo "cond:$COND")
echo "=== $(date -u +%FT%TZ) eval $COND episodes 0..$((N-1)) groups $G x $E"
$PY -u scripts/rollout.py --condition "$COND" --overlay "$OV" --env-seed 17 --episodes $(seq 0 $((N-1))) \
  --groups "$G" --envs-per-group "$E" --out "runs/eval/$COND"
echo "=== $(date -u +%FT%TZ) eval $COND finished"
