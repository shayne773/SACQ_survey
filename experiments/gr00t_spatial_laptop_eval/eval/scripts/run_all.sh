#!/bin/bash
# Full LIBERO Spatial run: per task, BF16 then SACQ W1.5A16; resumable (completed episode files are skipped).
cd "${SACQ_LAB:?set SACQ_LAB to the lab root (see README)}/eval"
export GALLIUM_DRIVER=llvmpipe  # first run on Mesa d3d12 hit reset-hash failures; cause unexplained (see report). CPU llvmpipe used for all reported episodes.
(while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)" >> full/vram.txt; sleep 5; done) & MON=$!
trap "kill $MON" EXIT
for t in 0 1 2 3 4 5 6 7 8 9; do
  for c in bf16 sacq; do
    if [ -f full/$c/completed_task$t ]; then continue; fi
    echo "$(date -Is) START task$t $c"
    ../policy/bin/python scripts/run_eval.py --condition $c --task-index $t --root $PWD/full/$c >> full/logs/task${t}_$c.log 2>&1; rc=$?
    if [ $rc -eq 0 ]; then touch full/$c/completed_task$t; echo "$(date -Is) DONE task$t $c"; else echo "$(date -Is) FAIL task$t $c rc=$rc"; fi
  done
done
echo ALL_FINISHED
