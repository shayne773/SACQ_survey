#!/bin/bash
# Second concurrent worker (owner request: "run two batches at once"). Walks batches from the top end
# (task9 sacq, task9 bf16, task8 sacq, ...). run_all.sh (worker 1) walks upward and skips any batch
# whose full/$c/completed_task$t exists, so worker 2 touches that file as its claim/skip marker
# (after an atomic mkdir claim). It only claims a batch >= 2 positions ahead of worker 1's last START,
# so worker 1 cannot reach it while the claim is being made. Real completion = DONE line here + 50 COMPLETE episodes.
cd "${SACQ_LAB:?set SACQ_LAB to the lab root (see README)}/eval"
export GALLIUM_DRIVER=llvmpipe
(while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)" >> full/vram_w2.txt; sleep 5; done) & MON=$!
trap "kill $MON" EXIT
for t in 9 8 7 6 5 4 3 2 1 0; do
  for c in sacq bf16; do
    [ "$c" = sacq ] && k=$(( 2 * t + 1 )) || k=$(( 2 * t ))
    last=$(grep ' START ' full/run_all.log | tail -1 | awk '{print $3, $4}')
    lt=${last#task}; lt=${lt%% *}; lc=${last##* }
    [ "$lc" = sacq ] && k1=$(( 2 * lt + 1 )) || k1=$(( 2 * lt ))
    if [ $k -le $(( k1 + 1 )) ]; then echo "$(date -Is) STOP w2 at task$t $c (worker1 at task$lt $lc)"; exit 0; fi
    if [ -f full/$c/completed_task$t ]; then continue; fi
    mkdir full/$c/w2claim_task$t 2>/dev/null || continue
    touch full/$c/completed_task$t
    echo "$(date -Is) START task$t $c"
    ../policy/bin/python scripts/run_eval.py --condition $c --task-index $t --root $PWD/full/$c >> full/logs/task${t}_$c.log 2>&1; rc=$?
    if [ $rc -eq 0 ]; then touch full/$c/w2claim_task$t/done; echo "$(date -Is) DONE task$t $c"; else echo "$(date -Is) FAIL task$t $c rc=$rc"; fi
  done
done
echo W2_FINISHED
