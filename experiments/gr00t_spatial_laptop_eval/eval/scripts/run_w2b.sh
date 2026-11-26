#!/bin/bash
# Worker 2, gated version (replaces run_w2.sh's loop after its in-flight batch). Same claim rules as run_w2.sh,
# plus a load gate: a model load briefly needs ~8 GB extra VRAM, so two simultaneous loads could exceed 16 GB.
# Worker 2 therefore starts a batch only when worker 1 is mid-batch: its last START >= 150 s ago and its current
# batch has < 44 episodes done (so it will not start loading during our ~80 s load window).
# Arg 1 (optional): pid of an in-flight worker-2 batch "task<t> <c>" to wait for and record first.
cd "${SACQ_LAB:?set SACQ_LAB to the lab root (see README)}/eval"
export GALLIUM_DRIVER=llvmpipe
(while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)" >> full/vram_w2.txt; sleep 5; done) & MON=$!
trap "kill $MON" EXIT
count() { ../sim/bin/python -c "
import json,glob,sys
t=json.load(open('tasks.json'))[int(sys.argv[1])]['task']
print(sum('\"COMPLETE\"' in open(f).read() for f in glob.glob('full/'+sys.argv[2]+'/episodes/final_seed17/libero_spatial/'+t+'/*.json')))" $1 $2; }
if [ -n "$1" ]; then
  pid=$1; t=$2; c=$3
  while kill -0 $pid 2>/dev/null; do sleep 10; done
  n=$(count $t $c)
  if [ "$n" = 50 ]; then touch full/$c/w2claim_task$t/done; echo "$(date -Is) DONE task$t $c (in-flight batch from run_w2.sh)"; else echo "$(date -Is) FAIL task$t $c only $n/50 complete (in-flight batch from run_w2.sh)"; fi
fi
w1() { last=$(grep ' START ' full/run_all.log | tail -1); lt=$(echo "$last" | awk '{print $3}'); lt=${lt#task}; lc=$(echo "$last" | awk '{print $4}');
       lts=$(date -d "$(echo "$last" | awk '{print $1}')" +%s); [ "$lc" = sacq ] && k1=$(( 2 * lt + 1 )) || k1=$(( 2 * lt )); }
for t in 9 8 7 6 5 4 3 2 1 0; do
  for c in sacq bf16; do
    [ "$c" = sacq ] && k=$(( 2 * t + 1 )) || k=$(( 2 * t ))
    if [ -f full/$c/completed_task$t ]; then continue; fi
    while true; do
      w1
      if [ $k -le $(( k1 + 1 )) ]; then echo "$(date -Is) STOP w2 at task$t $c (worker1 at task$lt $lc)"; exit 0; fi
      grep -q ALL_FINISHED full/run_all.log && break
      age=$(( $(date +%s) - lts )); n=$(count $lt $lc)
      [ $age -ge 150 ] && [ $n -lt 44 ] && break
      sleep 15
    done
    mkdir full/$c/w2claim_task$t 2>/dev/null || continue
    touch full/$c/completed_task$t
    echo "$(date -Is) START task$t $c"
    ../policy/bin/python scripts/run_eval.py --condition $c --task-index $t --root $PWD/full/$c >> full/logs/task${t}_$c.log 2>&1; rc=$?
    if [ $rc -eq 0 ]; then touch full/$c/w2claim_task$t/done; echo "$(date -Is) DONE task$t $c"; else echo "$(date -Is) FAIL task$t $c rc=$rc"; fi
  done
done
echo W2_FINISHED
