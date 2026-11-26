#!/bin/bash
# memwatch.sh LOG [interval] [kill_threshold_used_gb]
# Logs used/available RAM and per-process RSS of the lab's python processes; terminates simulator/pipeline python
# processes if used RAM exceeds the threshold (guard against the WSL OOM killer).
LOG=$1; INT=${2:-5}; LIMIT=${3:-22}
PAT="openpi/.venv/bin/python"
while true; do
  read -r used avail <<<"$(free -m | awk '/Mem:/{print $3, $7}')"
  n=$(pgrep -fc "$PAT")
  rss=$(ps -eo rss,args | grep "[o]penpi/.venv/bin/python" | awk '{s+=$1} END{print int(s/1024)}')
  top=$(ps -eo rss,args --sort=-rss | grep "[o]penpi/.venv/bin/python" | head -4 | awk '{printf "%dMB ", $1/1024}')
  echo "$(date +%H:%M:%S) used=${used}MB avail=${avail}MB py_procs=$n py_rss_sum=${rss}MB top=$top" >> "$LOG"
  if [ "$used" -gt $((LIMIT*1024)) ]; then
    echo "$(date +%H:%M:%S) GUARD: used ${used}MB > ${LIMIT}GB, terminating python runs" >> "$LOG"
    pkill -TERM -f "scripts/[a-z_0-9]+\.py" ; sleep 5
    pkill -KILL -f "scripts/[a-z_0-9]+\.py"
  fi
  sleep "$INT"
done
