#!/bin/bash
# diskguard.sh LOG [interval_s] [stop_below_gb] [resume_above_gb]
# The WSL ext4.vhdx lives on C:, so lab writes grow it; WSL crashed 2026-11-24 21:16Z with C: full. This guard logs
# free space on /mnt/c and / and, when C: free drops below the stop threshold, SIGSTOPs every lab python script
# (pipeline steps and simulator workers; a stopped process loses nothing) and appends one blocked: line to the
# status file; it SIGCONTs them and appends resolved: once C: free is back above the resume threshold.
LOG=$1; INT=${2:-30}; STOP=${3:-20}; RESUME=${4:-25}
STATUS=${SACQ_STATUS_FILE:-/dev/null}  # optional status file for the blocked:/resolved: lines
PAT="scripts/[a-z_0-9]+\.py"
stopped=0
while true; do
  c=$(df -BG --output=avail /mnt/c | tail -1 | tr -dc 0-9)
  r=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
  echo "$(date -u +%FT%TZ) c_free=${c}G root_free=${r}G stopped=$stopped" >> "$LOG"
  if [ "$stopped" = 0 ] && [ "$c" -lt "$STOP" ]; then
    pkill -STOP -f "$PAT"; stopped=1
    echo "$(date -u +%FT%TZ) GUARD: C: free ${c}G < ${STOP}G, SIGSTOPped lab python" >> "$LOG"
    echo "blocked: [key=sacq-pi05-cdrive] C: free space ${c} GB < ${STOP} GB (WSL vhdx grows on C:); pipeline paused (SIGSTOP), resumes by itself once C: free > ${RESUME} GB" >> "$STATUS"
  elif [ "$stopped" = 1 ] && [ "$c" -gt "$RESUME" ]; then
    pkill -CONT -f "$PAT"; stopped=0
    echo "$(date -u +%FT%TZ) GUARD: C: free ${c}G > ${RESUME}G, SIGCONT lab python" >> "$LOG"
    echo "resolved: [key=sacq-pi05-cdrive] C: free space back to ${c} GB; pipeline resumed" >> "$STATUS"
  elif [ "$stopped" = 1 ]; then
    pkill -STOP -f "$PAT"   # also catch processes started after the first stop
  fi
  sleep "$INT"
done
