#!/bin/bash
# (Re)start the RAM guard; pidfile avoids pkill -f matching the caller's own command line.
cd "$(dirname "$0")/.."
if [ -f logs/memwatch.pid ] && kill -0 "$(cat logs/memwatch.pid)" 2>/dev/null; then kill "$(cat logs/memwatch.pid)"; fi
nohup tools/memwatch.sh logs/memwatch.log "${1:-5}" "${2:-22}" >/dev/null 2>&1 &
echo $! > logs/memwatch.pid
echo "guard pid $(cat logs/memwatch.pid)"
