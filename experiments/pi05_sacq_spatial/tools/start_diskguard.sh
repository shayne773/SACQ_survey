#!/bin/bash
cd "$(dirname "$0")/.."
if [ -f logs/diskguard.pid ] && kill -0 "$(cat logs/diskguard.pid)" 2>/dev/null; then kill "$(cat logs/diskguard.pid)"; fi
nohup tools/diskguard.sh logs/diskguard.log "${1:-30}" "${2:-20}" "${3:-25}" >/dev/null 2>&1 &
echo $! > logs/diskguard.pid
echo "diskguard pid $(cat logs/diskguard.pid)"
