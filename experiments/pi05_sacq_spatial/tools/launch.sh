#!/bin/bash
# tools/launch.sh NAME CMD...  -> detached run, log in logs/NAME.log, pid in logs/NAME.pid
cd "$(dirname "$0")/.."
NAME=$1; shift
setsid nohup "$@" > "logs/$NAME.log" 2>&1 < /dev/null &
echo $! > "logs/$NAME.pid"
echo "launched $NAME pid $(cat logs/$NAME.pid)"
