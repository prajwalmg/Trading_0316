#!/usr/bin/env bash
# scripts/run_multi.sh — Auto-restarting multi-asset paper trading.
# Usage: bash scripts/run_multi.sh &
set -euo pipefail
cd "$(dirname "$0")/.."

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

LOGFILE="logs/multi_asset_$(date +%Y%m%d).log"
mkdir -p logs

while true; do
    echo "$(date '+%Y-%m-%d %H:%M:%S'): Starting multi-asset system" >> "$LOGFILE"
    venv/bin/python main.py --mode multi >> "$LOGFILE" 2>&1
    EXIT=$?

    # 130=Ctrl-C (SIGINT), 143=SIGTERM → manual stop, do not restart
    if [ "$EXIT" -eq 130 ] || [ "$EXIT" -eq 143 ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S'): Manual stop (exit $EXIT)" >> "$LOGFILE"
        break
    fi

    echo "$(date '+%Y-%m-%d %H:%M:%S'): Crashed (exit $EXIT) — restarting in 60s..." >> "$LOGFILE"
    sleep 60
done
