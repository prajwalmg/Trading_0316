#!/usr/bin/env bash
# scripts/validate_and_restart.sh — Phase 6: validate models + QuestDB, restart trading.
# Usage: bash scripts/validate_and_restart.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

LOGFILE="logs/restart_$(date +%Y%m%d_%H%M).log"
mkdir -p logs

log() { echo "$(date '+%Y-%m-%d %H:%M:%S'): $*" | tee -a "$LOGFILE"; }

log "=== Phase 6: Validate and Restart ==="

# ── 6A: Pre-launch validation ─────────────────────────────────────────────────
log "6A: Checking swing model count..."
SWING_COUNT=$(ls models/saved/swing_*_ensemble.pkl 2>/dev/null | wc -l | tr -d ' ')
PROD_COUNT=$(ls models/saved/*_ensemble.pkl 2>/dev/null | grep -v swing_ | grep -v '\.bak' | wc -l | tr -d ' ')
log "  swing_ models: $SWING_COUNT"
log "  production models: $PROD_COUNT"

log "6A: Checking intraday model count..."
INTRADAY_COUNT=$(ls models/saved/intraday_*_ensemble.pkl 2>/dev/null | wc -l | tr -d ' ')
log "  intraday_ models: $INTRADAY_COUNT"

log "6A: Checking QuestDB..."
QDB_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:9000/exec?query=SELECT+count()+FROM+ohlcv" 2>/dev/null || echo "000")
if [ "$QDB_STATUS" = "200" ]; then
    QDB_ROWS=$(curl -s "http://localhost:9000/exec?query=SELECT+count()+FROM+ohlcv" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['dataset'][0][0])" 2>/dev/null || echo "0")
    log "  QuestDB: OK ($QDB_ROWS rows in ohlcv)"
else
    log "  QuestDB: UNAVAILABLE (HTTP $QDB_STATUS) — continuing without it"
fi

# ── 6B: Restart swing (paper) system ─────────────────────────────────────────
log "6B: Stopping existing paper trading..."
PAPER_PID=$(pgrep -f "main.py --mode paper" | head -1 || true)
if [ -n "$PAPER_PID" ]; then
    kill "$PAPER_PID" && log "  Killed paper PID $PAPER_PID"
    sleep 3
else
    log "  No paper process found"
fi

# Also check for --mode swing
SWING_PID=$(pgrep -f "main.py --mode swing" | head -1 || true)
if [ -n "$SWING_PID" ]; then
    kill "$SWING_PID" && log "  Killed swing PID $SWING_PID"
    sleep 3
fi

log "6B: Starting swing system..."
bash scripts/run_paper.sh &
SWING_NEW_PID=$!
echo $SWING_NEW_PID > logs/paper.pid
log "  Swing started (background PID $SWING_NEW_PID)"
sleep 5

# ── 6C: Restart intraday system ───────────────────────────────────────────────
log "6C: Stopping existing intraday trading..."
INTRADAY_PID=$(pgrep -f "main.py --mode intraday" | head -1 || true)
if [ -n "$INTRADAY_PID" ]; then
    kill "$INTRADAY_PID" && log "  Killed intraday PID $INTRADAY_PID"
    sleep 3
else
    log "  No intraday process found"
fi

log "6C: Starting intraday system..."
bash scripts/run_intraday.sh &
INTRA_NEW_PID=$!
echo $INTRA_NEW_PID > logs/intraday.pid
log "  Intraday started (background PID $INTRA_NEW_PID)"
sleep 5

# ── Final verification ────────────────────────────────────────────────────────
log "Final check: running trading processes..."
ps aux | grep "main.py --mode" | grep -v grep | tee -a "$LOGFILE" || true

log "=== Restart complete ==="
log "  Log: $LOGFILE"
