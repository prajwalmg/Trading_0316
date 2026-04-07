#!/usr/bin/env bash
# Quick status check for the trading system
# Usage: bash scripts/status.sh

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB="$BASE_DIR/data/trades.db"
LOG="$BASE_DIR/logs/system.log"
SEP="============================================================"

echo "$SEP"
echo "  TRADING SYSTEM STATUS  —  $(date '+%Y-%m-%d %H:%M:%S')"
echo "$SEP"

# ── 1. Process PIDs ──────────────────────────────────────────────
echo ""
echo "--- 1. PROCESS STATUS ---"

for MODE in intraday swing; do
    PID_FILE="$BASE_DIR/logs/${MODE}.pid"
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE" 2>/dev/null | tr -d '[:space:]')
        if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
            echo "  $MODE  : RUNNING  (pid=$PID)"
        else
            echo "  $MODE  : STOPPED  (last pid=${PID:-N/A})"
        fi
    else
        echo "  $MODE  : STOPPED  (no pid file)"
    fi
done

# Also check multi-asset
MULTI_PID_FILE="$BASE_DIR/logs/multi_asset.pid"
if [ -f "$MULTI_PID_FILE" ]; then
    PID=$(cat "$MULTI_PID_FILE" 2>/dev/null | tr -d '[:space:]')
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        echo "  multi  : RUNNING  (pid=$PID)"
    fi
fi

# Cross-check with ps
INTRADAY_PS=$(ps aux 2>/dev/null | grep 'mode intraday' | grep -v grep)
SWING_PS=$(ps aux 2>/dev/null | grep 'mode paper' | grep -v grep)
if [ -n "$INTRADAY_PS" ]; then
    echo "  (ps confirms intraday running)"
fi
if [ -n "$SWING_PS" ]; then
    echo "  (ps confirms swing/paper running)"
fi

# ── 2. Last 5 trades ─────────────────────────────────────────────
echo ""
echo "--- 2. LAST 5 TRADES ---"

if [ -f "$DB" ]; then
    sqlite3 -column -header "$DB" \
        "SELECT id, instrument, direction,
                printf('%.5f', entry) AS entry,
                printf('%.5f', exit_price) AS exit_price,
                printf('%+.4f', pnl) AS pnl,
                system,
                substr(exit_time, 1, 19) AS exit_time
         FROM trades
         ORDER BY id DESC
         LIMIT 5;" 2>/dev/null || echo "  (sqlite3 not available or query error)"
else
    echo "  trades.db not found at $DB"
fi

# ── 3. NAV = 10000 + sum(pnl) ────────────────────────────────────
echo ""
echo "--- 3. NAV ESTIMATE ---"

if [ -f "$DB" ]; then
    TOTAL_PNL=$(sqlite3 "$DB" \
        "SELECT COALESCE(sum(pnl), 0) FROM trades WHERE pnl IS NOT NULL;" 2>/dev/null)
    TOTAL_TRADES=$(sqlite3 "$DB" \
        "SELECT count(*) FROM trades WHERE pnl IS NOT NULL;" 2>/dev/null)
    WIN_TRADES=$(sqlite3 "$DB" \
        "SELECT count(*) FROM trades WHERE pnl > 0;" 2>/dev/null)

    if [ -n "$TOTAL_PNL" ]; then
        NAV=$(python3 -c "print(f'  NAV = 10000 + ({float(\"$TOTAL_PNL\"):.4f}) = {10000 + float(\"$TOTAL_PNL\"):.2f}')" 2>/dev/null \
              || echo "  NAV = 10000 + $TOTAL_PNL  (compute python3 for exact value)")
        echo "$NAV"
        if [ -n "$TOTAL_TRADES" ] && [ "$TOTAL_TRADES" -gt 0 ] && [ -n "$WIN_TRADES" ]; then
            WR=$(python3 -c "print(f'  Win rate: {int(\"$WIN_TRADES\")}/{int(\"$TOTAL_TRADES\")} = {int(\"$WIN_TRADES\")/int(\"$TOTAL_TRADES\"):.1%}')" 2>/dev/null)
            echo "$WR"
        fi
    fi
else
    echo "  trades.db not found"
fi

# ── 4. Last 5 relevant log lines ─────────────────────────────────
echo ""
echo "--- 4. RECENT LOG (last 5 relevant lines) ---"

# Prefer system.log; fall back to today's dated log
TODAY=$(date '+%Y%m%d')
DATED_LOG="$BASE_DIR/logs/paper_trading_${TODAY}.log"

if [ -f "$LOG" ]; then
    tail -200 "$LOG" | grep -E 'ERROR|WARN|TRADE|SIGNAL|BUY|SELL|INFO' \
        | grep -v 'yfinance' \
        | tail -5 \
        | sed 's/^/  /'
elif [ -f "$DATED_LOG" ]; then
    tail -200 "$DATED_LOG" | grep -E 'ERROR|WARN|TRADE|SIGNAL|BUY|SELL|INFO' \
        | tail -5 \
        | sed 's/^/  /'
else
    echo "  No log file found"
fi

echo ""
echo "$SEP"
