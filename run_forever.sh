#!/usr/bin/env bash
# ================================================================
#  run_forever.sh
#  Auto-restart wrapper for Trading Firm OS.
#
#  Usage:
#    chmod +x run_forever.sh
#    ./run_forever.sh                    # paper mode (default)
#    ./run_forever.sh --mode paper --capital 10000
#
#  Behaviour:
#    - Activates Python venv if present (./venv or ./.venv)
#    - Restarts automatically on crash (exit code ≠ 0)
#    - Exits cleanly if main.py returns 0 (user pressed Ctrl+C)
#    - Stops after MAX_RESTARTS consecutive crashes
#    - Waits RESTART_DELAY seconds between restarts
#    - Logs each restart to logs/restart.log
# ================================================================

MAX_RESTARTS=10
RESTART_DELAY=30
RESTART_COUNT=0
LOG_FILE="logs/restart.log"
ARGS="${@:---mode paper}"

mkdir -p logs

# ── Activate virtual environment if available ─────────────────
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "[run_forever] venv activated"
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "[run_forever] .venv activated"
fi

echo "========================================"                >> "$LOG_FILE"
echo "  run_forever.sh started: $(date -u)"                   >> "$LOG_FILE"
echo "  Args: $ARGS"                                           >> "$LOG_FILE"
echo "========================================"                >> "$LOG_FILE"

while true; do
    echo "[run_forever] Starting main.py  (restart #${RESTART_COUNT})  $(date -u)"
    echo "$(date -u) — START (restart #${RESTART_COUNT})"     >> "$LOG_FILE"

    python main.py $ARGS
    EXIT_CODE=$?

    echo "$(date -u) — EXIT code=${EXIT_CODE}"                >> "$LOG_FILE"

    # Clean exit (Ctrl+C or intentional shutdown)
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[run_forever] Clean exit (code 0). Goodbye."
        echo "$(date -u) — Clean exit."                       >> "$LOG_FILE"
        exit 0
    fi

    RESTART_COUNT=$((RESTART_COUNT + 1))

    if [ $RESTART_COUNT -ge $MAX_RESTARTS ]; then
        echo "[run_forever] Max restarts ($MAX_RESTARTS) reached. Giving up."
        echo "$(date -u) — ABORT after $MAX_RESTARTS restarts." >> "$LOG_FILE"
        exit 1
    fi

    echo "[run_forever] Crash detected (code $EXIT_CODE). " \
         "Restarting in ${RESTART_DELAY}s... " \
         "(${RESTART_COUNT}/${MAX_RESTARTS})"
    echo "$(date -u) — Restart in ${RESTART_DELAY}s " \
         "(${RESTART_COUNT}/${MAX_RESTARTS})"                  >> "$LOG_FILE"

    sleep "$RESTART_DELAY"
done
