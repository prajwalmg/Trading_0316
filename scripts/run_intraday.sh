#!/bin/bash
# Persistent intraday trading runner
# Auto-restarts on crash

cd /Users/mgprajwal/Downloads/trading_firm
source venv/bin/activate
export DYLD_LIBRARY_PATH="/Users/mgprajwal/.pyenv/versions/3.11.9-arm64/lib:$DYLD_LIBRARY_PATH"

LOGFILE="logs/intraday_trading_$(date +%Y%m%d).log"

echo "$(date): Starting intraday trading" >> $LOGFILE

while true; do
  echo "$(date): Launching main.py --mode intraday" >> $LOGFILE

  OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  python main.py --mode intraday --poll 300 \
    >> $LOGFILE 2>&1

  EXIT_CODE=$?
  echo "$(date): Exited with code $EXIT_CODE" >> $LOGFILE

  if [ $EXIT_CODE -eq 130 ] || [ $EXIT_CODE -eq 143 ]; then
    echo "$(date): Manual stop — not restarting" >> $LOGFILE
    break
  fi

  echo "$(date): Restarting in 60s..." >> $LOGFILE
  sleep 60
done
