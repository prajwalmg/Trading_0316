#!/bin/bash
# Run on 1st of each month at 2am
# Retrains all models on latest data

cd /Users/mgprajwal/Downloads/trading_firm
source venv/bin/activate

LOGFILE="logs/retrain_$(date +%Y%m%d).log"
echo "$(date): Starting monthly retrain" >> $LOGFILE

# Clear pipeline cache to force fresh data fetch
find data/cache -name "*.parquet" -not -path "*/dukascopy/*" -mtime +1 -delete

OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
python main.py --mode train \
  --instruments \
    EURUSD=X GBPUSD=X EURJPY=X USDCAD=X \
    GBPJPY=X AUDUSD=X EURGBP=X USDJPY=X \
  >> $LOGFILE 2>&1

if [ $? -eq 0 ]; then
  python -c "
import sys; sys.path.insert(0,'.')
from notifications.telegram import _send
from datetime import datetime
_send('✅ Monthly retrain complete\n' + datetime.now().strftime('%d %b %Y') + '\nAll 8 models updated')
  "
else
  python -c "
import sys; sys.path.insert(0,'.')
from notifications.telegram import _send
_send('❌ Monthly retrain FAILED\nCheck logs/retrain_$(date +%Y%m%d).log')
  "
fi
