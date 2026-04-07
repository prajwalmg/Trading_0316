#!/usr/bin/env bash
# scripts/retrain_new_models.sh — Retrain equity, crypto, and commodity models.
# Safe to run while paper/multi-asset trading is active.
# After training, models are hot-swapped via ModelRegistry.
#
# Usage:
#   bash scripts/retrain_new_models.sh                        # all non-forex instruments
#   bash scripts/retrain_new_models.sh AAPL MSFT NVDA BTC-USD # specific tickers

set -euo pipefail
cd "$(dirname "$0")/.."

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

LOG="logs/retrain_new_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

# Default: all non-forex tickers
DEFAULT_TICKERS="AAPL MSFT NVDA GOOGL META AMZN TSLA JPM SPY GS EWG SAP \
BTC-USD ETH-USD SOL-USD BNB-USD XRP-USD ADA-USD AVAX-USD \
GC=F SI=F NG=F HG=F CL=F ZW=F"

TICKERS="${*:-$DEFAULT_TICKERS}"

echo "$(date '+%Y-%m-%d %H:%M:%S'): Starting retrain for: $TICKERS" | tee -a "$LOG"

venv/bin/python - "$TICKERS" <<'PYEOF' 2>&1 | tee -a "$LOG"
import sys, os, logging

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("retrain")
sys.path.insert(0, ".")

from config.settings import ASSET_CLASS_MAP
from data.pipeline   import DataPipeline
from signals.features import get_X_y
from signals.ensemble import StackedEnsemble
from signals.regime   import RegimeTracker
from models.registry  import ModelRegistry

TICKERS = sys.argv[1].split()

def get_label_params(ticker):
    ac = ASSET_CLASS_MAP.get(ticker, "equity")
    return {
        "forex":     {"sl_mult": 1.5, "tp_mult": 1.5, "forward_bars": 20},
        "equity":    {"sl_mult": 2.0, "tp_mult": 2.0, "forward_bars": 20},
        "commodity": {"sl_mult": 1.5, "tp_mult": 1.5, "forward_bars": 20},
        "crypto":    {"sl_mult": 2.0, "tp_mult": 2.0, "forward_bars": 20},
    }.get(ac, {"sl_mult": 1.5, "tp_mult": 1.5, "forward_bars": 20})

logger.info(f"Fetching data for {len(TICKERS)} instruments...")
pipeline = DataPipeline()
pipeline.refresh_all(tickers=TICKERS, training_mode=True, interval="1h", days=730)

logger.info("Pretraining HMM regime models...")
regime_trainer = RegimeTracker()
regime_trainer.pretrain_all(TICKERS)

registry = ModelRegistry()
trained, failed = [], []

for ticker in TICKERS:
    try:
        df = pipeline.get(ticker)
        if df.empty or len(df) < 200:
            logger.warning(f"{ticker}: insufficient data ({len(df)} bars) — skip")
            failed.append(ticker)
            continue

        X, y, _ = get_X_y(df, **get_label_params(ticker), swing=True, ticker=ticker)
        if len(X) < 200:
            logger.warning(f"{ticker}: only {len(X)} samples — skip")
            failed.append(ticker)
            continue

        import numpy as np
        unique, counts = np.unique(y, return_counts=True)
        logger.info(f"{ticker}: {len(X)} samples | classes: {dict(zip(unique, counts))}")

        model = StackedEnsemble(instrument=ticker)
        model.train(X, y)
        model.save()

        ac = ASSET_CLASS_MAP.get(ticker, "equity")
        registry.register(
            ticker, ac, f"models/saved/{ticker}_ensemble.pkl",
            metrics={"samples": len(X)},
        )
        trained.append(ticker)
        logger.info(f"{ticker}: trained and registered ✓")

    except Exception as e:
        logger.error(f"{ticker}: FAILED — {e}")
        failed.append(ticker)

print(f"\n{'='*60}")
print(f"Retrain complete: {len(trained)} trained, {len(failed)} failed")
if failed:
    print(f"Failed: {failed}")
print(f"Registry summary: {registry.summary()}")
PYEOF

echo "$(date '+%Y-%m-%d %H:%M:%S'): Retrain script done — log at $LOG"
