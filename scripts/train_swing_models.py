#!/usr/bin/env python3
"""scripts/train_swing_models.py — Train swing models with MTF features.

Saves as swing_{INSTRUMENT}_ensemble.pkl.
Quality gate: Sharpe > 1.0 AND trades >= 30.
Never overwrites existing production models (those without swing_ prefix).

Usage:
    venv/bin/python scripts/train_swing_models.py [--asset-class forex|equity|crypto|commodity|all]
"""
import sys, os, argparse, logging, time, pickle
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/train_swing_{datetime.now().strftime('%Y%m%d_%H%M')}.log"),
    ]
)
logger = logging.getLogger("train_swing")

# Swing instruments: 17 forex + 5 equity + 3 crypto + 2 commodity
SWING_INSTRUMENTS = {
    # Forex (17)
    "EURUSD=X":  "forex",
    "GBPUSD=X":  "forex",
    "USDJPY=X":  "forex",
    "AUDUSD=X":  "forex",
    "USDCAD=X":  "forex",
    "NZDUSD=X":  "forex",
    "USDCHF=X":  "forex",
    "EURGBP=X":  "forex",
    "EURJPY=X":  "forex",
    "GBPJPY=X":  "forex",
    "AUDJPY=X":  "forex",
    "EURAUD=X":  "forex",
    "GBPAUD=X":  "forex",
    "CADJPY=X":  "forex",
    "CHFJPY=X":  "forex",
    "EURCAD=X":  "forex",
    "GBPCAD=X":  "forex",
    # Equity (5)
    "AAPL":      "equity",
    "MSFT":      "equity",
    "NVDA":      "equity",
    "TSLA":      "equity",
    "SPY":       "equity",
    # Crypto (3)
    "BTC-USD":   "crypto",
    "ETH-USD":   "crypto",
    "SOL-USD":   "crypto",
    # Commodity (2)
    "GC=F":      "commodity",
    "CL=F":      "commodity",
}

MODEL_DIR = "models/saved"


def get_data(ticker: str, asset_class: str, years: int = 2):
    """Fetch 1h, 4h, 1d data for training."""
    import pandas as pd
    from data.unified_pipeline import UnifiedDataPipeline, _resample_to_htf
    udp = UnifiedDataPipeline()
    df_1h = udp.get(ticker, "1h", years=years)
    if df_1h.empty:
        logger.warning(f"{ticker}: no 1h data from pipeline, trying yfinance directly")
        try:
            import yfinance as yf
            tf_map = {"forex": "1h", "equity": "1h", "crypto": "1h", "commodity": "1h"}
            df_1h = yf.download(ticker, period=f"{min(years*365,729)}d", interval="1h",
                                progress=False, auto_adjust=True)
            if not df_1h.empty:
                if hasattr(df_1h.columns, 'get_level_values'):
                    df_1h.columns = df_1h.columns.get_level_values(0)
                df_1h.columns = [c.lower() for c in df_1h.columns]
                df_1h.index = df_1h.index.tz_localize(None) if df_1h.index.tz else df_1h.index
        except Exception as e:
            logger.error(f"{ticker}: yfinance fallback failed: {e}")
    if df_1h.empty:
        return None, None, None
    df_4h = _resample_to_htf(df_1h, "4h")
    df_1d = _resample_to_htf(df_1h, "1d")
    logger.info(f"{ticker}: 1h={len(df_1h)}, 4h={len(df_4h)}, 1d={len(df_1d)} bars")
    return df_1h, df_4h, df_1d


def build_training_data(ticker: str, asset_class: str, df_1h, df_4h, df_1d):
    """Build feature matrix and labels."""
    import numpy as np
    from signals.features import build_features, SWING_FEATURE_COLS
    try:
        df_feat = build_features(df_1h.copy(), add_labels=True, drop_na=True,
                                 swing=True, df_4h=df_4h, df_1d=df_1d)
        if df_feat.empty:
            return None, None
        mask = df_feat["label"] != 0
        df_feat = df_feat[mask]
        feat_cols = [c for c in SWING_FEATURE_COLS if c in df_feat.columns]
        X = df_feat[feat_cols].values
        y = df_feat["label"].values
        min_samples = 300 if asset_class in ("equity", "commodity") else 500
        if len(X) < min_samples:
            logger.warning(f"{ticker}: only {len(X)} training samples — skipping")
            return None, None
        logger.info(f"{ticker}: training with X={X.shape}, y dist={dict(zip(*np.unique(y, return_counts=True)))}")
        return X, y
    except Exception as e:
        logger.error(f"{ticker}: build_features error: {e}", exc_info=True)
        return None, None


def run_quality_gate(ticker: str, model, df_1h, swing: bool = True) -> dict:
    """Run backtest and check quality gate: Sharpe > 1.0 AND trades >= 30."""
    try:
        from backtest.engine import run_backtest_single
        result = run_backtest_single(df_1h, model, swing=swing, ticker=ticker)
        metrics = result.get("metrics", {})
        sharpe = metrics.get("sharpe_ratio", 0.0)
        trades = metrics.get("total_trades", 0)
        return {"sharpe": sharpe, "trades": trades, "pass": sharpe > 1.0 and trades >= 30}
    except Exception as e:
        logger.warning(f"{ticker}: backtest failed ({e}), using OOF accuracy fallback")
        return {"sharpe": 0.0, "trades": 0, "pass": False}


def train_instrument(ticker: str, asset_class: str, dry_run: bool = False) -> bool:
    """Train a swing model for one instrument. Returns True if saved."""
    logger.info(f"{'='*60}")
    logger.info(f"Training swing model: {ticker} ({asset_class})")
    t0 = time.time()

    # Check if model already trained and not forcing retrain
    out_path = Path(MODEL_DIR) / f"swing_{ticker}_ensemble.pkl"
    if out_path.exists():
        logger.info(f"{ticker}: model already exists at {out_path}, skipping")
        return False

    # Fetch data
    df_1h, df_4h, df_1d = get_data(ticker, asset_class, years=4)
    if df_1h is None:
        logger.error(f"{ticker}: no data available")
        return False

    # Build features
    X, y = build_training_data(ticker, asset_class, df_1h, df_4h, df_1d)
    if X is None:
        return False

    if dry_run:
        logger.info(f"{ticker}: DRY RUN — would train with X={X.shape}")
        return False

    # Train
    from signals.ensemble import StackedEnsemble
    model = StackedEnsemble(instrument=f"swing_{ticker}", swing=True)
    try:
        cv_metrics = model.train(X, y)
        logger.info(f"{ticker}: training done in {time.time()-t0:.0f}s, metrics={cv_metrics}")
    except Exception as e:
        logger.error(f"{ticker}: training failed: {e}", exc_info=True)
        return False

    # Quality gate
    gate = run_quality_gate(ticker, model, df_1h, swing=True)
    logger.info(f"{ticker}: quality gate — sharpe={gate['sharpe']:.3f}, trades={gate['trades']}, pass={gate['pass']}")

    if not gate["pass"]:
        logger.warning(f"{ticker}: QUALITY GATE FAILED — model NOT saved")
        return False

    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"{ticker}: model saved to {out_path}")

    # Register
    try:
        from models.registry import ModelRegistry
        from config.settings import ASSET_CLASS_MAP
        reg = ModelRegistry()
        reg.register(
            f"swing_{ticker}", asset_class, str(out_path),
            metrics={"sharpe": gate["sharpe"], "trades": gate["trades"]},
        )
    except Exception as e:
        logger.warning(f"{ticker}: registry update failed: {e}")

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-class", default="all",
                        choices=["all", "forex", "equity", "crypto", "commodity"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Retrain even if model exists")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)

    instruments = {
        k: v for k, v in SWING_INSTRUMENTS.items()
        if args.asset_class == "all" or v == args.asset_class
    }

    logger.info(f"Training {len(instruments)} swing instruments (asset_class={args.asset_class})")

    if args.force:
        # Remove existing swing models if force
        for ticker in instruments:
            p = Path(MODEL_DIR) / f"swing_{ticker}_ensemble.pkl"
            if p.exists():
                p.unlink()
                logger.info(f"Removed existing {p} (--force)")

    saved = 0
    failed = 0
    for ticker, asset_class in instruments.items():
        try:
            ok = train_instrument(ticker, asset_class, dry_run=args.dry_run)
            if ok:
                saved += 1
        except Exception as e:
            logger.error(f"{ticker}: unexpected error: {e}", exc_info=True)
            failed += 1

    logger.info(f"Done: {saved} saved, {failed} failed, {len(instruments)-saved-failed} skipped")


if __name__ == "__main__":
    main()
