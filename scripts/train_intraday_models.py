#!/usr/bin/env python3
"""scripts/train_intraday_models.py — Train intraday models with HTF context features.

Saves as intraday_{INSTRUMENT}_ensemble.pkl.
Quality gate: Sharpe > 1.0 AND trades >= 30.

Usage:
    venv/bin/python scripts/train_intraday_models.py [--pair EURUSD=X|all]
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
        logging.FileHandler(f"logs/train_intraday_{datetime.now().strftime('%Y%m%d_%H%M')}.log"),
    ]
)
logger = logging.getLogger("train_intraday")

# 10 intraday instruments (all forex 5m)
INTRADAY_INSTRUMENTS = {
    "EURUSD=X": "forex",
    "GBPUSD=X": "forex",
    "USDJPY=X": "forex",
    "AUDUSD=X": "forex",
    "USDCAD=X": "forex",
    "NZDUSD=X": "forex",
    "USDCHF=X": "forex",
    "EURGBP=X": "forex",
    "EURJPY=X": "forex",
    "GBPJPY=X": "forex",
}

MODEL_DIR = "models/saved"


def get_data(ticker: str, years_5m: int = 1, years_1h: int = 2):
    """Fetch 5m (base), 1h (context), and 4h (context) data."""
    import pandas as pd
    import yfinance as yf
    from data.unified_pipeline import UnifiedDataPipeline, _resample_to_htf

    udp = UnifiedDataPipeline()

    # 5m data: try yfinance FIRST to avoid Dukascopy hanging indefinitely
    # on pairs with no 5m cache (NZDUSD, USDCHF, etc.)
    df_5m = pd.DataFrame()
    try:
        df_5m = yf.download(ticker, period="60d", interval="5m",
                            progress=False, auto_adjust=True)
        if not df_5m.empty:
            if hasattr(df_5m.columns, 'get_level_values'):
                df_5m.columns = df_5m.columns.get_level_values(0)
            df_5m.columns = [c.lower() for c in df_5m.columns]
            df_5m.index = df_5m.index.tz_localize(None) if df_5m.index.tz else df_5m.index
            logger.info(f"{ticker}: 5m via yfinance — {len(df_5m)} bars")
    except Exception as e:
        logger.error(f"{ticker}: 5m yfinance failed: {e}")

    if df_5m.empty:
        # Last resort: unified pipeline (may call Dukascopy — can be slow)
        logger.warning(f"{ticker}: yfinance 5m empty, trying pipeline")
        df_5m = udp.get(ticker, "5m", years=years_5m)

    # 1h data (HTF context)
    df_1h = udp.get(ticker, "1h", years=years_1h)
    if df_1h.empty:
        # Resample from 5m if available
        if not df_5m.empty:
            df_1h = df_5m.resample("1h", closed="left", label="left").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum",
            }).dropna(subset=["close"])

    # 4h data (HTF context) - resample from 1h
    df_4h = pd.DataFrame()
    if not df_1h.empty:
        from data.unified_pipeline import _resample_to_htf
        df_4h = _resample_to_htf(df_1h, "4h")

    logger.info(f"{ticker}: 5m={len(df_5m)}, 1h={len(df_1h)}, 4h={len(df_4h)} bars")
    return df_5m, df_1h, df_4h


def build_training_data(ticker: str, df_5m, df_1h, df_4h):
    """Build intraday feature matrix and labels."""
    import numpy as np
    from signals.features_intraday import build_features_intraday, get_X_y_intraday
    try:
        df_feat = build_features_intraday(
            df_5m.copy(), add_labels=True, drop_na=True,
            df_1h=df_1h, df_4h=df_4h,
        )
        if df_feat.empty:
            return None, None
        X, y, df_feat = get_X_y_intraday(df_feat)
        if len(X) < 500:
            logger.warning(f"{ticker}: only {len(X)} training samples — skipping")
            return None, None
        logger.info(f"{ticker}: training with X={X.shape}, y dist={dict(zip(*np.unique(y, return_counts=True)))}")
        return X, y
    except Exception as e:
        logger.error(f"{ticker}: build_features_intraday error: {e}", exc_info=True)
        return None, None


def run_quality_gate(ticker: str, model, df_5m) -> dict:
    """Simple walk-forward quality gate using backtest engine."""
    try:
        from backtest.engine import run_backtest_single
        result = run_backtest_single(df_5m, model, swing=False, ticker=ticker)
        metrics = result.get("metrics", {})
        sharpe = metrics.get("sharpe_ratio", 0.0)
        trades = metrics.get("total_trades", 0)
        return {"sharpe": sharpe, "trades": trades, "pass": sharpe > 1.0 and trades >= 30}
    except Exception as e:
        logger.warning(f"{ticker}: backtest failed ({e})")
        return {"sharpe": 0.0, "trades": 0, "pass": False}


def train_instrument(ticker: str, asset_class: str, dry_run: bool = False) -> bool:
    """Train an intraday model for one instrument. Returns True if saved."""
    logger.info(f"{'='*60}")
    logger.info(f"Training intraday model: {ticker} ({asset_class})")
    t0 = time.time()

    out_path = Path(MODEL_DIR) / f"intraday_{ticker}_ensemble.pkl"
    if out_path.exists():
        logger.info(f"{ticker}: model already exists at {out_path}, skipping")
        return False

    df_5m, df_1h, df_4h = get_data(ticker)
    if df_5m.empty:
        logger.error(f"{ticker}: no 5m data available")
        return False

    X, y = build_training_data(ticker, df_5m, df_1h, df_4h)
    if X is None:
        return False

    if dry_run:
        logger.info(f"{ticker}: DRY RUN — would train with X={X.shape}")
        return False

    from signals.ensemble import StackedEnsemble
    model = StackedEnsemble(instrument=f"intraday_{ticker}", swing=False)
    try:
        cv_metrics = model.train(X, y)
        logger.info(f"{ticker}: training done in {time.time()-t0:.0f}s, metrics={cv_metrics}")
    except Exception as e:
        logger.error(f"{ticker}: training failed: {e}", exc_info=True)
        return False

    gate = run_quality_gate(ticker, model, df_5m)
    logger.info(f"{ticker}: quality gate — sharpe={gate['sharpe']:.3f}, trades={gate['trades']}, pass={gate['pass']}")

    if not gate["pass"]:
        logger.warning(f"{ticker}: QUALITY GATE FAILED — model NOT saved")
        return False

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"{ticker}: model saved to {out_path}")

    try:
        from models.registry import ModelRegistry
        reg = ModelRegistry()
        reg.register(
            f"intraday_{ticker}", asset_class, str(out_path),
            metrics={"sharpe": gate["sharpe"], "trades": gate["trades"]},
        )
    except Exception as e:
        logger.warning(f"{ticker}: registry update failed: {e}")

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default="all", help="Specific pair or 'all'")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)

    instruments = INTRADAY_INSTRUMENTS
    if args.pair != "all":
        instruments = {k: v for k, v in instruments.items() if k == args.pair}

    logger.info(f"Training {len(instruments)} intraday instruments")

    if args.force:
        for ticker in instruments:
            p = Path(MODEL_DIR) / f"intraday_{ticker}_ensemble.pkl"
            if p.exists():
                p.unlink()
                logger.info(f"Removed {p} (--force)")

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
