#!/usr/bin/env python3
"""scripts/promote_models.py — Compare new swing models vs production, promote if better.

Rules:
  - Compare new swing_{ticker}_ensemble.pkl vs production {ticker}_ensemble.pkl
  - Promote only if new Sharpe > prod Sharpe (or prod model doesn't exist)
  - Always backup production model as {ticker}_ensemble.pkl.bak before replacing
  - Registers promoted model under original ticker name in ModelRegistry

Usage:
    venv/bin/python scripts/promote_models.py [--asset-class forex|all] [--dry-run]
"""
import sys, os, pickle, logging, argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/promote_{datetime.now().strftime('%Y%m%d_%H%M')}.log"),
    ]
)
logger = logging.getLogger("promote_models")

MODEL_DIR = Path("models/saved")

SWING_INSTRUMENTS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X",
    "NZDUSD=X", "USDCHF=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X",
    "AUDJPY=X", "EURAUD=X", "GBPAUD=X", "CADJPY=X", "CHFJPY=X",
    "EURCAD=X", "GBPCAD=X",
    "AAPL", "MSFT", "NVDA", "TSLA", "SPY",
    "BTC-USD", "ETH-USD", "SOL-USD",
    "GC=F", "CL=F",
]


def get_model_sharpe(model_path: Path) -> float:
    """Load model and return its sharpe from registry metrics, or 0.0."""
    try:
        from models.registry import ModelRegistry
        reg = ModelRegistry()
        # Try to find in registry by matching path
        for ticker, info in reg._data["instruments"].items():
            if Path(info.get("model_path", "")) == model_path:
                return info.get("metrics", {}).get("sharpe", 0.0)
    except Exception:
        pass

    # Fallback: load and check _cv_metrics
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        cv = getattr(model, "_cv_metrics", {})
        # OOF accuracy is a proxy — not Sharpe, but use ensemble accuracy
        meta_acc = cv.get("ensemble", {}).get("meta_accuracy", 0.0)
        return meta_acc  # 0.0 if no Sharpe stored
    except Exception:
        return 0.0


def run_quick_backtest(ticker: str, model_path: Path) -> float:
    """Run a quick backtest and return Sharpe."""
    try:
        import pickle
        import yfinance as yf
        from backtest.engine import run_backtest_single

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        df = yf.download(ticker, period="730d", interval="1h",
                         progress=False, auto_adjust=True)
        if df.empty:
            return 0.0
        if hasattr(df.columns, "get_level_values"):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]

        result = run_backtest_single(df, model, swing=True, ticker=ticker)
        return result.get("metrics", {}).get("sharpe_ratio", 0.0)
    except Exception as e:
        logger.warning(f"{ticker}: backtest failed: {e}")
        return 0.0


def promote(ticker: str, dry_run: bool = False) -> str:
    """Compare and optionally promote swing model. Returns action taken."""
    new_path  = MODEL_DIR / f"swing_{ticker}_ensemble.pkl"
    prod_path = MODEL_DIR / f"{ticker}_ensemble.pkl"
    bak_path  = MODEL_DIR / f"{ticker}_ensemble.pkl.bak"

    if not new_path.exists():
        return f"SKIP: no new model at {new_path}"

    # Get new model Sharpe from registry or backtest
    from models.registry import ModelRegistry
    reg = ModelRegistry()
    new_info = reg.get_info(f"swing_{ticker}")
    new_sharpe = new_info.get("metrics", {}).get("sharpe", None)

    if new_sharpe is None:
        logger.info(f"{ticker}: no Sharpe in registry — running backtest...")
        new_sharpe = run_quick_backtest(ticker, new_path)

    # Get prod model Sharpe
    prod_sharpe = 0.0
    if prod_path.exists():
        prod_info = reg.get_info(ticker)
        prod_sharpe = prod_info.get("metrics", {}).get("sharpe", None)
        if prod_sharpe is None:
            prod_sharpe = run_quick_backtest(ticker, prod_path)

    logger.info(f"{ticker}: new_sharpe={new_sharpe:.3f}, prod_sharpe={prod_sharpe:.3f}")

    if new_sharpe <= prod_sharpe:
        return f"KEEP_PROD: new={new_sharpe:.3f} <= prod={prod_sharpe:.3f}"

    if dry_run:
        return f"DRY_RUN_PROMOTE: new={new_sharpe:.3f} > prod={prod_sharpe:.3f}"

    # Backup production
    import shutil
    if prod_path.exists():
        shutil.copy2(prod_path, bak_path)
        logger.info(f"{ticker}: backed up to {bak_path}")

    # Promote: copy new → prod
    shutil.copy2(new_path, prod_path)
    logger.info(f"{ticker}: promoted swing_{ticker} → {ticker}_ensemble.pkl")

    # Update registry
    try:
        asset_class = new_info.get("asset_class", "forex")
        reg.register(ticker, asset_class, str(prod_path),
                     metrics={"sharpe": new_sharpe})
    except Exception as e:
        logger.warning(f"{ticker}: registry update failed: {e}")

    return f"PROMOTED: new={new_sharpe:.3f} > prod={prod_sharpe:.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-class", default="all",
                        choices=["all", "forex", "equity", "crypto", "commodity"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)

    from config.settings import ASSET_CLASS_MAP
    instruments = [
        t for t in SWING_INSTRUMENTS
        if args.asset_class == "all" or ASSET_CLASS_MAP.get(t, "forex") == args.asset_class
    ]

    logger.info(f"Evaluating {len(instruments)} instruments for promotion")
    results = {}
    for ticker in instruments:
        action = promote(ticker, dry_run=args.dry_run)
        results[ticker] = action
        logger.info(f"  {ticker}: {action}")

    promoted = [t for t, a in results.items() if "PROMOTED" in a]
    logger.info(f"\nSummary: {len(promoted)}/{len(instruments)} promoted")
    if promoted:
        logger.info(f"  Promoted: {promoted}")


if __name__ == "__main__":
    main()
