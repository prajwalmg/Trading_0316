"""
================================================================
  main.py — TradingFirm OS Entry Point

  Modes:
    python main.py --mode paper          # Paper trade all instruments
    python main.py --mode train          # Train ensemble models
    python main.py --mode backtest       # Run full backtest
    python main.py --mode walkforward    # Walk-forward validation
    python main.py --mode dashboard      # Show dashboard (no trading)

  Optional flags:
    --instruments EUR_USD AAPL GC=F      # Override instrument list
    --capital 10000                      # Override starting capital
================================================================
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ── Logging first ────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/system.log"),
    ],
)
logger = logging.getLogger("trading_firm.main")

import sys
sys.path.insert(0, os.path.dirname(__file__))

from config.settings   import *
from data.pipeline     import DataPipeline
from signals.features  import build_features, get_X_y, FEATURE_COLS
from signals.ensemble  import StackedEnsemble
from signals.regime    import RegimeTracker
from risk.engine       import RiskEngine
from execution.broker  import get_broker, PaperBroker
from portfolio.manager import PortfolioManager
from dashboard.cli     import Dashboard


# ══════════════════════════════════════════════════════════════
#  MODE: TRAIN
# ══════════════════════════════════════════════════════════════

def run_train(instruments: list, capital: float):
    logger.info("=" * 60)
    logger.info("MODE: TRAIN")
    logger.info("=" * 60)

    pipeline = DataPipeline()
    pipeline.refresh_all(tickers=instruments)

    trained = []
    for ticker in instruments:
        df_raw = pipeline.get(ticker)
        if df_raw.empty:
            logger.warning(f"{ticker}: no data — skipping")
            continue

        X, y, df_feat = get_X_y(df_raw)
        if len(X) < 200:
            logger.warning(f"{ticker}: only {len(X)} samples — skipping")
            continue

        unique, counts = np.unique(y, return_counts=True)
        logger.info(f"{ticker}: {len(X)} samples | classes: {dict(zip(unique, counts))}")

        model = StackedEnsemble(instrument=ticker)
        metrics = model.train(X, y)
        model.save()
        trained.append(ticker)

    print(f"\n✓ Trained {len(trained)} models: {trained}")
    print(f"  Models saved to: {MODEL_DIR}/")


# ══════════════════════════════════════════════════════════════
#  MODE: BACKTEST
# ══════════════════════════════════════════════════════════════

def run_backtest(instruments: list, capital: float):
    from backtest.engine import run_backtest_single
    logger.info("=" * 60)
    logger.info("MODE: BACKTEST")
    logger.info("=" * 60)

    pipeline = DataPipeline()
    pipeline.refresh_all(tickers=instruments)

    all_results = {}
    for ticker in instruments:
        df_raw = pipeline.get(ticker)
        if df_raw.empty or len(df_raw) < 300:
            continue

        split    = int(len(df_raw) * 0.70)
        df_train = df_raw.iloc[:split]
        df_test  = df_raw.iloc[split:]

        X_tr, y_tr, _ = get_X_y(df_train)
        if len(X_tr) < 100:
            continue

        model = StackedEnsemble(instrument=ticker)
        model.train(X_tr, y_tr)

        result = run_backtest_single(df_test, model, capital)
        all_results[ticker] = result

        print(f"\n{'─'*50}")
        print(f"  {ticker} Backtest Results")
        print(f"{'─'*50}")
        for k, v in result["metrics"].items():
            print(f"  {k:<22} {v}")

    # Portfolio summary
    if all_results:
        print(f"\n{'═'*50}")
        print("  PORTFOLIO BACKTEST SUMMARY")
        print(f"{'═'*50}")
        sharpes = [r["metrics"].get("sharpe_ratio", 0) for r in all_results.values()]
        print(f"  Instruments traded:   {len(all_results)}")
        print(f"  Avg Sharpe ratio:     {np.mean(sharpes):.3f}")


# ══════════════════════════════════════════════════════════════
#  MODE: PAPER / LIVE TRADING LOOP
# ══════════════════════════════════════════════════════════════

def run_paper(instruments: list, capital: float, poll_seconds: int = 60):
    dash      = Dashboard()
    dash.render_startup()
    time.sleep(2)

    # Initialise components
    logger.info("Initialising DataPipeline...")
    pipeline = DataPipeline()
    pipeline.refresh_all(tickers=instruments)

    logger.info("Initialising RiskEngine...")
    risk     = RiskEngine(initial_capital=capital)

    logger.info("Initialising Broker (paper)...")
    broker   = get_broker("paper")

    logger.info("Initialising PortfolioManager...")
    portfolio = PortfolioManager(risk)

    # Load or train models
    models = {}
    for ticker in instruments:
        try:
            models[ticker] = StackedEnsemble.load(ticker)
            logger.info(f"Model loaded: {ticker}")
        except FileNotFoundError:
            df_raw = pipeline.get(ticker)
            if df_raw.empty or len(df_raw) < 200:
                logger.warning(f"{ticker}: not enough data to train — skipping")
                continue
            X, y, _ = get_X_y(df_raw)
            model    = StackedEnsemble(instrument=ticker)
            model.train(X, y)
            model.save()
            models[ticker] = model
            logger.info(f"Model trained: {ticker}")

    logger.info(f"Trading loop starting | {len(models)} instruments | capital={capital:.2f}")
    cycle = 0

    try:
        while True:
            cycle += 1
            logger.info(f"\n{'─'*40} Cycle {cycle} {'─'*40}")

            # ── Refresh prices ─────────────────────────────────
            pipeline.refresh_all(tickers=list(models.keys()), use_cache=False)

            for ticker, model in models.items():
                df_raw = pipeline.get(ticker)
                if df_raw.empty:
                    continue

                # Update broker paper price
                latest_price = pipeline.get_latest_price(ticker)
                if isinstance(broker, PaperBroker) and latest_price > 0:
                    broker.update_price(ticker, latest_price)

                # Build features + get ML signal
                df_feat = build_features(df_raw, add_labels=False, drop_na=True)
                if df_feat.empty:
                    continue

                sig_dict = model.signal_for_latest_bar(df_raw)
                ml_signal = sig_dict["signal"]
                ml_conf   = sig_dict["confidence"]

                logger.info(
                    f"{ticker}: ML signal={ml_signal:+d} | "
                    f"conf={ml_conf:.2%} | "
                    f"tradeable={sig_dict['tradeable']}"
                )

                if not sig_dict["tradeable"]:
                    continue

                # Aggregate strategies via portfolio manager
                agg = portfolio.aggregate_signals(df_feat, ticker, ml_signal, ml_conf)

                # Execution decision
                execute, reason, sizing = portfolio.should_execute(agg, df_feat)

                if execute:
                    direction = agg["signal"]
                    units     = sizing["units"] * direction
                    sl        = sizing.get("sl")
                    tp        = sizing.get("tp")

                    resp = broker.market_order(ticker, units, sl, tp)
                    logger.info(f"Order sent: {ticker} | {resp}")

                    # Register with risk engine
                    risk.open_trade({
                        "instrument": ticker,
                        "direction":  direction,
                        "units":      sizing["units"],
                        "entry":      sizing["entry"],
                        "sl":         sl,
                        "tp":         tp,
                    })
                else:
                    logger.info(f"{ticker}: blocked — {reason}")

            # ── Update risk NAV ────────────────────────────────
            acc = broker.get_account()
            risk.update_nav(acc["nav"])

            # ── Render dashboard ───────────────────────────────
            open_pos  = broker.get_open_positions()
            trade_hist = broker.trade_log if isinstance(broker, PaperBroker) else pd.DataFrame()
            regime_map = portfolio.regime_tracker.all_current()
            port_sum   = portfolio.portfolio_summary()

            dash.render(
                account=acc,
                risk_report=risk.report(),
                open_positions=open_pos,
                trade_history=trade_hist,
                regime_map=regime_map,
                portfolio_summary=port_sum,
                cycle=cycle,
            )

            # ── Periodic rebalance ─────────────────────────────
            portfolio.rebalance_weights()

            time.sleep(poll_seconds)

    except KeyboardInterrupt:
        print("\n\nStopped by user.")
        acc = broker.get_account()
        print(f"Final NAV: ${acc['nav']:,.2f}")
        print(f"Total trades: {len(broker.trade_log) if isinstance(broker, PaperBroker) else '?'}")


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="TradingFirm OS")
    p.add_argument("--mode", choices=["paper","train","backtest","walkforward"],
                   default="paper")
    p.add_argument("--instruments", nargs="+",
                   default=FOREX_PAIRS[:4] + EQUITY_TICKERS[:2] + COMMODITY_TICKERS[:2])
    p.add_argument("--capital", type=float, default=INITIAL_CAPITAL)
    p.add_argument("--poll",    type=int,   default=60,
                   help="Seconds between cycles (paper mode)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "train":
        run_train(args.instruments, args.capital)

    elif args.mode == "backtest":
        run_backtest(args.instruments, args.capital)

    elif args.mode in ("paper", "live"):
        run_paper(args.instruments, args.capital, args.poll)
