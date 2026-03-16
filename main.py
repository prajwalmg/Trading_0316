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
import yfinance as yf
from yfinance import ticker

from execution import broker

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
from signals.features  import build_features, get_X_y, FEATURE_COLS, SWING_FEATURE_COLS, add_swing_features
from signals.ensemble  import StackedEnsemble
from signals.regime    import RegimeTracker
from risk.engine       import RiskEngine
from execution.broker  import get_broker, PaperBroker
from portfolio.manager import PortfolioManager
from dashboard.cli     import Dashboard

def get_label_params(ticker: str) -> dict:
    """Return asset-class specific label parameters."""
    from config.settings import ASSET_CLASS_MAP
    asset_class = ASSET_CLASS_MAP.get(ticker, "equity")
    return {
        "forex":     {"sl_mult": 1.5, "tp_mult": 2.0, "forward_bars": 24},
        "equity":    {"sl_mult": 2.0, "tp_mult": 3.0, "forward_bars": 48},
        "commodity": {"sl_mult": 1.5, "tp_mult": 2.5, "forward_bars": 36},
        "crypto":    {"sl_mult": 2.0, "tp_mult": 3.0, "forward_bars": 48},
    }.get(asset_class, {"sl_mult": 1.5, "tp_mult": 2.0, "forward_bars": 24})


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

        X, y, df_feat = get_X_y(df_raw, **get_label_params(ticker), swing=True)
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


def walk_forward_backtest_swing(
    df_raw:    pd.DataFrame,
    ticker:    str,
    n_windows: int = 4,
) -> dict:
    """Rolling walk-forward backtest for swing system."""
    from backtest.engine import run_backtest_single

    window  = len(df_raw) // (n_windows + 1)
    params  = get_label_params(ticker)
    pfs, wrs, dds = [], [], []
    all_trades = 0

    for i in range(n_windows):
        train_end  = (i + 1) * window
        test_start = train_end
        test_end   = test_start + window

        df_train = df_raw.iloc[:train_end]
        df_test  = df_raw.iloc[test_start:test_end]

        X_tr, y_tr, _ = get_X_y(df_train, **params, swing=True)
        if len(X_tr) < 200:
            continue

        wf_model = StackedEnsemble(instrument=f"{ticker}_wf_{i}")
        wf_model.train(X_tr, y_tr)

        result = run_backtest_single(df_test, wf_model, 10000, swing=True)
        pf     = result["metrics"].get("profit_factor", 0)
        wr     = result["metrics"].get("win_rate", 0)
        dd     = result["metrics"].get("max_drawdown", 0)
        trades = result["metrics"].get("total_trades", 0)

        pfs.append(pf)
        wrs.append(wr)
        dds.append(dd)
        all_trades += trades

        wr_display = f"{wr:.1%}" if isinstance(wr, float) else str(wr)
        dd_display = f"{dd:.1%}" if isinstance(dd, float) else str(dd)
        print(f"    Window {i+1}/{n_windows}: "
              f"PF={pf:.3f} | WR={wr_display} | "
              f"DD={dd_display} | Trades={trades}")

    if not pfs:
        return {}

    return {
        "pf_mean":      round(np.mean(pfs), 3),
        "pf_std":       round(np.std(pfs),  3),
        "pf_min":       round(np.min(pfs),  3),
        "wr_mean":      round(np.mean([w for w in wrs if isinstance(w, float)]), 3),
        "dd_worst":     round(np.min([d for d in dds if isinstance(d, (int, float))]) if any(isinstance(d, (int, float)) for d in dds) else 0, 3),
        "total_trades": all_trades,
    }


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

        X_tr, y_tr, _ = get_X_y(df_train, **get_label_params(ticker), swing=True)
        if len(X_tr) < 100:
            continue

        #model = StackedEnsemble(instrument=ticker)
        #model.train(X_tr, y_tr)

        try:
            model = StackedEnsemble.load(ticker)
            logger.info(f"Model loaded for backtest: {ticker}")
        except FileNotFoundError:
            model = StackedEnsemble(instrument=ticker)
            model.train(X_tr, y_tr)

        result = run_backtest_single(df_test, model, capital, swing=True)
        all_results[ticker] = result

        print(f"\n{'─'*50}")
        print(f"  {ticker} Backtest Results")
        print(f"{'─'*50}")
        for k, v in result["metrics"].items():
            print(f"  {k:<22} {v}")
        
        # Walk-forward validation
        print(f"\n  Walk-Forward Validation ({ticker}):")
        wf = walk_forward_backtest_swing(df_raw, ticker, n_windows=3)
        if wf:
            print(f"  PF: {wf['pf_mean']:.3f} ± {wf['pf_std']:.3f} | "
                  f"Min PF: {wf['pf_min']:.3f} | "
                  f"Worst DD: {wf['dd_worst']:.1%} | "
                  f"Trades: {wf['total_trades']}")
        else:
            print(f"  Walk-forward: insufficient data")


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
            X, y, _ = get_X_y(df_raw, **get_label_params(ticker), swing=True)
            model    = StackedEnsemble(instrument=ticker)
            model.train(X, y)
            model.save()
            models[ticker] = model
            logger.info(f"Model trained: {ticker}")

    logger.info(f"Trading loop starting | {len(models)} instruments | capital={capital:.2f}")
    cycle = 0
    daily_trade_count = {}

    try:
        while True:
            cycle += 1
            logger.info(f"\n{'─'*40} Cycle {cycle} {'─'*40}")

            # ── Refresh prices ─────────────────────────────────
            pipeline.refresh_all(tickers=list(models.keys()))

            # ── Trailing stops ────────────────────────────────────────────
            for pos in broker.get_open_positions():
                tkr           = pos["instrument"]
                current_price = broker.get_last_price(tkr)
                if current_price > 0:
                    new_sl = risk.update_trailing_stop(pos, current_price)
                    if new_sl != pos.get("sl"):
                        broker.update_stop_loss(tkr, new_sl)
                        logger.info(f"{tkr}: trailing stop → {new_sl:.5f}")

            # ── Partial profit taking ─────────────────────────────────────
            for pos in broker.get_open_positions():
                tkr           = pos["instrument"]
                entry         = pos["entry"]
                direction     = pos.get("direction", 1)
                current_price = broker.get_last_price(tkr)
                atr           = pos.get("atr", entry * 0.001)
                if current_price <= 0:
                    continue
                move = (current_price - entry) * direction
                if move >= atr and not pos.get("partial_closed"):
                    result = broker.partial_close(tkr, close_fraction=0.5)
                    if result["status"] == "partial_closed":
                        pos["partial_closed"] = True
                        logger.info(f"{tkr}: partial TP | P&L: {result['pnl']:+.2f}")

            # ── Max hold time ─────────────────────────────────────────────
            for pos in broker.get_open_positions():
                tkr       = pos["instrument"]
                bars_open = broker.get_bars_open(tkr, cycle)
                if bars_open >= 120:
                    broker.close_position(tkr, reason="max_hold_time")
                    risk.close_trade(tkr)
                    logger.info(f"{tkr}: closed — max hold time reached")

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
                
                # ── HTF trend filter ──────────────────────────────────────
                try:
                    htf_df    = yf.download(ticker, interval="4h", period="30d",                
                            progress=False, multi_level_index=False)
                    htf_df.columns = [c.lower() for c in htf_df.columns]
                    htf_c     = htf_df["close"]
                    ema20     = htf_c.ewm(span=20).mean()
                    ema50     = htf_c.ewm(span=50).mean()
                    if htf_c.iloc[-1] > ema20.iloc[-1] > ema50.iloc[-1]:                
                        htf_trend = 1
                    elif htf_c.iloc[-1] < ema20.iloc[-1] < ema50.iloc[-1]:
                        htf_trend = -1
                    else:
                        htf_trend = 0
                except Exception:
                    htf_trend = 0

                if htf_trend != 0 and htf_trend != ml_signal:
                    logger.info(f"{ticker}: against HTF trend ({htf_trend:+d}) — skip")
                    continue

                # ── Max 3 trades per instrument per day ──────────────────
                now      = datetime.now(timezone.utc)
                today    = now.strftime("%Y-%m-%d")
                day_key  = f"{ticker}_{today}"
                if daily_trade_count.get(day_key, 0) >= 3:              
                    logger.info(f"{ticker}: max 3 trades/day reached — skip")
                    continue

                # ── Rolling win rate gate ─────────────────────────────────
                import glob, json
                _wins, _total = 0, 0
                _log_path = f"logs/feedback/{ticker}_outcomes.jsonl"
                if os.path.exists(_log_path):
                    _lines = open(_log_path).readlines()[-20:]
                    for _line in _lines:
                        try:
                            _r = json.loads(_line)
                            _total += 1
                            if _r.get("won"):
                                _wins += 1
                        except Exception:
                            pass
                if _total >= 10 and (_wins / _total) < 0.35:
                    logger.info(f"{ticker}: paused — rolling WR {_wins/_total:.1%} < 35%")
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

                    # Increment daily trade counter
                    daily_trade_count[day_key] = daily_trade_count.get(day_key, 0) + 1

                    # Store metadata for feedback logging
                    if ticker in broker.positions:
                        broker.positions[ticker]["confidence"] = ml_conf
                        broker.positions[ticker]["regime"]     = agg.get("regime", "unknown")
                        broker.positions[ticker]["htf_trend"]  = htf_trend
                        broker.positions[ticker]["atr"]        = sig_dict.get("atr", 0)

                    # Register with risk engine
                    risk.open_trade({
                        "instrument": ticker,
                        "direction":  direction,
                        "units":      sizing["units"],
                        "entry":      sizing["entry"],
                        "sl":         sl,
                        "tp":         tp,
                        "confidence": ml_conf,
                        "regime":     agg.get("regime", "unknown"),
                        "htf_trend":  htf_trend,
                        "atr":        sig_dict.get("atr", 0),
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

            # Log trades to CSV after each cycle
            if isinstance(broker, PaperBroker) and not broker.trade_log.empty:
                broker.trade_log.to_csv("logs/paper_trades.csv", index=False)


            # ── Periodic rebalance ─────────────────────────────
            portfolio.rebalance_weights()

            for i in range(poll_seconds):
                time.sleep(1)
                if i % 60 == 0 and i > 0:
                    logger.info(f"Next cycle in {poll_seconds - i}s...")


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
                   default=FOREX_PAIRS + EQUITY_TICKERS + COMMODITY_TICKERS + CRYPTO_TICKERS)
    p.add_argument("--capital", type=float, default=INITIAL_CAPITAL)
    p.add_argument("--poll",    type=int,   default=3600,
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
