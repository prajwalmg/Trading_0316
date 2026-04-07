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

# ── Thread-count limits (must be set before any ML imports) ──
# Prevents OpenMP/MKL/LGBM threads from deadlocking PyTorch
# on macOS when multiple models predict in sequence.
import os as _os
_os.environ.setdefault("OMP_NUM_THREADS",      "1")
_os.environ.setdefault("MKL_NUM_THREADS",      "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta

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

def get_min_confidence(ticker: str) -> float:
    """Return per-instrument MIN_CONFIDENCE, falling back to global setting."""
    from config.settings import INSTRUMENT_MIN_CONFIDENCE, MIN_CONFIDENCE
    return INSTRUMENT_MIN_CONFIDENCE.get(ticker, MIN_CONFIDENCE)


def get_label_params(ticker: str) -> dict:
    """Return asset-class specific label parameters for DAILY bars."""
    from config.settings import ASSET_CLASS_MAP
    asset_class = ASSET_CLASS_MAP.get(ticker, "equity")
    return {
        "forex":     {"sl_mult": 1.5, "tp_mult": 1.5, "forward_bars": 20},
        "equity":    {"sl_mult": 2.0, "tp_mult": 2.0, "forward_bars": 20},
        "commodity": {"sl_mult": 1.5, "tp_mult": 1.5, "forward_bars": 20},
        "crypto":    {"sl_mult": 2.0, "tp_mult": 2.0, "forward_bars": 20},
    }.get(asset_class, {"sl_mult": 1.5, "tp_mult": 1.5, "forward_bars": 20})


# ══════════════════════════════════════════════════════════════
#  MODE: TRAIN
# ══════════════════════════════════════════════════════════════

def run_train(instruments: list, capital: float):
    logger.info("=" * 60)
    logger.info("MODE: TRAIN")
    logger.info("=" * 60)

    pipeline = DataPipeline()
    pipeline.refresh_all(
        tickers=instruments,
        training_mode=True,
        interval="1h",
        days=1825,   # 5 years of daily bars for training
    )

    # ── HMM regime pretraining ────────────────────────────────
    logger.info("Pretraining HMM regime models...")
    from signals.regime import RegimeTracker as _RT
    _regime_trainer = _RT()
    _regime_trainer.pretrain_all(instruments)

    trained = []
    for ticker in instruments:
        df_raw = pipeline.get(ticker)
        if df_raw.empty:
            logger.warning(f"{ticker}: no data — skipping")
            continue

        X, y, df_feat = get_X_y(df_raw, **get_label_params(ticker), swing=True, ticker=ticker)
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
    df_raw:          pd.DataFrame,
    ticker:          str,
    n_windows:       int  = 5,
    sweep_thresholds: bool = False,
) -> dict:
    """Rolling walk-forward backtest for swing system.

    If sweep_thresholds=True, reuses each WF-trained model to evaluate
    confidence thresholds [0.54..0.65] at no additional training cost,
    and returns the best threshold found (max avg Sharpe, trades>=30, PF>=1.0).
    """
    from backtest.engine import run_backtest_single

    window  = len(df_raw) // (n_windows + 1)
    params  = get_label_params(ticker)
    pfs, wrs, dds, sharpes = [], [], [], []
    all_trades = 0

    # Per-threshold accumulators when sweeping
    _thresh_list = [round(0.54 + k * 0.01, 2) for k in range(12)]  # 0.54..0.65
    _thresh_sharpes: dict = {t: [] for t in _thresh_list}
    _thresh_pfs:     dict = {t: [] for t in _thresh_list}
    _thresh_trades:  dict = {t: 0  for t in _thresh_list}

    for i in range(n_windows):
        train_end  = (i + 1) * window
        test_start = train_end
        test_end   = test_start + window

        df_train = df_raw.iloc[:train_end]
        df_test  = df_raw.iloc[test_start:test_end]

        X_tr, y_tr, _ = get_X_y(df_train, **params, swing=True, ticker=ticker)
        if len(X_tr) < 200:
            continue

        wf_model = StackedEnsemble(instrument=f"{ticker}_wf_{i}")
        wf_model.train(X_tr, y_tr)

        result = run_backtest_single(df_test, wf_model, 10000, swing=True, ticker=ticker, min_conf=get_min_confidence(ticker))
        pf     = result["metrics"].get("profit_factor", 0)
        wr     = result["metrics"].get("win_rate", 0)
        dd     = result["metrics"].get("max_drawdown", 0)
        trades = result["metrics"].get("total_trades", 0)
        sharpe = result["metrics"].get("sharpe_ratio", 0)

        pfs.append(pf); wrs.append(wr); dds.append(dd)
        sharpes.append(sharpe); all_trades += trades

        wr_display = f"{wr:.1%}" if isinstance(wr, float) else str(wr)
        dd_display = f"{dd:.1%}" if isinstance(dd, float) else str(dd)
        print(f"    Window {i+1}/{n_windows}: "
              f"PF={pf:.3f} | WR={wr_display} | "
              f"DD={dd_display} | Sharpe={sharpe:.3f} | Trades={trades}")

        # Threshold sweep — reuse same trained model, no extra training
        if sweep_thresholds:
            for thresh in _thresh_list:
                r = run_backtest_single(df_test, wf_model, 10000, swing=True,
                                        ticker=ticker, min_conf=thresh)
                _thresh_sharpes[thresh].append(r["metrics"].get("sharpe_ratio", 0))
                _thresh_pfs[thresh].append(r["metrics"].get("profit_factor", 0))
                _thresh_trades[thresh] += r["metrics"].get("total_trades", 0)

    if not pfs:
        return {}

    result_dict = {
        "pf_mean":      round(np.mean(pfs), 3),
        "pf_std":       round(np.std(pfs),  3),
        "pf_min":       round(np.min(pfs),  3),
        "sharpe_mean":  round(np.mean(sharpes), 3),
        "wr_mean":      round(np.mean([w for w in wrs if isinstance(w, float)]), 3),
        "dd_worst":     round(np.min([d for d in dds if isinstance(d, (int, float))]) if any(isinstance(d, (int, float)) for d in dds) else 0, 3),
        "total_trades": all_trades,
    }

    if sweep_thresholds:
        best_thresh = get_min_confidence(ticker)
        best_sharpe = -999.0
        print(f"\n  Threshold sweep ({ticker}):")
        for thresh in _thresh_list:
            ss = _thresh_sharpes[thresh]
            ps = _thresh_pfs[thresh]
            tt = _thresh_trades[thresh]
            if not ss:
                continue
            avg_s  = float(np.mean(ss))
            avg_pf = float(np.mean(ps))
            flag = ""
            if tt >= 30 and avg_pf >= 1.0 and avg_s > best_sharpe:
                best_sharpe = avg_s
                best_thresh = thresh
                flag = " ◀ best"
            print(f"    thresh={thresh:.2f}: Sharpe={avg_s:.3f} | PF={avg_pf:.3f} | Trades={tt}{flag}")
        result_dict["best_threshold"] = best_thresh
        result_dict["best_sharpe"]    = round(best_sharpe, 3)

    return result_dict


# ══════════════════════════════════════════════════════════════
#  MODE: BACKTEST
# ══════════════════════════════════════════════════════════════

def run_backtest(instruments: list, capital: float):
    from backtest.engine import run_backtest_single
    logger.info("=" * 60)
    logger.info("MODE: BACKTEST")
    logger.info("=" * 60)

    pipeline = DataPipeline()
    pipeline.refresh_all(tickers=instruments, training_mode=True)

    all_results = {}
    for ticker in instruments:
        df_raw = pipeline.get(ticker)
        if df_raw.empty or len(df_raw) < 300:
            continue

        split    = int(len(df_raw) * 0.70)
        df_train = df_raw.iloc[:split]
        df_test  = df_raw.iloc[split:]

        X_tr, y_tr, _ = get_X_y(df_train, **get_label_params(ticker), swing=True, ticker=ticker)
        if len(X_tr) < 100:
            continue

        #model = StackedEnsemble(instrument=ticker)
        #model.train(X_tr, y_tr)

        model = StackedEnsemble(instrument=ticker)
        model.train(X_tr, y_tr)
        logger.info(f"Model retrained on 70% split for backtest: {ticker}")

        result = run_backtest_single(df_test, model, capital, swing=True, ticker=ticker, min_conf=get_min_confidence(ticker), use_circuit_breaker=False)
        all_results[ticker] = result

        print(f"\n{'─'*50}")
        print(f"  {ticker} Backtest Results")
        print(f"{'─'*50}")
        for k, v in result["metrics"].items():
            print(f"  {k:<22} {v}")
        
        # Walk-forward validation with threshold sweep
        print(f"\n  Walk-Forward Validation ({ticker}):")
        wf = walk_forward_backtest_swing(df_raw, ticker, sweep_thresholds=True)
        if wf:
            print(f"  PF: {wf['pf_mean']:.3f} ± {wf['pf_std']:.3f} | "
                  f"Min PF: {wf['pf_min']:.3f} | "
                  f"Sharpe: {wf.get('sharpe_mean', 0):.3f} | "
                  f"Worst DD: {wf['dd_worst']:.1%} | "
                  f"Trades: {wf['total_trades']}")
            if "best_threshold" in wf:
                current = get_min_confidence(ticker)
                best    = wf["best_threshold"]
                print(f"  Optimal threshold: {best:.2f} (current: {current:.2f}, "
                      f"best Sharpe: {wf['best_sharpe']:.3f})")
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
    # ── Runtime overrides (auto-tuned MIN_CONFIDENCE, etc.) ────
    import json as _json
    _overrides_path = "config/runtime_overrides.json"
    if os.path.exists(_overrides_path):
        try:
            with open(_overrides_path) as _f:
                _ov = _json.load(_f)
            import config.settings as _cfg
            for _k, _v in _ov.items():
                if hasattr(_cfg, _k):
                    setattr(_cfg, _k, _v)
                    logger.info(f"Runtime override applied: {_k}={_v}")
        except Exception as _e:
            logger.warning(f"Could not load runtime overrides: {_e}")

    from utils.alerts import alert_startup
    from notifications.telegram import (
        system_started, trade_opened, trade_closed,
        daily_report, drawdown_alert, system_error,
    )
    alert_startup()

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
    portfolio = PortfolioManager(risk, pipeline=pipeline)

    # Log upcoming high-impact economic events
    from data.calendar import log_upcoming_events
    log_upcoming_events()

    # Load or train models — prefer swing_{ticker} models (MTF-trained)
    from config.settings import SWING_EXCLUDED_INSTRUMENTS
    models = {}
    for ticker in instruments:
        if ticker in SWING_EXCLUDED_INSTRUMENTS:
            logger.info(f"Skipping {ticker} — in SWING_EXCLUDED_INSTRUMENTS (no valid WF threshold)")
            continue
        loaded = False
        for name in (f"swing_{ticker}", ticker):
            try:
                models[ticker] = StackedEnsemble.load(name)
                logger.info(f"Model loaded: {name} → {ticker}")
                loaded = True
                break
            except (FileNotFoundError, Exception):
                continue
        if not loaded:
            df_raw = pipeline.get(ticker)
            if df_raw.empty or len(df_raw) < 200:
                logger.warning(f"{ticker}: not enough data to train — skipping")
                continue
            X, y, _ = get_X_y(df_raw, **get_label_params(ticker), swing=True, ticker=ticker)
            model    = StackedEnsemble(instrument=ticker)
            model.train(X, y)
            model.save()
            models[ticker] = model
            logger.info(f"Model trained: {ticker}")

    logger.info(f"Trading loop starting | {len(models)} instruments | capital={capital:.2f}")
    system_started(len(models), capital)

    cycle = 0
    daily_trade_count = {}
    _peak_nav = capital
    _daily_report_day = -1   # track last day daily_report was sent

    _trades_seen = 0   # tracks how many broker trades we've already attributed

    # Swing circuit breaker — relaxed relative to intraday
    from execution.risk_manager import SessionRiskManager as _SRM
    _swing_risk = _SRM(
        initial_nav               = capital,
        daily_loss_pct            = 0.05,   # 5 % daily loss → halt swing
        max_trades_per_instrument = 2,
        max_consec_losses         = 2,
        max_total_daily           = 8,
    )
    logger.info("[swing] SessionRiskManager active | daily_loss=5% | max_per_inst=2 | daily_cap=8")

    try:
        while True:
            cycle += 1
            logger.info(f"\n{'─'*40} Cycle {cycle} {'─'*40}")

            # ── Refresh prices ─────────────────────────────────
            pipeline.refresh_all(tickers=list(models.keys()))

            # ── Swing circuit breaker ─────────────────────────────────────
            _swing_nav = broker.get_account()["nav"]
            _swing_risk.set_session_start(_swing_nav)
            if _swing_risk.halted:
                logger.warning(
                    f"[swing] Session halted: {_swing_risk.halt_reason} "
                    f"— skipping all signals this cycle"
                )
                time.sleep(poll_seconds)
                continue

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
                    # Attribute P&L to driving strategy before closing
                    _strat = pos.get("driving_strategy", "regime_adaptive")
                    _entry = pos.get("entry", 0)
                    _price = broker.get_last_price(tkr) or _entry
                    _pnl   = pos.get("direction", 1) * (_price - _entry) * pos.get("units", 0)
                    portfolio.record_strategy_pnl(_strat, _pnl)

                    broker.close_position(tkr, reason="max_hold_time")
                    risk.close_trade(tkr)
                    logger.info(f"{tkr}: closed — max hold time reached")

            # ── Session drawdown circuit breaker ─────────────────
            _session_halted = risk.check_session_drawdown()
            if _session_halted:
                logger.warning("Session DD limit hit — skipping all signals this cycle")

            for ticker, model in models.items():
                if _session_halted:
                    break
                df_raw = pipeline.get(ticker)
                if df_raw.empty:
                    continue

                # Detect stale data
                if not df_raw.empty:
                    try:
                        last_bar = df_raw.index[-1]
                        now = pd.Timestamp.now(tz='UTC')
                        last_bar_utc = last_bar.tz_convert('UTC') if last_bar.tzinfo else last_bar.tz_localize('UTC')
                        age_hours = (now - last_bar_utc).total_seconds() / 3600
                        is_weekend = now.weekday() >= 5

                        if age_hours > 3 and not is_weekend:
                            logger.warning(f"{ticker}: stale data ({age_hours:.1f}h old) — skip")
                            continue

                        if len(df_raw) > 1 and df_raw['close'].duplicated(keep=False).all():
                            logger.warning(f"{ticker}: duplicate bars detected — skip")
                            continue
                    except Exception:
                        pass  # Don't block on guard failure

                # Update broker paper price
                latest_price = pipeline.get_latest_price(ticker)
                if isinstance(broker, PaperBroker) and latest_price > 0:
                    broker.update_price(ticker, latest_price)

                # ── Fear & Greed sentiment ────────────────────────────────
                from data.alternative import get_fear_greed
                from config.settings import ASSET_CLASS_MAP
                _ac  = ASSET_CLASS_MAP.get(ticker, "equity")
                _fg  = get_fear_greed("crypto" if _ac == "crypto" else "market")

                # Compute HTF timeframes for MTF feature engineering
                try:
                    from data.unified_pipeline import _resample_to_htf
                    _df_4h = _resample_to_htf(df_raw, "4h")
                    _df_1d = _resample_to_htf(df_raw, "1d")
                except Exception:
                    _df_4h = _df_1d = None

                # Build features + get ML signal
                # swing=True generates SWING_FEATURE_COLS (58 features) for
                # the new MTF-trained swing models.
                df_feat = build_features(
                    df_raw,
                    add_labels=False,
                    drop_na=True,
                    swing=True,
                    fg_norm=_fg["fg_norm"],
                    fg_contrarian=_fg["fg_contrarian"],
                    ticker=ticker,
                    df_4h=_df_4h,
                    df_1d=_df_1d,
                )
                if df_feat.empty:
                    continue

                sig_dict = model.signal_for_latest_bar(df_feat)
                ml_signal = sig_dict["signal"]
                ml_conf   = sig_dict["confidence"]

                # Log current HMM regime state
                _rt       = portfolio.regime_tracker
                _regime   = _rt.detect(df_raw, ticker)
                _conf_r   = _rt.get_regime_confidence(ticker)
                _vol_fct  = _rt.get_vol_forecast(ticker)
                _in_trans = _rt.is_in_transition(ticker)
                logger.info(
                    f"{ticker}: ML signal={ml_signal:+d} | "
                    f"conf={ml_conf:.2%} | "
                    f"tradeable={sig_dict['tradeable']} | "
                    f"regime={_regime} ({_conf_r:.0%}) | "
                    f"vol_fcast={_vol_fct:.4f} | "
                    f"transition={_in_trans}"
                )

                # ── News blackout filter ──────────────────────────────────
                from data.calendar import is_news_blackout
                in_blackout, blackout_reason = is_news_blackout(ticker)
                if in_blackout:
                    logger.info(f"{ticker}: {blackout_reason} — skipping")
                    continue

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
                import json
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

                # Swing circuit breaker — per-instrument gate
                _sw_ok, _sw_reason = _swing_risk.can_trade(ticker, _swing_nav)
                if not _sw_ok:
                    logger.info(f"[swing] {ticker}: session risk blocked — {_sw_reason}")
                    continue

                # Execution decision
                execute, reason, sizing = portfolio.should_execute(agg, df_feat)

                if execute:
                    direction = agg["signal"]
                    units     = sizing["units"] * direction
                    sl        = sizing.get("sl")
                    tp        = sizing.get("tp")

                    # Determine asset class for this ticker
                    from data.unified_pipeline import INSTRUMENT_REGISTRY as _IR
                    _ac = _IR.get(ticker, {}).get("asset_class", "forex")

                    resp = broker.market_order(
                        ticker, units, sl, tp,
                        system      = 'swing',
                        asset_class = _ac,
                        timeframe   = '1h',
                        confidence  = ml_conf,
                        regime      = str(agg.get("regime", "unknown")),
                        atr         = float(sig_dict.get("atr", 0)),
                        risk_pct    = sizing.get("risk_pct"),
                        risk_amount = sizing.get("risk_amount"),
                    )
                    if resp.get("status") == "filled":
                        _swing_risk.record_trade_open(ticker)
                    logger.info(f"Order sent: {ticker} | {resp}")

                    try:
                        from utils.alerts import alert_trade_opened
                        alert_trade_opened(
                            ticker     = ticker,
                            direction  = direction,
                            units      = sizing["units"],
                            entry      = sizing["entry"],
                            sl         = sl or 0.0,
                            tp         = tp or 0.0,
                            confidence = ml_conf,
                            strategy   = agg.get("regime", ""),
                        )
                    except Exception:
                        pass

                    # Increment daily trade counter
                    daily_trade_count[day_key] = daily_trade_count.get(day_key, 0) + 1

                    # Store metadata for feedback logging
                    if ticker in broker.positions:
                        broker.positions[ticker]["confidence"] = ml_conf
                        broker.positions[ticker]["regime"]     = agg.get("regime", "unknown")
                        broker.positions[ticker]["htf_trend"]  = htf_trend
                        broker.positions[ticker]["atr"]        = sig_dict.get("atr", 0)
                        # Store the dominant strategy so P&L can be attributed
                        # back to it when the position closes.
                        votes = agg.get("strategy_votes", {})
                        broker.positions[ticker]["driving_strategy"] = max(
                            (s for s, v in votes.items() if v == agg["signal"]),
                            default="regime_adaptive",
                        )

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

            # ── Attribute P&L for SL/TP-closed trades this cycle ──────────
            # broker.trades grows when _close_position() fires (SL/TP hits
            # inside update_price()).  _trades_seen tracks the watermark so
            # we only process genuinely new entries — no double-counting.
            if isinstance(broker, PaperBroker):
                _new_trades = broker.trades[_trades_seen:]
                for _t in _new_trades:
                    _pnl = _t.get("pnl", 0.0)
                    portfolio.record_strategy_pnl("regime_adaptive", _pnl)
                    _swing_risk.record_trade_close(_t["instrument"], _pnl)
                _trades_seen = len(broker.trades)

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
                regime_tracker=portfolio.regime_tracker,
            )

            # Log trades to CSV after each cycle
            if isinstance(broker, PaperBroker) and not broker.trade_log.empty:
                broker.trade_log.to_csv("logs/paper_trades.csv", index=False)

            # ── Daily HTML report (every 24 cycles) ────────────────
            if cycle % 24 == 0:
                try:
                    from utils.report import generate_daily_report
                    generate_daily_report(
                        risk_engine   = risk,
                        trade_history = broker.trade_log if isinstance(broker, PaperBroker) else pd.DataFrame(),
                        open_positions = broker.get_open_positions(),
                        instruments   = list(models.keys()),
                    )
                except Exception as _re:
                    logger.debug(f"Daily report skipped: {_re}")

                # ── Daily summary Telegram alert ─────────────────────
                try:
                    from utils.alerts import alert_daily_summary
                    _rpt = risk.report()
                    _th  = broker.trade_log if isinstance(broker, PaperBroker) else pd.DataFrame()
                    _wr  = 0.0
                    if not _th.empty and "pnl" in _th.columns:
                        _wr = (_th["pnl"] > 0).mean()
                    _dd_str = _rpt.get("drawdown", "0%").replace("%", "")
                    _dd_val = float(_dd_str) / 100 if _dd_str else 0.0
                    alert_daily_summary(
                        nav           = _rpt["nav"],
                        daily_pnl     = _rpt["daily_pnl"],
                        daily_pnl_pct = _rpt["daily_pnl"] / max(_rpt["nav"], 1),
                        n_trades      = _rpt["total_trades"],
                        win_rate      = _wr,
                        max_dd        = _dd_val,
                    )
                except Exception:
                    pass

                # ── Telegram daily report ─────────────────────────
                try:
                    _rpt  = risk.report()
                    _nav  = _rpt.get("nav", capital)
                    _th   = broker.trade_log if isinstance(broker, PaperBroker) else pd.DataFrame()
                    _wr   = float((_th["pnl"] > 0).mean()) if not _th.empty and "pnl" in _th.columns else 0.0
                    _dpnl = _rpt.get("daily_pnl", 0.0)
                    _tpnl = _nav - capital
                    _nopen = len(broker.get_open_positions())
                    _today = datetime.now().day
                    if _today != _daily_report_day:
                        try:
                            from data.trade_db import get_system_stats as _gss
                            _sys_stats = _gss(days=1)
                        except Exception:
                            _sys_stats = None
                        daily_report(
                            nav          = _nav,
                            daily_pnl    = _dpnl,
                            total_trades = len(_th) if not _th.empty else 0,
                            win_rate     = _wr,
                            total_pnl    = _tpnl,
                            open_pos     = _nopen,
                            system_stats = _sys_stats,
                        )
                        _daily_report_day = _today
                except Exception:
                    pass

            # ── Drawdown alert (every cycle) ──────────────────
            try:
                _cur_nav = broker.get_account().get("nav", capital)
                _peak_nav = max(_peak_nav, _cur_nav)
                _dd = (_peak_nav - _cur_nav) / _peak_nav if _peak_nav > 0 else 0.0
                if _dd > 0.03:
                    drawdown_alert(_dd)
            except Exception:
                pass

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
    except Exception as _exc:
        try:
            system_error(str(_exc))
        except Exception:
            pass
        raise


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════
#  INTRADAY 5m PAPER TRADING
# ══════════════════════════════════════════════════════════════

def _get_live_5m_bars(ticker: str, n: int = 500) -> pd.DataFrame:
    """Fetch fresh 5m bars from yfinance for live intraday inference."""
    import yfinance as yf
    try:
        df = yf.download(ticker, period="5d", interval="5m",
                         progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        return df.tail(n)
    except Exception as e:
        logger.warning(f"Live 5m fetch {ticker}: {e}")
        return pd.DataFrame()

# Per-pair confidence thresholds — updated as models are trained.
INTRADAY_MIN_CONF: dict = {
    "USDJPY=X": 0.52,
    "GBPUSD=X": 0.54,
    "EURUSD=X": 0.54,
    "AUDUSD=X": 0.54,
    "USDCAD=X": 0.54,
    "NZDUSD=X": 0.54,
    "USDCHF=X": 0.54,
    "EURGBP=X": 0.54,
    "EURJPY=X": 0.53,
    "GBPJPY=X": 0.53,
}

# All 10 intraday forex pairs
INTRADAY_PAIRS: list = [
    # NZDUSD=X and USDCHF=X removed — no valid confidence threshold
    # (all WF Sharpe < 0 or trades < 30 in sweep)
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X",
    "EURJPY=X", "GBPJPY=X",
]


def run_intraday(capital: float = INITIAL_CAPITAL, poll_seconds: int = 300):
    """Intraday 5m paper trading loop with HTF context features.

    Loads intraday_{ticker}_ensemble.pkl models (trained with 5m+1h+4h features).
    Falls back to old intraday_{ticker_no_X} naming if new model not found.
    Writes trade log to logs/intraday_trades.csv.
    """
    from data.pipeline import get_intraday_5m
    from signals.features_intraday import build_features_intraday, INTRADAY_FEATURE_COLS
    from backtest.engine_intraday import _FOREX_SPREAD_PIPS, _DEFAULT_SPREAD_PIPS
    from config.settings import MAX_RISK_PCT, SLIPPAGE_BPS, COMMISSION_BPS
    from data.unified_pipeline import UnifiedDataPipeline, _resample_to_htf
    from execution.broker import PaperBroker
    from execution.risk_manager import SessionRiskManager

    if not INTRADAY_PAIRS:
        logger.error("INTRADAY_PAIRS is empty.")
        return

    udp    = UnifiedDataPipeline()
    broker = PaperBroker(initial_capital=capital)

    # Load models — try intraday_{ticker} first, then old naming convention
    models = {}
    for ticker in INTRADAY_PAIRS:
        old_name = f"intraday_{ticker.replace('=X','')}"
        for name in (f"intraday_{ticker}", old_name):
            try:
                m = StackedEnsemble.load(name)
                models[ticker] = m
                logger.info(f"[intraday] Loaded model: {name}")
                break
            except (FileNotFoundError, Exception):
                continue
        if ticker not in models:
            logger.warning(f"[intraday] No model for {ticker} — skipping")

    if not models:
        logger.error("[intraday] No models loaded — aborting.")
        return

    logger.info(
        f"[intraday] Starting 5m paper loop | "
        f"pairs={list(models)} | capital={capital:.2f} | poll={poll_seconds}s"
    )

    trade_log_path = "logs/intraday_trades.csv"
    os.makedirs("logs", exist_ok=True)
    if not os.path.exists(trade_log_path):
        pd.DataFrame(columns=[
            "timestamp","ticker","direction","entry","exit",
            "pnl","confidence","nav","cycle",
        ]).to_csv(trade_log_path, index=False)

    cycle = 0
    NAV_FLOOR = capital * 0.50   # hard stop at 50 % of starting capital

    # Macro event gate — halves position size on high-impact days
    try:
        from data.macro_calendar import MacroCalendar
        _macro_cal = MacroCalendar()
    except Exception:
        _macro_cal = None

    session_risk = SessionRiskManager(
        initial_nav               = capital,
        daily_loss_pct            = 0.03,
        max_trades_per_instrument = 3,
        max_consec_losses         = 2,
        max_total_daily           = 10,
    )
    logger.info(
        f"[intraday] SessionRiskManager active | "
        f"daily_loss=3% | max_per_inst=3 | consec_loss=2 | daily_cap=10 | "
        f"nav_floor={NAV_FLOOR:.2f}"
    )

    try:
        while True:
            cycle += 1
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            nav = broker.get_account()["nav"]

            # ── NAV floor — hard stop ─────────────────────────────────────
            if nav <= NAV_FLOOR:
                logger.critical(
                    f"🚨 NAV FLOOR HIT: {nav:.2f} ≤ {NAV_FLOOR:.2f} "
                    f"— shutting down intraday system"
                )
                break

            # ── Session risk gate ─────────────────────────────────────────
            session_risk._check_date_reset()   # clears stale halt on new UTC day
            session_risk.set_session_start(nav)
            if session_risk.halted:
                _now = datetime.now(timezone.utc)
                _midnight = (_now + timedelta(days=1)).replace(
                    hour=0, minute=5, second=0, microsecond=0
                )
                _sleep = max((_midnight - _now).total_seconds(), poll_seconds)
                logger.warning(
                    f"[intraday] Session halted: {session_risk.halt_reason} "
                    f"— sleeping {_sleep/3600:.1f}h to midnight UTC"
                )
                time.sleep(_sleep)
                continue

            logger.info(
                f"[intraday] Cycle {cycle} | {ts} | "
                f"NAV={nav:.2f} | Open={len(broker.positions)} | "
                f"risk={session_risk.status()}"
            )

            # Track closed trades so we can call record_trade_close() below
            _prev_trade_count = len(broker.trades)

            # ── Check SL/TP and time expiry on open positions ────────────
            for ticker in list(models.keys()):
                try:
                    df_chk = _get_live_5m_bars(ticker, n=5)
                    if df_chk.empty:
                        continue
                    bar       = df_chk.iloc[-1]
                    bar_high  = float(bar.get("high",  bar["close"]))
                    bar_low   = float(bar.get("low",   bar["close"]))
                    bar_close = float(bar["close"])
                    if ticker in broker.positions:
                        pos = broker.positions[ticker]
                        # Feed intrabar extremes so SL/TP triggers on bar high/low,
                        # not just the closing price (prevents missed intrabar touches)
                        if pos["direction"] == 1:          # long: SL=low, TP=high
                            broker.update_price(ticker, bar_low)
                            if ticker in broker.positions:
                                broker.update_price(ticker, bar_high)
                        else:                              # short: SL=high, TP=low
                            broker.update_price(ticker, bar_high)
                            if ticker in broker.positions:
                                broker.update_price(ticker, bar_low)
                        if ticker in broker.positions:
                            broker.update_price(ticker, bar_close)   # unrealised P&L
                    else:
                        broker.update_price(ticker, bar_close)
                except Exception:
                    pass
            # Time expiry — close any intraday position held > 20 bars (100 min)
            for tkr, pos in list(broker.positions.items()):
                age_min = (datetime.now(timezone.utc) - pos["time"]).total_seconds() / 60
                if age_min >= 100:
                    price = broker._prices.get(tkr, pos["entry"])
                    broker._close_position(tkr, price, reason="timeout")
                    logger.info(f"[intraday] {tkr}: timeout close after {age_min:.0f}m")

            # Notify session risk of any new closes (SL/TP/timeout)
            for _t in broker.trades[_prev_trade_count:]:
                session_risk.record_trade_close(_t["instrument"], _t["pnl"])

            for ticker, model in models.items():
                try:
                    # Fetch fresh live 5m bars; fallback to cached parquet
                    df_raw = _get_live_5m_bars(ticker, n=500)
                    if df_raw.empty:
                        df_raw = get_intraday_5m(ticker, max_bars=500)
                    if df_raw.empty or len(df_raw) < 50:
                        logger.warning(f"[intraday] {ticker}: insufficient bars")
                        continue

                    # Get 1h and 4h HTF context for MTF features
                    try:
                        _df_1h_ctx = udp.get(ticker, "1h", years=1)
                        _df_4h_ctx = _resample_to_htf(_df_1h_ctx, "4h") if not _df_1h_ctx.empty else None
                    except Exception:
                        _df_1h_ctx = _df_4h_ctx = None

                    df_feat = build_features_intraday(
                        df_raw, ticker=ticker, add_labels=False, drop_na=True,
                        df_1h=_df_1h_ctx, df_4h=_df_4h_ctx,
                    )
                    if df_feat.empty:
                        continue

                    # Use model's full feature set (handles HTF columns via _full_feature_cols)
                    _full_cols = getattr(model, "_full_feature_cols", None) or INTRADAY_FEATURE_COLS
                    _avail = [c for c in _full_cols if c in df_feat.columns]
                    X = df_feat.reindex(columns=_full_cols, fill_value=0.0).values
                    sigs, confs, _ = model.predict_with_confidence(X)

                    # Signal for the latest bar
                    sig  = int(sigs[-1])
                    conf = float(confs[-1])
                    min_conf = INTRADAY_MIN_CONF.get(ticker, 0.55)

                    if sig == 0 or conf < min_conf:
                        logger.info(
                            f"[intraday] {ticker}: sig={sig} conf={conf:.3f} < {min_conf} → skip | last={df_feat.index[-1]}"
                        )
                        continue

                    entry = float(df_feat["close"].iloc[-1])
                    atr   = float(df_feat["atr_1h"].iloc[-1]) if "atr_1h" in df_feat.columns \
                            else float(df_feat["atr_5m"].iloc[-1]) * 3

                    sl = entry - sig * atr * 1.0
                    tp = entry + sig * atr * 1.5

                    # Risk-based position sizing: risk 1% of NAV over the SL distance
                    sl_dist = abs(entry - sl)
                    if sl_dist <= 0:
                        logger.warning(f"[intraday] {ticker}: SL distance zero — skip")
                        continue
                    risk_amt = nav * MAX_RISK_PCT
                    units    = risk_amt / sl_dist
                    # Safety rail: cap at 30× NAV leverage (standard retail forex limit)
                    # Do NOT use a tight notional % — it destroys sizing on high-price pairs (JPY)
                    max_units = (nav * 30) / max(entry, 1e-9)
                    if units > max_units:
                        units = max_units
                    pos_val = units * entry
                    logger.info(
                        f"[intraday] {ticker}: sizing: risk_amt={risk_amt:.2f} "
                        f"sl_dist={sl_dist:.5f} units={units:.1f} "
                        f"notional={pos_val:.2f} leverage={pos_val/max(nav,1):.1f}x"
                    )

                    direction = "Long" if sig == 1 else "Short"
                    logger.info(
                        f"[intraday] {ticker}: {direction} | entry={entry:.5f} "
                        f"sl={sl:.5f} tp={tp:.5f} | conf={conf:.3f} | "
                        f"risk=€{risk_amt:.2f} | units={units:.2f} | "
                        f"pos_val={pos_val:.2f} | last={df_feat.index[-1]}"
                    )

                    # Skip if already in a position for this ticker
                    if ticker in broker.positions:
                        logger.info(f"[intraday] {ticker}: already open — skip")
                        continue

                    # Session risk gate
                    _ok, _reason = session_risk.can_trade(ticker, nav)
                    if not _ok:
                        logger.info(f"[intraday] {ticker}: session risk blocked — {_reason}")
                        continue

                    # Macro event gate — halve size on high-impact days
                    if _macro_cal is not None:
                        _macro_mult = _macro_cal.position_size_multiplier()
                        if _macro_mult < 1.0:
                            _is_hi, _ev = _macro_cal.is_high_impact_day()
                            logger.info(
                                f"[intraday] {ticker}: macro event '{_ev}' → "
                                f"size ×{_macro_mult:.1f}"
                            )
                        units = units * _macro_mult

                    # Execute via PaperBroker (triggers SL/TP tracking + Telegram)
                    broker.update_price(ticker, entry)
                    resp = broker.market_order(
                        ticker, int(units) * sig, sl, tp,
                        system      = 'intraday',
                        asset_class = 'forex',
                        timeframe   = '5M',
                        confidence  = conf,
                        regime      = 'intraday_5m',
                        atr         = atr,
                        risk_pct    = MAX_RISK_PCT * 100,
                        risk_amount = risk_amt,
                    )
                    if resp.get("status") == "filled":
                        session_risk.record_trade_open(ticker)
                    logger.info(
                        f"[intraday] {ticker}: order → {resp.get('status')} "
                        f"fill={resp.get('fill_price', entry):.5f} | "
                        f"Open positions: {len(broker.positions)}"
                    )

                    row = {
                        "timestamp": ts, "ticker": ticker, "direction": direction,
                        "entry": round(entry, 5), "exit": None,
                        "pnl": None, "confidence": round(conf, 3), "nav": round(nav, 2),
                        "cycle": cycle,
                    }
                    pd.DataFrame([row]).to_csv(
                        trade_log_path, mode="a", header=False, index=False
                    )

                except Exception as exc:
                    logger.error(f"[intraday] {ticker}: {exc}")

            logger.info(f"[intraday] Cycle {cycle} done. Sleeping {poll_seconds}s …")
            time.sleep(poll_seconds)

    except KeyboardInterrupt:
        logger.info("[intraday] Stopped by user.")


# ══════════════════════════════════════════════════════════════
#  MODE: MULTI-ASSET PAPER TRADING  — helpers
# ══════════════════════════════════════════════════════════════

def _is_market_hours(dt=None) -> bool:
    """Returns True if US equity market is open (NYSE: 14:30–21:00 UTC, weekdays)."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    if dt.weekday() >= 5:
        return False
    market_open  = dt.replace(hour=14, minute=30, second=0, microsecond=0)
    market_close = dt.replace(hour=21, minute=0,  second=0, microsecond=0)
    return market_open <= dt <= market_close


MAX_POSITIONS_PER_CLASS = {"forex": 4, "equity": 3, "crypto": 2, "commodity": 2}
MAX_TOTAL_POSITIONS     = 8


def _get_data_for_instrument(ticker: str, asset_class: str) -> pd.DataFrame:
    """Route data request to the correct source based on asset class.
    Returns at least 500 bars for feature engineering, or empty DataFrame.
    """
    try:
        if asset_class == "forex":
            pipeline = DataPipeline()
            return pipeline.get(ticker)

        elif asset_class == "equity":
            try:
                from data.massive import MassiveClient
                mc = MassiveClient()
                df = mc.fetch_bars(ticker, "hour", 1, years=5)
            except Exception:
                df = pd.DataFrame()
            if df.empty:
                from data.fmp import FMPClient
                df = FMPClient().get_ohlcv(ticker, "1hour", years=2)
            if df.empty:
                import yfinance as yf
                df = yf.download(ticker, period="730d", interval="1h",
                                 progress=False, auto_adjust=True)
                if not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df.columns = [c.lower() for c in df.columns]
            return df

        elif asset_class == "crypto":
            from data.crypto_data import CryptoDataClient
            cc = CryptoDataClient()
            df = cc.get_cached(ticker, "1h")
            if df.empty:
                df = cc.fetch_ohlcv(ticker, "1h", years=2)
            return df

        elif asset_class == "commodity":
            try:
                from data.fmp import FMPClient
                df = FMPClient().get_ohlcv(ticker, "1hour", years=2)
            except Exception:
                df = pd.DataFrame()
            if df.empty:
                import yfinance as yf
                df = yf.download(ticker, period="730d", interval="1h",
                                 progress=False, auto_adjust=True)
                if not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df.columns = [c.lower() for c in df.columns]
            return df

        else:
            logger.warning(f"Unknown asset class: {asset_class} for {ticker}")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"Data fetch failed {ticker} ({asset_class}): {e}")
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════
#  MODE: MULTI-ASSET PAPER TRADING
# ══════════════════════════════════════════════════════════════

def run_multi_asset(capital: float, poll_seconds: int = 3600):
    """Paper-trade all instruments registered in ModelRegistry.

    Differences from run_paper:
      - Instruments come from ModelRegistry (hot-swappable, not CLI arg)
      - US equities skipped outside 14:30–21:00 UTC
      - Asset-class-aware stale data guard (crypto: 2h, others: 3h weekdays)
      - Per-class position limits (forex≤4, equity≤3, crypto≤2, commodity≤2)
      - Separate trade log: logs/multi_asset_trades.csv
    """
    from models.registry import ModelRegistry

    logger.info("=" * 60)
    logger.info("MODE: MULTI-ASSET PAPER TRADING")
    logger.info("=" * 60)

    registry = ModelRegistry()
    n_new    = registry.auto_register_existing()
    logger.info(f"Registry bootstrap: {n_new} new entries")
    summary  = registry.summary()
    logger.info(f"Registry summary: {summary['by_class']} | active={summary['active']}")

    active_info = registry.get_active_instruments()   # dict {ticker: info}
    if not active_info:
        logger.error("No active instruments in registry — aborting")
        return

    # ── Load models ───────────────────────────────────────────
    models: dict = {}   # {ticker: StackedEnsemble}
    for ticker in active_info:
        try:
            models[ticker] = registry.load_model(ticker)
            logger.info(f"Loaded model: {ticker}")
        except Exception as e:
            logger.warning(f"{ticker}: model load failed ({e}) — skipping")

    if not models:
        logger.error("No models loaded — aborting")
        return

    # ── Infrastructure ────────────────────────────────────────
    broker    = PaperBroker(capital)
    risk      = RiskEngine(capital)
    portfolio = PortfolioManager(list(models.keys()), capital)
    dash      = Dashboard()

    trade_log_path = "logs/multi_asset_trades.csv"
    os.makedirs("logs", exist_ok=True)
    if not os.path.exists(trade_log_path):
        pd.DataFrame(columns=[
            "timestamp", "ticker", "asset_class", "direction",
            "entry", "sl", "tp", "confidence", "nav", "cycle",
        ]).to_csv(trade_log_path, index=False)

    cycle     = 0
    _peak_nav = capital

    try:
        while True:
            cycle += 1
            now_utc = datetime.now(timezone.utc)
            ts      = now_utc.strftime("%Y-%m-%d %H:%M")
            logger.info(f"\n{'─'*40} [multi] Cycle {cycle} | {ts} {'─'*40}")

            # ── Session drawdown guard ────────────────────────
            _halted = risk.check_session_drawdown()
            if _halted:
                logger.warning("[multi] Session DD limit — skipping signals")

            # Count open positions per asset class
            open_by_class: dict = {}
            for pos in broker.get_open_positions():
                _pac = active_info.get(pos["instrument"], {}).get("asset_class", "unknown")
                open_by_class[_pac] = open_by_class.get(_pac, 0) + 1
            total_open = sum(open_by_class.values())

            for ticker, model in models.items():
                if _halted:
                    break

                info = active_info.get(ticker, {})
                ac   = info.get("asset_class", "equity")

                # ── Market hours gate (equities only) ─────────
                if ac == "equity" and not _is_market_hours(now_utc):
                    continue

                # ── Per-class position limits ──────────────────
                if open_by_class.get(ac, 0) >= MAX_POSITIONS_PER_CLASS.get(ac, 2):
                    continue
                if total_open >= MAX_TOTAL_POSITIONS:
                    break

                # ── Get data ───────────────────────────────────
                df_raw = _get_data_for_instrument(ticker, ac)
                if df_raw.empty or len(df_raw) < 50:
                    logger.warning(f"[multi] {ticker}: no data")
                    continue

                # ── Stale-data guard (asset-class aware) ───────
                try:
                    last_bar = df_raw.index[-1]
                    now_ts   = pd.Timestamp.now(tz="UTC")
                    lb_utc   = (last_bar.tz_convert("UTC")
                                if getattr(last_bar, "tzinfo", None)
                                else pd.Timestamp(last_bar, tz="UTC"))
                    age_h    = (now_ts - lb_utc).total_seconds() / 3600
                    stale_th = 2.0 if ac == "crypto" else 3.0
                    is_weekend = now_utc.weekday() >= 5
                    if age_h > stale_th and (ac == "crypto" or not is_weekend):
                        logger.warning(f"[multi] {ticker}: stale ({age_h:.1f}h) — skip")
                        continue
                except Exception:
                    pass

                # ── Update paper price ─────────────────────────
                try:
                    latest_price = float(df_raw["close"].iloc[-1])
                    if latest_price > 0:
                        broker.update_price(ticker, latest_price)
                except Exception:
                    pass

                # ── Build features ─────────────────────────────
                try:
                    from data.alternative import get_fear_greed
                    _fg = get_fear_greed("crypto" if ac == "crypto" else "market")
                except Exception:
                    _fg = {"fg_norm": 0.0, "fg_contrarian": 0.0}

                try:
                    df_feat = build_features(
                        df_raw, add_labels=False, drop_na=True,
                        fg_norm=_fg.get("fg_norm", 0.0),
                        fg_contrarian=_fg.get("fg_contrarian", 0.0),
                        ticker=ticker,
                    )
                except Exception as _fe:
                    logger.warning(f"[multi] {ticker}: build_features error: {_fe}")
                    continue
                if df_feat.empty:
                    continue

                # ── ML signal ─────────────────────────────────
                try:
                    sig_dict  = model.signal_for_latest_bar(df_feat)
                except Exception as _se:
                    logger.warning(f"[multi] {ticker}: signal error: {_se}")
                    continue

                ml_signal = sig_dict["signal"]
                ml_conf   = sig_dict["confidence"]
                min_conf  = info.get("min_conf", get_min_confidence(ticker))

                logger.info(
                    f"[multi] {ticker} ({ac}): sig={ml_signal:+d} "
                    f"conf={ml_conf:.2%} thr={min_conf:.2f}"
                )

                if ml_signal == 0 or ml_conf < min_conf:
                    continue
                if not sig_dict.get("tradeable", True):
                    continue

                # ── Size & levels ──────────────────────────────
                entry     = float(df_feat["close"].iloc[-1])
                atr       = float(sig_dict.get("atr", entry * 0.01))
                atr_mult  = ASSET_ATR_MULTIPLIER.get(ac, 2.0)
                risk_pct  = ASSET_RISK_PCT.get(ac, 0.01)
                nav       = broker.get_account()["nav"]
                sl        = entry - ml_signal * atr * atr_mult
                tp        = entry + ml_signal * atr * atr_mult * 1.5
                risk_amt  = nav * risk_pct
                units     = risk_amt / (atr * atr_mult + 1e-9)
                direction = "Long" if ml_signal == 1 else "Short"

                # ── Risk gate ──────────────────────────────────
                ok, reason = risk.can_trade(
                    instrument  = ticker,
                    signal      = ml_signal,
                    confidence  = ml_conf,
                    atr         = atr,
                    atr_mean    = atr,
                    hour_utc    = now_utc.hour,
                )
                if not ok:
                    logger.info(f"[multi] {ticker}: blocked — {reason}")
                    continue

                # ── Execute ────────────────────────────────────
                fill = broker.market_order(
                    ticker, int(units) * ml_signal, sl, tp,
                    system      = 'multi',
                    asset_class = ac,
                    timeframe   = '1h',
                    confidence  = ml_conf,
                    regime      = 'unknown',
                    atr         = atr,
                )
                if fill.get("status") != "filled":
                    logger.warning(f"[multi] {ticker}: order rejected — {fill}")
                    continue

                entry_filled = fill.get("fill_price", entry)
                risk.open_trade({
                    "instrument": ticker,
                    "direction":  ml_signal,
                    "units":      int(units),
                    "entry":      entry_filled,
                    "sl":         sl,
                    "tp":         tp,
                    "confidence": ml_conf,
                    "regime":     "unknown",
                    "atr":        atr,
                })
                open_by_class[ac] = open_by_class.get(ac, 0) + 1
                total_open       += 1

                logger.info(
                    f"[multi] {ticker}: {direction} fill={entry_filled:.5f} "
                    f"sl={sl:.5f} tp={tp:.5f} conf={ml_conf:.3f}"
                )
                pd.DataFrame([{
                    "timestamp":   ts, "ticker": ticker, "asset_class": ac,
                    "direction":   direction, "entry": round(entry_filled, 5),
                    "sl":          round(sl, 5), "tp": round(tp, 5),
                    "confidence":  round(ml_conf, 3), "nav": round(nav, 2),
                    "cycle":       cycle,
                }]).to_csv(trade_log_path, mode="a", header=False, index=False)

            # ── Update NAV & trailing stops ───────────────────
            acc = broker.get_account()
            risk.update_nav(acc["nav"])

            for pos in broker.get_open_positions():
                tkr = pos["instrument"]
                cp  = broker.get_last_price(tkr)
                if cp > 0:
                    new_sl = risk.update_trailing_stop(pos, cp)
                    if new_sl != pos.get("sl"):
                        broker.update_stop_loss(tkr, new_sl)

            # ── Dashboard ─────────────────────────────────────
            try:
                dash.render(
                    account=acc,
                    risk_report=risk.report(),
                    open_positions=broker.get_open_positions(),
                    trade_history=broker.trade_log if isinstance(broker, PaperBroker) else pd.DataFrame(),
                    regime_map={},
                    portfolio_summary={},
                    cycle=cycle,
                    regime_tracker=portfolio.regime_tracker,
                )
            except Exception:
                pass

            # ── Drawdown alert ────────────────────────────────
            try:
                cur_nav   = acc.get("nav", capital)
                _peak_nav = max(_peak_nav, cur_nav)
                dd = (_peak_nav - cur_nav) / _peak_nav if _peak_nav > 0 else 0.0
                if dd > 0.03:
                    from utils.alerts import drawdown_alert
                    drawdown_alert(dd)
            except Exception:
                pass

            logger.info(f"[multi] Cycle {cycle} done. Sleeping {poll_seconds}s …")
            for i in range(poll_seconds):
                time.sleep(1)
                if i % 60 == 0 and i > 0:
                    logger.info(f"[multi] Next cycle in {poll_seconds - i}s …")

    except KeyboardInterrupt:
        print("\n[multi] Stopped by user.")
        acc = broker.get_account()
        print(f"Final NAV: ${acc['nav']:,.2f}")


run_swing = run_paper   # alias: --mode paper and --mode swing both call run_paper


def parse_args():
    p = argparse.ArgumentParser(description="TradingFirm OS")
    p.add_argument("--mode", choices=["paper","swing","train","backtest","walkforward","intraday","multi"],
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

    elif args.mode in ("paper", "swing", "live"):
        run_paper(args.instruments, args.capital, args.poll)

    elif args.mode == "intraday":
        run_intraday(args.capital, args.poll)

    elif args.mode == "multi":
        run_multi_asset(args.capital, args.poll)
