"""
================================================================
  INTRADAY TRADING SYSTEM — 5-minute bars
  Supports: Forex pairs + Crypto (24/7)
  Data source: yfinance only (train and live same source)
================================================================
"""

import os
import time
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone

from signals.ensemble  import StackedEnsemble
from signals.features  import get_X_y, FEATURE_COLS
from risk.engine       import RiskEngine
from execution.broker  import PaperBroker
from dashboard.cli     import Dashboard
from config.settings   import INITIAL_CAPITAL, MIN_CONFIDENCE

logger = logging.getLogger("trading_firm.intraday")

# ================================================================
# INSTRUMENT DEFINITIONS
# ================================================================

FOREX_PAIRS = {
    #"USDCHF": "USDCHF=X",
    #"AUDUSD": "AUDUSD=X",
    #"EURUSD": "EURUSD=X",
    #"GBPUSD": "GBPUSD=X",
}

CRYPTO_PAIRS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    #"BNB": "BNB-USD",
}

ALL_PAIRS    = {**FOREX_PAIRS, **CRYPTO_PAIRS}
PAIRS        = list(ALL_PAIRS.keys())
MODEL_SUFFIX = "5m"

# ================================================================
# SESSION THRESHOLD
# ================================================================

def get_session_threshold(hour_utc: int, is_crypto: bool = False) -> float:
    """
    Return minimum confidence threshold based on session and asset type.
    Crypto uses flat threshold — no session concept.
    """
    if is_crypto:
        return 0.55
    if 13 <= hour_utc < 17:   # London/NY overlap — most liquid
        return 0.52
    elif 7 <= hour_utc < 13:  # London session
        return 0.55
    elif 17 <= hour_utc < 21: # NY afternoon
        return 0.55
    else:                      # Asian/off-hours — very strict
        return 0.70


# ================================================================
# SHARED DATA FETCH
# ================================================================

def fetch_intraday_data(
    pair:      str,
    yf_ticker: str,
    is_crypto: bool = False,
    period:    str  = "60d",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch 5-min bars from yfinance or local cache.
    Cache is updated daily and grows beyond 60-day yfinance limit.
    """
    from data.pipeline import update_intraday_cache

    # Try cache first during paper trading
    if use_cache:
        cache_path = f"data/cache/intraday/{pair}_5min.parquet"
        if os.path.exists(cache_path):
            try:
                df_cached = update_intraday_cache(pair, yf_ticker, is_crypto)
                if not df_cached.empty and len(df_cached) > 500:
                    logger.info(f"{pair}: using cache — {len(df_cached)} bars")
                    return df_cached
            except Exception as e:
                logger.warning(f"{pair}: cache read failed — {e}, falling back to yfinance")

    # Fall back to direct yfinance fetch
    for attempt in range(3):
        try:
            df = yf.download(
                yf_ticker, interval="5m", period=period,
                progress=False, multi_level_index=False
            )
            if not df.empty:
                break
        except Exception as e:
            logger.warning(f"{pair}: fetch attempt {attempt+1} failed — {e}")
            time.sleep(2 ** attempt)
    else:
        logger.error(f"{pair}: all fetch attempts failed")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    df.columns = [c.lower() for c in df.columns]
    df.index   = df.index.tz_localize(None)

    if not is_crypto:
        df = df[
            (df.index.weekday < 5) &
            (df.index.hour >= 7)   &
            (df.index.hour < 21)
        ]

    logger.info(f"{pair}: {len(df)} bars | {df.index[0].date()} → {df.index[-1].date()}")
    return df



# ================================================================
# TRAIN MODE
# ================================================================

def run_train():
    """Train ML models for all pairs using yfinance 5-min data."""
    logger.info("=" * 60)
    logger.info("INTRADAY — MODE: TRAIN (yfinance 5-min)")
    logger.info("=" * 60)

    for pair, yf_ticker in ALL_PAIRS.items():
        is_crypto = pair in CRYPTO_PAIRS
        logger.info(f"\n{'─'*50}")
        logger.info(f"Training {pair} ({'crypto 24/7' if is_crypto else 'forex session'})")

        df_raw = fetch_intraday_data(pair, yf_ticker, is_crypto, use_cache=False)

        if df_raw.empty or len(df_raw) < 500:
            logger.warning(f"{pair}: insufficient data — skipping")
            continue

        label_params = {
            "forex":     {"sl_mult": 1.5, "tp_mult": 2.0, "forward_bars": 12},
            "crypto":    {"sl_mult": 3.0, "tp_mult": 4.0, "forward_bars": 24},
        }
        params = label_params["crypto"] if is_crypto else label_params["forex"]
        X, y, df_feat = get_X_y(df_raw, **params)

        if len(X) < 500:
            logger.warning(f"{pair}: only {len(X)} samples after features — skipping")
            continue

        unique, counts = np.unique(y, return_counts=True)
        logger.info(
            f"{pair}: {len(X)} samples | "
            f"classes: {dict(zip(unique.tolist(), counts.tolist()))}"
        )

        model = StackedEnsemble(instrument=f"{pair}_{MODEL_SUFFIX}")
        model.train(X, y)
        model.save()
        logger.info(f"{pair}: ✓ saved → {pair}_{MODEL_SUFFIX}")

    print(f"\n✓ Intraday models saved to models/saved/ with _{MODEL_SUFFIX} suffix")

# ================================================================
# WALK-FORWARD BACKTEST
# ================================================================

def walk_forward_backtest(
    df_raw:    pd.DataFrame,
    pair:      str,
    is_crypto: bool,
    n_windows: int = 4,
) -> dict:
    """
    Rolling walk-forward backtest.
    Trains on expanding window, tests on next period.
    Gives realistic estimate of live performance.
    """
    from backtest.engine import run_backtest_single

    window = len(df_raw) // (n_windows + 1)

    label_params = {
        "crypto": {"sl_mult": 3.0, "tp_mult": 4.0, "forward_bars": 24},
        "forex":  {"sl_mult": 1.5, "tp_mult": 2.0, "forward_bars": 12},
    }
    params = label_params["crypto"] if is_crypto else label_params["forex"]

    pfs, wrs, dds = [], [], []
    all_trades = 0

    for i in range(n_windows):
        train_end  = (i + 1) * window
        test_start = train_end
        test_end   = test_start + window

        df_train = df_raw.iloc[:train_end]
        df_test  = df_raw.iloc[test_start:test_end]

        X_tr, y_tr, _ = get_X_y(df_train, **params)
        if len(X_tr) < 500:
            logger.warning(f"{pair} WF window {i+1}: insufficient samples — skipping")
            continue

        wf_model = StackedEnsemble(instrument=f"{pair}_wf_{i}")
        wf_model.train(X_tr, y_tr)

        result = run_backtest_single(df_test, wf_model, INITIAL_CAPITAL)
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
        "wr_mean":      np.mean([w if isinstance(w, float) else 0 for w in wrs]),
        "dd_worst":     round(np.min([d for d in dds if isinstance(d, (int, float))]) if any(isinstance(d, (int, float)) for d in dds) else 0, 3),
        "total_trades": all_trades,
    }



# ================================================================
# BACKTEST MODE
# ================================================================

def run_backtest():
    """Backtest intraday models on yfinance 5-min data."""
    from backtest.engine import run_backtest_single

    logger.info("=" * 60)
    logger.info("INTRADAY — MODE: BACKTEST (yfinance 5-min)")
    logger.info("=" * 60)

    all_results = {}

    for pair, yf_ticker in ALL_PAIRS.items():
        is_crypto = pair in CRYPTO_PAIRS

        df_raw = fetch_intraday_data(pair, yf_ticker, is_crypto, use_cache=False)

        if df_raw.empty or len(df_raw) < 500:
            logger.warning(f"{pair}: insufficient data — skipping")
            continue

        # 70/30 split
        split    = int(len(df_raw) * 0.70)
        df_train = df_raw.iloc[:split]
        df_test  = df_raw.iloc[split:]

        X_tr, y_tr, _ = get_X_y(df_train)

        if len(X_tr) < 500:
            logger.warning(f"{pair}: insufficient training samples — skipping")
            continue

        try:
            model = StackedEnsemble.load(f"{pair}_{MODEL_SUFFIX}")
            logger.info(f"{pair}: loaded saved model")
        except FileNotFoundError:
            logger.info(f"{pair}: no saved model — training now")
            model = StackedEnsemble(instrument=f"{pair}_{MODEL_SUFFIX}")
            model.train(X_tr, y_tr)

        result = run_backtest_single(df_test, model, INITIAL_CAPITAL)
        all_results[pair] = result

        print(f"\n{'─'*50}")
        print(f"  {pair} ({'crypto' if is_crypto else 'forex'}) Backtest")
        print(f"{'─'*50}")
        for k, v in result["metrics"].items():
            print(f"  {k:<22} {v}")
        
        print(f"\n  Walk-Forward Validation ({pair}):")
        wf = walk_forward_backtest(df_raw, pair, is_crypto)
        if wf:
            print(f"  PF: {wf['pf_mean']:.3f} ± {wf['pf_std']:.3f} | "
                f"Min PF: {wf['pf_min']:.3f} | "
                f"Worst DD: {wf['dd_worst']:.1%} | "
                f"Trades: {wf['total_trades']}")

    if all_results:
        pfs = [
            r["metrics"].get("profit_factor", 0)
            for r in all_results.values()
            if isinstance(r["metrics"].get("profit_factor"), (int, float))
        ]
        print(f"\n{'═'*50}")
        print(f"  PAIRS TESTED:      {len(all_results)}")
        print(f"  AVG PROFIT FACTOR: {sum(pfs)/max(len(pfs),1):.3f}")
        print(f"  PAIRS > 1.3 PF:    {sum(1 for p in pfs if p > 1.3)}")
        print(f"{'═'*50}")


_htf_cache = {}   # cache HTF trends to avoid fetching every cycle

def get_htf_trend(pair: str, yf_ticker: str) -> int:
    """
    Returns +1 uptrend, -1 downtrend, 0 unclear.
    Cached per cycle to avoid excessive yfinance calls.
    """
    global _htf_cache
    cache_key = f"{pair}_{datetime.now(timezone.utc).strftime('%Y%m%d%H')}"

    if cache_key in _htf_cache:
        return _htf_cache[cache_key]

    try:
        df = yf.download(yf_ticker, interval="1h", period="7d",
                         progress=False, multi_level_index=False)
        if df.empty:
            return 0
        df.columns = [c.lower() for c in df.columns]
        c     = df["close"]
        ema20 = c.ewm(span=20).mean()
        ema50 = c.ewm(span=50).mean()

        if c.iloc[-1] > ema20.iloc[-1] > ema50.iloc[-1]:
            trend = 1
        elif c.iloc[-1] < ema20.iloc[-1] < ema50.iloc[-1]:
            trend = -1
        else:
            trend = 0

        _htf_cache[cache_key] = trend
        return trend
    except Exception:
        return 0

# ================================================================
# PAPER TRADING MODE
# ================================================================

def run_paper(poll_seconds: int = 300, capital: float = INITIAL_CAPITAL):
    """
    Intraday paper trading loop.
    - Forex: trades 7am-9pm UTC weekdays only
    - Crypto: trades 24/7
    - Both use yfinance 5-min data — same source as training
    """
    dash = Dashboard()
    dash.render_startup()
    time.sleep(1)

    risk   = RiskEngine(initial_capital=capital)
    broker = PaperBroker(initial_capital=capital)

    # ── Load models ───────────────────────────────────────────
    models = {}
    for pair in ALL_PAIRS:
        try:
            models[pair] = StackedEnsemble.load(f"{pair}_{MODEL_SUFFIX}")
            logger.info(f"Loaded model: {pair}")
        except FileNotFoundError:
            logger.warning(f"No model for {pair} — run train first")

    if not models:
        logger.error("No intraday models found. Run: python intraday_forex.py --mode train")
        return

    logger.info(
        f"Intraday paper loop starting | "
        f"{len(models)} pairs | poll={poll_seconds}s"
    )

    cycle = 0
    daily_trade_count = {}  # track trades per day to avoid overtrading in volatile sessions
    try:
        while True:
            cycle += 1
            now = datetime.now(timezone.utc)

            logger.info(f"Cycle {cycle} | {now.strftime('%H:%M:%S UTC')}")

            # ── Weekend handling ──────────────────────────────
            if now.weekday() >= 5:
                # Close all forex positions on weekend
                for pos in broker.get_open_positions():
                    if pos["instrument"] in FOREX_PAIRS:
                        broker.close_position(pos["instrument"], reason="weekend_close")
                        logger.info(f"{pos['instrument']}: closed — weekend")

                # If no crypto models loaded, skip entirely
                if not any(p in CRYPTO_PAIRS for p in models):
                    logger.info(f"Weekend — no crypto models, sleeping {poll_seconds}s")
                    time.sleep(poll_seconds)
                    continue

            # Friday evening — close forex before weekend
            if now.weekday() == 4 and now.hour >= 20:
                for pos in broker.get_open_positions():
                    if pos["instrument"] in FOREX_PAIRS:
                        broker.close_position(pos["instrument"], reason="weekend_close")
                        logger.info(f"{pos['instrument']}: closed before weekend")

            # ── Live price fetch ──────────────────────────────
            live_prices = {}
            for pair, yf_ticker in ALL_PAIRS.items():
                if pair not in models:
                    continue
                for attempt in range(3):
                    try:
                        tick = yf.download(
                            yf_ticker, interval="1m", period="1d",
                            progress=False, multi_level_index=False
                        )
                        if not tick.empty:
                            tick.columns = [c.lower() for c in tick.columns]
                            live_prices[pair] = float(tick["close"].iloc[-1])
                            broker.update_price(pair, live_prices[pair])
                            break
                    except Exception as e:
                        logger.warning(f"{pair}: price fetch attempt {attempt+1} failed — {e}")
                        time.sleep(2 ** attempt)

            # ── Trailing stops ────────────────────────────────
            for pos in broker.get_open_positions():
                tkr           = pos["instrument"]
                current_price = live_prices.get(tkr, 0)
                if current_price <= 0:
                    continue
                direction  = pos.get("direction", 1)
                current_sl = pos.get("sl", 0)
                atr        = pos.get("atr", current_price * 0.0005)
                if direction == 1:
                    new_sl = max(current_price - atr, current_sl)
                else:
                    new_sl = min(current_price + atr, current_sl)
                if new_sl != current_sl:
                    broker.update_stop_loss(tkr, new_sl)
                    logger.info(f"{tkr}: trailing stop → {new_sl:.5f}")

            # ── Partial profit taking ─────────────────────────
            for pos in broker.get_open_positions():
                tkr           = pos["instrument"]
                current_price = live_prices.get(tkr, 0)
                if current_price <= 0 or pos.get("partial_closed"):
                    continue
                entry     = pos["entry"]
                direction = pos.get("direction", 1)
                atr       = pos.get("atr", entry * 0.0005)
                move      = (current_price - entry) * direction
                if move >= atr:
                    result = broker.partial_close(tkr, close_fraction=0.5)
                    if result["status"] == "partial_closed":
                        pos["partial_closed"] = True
                        logger.info(f"{tkr}: partial TP | +{result['pnl']:.2f}")

            # ── Max hold time — 12 bars (1 hour) ─────────────
            for pos in broker.get_open_positions():
                tkr       = pos["instrument"]
                bars_open = broker.get_bars_open(tkr, cycle)
                if bars_open >= 12:
                    broker.close_position(tkr, reason="max_hold_intraday")
                    logger.info(f"{tkr}: closed — 1hr max hold reached")

            # ── Signal generation ─────────────────────────────
            for pair, model in models.items():
                is_crypto = pair in CRYPTO_PAIRS

                # Session filter for forex
                if not is_crypto:
                    if now.weekday() >= 5:
                        continue
                    if now.hour < 7 or now.hour >= 21:
                        logger.info(f"{pair}: outside session hours")
                        continue

                # Fetch latest bars for features
                df_raw = fetch_intraday_data(
                    pair, ALL_PAIRS[pair], is_crypto, period="60d"
                )

                if df_raw.empty:
                    logger.warning(f"{pair}: no data — skipping")
                    continue

                # Use data price as fallback if live fetch failed
                if pair not in live_prices:
                    fallback = float(df_raw["close"].iloc[-1])
                    live_prices[pair] = fallback
                    broker.update_price(pair, fallback)
                    logger.info(f"{pair}: using data fallback price {fallback:.5f}")

                # Append latest live bar
                bar_time = now.replace(second=0, microsecond=0, tzinfo=None)
                last_bar = pd.DataFrame([{
                    "open":   live_prices[pair],
                    "high":   live_prices[pair],
                    "low":    live_prices[pair],
                    "close":  live_prices[pair],
                    "volume": 0,
                }], index=pd.DatetimeIndex([bar_time]))

                df_live = pd.concat([df_raw, last_bar])
                df_live = df_live[~df_live.index.duplicated(keep="last")]

                # Generate signal
                sig_dict = model.signal_for_latest_bar(df_live)
                ml_sig   = sig_dict["signal"]
                ml_conf  = sig_dict["confidence"]

                logger.info(
                    f"{pair}: signal={ml_sig:+d} | "
                    f"conf={ml_conf:.2%} | "
                    f"price={live_prices.get(pair, '?')}"
                )

                # Confidence threshold
                threshold = get_session_threshold(now.hour, is_crypto)
                if ml_sig == 0 or ml_conf < threshold:
                    logger.info(
                        f"{pair}: below threshold "
                        f"({ml_conf:.2%} < {threshold:.2%})"
                    )
                    continue

                if pair not in live_prices:
                    logger.warning(f"{pair}: no live price — skipping")
                    continue
                
                # ── HTF trend filter ──────────────────────────────────────
                htf_trend = get_htf_trend(pair, ALL_PAIRS[pair])
                if htf_trend != 0 and htf_trend != ml_sig:
                    logger.info(f"{pair}: signal against HTF trend ({htf_trend:+d}) — skip")
                    continue

                # ── Max 2 trades per instrument per day ──────────
                today     = now.strftime("%Y-%m-%d")
                day_key   = f"{pair}_{today}"
                if daily_trade_count.get(day_key, 0) >= 2:
                    logger.info(f"{pair}: max 2 trades/day reached — skip")
                    continue

                # Risk checks
                atr      = sig_dict.get("atr", 0.001)
                atr_norm = sig_dict.get("atr_norm", 1.0)
                atr_mean = atr / (atr_norm + 1e-9)
                entry    = live_prices[pair]

                allowed, reason = risk.can_trade(
                    instrument=pair,
                    signal=ml_sig,
                    confidence=ml_conf,
                    atr=atr,
                    atr_mean=atr_mean,
                    hour_utc=now.hour,
                )

                if not allowed:
                    logger.info(f"{pair}: blocked — {reason}")
                    continue

                # Position sizing
                regime  = "trending_up" if ml_conf > 0.70 else "ranging"
                sizing  = risk.position_size(
                    instrument=pair,
                    entry=entry,
                    atr=atr,
                    confidence=ml_conf,
                    regime=regime,
                )

                sl, tp = risk.sl_tp(entry, ml_sig, atr, pair)
                units  = sizing["units"] * ml_sig

                resp = broker.market_order(pair, units, sl, tp)

                if pair in broker.positions:
                    broker.positions[pair]["confidence"] = ml_conf
                    broker.positions[pair]["regime"]     = "trending_up" if ml_conf > 0.70 else "ranging"
                    broker.positions[pair]["htf_trend"]  = htf_trend
                    broker.positions[pair]["atr"]        = atr

                # Increment daily trade counter
                daily_trade_count[day_key] = daily_trade_count.get(day_key, 0) + 1

                risk.open_trade({
                    "instrument": pair,
                    "direction":  ml_sig,
                    "units":      sizing["units"],
                    "entry":      entry,
                    "sl":         sl,
                    "tp":         tp,
                    "atr":        atr,
                    "open_bar":   cycle,
                    "confidence": ml_conf,
                    "regime":     "trending_up" if ml_conf > 0.70 else "ranging",
                    "htf_trend":  htf_trend,
                })

                logger.info(
                    f"ORDER: {pair} | "
                    f"{'BUY' if ml_sig == 1 else 'SELL'} {sizing['units']} | "
                    f"entry={entry:.5f} sl={sl:.5f} tp={tp:.5f}"
                )

            # ── Dashboard update ──────────────────────────────
            acc       = broker.get_account()
            open_pos  = broker.get_open_positions()
            trade_log = broker.trade_log

            risk.update_nav(acc["nav"])

            if not trade_log.empty:
                os.makedirs("logs", exist_ok=True)
                trade_log.to_csv("logs/intraday_paper_trades.csv", index=False)

            regime_map = {}
            for p in models:
                regime_map[p] = "crypto" if p in CRYPTO_PAIRS else "ranging"

            dash.render(
                account=acc,
                risk_report=risk.report(),
                open_positions=open_pos,
                trade_history=trade_log,
                regime_map=regime_map,
                portfolio_summary={
                    "dominant_regime":    "Intraday",
                    "instrument_regimes": regime_map,
                    "strategy_weights":   {"ml_ensemble": 1.0},
                    "strategy_pnl":       {"ml_ensemble": risk.total_pnl},
                },
                cycle=cycle,
            )

            logger.info(f"Next cycle in {poll_seconds}s...")

            # Sleep with countdown
            for remaining in range(poll_seconds, 0, -1):
                time.sleep(1)
                if remaining % 60 == 0:
                    logger.info(f"Next cycle in {remaining}s...")

    except KeyboardInterrupt:
        print("\n\nIntraday paper trading stopped.")
        acc = broker.get_account()
        print(f"Final NAV:    ${acc['nav']:,.2f}")
        print(f"Total trades: {len(broker.trade_log)}")


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[
            logging.FileHandler("logs/intraday.log"),
            logging.StreamHandler(),
        ],
    )

    p = argparse.ArgumentParser(description="Intraday Trading System")
    p.add_argument("--mode",    choices=["train", "backtest", "paper"], default="paper")
    p.add_argument("--poll",    type=int,   default=300)
    p.add_argument("--capital", type=float, default=INITIAL_CAPITAL)
    args = p.parse_args()

    os.makedirs("logs",         exist_ok=True)
    os.makedirs("models/saved", exist_ok=True)

    if args.mode == "train":
        run_train()
    elif args.mode == "backtest":
        run_backtest()
    elif args.mode == "paper":
        run_paper(poll_seconds=args.poll, capital=args.capital)