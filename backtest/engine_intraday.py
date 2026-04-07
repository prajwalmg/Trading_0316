"""
================================================================
  backtest/engine_intraday.py — Intraday 5m Backtest Engine
================================================================
  Mirror of engine.py but uses build_features_intraday and
  INTRADAY_FEATURE_COLS. Does NOT touch engine.py.
================================================================
"""

import logging
import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.settings import (
    SLIPPAGE_BPS, COMMISSION_BPS, MAX_RISK_PCT, INITIAL_CAPITAL,
)
from signals.features_intraday import (
    build_features_intraday, INTRADAY_FEATURE_COLS,
)

logger = logging.getLogger("trading_firm.backtest_intraday")

# 5m bars per year (forex ~252 trading days × 24h × 12 bars/h)
BARS_PER_YEAR_5M = 105_120

# Same spread table as swing engine
_FOREX_SPREAD_PIPS: dict = {
    "EURUSD=X": 0.5, "GBPUSD=X": 0.6, "USDJPY=X": 0.6, "USDCHF=X": 0.8,
    "AUDUSD=X": 0.7, "NZDUSD=X": 0.9, "USDCAD=X": 0.8,
    "EURGBP=X": 1.0, "EURJPY=X": 1.0, "GBPJPY=X": 1.5,
}
_DEFAULT_SPREAD_PIPS = 1.5


def _compute_metrics(equity: pd.Series, trades: pd.DataFrame) -> dict:
    ret     = equity.pct_change().dropna()
    tot_ret = equity.iloc[-1] / equity.iloc[0] - 1

    if len(equity) > 1 and hasattr(equity.index, "to_series"):
        days  = max((equity.index[-1] - equity.index[0]).days, 1)
        years = days / 365.25
    else:
        years = max(len(equity) / BARS_PER_YEAR_5M, 1 / 365)

    ann_ret    = max(min((1 + tot_ret) ** (1 / max(years, 0.01)) - 1, 10.0), -1.0)
    actual_bpy = len(equity) / max(years, 0.01)
    sharpe     = (ret.mean() / (ret.std() + 1e-9)) * np.sqrt(min(actual_bpy, BARS_PER_YEAR_5M))

    neg_ret = ret[ret < 0]
    sortino = (ret.mean() / (neg_ret.std() + 1e-9)) * np.sqrt(BARS_PER_YEAR_5M) if len(neg_ret) > 1 else 0
    max_dd  = (equity / equity.cummax() - 1).min()
    calmar  = ann_ret / (abs(max_dd) + 1e-9)

    if not trades.empty and "pnl" in trades.columns:
        w  = trades[trades["pnl"] > 0]["pnl"]
        l  = trades[trades["pnl"] < 0]["pnl"]
        wr = len(w) / len(trades)
        pf = w.sum() / (abs(l.sum()) + 1e-9)
        ex = wr * w.mean() - (1 - wr) * abs(l.mean()) if len(w) and len(l) else 0
    else:
        wr = pf = ex = 0

    return {
        "total_return":  f"{tot_ret:.2%}",
        "annual_return": f"{ann_ret:.2%}",
        "sharpe_ratio":  round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "calmar_ratio":  round(calmar, 3),
        "max_drawdown":  f"{max_dd:.2%}",
        "win_rate":      f"{wr:.2%}",
        "profit_factor": round(pf, 3),
        "expectancy":    round(ex, 4),
        "total_trades":  len(trades),
    }


def run_backtest_intraday(
    df_raw:   pd.DataFrame,
    model,
    capital:  float = INITIAL_CAPITAL,
    min_conf: float = 0.55,
    ticker:   str   = "",
    sl_mult:  float = 1.0,
    tp_mult:  float = 1.0,
    forward_bars: int = 12,
) -> dict:
    """
    Intraday 5m backtest. Builds features via build_features_intraday,
    then runs triple-barrier simulation using the intraday model.
    """
    df_feat = build_features_intraday(
        df_raw, ticker=ticker, add_labels=False, drop_na=True,
        sl_mult=sl_mult, tp_mult=tp_mult, forward_bars=forward_bars,
    )
    if df_feat.empty or len(df_feat) < 50:
        return {"metrics": {}, "equity": pd.Series(), "trades": pd.DataFrame()}

    X = df_feat[INTRADAY_FEATURE_COLS].values
    try:
        sigs, confs, _ = model.predict_with_confidence(X)
    except Exception as e:
        logger.error(f"predict_with_confidence failed: {e}")
        return {"metrics": {}, "equity": pd.Series(), "trades": pd.DataFrame()}

    # Cost model
    cost_rate = (SLIPPAGE_BPS + COMMISSION_BPS) / 10_000
    if ticker.endswith("=X"):
        avg_price   = float(df_feat["close"].mean())
        spread_pips = _FOREX_SPREAD_PIPS.get(ticker, _DEFAULT_SPREAD_PIPS)
        pip_size    = 0.01 if avg_price > 20 else 0.0001
        cost_rate  += (spread_pips * pip_size) / avg_price

    trades = []
    equity = [capital]
    nav    = capital
    i      = 0
    n      = len(df_feat)

    while i < n - 1:
        sig  = int(sigs[i])
        conf = float(confs[i])

        if sig == 0 or conf < min_conf:
            equity.append(nav)
            i += 1
            continue

        entry    = float(df_feat["close"].iloc[i])
        # Use 1h ATR for SL/TP sizing — matches label construction and keeps
        # cost-to-barrier ratio viable (1h ATR ≈ 10× 5m ATR).
        # Fall back to 3× atr_5m if atr_1h not available.
        if "atr_1h" in df_feat.columns:
            atr = float(df_feat["atr_1h"].iloc[i])
        elif "atr_5m" in df_feat.columns:
            atr = float(df_feat["atr_5m"].iloc[i]) * 3
        else:
            atr = entry * 0.0005
        if atr <= 0 or np.isnan(atr):
            equity.append(nav)
            i += 1
            continue

        sl = entry - sig * atr * sl_mult
        tp = entry + sig * atr * tp_mult

        # Confidence scalar (same as swing engine)
        conf_above  = conf - min_conf
        conf_range  = 1.0 - min_conf
        conf_scalar = 0.5 + (conf_above / (conf_range + 1e-9)) * 1.0
        conf_scalar = max(0.5, min(1.5, conf_scalar))

        risk_amt  = nav * MAX_RISK_PCT
        units     = risk_amt / (atr * sl_mult + 1e-9) * conf_scalar
        pos_val   = min(units * entry, nav * 2)   # cap 2× nav for intraday

        exit_price = None
        exit_bar   = i + 1
        horizon    = min(i + forward_bars + 1, n)

        for j in range(i + 1, horizon):
            lo = float(df_feat["low"].iloc[j])
            hi = float(df_feat["high"].iloc[j])
            if sig == 1:
                if lo <= sl: exit_price = sl; break
                if hi >= tp: exit_price = tp; break
            else:
                if hi >= sl: exit_price = sl; break
                if lo <= tp: exit_price = tp; break
            exit_bar = j

        if exit_price is None:
            exit_price = float(df_feat["close"].iloc[min(exit_bar, n - 1)])

        gross_ret = sig * (exit_price - entry) / (entry + 1e-9)
        net_pnl   = pos_val * gross_ret - pos_val * cost_rate
        nav      += net_pnl
        equity.append(nav)

        trades.append({
            "entry_time": df_feat.index[i],
            "exit_time":  df_feat.index[min(exit_bar, n - 1)],
            "direction":  "Long" if sig == 1 else "Short",
            "entry":      round(entry, 5),
            "exit":       round(exit_price, 5),
            "pnl":        round(net_pnl, 4),
            "confidence": round(conf, 3),
            "nav":        round(nav, 2),
        })

        i = exit_bar + 1

    eq_series = pd.Series(
        equity[:n],
        index=df_feat.index[:len(equity)],
    )
    trades_df = pd.DataFrame(trades)
    metrics   = _compute_metrics(eq_series, trades_df)

    logger.info(
        f"[Intraday] {ticker}: {len(trades_df)} trades | "
        f"Sharpe={metrics['sharpe_ratio']} | "
        f"DD={metrics['max_drawdown']} | "
        f"WR={metrics['win_rate']}"
    )
    return {"metrics": metrics, "equity": eq_series, "trades": trades_df}
