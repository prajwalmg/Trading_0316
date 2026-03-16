"""
================================================================
  backtest/engine.py
  Vectorised backtest engine with full cost modelling.
  Computes: Sharpe, Sortino, Calmar, Max DD, Win Rate,
            Profit Factor, Expectancy, per-trade log.
================================================================
"""

import logging
import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.settings import (
    SL_ATR_MULT, TP_ATR_MULT, MIN_CONFIDENCE,
    SLIPPAGE_BPS, COMMISSION_BPS, MAX_RISK_PCT,
    INITIAL_CAPITAL, GRANULARITY,
)
from signals.features import build_features, FEATURE_COLS, SWING_FEATURE_COLS

logger = logging.getLogger("trading_firm.backtest")

BARS_PER_YEAR = {
    "1m": 525_600, "5m": 105_120, "15m": 35_040,
    "30m": 17_520, "60m": 8_760,  "1d": 252,
}


def compute_metrics(equity: pd.Series, trades: pd.DataFrame) -> dict:
    ret      = equity.pct_change().dropna()
    ann      = BARS_PER_YEAR.get(GRANULARITY, 105_120)
    tot_ret  = equity.iloc[-1] / equity.iloc[0] - 1
    #ann_ret  = (1 + tot_ret) ** (ann / max(len(equity), 1)) - 1
    #sharpe   = (ret.mean() / (ret.std() + 1e-9)) * np.sqrt(ann)
    # Use actual calendar days instead of bar count for annualisation
    if len(equity) > 1 and hasattr(equity.index, 'to_series'):
        days = max((equity.index[-1] - equity.index[0]).days, 1)
        years = days / 365.25
    else:
        years = max(len(equity) / ann, 1/365)

    ann_ret = (1 + tot_ret) ** (1 / max(years, 0.01)) - 1
    ann_ret = max(min(ann_ret, 10.0), -1.0)   # Cap at 1000% to prevent explosion

    # Sharpe: annualise by sqrt of actual bars per year in the test window
    actual_bpy = len(equity) / max(years, 0.01)
    sharpe = (ret.mean() / (ret.std() + 1e-9)) * np.sqrt(min(actual_bpy, ann))

    neg_ret  = ret[ret < 0]
    sortino  = (ret.mean() / (neg_ret.std() + 1e-9)) * np.sqrt(ann) if len(neg_ret) > 1 else 0
    max_dd   = (equity / equity.cummax() - 1).min()
    calmar   = ann_ret / (abs(max_dd) + 1e-9)

    if not trades.empty and "pnl" in trades.columns:
        w  = trades[trades["pnl"] > 0]["pnl"]
        l  = trades[trades["pnl"] < 0]["pnl"]
        wr = len(w) / len(trades)
        pf = w.sum() / (abs(l.sum()) + 1e-9)
        ex = wr * w.mean() - (1 - wr) * abs(l.mean()) if len(w) and len(l) else 0
    else:
        wr = pf = ex = 0

    return {
        "total_return":   f"{tot_ret:.2%}",
        "annual_return":  f"{ann_ret:.2%}",
        "sharpe_ratio":   round(sharpe, 3),
        "sortino_ratio":  round(sortino, 3),
        "calmar_ratio":   round(calmar, 3),
        "max_drawdown":   f"{max_dd:.2%}",
        "win_rate":       f"{wr:.2%}",
        "profit_factor":  round(pf, 3),
        "expectancy":     round(ex, 4),
        "total_trades":   len(trades),
    }


def run_backtest_single(
    df_raw:    pd.DataFrame,
    model,
    capital:   float = INITIAL_CAPITAL,
    min_conf:  float = MIN_CONFIDENCE,
    swing:     bool  = False,
) -> dict:
    df_feat = build_features(df_raw, add_labels=False, drop_na=True, swing=swing)
    if df_feat.empty:
        return {"metrics": {}, "equity": pd.Series(), "trades": pd.DataFrame()}

    active_cols = SWING_FEATURE_COLS if swing else FEATURE_COLS
    X           = df_feat[active_cols].values
    sigs, confs, _ = model.predict_with_confidence(X)

    cost_rate = (SLIPPAGE_BPS + COMMISSION_BPS) / 10_000

    # Add realistic forex spread for intraday (1.5 pips)
    # Only apply for forex — equities already have commission modelled
    avg_price    = float(df_feat["close"].mean())
    spread_pips  = 1.5
    pip_size     = 0.0001 if avg_price < 10 else 0.01
    spread_cost  = (spread_pips * pip_size) / avg_price
    cost_rate    = cost_rate + spread_cost


    trades   = []
    equity   = [capital]
    nav      = capital
    i        = 0

    while i < len(df_feat) - 1:
        sig  = int(sigs[i])
        conf = float(confs[i])

        if sig == 0 or conf < min_conf:
            equity.append(nav)
            i += 1
            continue

        entry = float(df_feat["close"].iloc[i])
        atr   = float(df_feat["atr"].iloc[i]) if "atr" in df_feat.columns else entry * 0.001
        sl    = entry - sig * atr * SL_ATR_MULT
        tp    = entry + sig * atr * TP_ATR_MULT

        risk_amt  = nav * MAX_RISK_PCT
        units     = risk_amt / (atr * SL_ATR_MULT + 1e-9)
        pos_val   = min(units * entry, nav * 5)

        exit_price = None
        exit_bar   = i + 1

        for j in range(i + 1, min(i + 50, len(df_feat))):
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
            exit_price = float(df_feat["close"].iloc[min(exit_bar, len(df_feat)-1)])

        gross_ret = sig * (exit_price - entry) / (entry + 1e-9)
        net_pnl   = pos_val * gross_ret - pos_val * cost_rate
        nav      += net_pnl
        equity.append(nav)

        trades.append({
            "entry_time": df_feat.index[i],
            "exit_time":  df_feat.index[min(exit_bar, len(df_feat)-1)],
            "direction":  "Long" if sig == 1 else "Short",
            "entry":      round(entry, 5),
            "exit":       round(exit_price, 5),
            "pnl":        round(net_pnl, 4),
            "confidence": round(conf, 3),
            "nav":        round(nav, 2),
        })

        i = exit_bar + 1

    eq_series = pd.Series(
        equity[:len(df_feat)],
        index=df_feat.index[:len(equity)]
    )
    trades_df = pd.DataFrame(trades)
    metrics   = compute_metrics(eq_series, trades_df)

    logger.info(
        f"Backtest: {len(trades_df)} trades | "
        f"Sharpe={metrics['sharpe_ratio']} | "
        f"DD={metrics['max_drawdown']} | "
        f"WR={metrics['win_rate']}"
    )
    return {"metrics": metrics, "equity": eq_series, "trades": trades_df}


# ── Walk-Forward Backtest ──────────────────────────────────────

def run_walkforward_backtest(
    df_raw:      pd.DataFrame,
    train_bars:  int   = 1080,   # ~45 days of 1h bars
    test_bars:   int   = 240,    # ~10 days of 1h bars
    capital:     float = INITIAL_CAPITAL,
    min_conf:    float = MIN_CONFIDENCE,
    swing:       bool  = False,
) -> dict:
    """
    Rolling walk-forward backtest: retrain the ensemble on each
    train window, then simulate forward on the next test window.

    This is the honest backtest — each out-of-sample segment is
    predicted by a model that has never seen it.

    Parameters
    ----------
    df_raw      : full raw OHLCV DataFrame
    train_bars  : number of bars in each training window
    test_bars   : number of bars in each out-of-sample test window
    capital     : starting capital (carries forward across folds)
    min_conf    : minimum ensemble confidence to take a trade

    Returns
    -------
    dict with keys:
      "metrics"       : aggregated metrics over all OOS periods
      "equity"        : full equity curve (pd.Series)
      "trades"        : all OOS trades (pd.DataFrame)
      "fold_metrics"  : per-fold metrics list
      "fold_count"    : number of folds completed
    """
    from signals.ensemble import StackedEnsemble
    from signals.features import get_X_y

    all_trades:  list = []
    equity_vals: list = []
    equity_idx:  list = []
    fold_metrics: list = []
    nav          = capital
    n            = len(df_raw)
    fold         = 0

    logger.info(
        f"Walk-forward backtest | "
        f"bars={n} | train={train_bars} | test={test_bars} | "
        f"expected_folds={(n - train_bars) // test_bars}"
    )

    start = 0
    while start + train_bars + test_bars <= n:
        train_df = df_raw.iloc[start : start + train_bars]
        test_df  = df_raw.iloc[start + train_bars : start + train_bars + test_bars]

        # ── Train on this window ──────────────────────────────
        X_tr, y_tr, _ = get_X_y(train_df, swing=swing)
        if len(X_tr) < 100:
            logger.warning(f"Fold {fold}: insufficient training samples ({len(X_tr)}) — skipping")
            start += test_bars
            continue

        model = StackedEnsemble(instrument=f"wf_fold_{fold}")
        try:
            model.train(X_tr, y_tr)
        except Exception as e:
            logger.error(f"Fold {fold}: training failed — {e}")
            start += test_bars
            continue

        # ── Simulate on out-of-sample test window ────────────
        result = run_backtest_single(
            test_df, model,
            capital=nav,
            min_conf=min_conf,
            swing=swing,
        )

        if result["trades"].empty:
            logger.info(f"Fold {fold}: no trades in OOS window")
            start += test_bars
            fold  += 1
            continue

        fold_trades = result["trades"].copy()
        fold_trades["fold"] = fold
        all_trades.append(fold_trades)

        # Carry NAV forward into the next fold
        if not result["equity"].empty:
            nav = float(result["equity"].iloc[-1])
            equity_vals.extend(result["equity"].tolist())
            equity_idx.extend(result["equity"].index.tolist())

        m = result["metrics"]
        fold_metrics.append({
            "fold":          fold,
            "train_start":   train_df.index[0],
            "train_end":     train_df.index[-1],
            "test_start":    test_df.index[0],
            "test_end":      test_df.index[-1],
            "trades":        m.get("total_trades", 0),
            "sharpe":        m.get("sharpe_ratio", 0),
            "max_dd":        m.get("max_drawdown", "0%"),
            "win_rate":      m.get("win_rate", "0%"),
            "total_return":  m.get("total_return", "0%"),
        })

        logger.info(
            f"Fold {fold} OOS | "
            f"trades={m.get('total_trades',0)} | "
            f"Sharpe={m.get('sharpe_ratio',0)} | "
            f"DD={m.get('max_drawdown','?')} | "
            f"WR={m.get('win_rate','?')}"
        )

        # Anchored: keep growing the training window each fold
        # (change to start += test_bars for rolling window)
        start += test_bars
        fold  += 1

    if not all_trades:
        logger.warning("Walk-forward: no trades generated across all folds")
        return {
            "metrics":      {},
            "equity":       pd.Series(dtype=float),
            "trades":       pd.DataFrame(),
            "fold_metrics": fold_metrics,
            "fold_count":   fold,
        }

    trades_df = pd.concat(all_trades, ignore_index=True)
    eq_series = pd.Series(equity_vals, index=equity_idx)

    metrics = compute_metrics(eq_series, trades_df)
    metrics["fold_count"] = fold

    logger.info(
        f"Walk-forward complete | {fold} folds | "
        f"Sharpe={metrics['sharpe_ratio']} | "
        f"DD={metrics['max_drawdown']} | "
        f"trades={metrics['total_trades']}"
    )

    return {
        "metrics":      metrics,
        "equity":       eq_series,
        "trades":       trades_df,
        "fold_metrics": fold_metrics,
        "fold_count":   fold,
    }
