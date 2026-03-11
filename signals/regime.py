"""
================================================================
  signals/regime.py
  Market regime detection — classifies each bar as:
    "trending_up"    : strong uptrend (ADX > 25, price above EMAs)
    "trending_down"  : strong downtrend
    "ranging"        : low ADX, price oscillating
    "high_volatility": realised vol spike (news events, data releases)

  Regime is used to:
    1. Select which strategy to weight more heavily
    2. Scale position sizes (reduce in high_volatility)
    3. Choose model hyperparameters per regime
================================================================
"""

import logging
import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.settings import (
    ADX_TREND_THRESHOLD,
    RVOL_HIGH_THRESHOLD,
    REGIME_LOOKBACK,
    EMA_FAST, EMA_SLOW,
)

logger = logging.getLogger("trading_firm.regime")

REGIMES = ["trending_up", "trending_down", "ranging", "high_volatility"]


def detect_regime(df: pd.DataFrame) -> pd.Series:
    """
    Classify each bar into one of four market regimes.

    Parameters
    ----------
    df : DataFrame with OHLCV columns (raw or feature-enriched)

    Returns
    -------
    pd.Series of regime strings, aligned to df index
    """
    c = df["close"]

    # ADX
    h, l = df["high"], df["low"]
    tr       = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr14    = tr.ewm(span=14, adjust=False).mean()
    plus_dm  = (h.diff()).clip(lower=0)
    minus_dm = (-l.diff()).clip(lower=0)
    plus_di  = 100 * plus_dm.ewm(span=14, adjust=False).mean() / (atr14 + 1e-9)
    minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / (atr14 + 1e-9)
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    adx      = dx.ewm(span=14, adjust=False).mean()

    # EMAs for direction
    ema_fast = c.ewm(span=EMA_FAST, adjust=False).mean()
    ema_slow = c.ewm(span=EMA_SLOW, adjust=False).mean()

    # Realised volatility ratio
    ret        = c.pct_change()
    rvol       = ret.rolling(REGIME_LOOKBACK // 5).std()
    rvol_mean  = rvol.rolling(REGIME_LOOKBACK).mean()
    rvol_ratio = rvol / (rvol_mean + 1e-9)

    # Classification logic
    regime = pd.Series("ranging", index=df.index)

    trending_up   = (adx > ADX_TREND_THRESHOLD) & (ema_fast > ema_slow)
    trending_down = (adx > ADX_TREND_THRESHOLD) & (ema_fast < ema_slow)
    high_vol      = rvol_ratio > RVOL_HIGH_THRESHOLD

    regime[trending_up]   = "trending_up"
    regime[trending_down] = "trending_down"
    regime[high_vol]      = "high_volatility"   # Overrides trend if vol spikes

    return regime


def get_current_regime(df: pd.DataFrame) -> str:
    """Return the regime of the most recent bar."""
    regimes = detect_regime(df)
    if regimes.empty:
        return "ranging"
    return regimes.iloc[-1]


def regime_position_scale(regime: str) -> float:
    """
    Return a position size multiplier based on current regime.
    This is applied on top of the base Kelly/fixed-fractional size.
    """
    scales = {
        "trending_up":    1.0,    # Full size — trend trading is where edge is clearest
        "trending_down":  1.0,    # Full size
        "ranging":        0.75,   # Slightly reduced — mean reversion has lower win rate
        "high_volatility": 0.30,  # Major reduction — protect capital during news events
    }
    return scales.get(regime, 0.75)


def strategy_weights_by_regime(regime: str) -> dict:
    """
    Adjust portfolio strategy weights based on current market regime.
    Returns dict matching STRATEGY_WEIGHTS keys in settings.py.
    """
    weights = {
        "trending_up": {
            "momentum":        0.50,
            "mean_reversion":  0.15,
            "breakout":        0.25,
            "regime_adaptive": 0.10,
        },
        "trending_down": {
            "momentum":        0.45,
            "mean_reversion":  0.20,
            "breakout":        0.20,
            "regime_adaptive": 0.15,
        },
        "ranging": {
            "momentum":        0.15,
            "mean_reversion":  0.55,
            "breakout":        0.15,
            "regime_adaptive": 0.15,
        },
        "high_volatility": {
            "momentum":        0.10,
            "mean_reversion":  0.10,
            "breakout":        0.05,
            "regime_adaptive": 0.75,   # Let adaptive strategy decide
        },
    }
    return weights.get(regime, {
        "momentum":        0.35,
        "mean_reversion":  0.30,
        "breakout":        0.20,
        "regime_adaptive": 0.15,
    })


class RegimeTracker:
    """
    Tracks regime history across all instruments.
    Used by portfolio manager to coordinate strategy weights.
    """

    def __init__(self):
        self._regimes:  dict = {}    # {ticker: current_regime}
        self._history:  dict = {}    # {ticker: pd.Series of regimes}

    def update(self, ticker: str, df: pd.DataFrame):
        regime_series         = detect_regime(df)
        self._history[ticker] = regime_series
        self._regimes[ticker] = regime_series.iloc[-1] if len(regime_series) else "ranging"

    def current(self, ticker: str) -> str:
        return self._regimes.get(ticker, "ranging")

    def all_current(self) -> dict:
        return dict(self._regimes)

    def dominant_regime(self) -> str:
        """
        Return the most common regime across all instruments.
        Used for portfolio-level decisions.
        """
        if not self._regimes:
            return "ranging"
        from collections import Counter
        counts = Counter(self._regimes.values())
        return counts.most_common(1)[0][0]

    def summary(self) -> pd.DataFrame:
        if not self._regimes:
            return pd.DataFrame()
        rows = [
            {"ticker": t, "regime": r}
            for t, r in self._regimes.items()
        ]
        return pd.DataFrame(rows).set_index("ticker")
