"""
================================================================
  signals/features.py
  Full feature engineering pipeline — 40+ features across:
    Trend, Momentum, Volatility, Candle Shape,
    Session/Time, Volume, Multi-timeframe
================================================================
"""

import logging
import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.settings import *

logger = logging.getLogger("trading_firm.features")


def _safe_div(a, b, fill=0.0):
    return a.div(b.replace(0, np.nan)).fillna(fill)


# ── Individual feature families ───────────────────────────────

def add_trend(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]

    # EMA cross (normalised)
    df["ema_fast"]   = c.ewm(span=EMA_FAST,  adjust=False).mean()
    df["ema_slow"]   = c.ewm(span=EMA_SLOW,  adjust=False).mean()
    df["ema_cross"]  = (df["ema_fast"] - df["ema_slow"]) / c

    # EMA 50 and 200 for longer context
    df["ema_50"]     = c.ewm(span=50,  adjust=False).mean()
    df["ema_200"]    = c.ewm(span=200, adjust=False).mean()
    df["ema_50_200"] = (df["ema_50"] - df["ema_200"]) / c  # golden/death cross

    # MACD
    exp1             = c.ewm(span=MACD_FAST,   adjust=False).mean()
    exp2             = c.ewm(span=MACD_SLOW,   adjust=False).mean()
    macd_line        = exp1 - exp2
    macd_sig         = macd_line.ewm(span=MACD_SIG, adjust=False).mean()
    df["macd"]       = macd_line / c
    df["macd_hist"]  = (macd_line - macd_sig) / c
    df["macd_cross"] = np.sign(macd_line - macd_sig)

    # ADX
    h, l = df["high"], df["low"]
    plus_dm  = (h.diff()).clip(lower=0)
    minus_dm = (-l.diff()).clip(lower=0)
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr14    = tr.ewm(span=14, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(span=14, adjust=False).mean() / (atr14 + 1e-9)
    minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / (atr14 + 1e-9)
    dx       = 100 * (_safe_div((plus_di - minus_di).abs(), plus_di + minus_di))
    df["adx"]      = dx.ewm(span=14, adjust=False).mean()
    df["plus_di"]  = plus_di
    df["minus_di"] = minus_di
    df["di_diff"]  = (plus_di - minus_di) / 100

    return df


def add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]

    # RSI
    delta    = c.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=RSI_PERIOD-1, adjust=False).mean()
    avg_loss = loss.ewm(com=RSI_PERIOD-1, adjust=False).mean()
    rs       = _safe_div(avg_gain, avg_loss)
    df["rsi"]      = 100 - (100 / (1 + rs))
    df["rsi_norm"] = (df["rsi"] - 50) / 50

    # RSI divergence (price vs RSI direction)
    df["rsi_div"] = np.sign(c.diff(3)) - np.sign(df["rsi"].diff(3))

    # Stochastic
    lo_k = df["low"].rolling(STOCH_K).min()
    hi_k = df["high"].rolling(STOCH_K).max()
    df["stoch_k"] = 100 * _safe_div(c - lo_k, hi_k - lo_k)
    df["stoch_d"] = df["stoch_k"].rolling(STOCH_D).mean()
    df["stoch_cross"] = np.sign(df["stoch_k"] - df["stoch_d"])

    # Rate of change — multiple horizons
    for n in [1, 3, 6, 12, 24]:
        df[f"roc_{n}"] = c.pct_change(n)

    # Williams %R
    df["williams_r"] = -100 * _safe_div(hi_k - c, hi_k - lo_k)

    # CCI (Commodity Channel Index)
    typical = (df["high"] + df["low"] + c) / 3
    sma_tp  = typical.rolling(20).mean()
    mad     = typical.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
    df["cci"] = _safe_div(typical - sma_tp, 0.015 * mad)

    return df


def add_volatility(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l = df["close"], df["high"], df["low"]

    # ATR — the most important feature for risk management
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    df["atr"]     = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    df["atr_pct"] = df["atr"] / c

    # Normalised ATR (relative to its own history)
    df["atr_norm"] = df["atr"] / df["atr"].rolling(50).mean()

    # Bollinger Bands
    bb_mid        = c.rolling(BB_PERIOD).mean()
    bb_std        = c.rolling(BB_PERIOD).std()
    bb_up         = bb_mid + BB_STD * bb_std
    bb_lo         = bb_mid - BB_STD * bb_std
    df["bb_width"]    = _safe_div(bb_up - bb_lo, bb_mid)
    df["bb_pos"]      = _safe_div(c - bb_lo, bb_up - bb_lo)  # 0=at lower, 1=at upper
    df["bb_squeeze"]  = (df["bb_width"] < df["bb_width"].rolling(50).mean()).astype(int)

    # Realised volatility at multiple windows
    ret = c.pct_change()
    for w in [5, 10, 20, 40]:
        df[f"rvol_{w}"] = ret.rolling(w).std()

    # Volatility ratio (current vs baseline)
    df["rvol_ratio"] = df["rvol_5"] / (df["rvol_40"] + 1e-9)

    # High-Low range
    df["hl_range"] = _safe_div(h - l, c)

    return df


def add_candle(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body   = (c - o).abs()
    candle = (h - l).replace(0, np.nan)

    df["body_ratio"]     = body / candle
    df["upper_wick"]     = (h - pd.concat([c, o], axis=1).max(axis=1)) / candle
    df["lower_wick"]     = (pd.concat([c, o], axis=1).min(axis=1) - l) / candle
    df["direction"]      = np.sign(c - o)
    df["close_pos"]      = (c - l) / candle          # 0=closed at low, 1=at high
    df["gap"]            = (o - c.shift(1)) / c.shift(1)

    # Consecutive direction (streak)
    dir_series        = np.sign(c - o)
    streak            = dir_series.groupby((dir_series != dir_series.shift()).cumsum()).cumcount() + 1
    df["bull_streak"] = np.where(dir_series == 1,  streak, 0)
    df["bear_streak"] = np.where(dir_series == -1, streak, 0)

    # Doji detection
    df["is_doji"] = ((body / candle) < 0.1).astype(int)

    return df


def add_session(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index

    # Cyclical time encoding
    df["hour_sin"]   = np.sin(2 * np.pi * idx.hour / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * idx.hour / 24)
    df["dow_sin"]    = np.sin(2 * np.pi * idx.dayofweek / 5)
    df["dow_cos"]    = np.cos(2 * np.pi * idx.dayofweek / 5)

    # Session flags (UTC)
    df["is_london"]   = idx.hour.isin(range(7, 16)).astype(int)
    df["is_ny"]       = idx.hour.isin(range(13, 21)).astype(int)
    df["is_overlap"]  = idx.hour.isin(range(13, 16)).astype(int)
    df["is_asian"]    = idx.hour.isin(range(0, 7)).astype(int)

    # Day of week
    df["is_monday"]  = (idx.dayofweek == 0).astype(int)
    df["is_friday"]  = (idx.dayofweek == 4).astype(int)

    return df


def add_volume(df: pd.DataFrame) -> pd.DataFrame:
    vol = df["volume"].replace(0, np.nan)
    c   = df["close"]

    df["vol_ratio"] = vol / vol.rolling(20).mean()
    df["vol_z"]     = (vol - vol.rolling(20).mean()) / (vol.rolling(20).std() + 1e-9)

    # OBV
    df["obv"]       = (np.sign(c.diff()) * vol).fillna(0).cumsum()
    df["obv_ema"]   = df["obv"].ewm(span=10).mean()
    df["obv_sig"]   = (df["obv"] - df["obv_ema"]) / (vol.rolling(20).mean() + 1e-9)

    # VWAP (intraday reset — approximated with rolling)
    df["vwap"]      = (vol * (df["high"] + df["low"] + c) / 3).rolling(20).sum() / (vol.rolling(20).sum() + 1e-9)
    df["vwap_dist"] = (c - df["vwap"]) / c

    return df


def add_target(df: pd.DataFrame, horizon: int = FEATURE_HORIZON) -> pd.DataFrame:
    """
    3-class target: 1=up, 0=flat, -1=down
    Based on forward close return over `horizon` bars.
    """
    fwd_ret       = df["close"].shift(-horizon) / df["close"] - 1
    df["future_ret"] = fwd_ret
    df["target"]     = 0
    df.loc[fwd_ret >  SIGNAL_THRESHOLD, "target"] =  1
    df.loc[fwd_ret < -SIGNAL_THRESHOLD, "target"] = -1
    return df


# ── Feature column registry ───────────────────────────────────

FEATURE_COLS = [
    # Trend
    "ema_cross", "ema_50_200", "macd", "macd_hist", "macd_cross",
    "adx", "plus_di", "minus_di", "di_diff",
    # Momentum
    "rsi_norm", "rsi_div", "stoch_k", "stoch_d", "stoch_cross",
    "roc_1", "roc_3", "roc_6", "roc_12", "roc_24",
    "williams_r", "cci",
    # Volatility
    "atr_pct", "atr_norm", "bb_width", "bb_pos", "bb_squeeze",
    "rvol_5", "rvol_10", "rvol_20", "rvol_ratio", "hl_range",
    # Candle
    "body_ratio", "upper_wick", "lower_wick", "direction",
    "close_pos", "gap", "bull_streak", "bear_streak", "is_doji",
    # Volume
    "vol_ratio", "vol_z", "obv_sig", "vwap_dist",
    # Session
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_london", "is_ny", "is_overlap",
    "is_monday", "is_friday",
]


def build_features(
    df:          pd.DataFrame,
    add_labels:  bool = True,
    horizon:     int  = FEATURE_HORIZON,
    drop_na:     bool = True,
) -> pd.DataFrame:
    """
    Full feature pipeline. Returns enriched DataFrame.

    Parameters
    ----------
    df         : raw OHLCV DataFrame
    add_labels : add forward-return target column
    horizon    : bars ahead for label
    drop_na    : drop rows with NaN in feature columns
    """
    if len(df) < MIN_BARS:
        logger.warning(f"Only {len(df)} bars — need {MIN_BARS} minimum")
        return pd.DataFrame()

    df = df.copy()
    df = add_trend(df)
    df = add_momentum(df)
    df = add_volatility(df)
    df = add_candle(df)
    df = add_session(df)
    df = add_volume(df)

    if add_labels:
        df = add_target(df, horizon=horizon)

    if drop_na:
        cols = FEATURE_COLS + (["target"] if add_labels else [])
        df   = df.dropna(subset=cols)

    return df


def get_X_y(df: pd.DataFrame) -> tuple:
    """Return (X, y, df_feat) ready for sklearn/xgb."""
    df_feat = build_features(df, add_labels=True)
    if df_feat.empty:
        return np.array([]), np.array([]), df_feat
    X = df_feat[FEATURE_COLS].values
    y = df_feat["target"].values
    return X, y, df_feat
