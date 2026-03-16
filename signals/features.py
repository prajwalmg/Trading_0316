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

"""
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
"""

def add_volume(df: pd.DataFrame) -> pd.DataFrame:
    vol = df["volume"].replace(0, np.nan)
    c   = df["close"]
    h   = df["high"]
    l   = df["low"]

    # If volume is all zero (forex), fill defaults so ratios don't explode
    if vol.isna().mean() > 0.5:
        for col in [
            "vol_ratio", "vol_z", "obv", "obv_ema", "obv_sig",
            "buy_volume", "sell_volume", "vol_imbalance", "vol_delta_z",
        ]:
            df[col] = 0.0
        df["vwap"]      = c
        df["vwap_dist"] = 0.0
        return df

    df["vol_ratio"] = vol / vol.rolling(20).mean()
    df["vol_z"]     = (vol - vol.rolling(20).mean()) / (vol.rolling(20).std() + 1e-9)

    df["obv"]       = (np.sign(c.diff()) * vol.fillna(0)).fillna(0).cumsum()
    df["obv_ema"]   = df["obv"].ewm(span=10).mean()
    df["obv_sig"]   = (df["obv"] - df["obv_ema"]) / (vol.rolling(20).mean().fillna(1) + 1e-9)

    df["vwap"]      = (vol * (h + l + c) / 3).rolling(20).sum() / (vol.rolling(20).sum() + 1e-9)
    df["vwap_dist"] = (c - df["vwap"]) / c

    # ── Volume profile ────────────────────────────────────────
    bar_range           = (h - l).replace(0, 1e-9)
    buy_ratio           = (c - l) / bar_range
    df["buy_volume"]    = vol * buy_ratio
    df["sell_volume"]   = vol * (1 - buy_ratio)

    df["vol_imbalance"] = (
        df["buy_volume"].rolling(10).sum() /
        (df["sell_volume"].rolling(10).sum() + 1e-9)
    ).clip(0, 5)

    vol_delta           = (df["buy_volume"] - df["sell_volume"]).rolling(20).sum()
    df["vol_delta_z"]   = (
        (vol_delta - vol_delta.rolling(50).mean()) /
        (vol_delta.rolling(50).std() + 1e-9)
    )

    # Add before return df:
    vol_cols = [
        "vol_ratio", "vol_z", "obv", "obv_ema", "obv_sig",
        "vwap", "vwap_dist", "buy_volume", "sell_volume",
        "vol_imbalance", "vol_delta_z",
    ]
    for col in vol_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

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


def add_microstructure(df: pd.DataFrame) -> pd.DataFrame:
    """Price microstructure features for intraday trading."""
    c = df["close"]
    h = df["high"]
    l = df["low"]
    o = df["open"]

    # Candle body ratio — how much of the bar was directional vs noise
    bar_range = (h - l).replace(0, 1e-9)
    df["body_ratio"]   = (c - o).abs() / bar_range
    df["upper_shadow"]  = (h - df[["open","close"]].max(axis=1)) / bar_range
    df["lower_shadow"]  = (df[["open","close"]].min(axis=1) - l) / bar_range

    # Price efficiency ratio — how efficiently price moved over last 10 bars
    net_move   = (c - c.shift(10)).abs()
    total_path = c.diff().abs().rolling(10).sum().replace(0, 1e-9)
    df["efficiency_ratio"] = net_move / total_path

    # Bar-over-bar momentum consistency
    df["mom_consistency"] = (
        c.diff().gt(0).astype(int)
         .rolling(10).mean()
    )

    # Volatility ratio — current bar range vs recent average range
    avg_range = bar_range.rolling(20).mean().replace(0, 1e-9)
    df["range_ratio"] = bar_range / avg_range

    return df


def add_stationary_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stationary features that have identical distributions
    regardless of window size — critical for train/live consistency.
    """
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"].fillna(0)

    # Returns — always stationary
    df["ret_1"]  = c.pct_change(1).clip(-0.1, 0.1)
    df["ret_5"]  = c.pct_change(5).clip(-0.2, 0.2)
    df["ret_20"] = c.pct_change(20).clip(-0.3, 0.3)

    # Position in recent range — bounded 0 to 1
    roll_high         = h.rolling(20).max()
    roll_low          = l.rolling(20).min()
    df["range_pos"]   = (c - roll_low) / (roll_high - roll_low + 1e-9)

    # ATR as percentage — stationary
    atr               = df["atr"] if "atr" in df.columns else c.rolling(14).std()
    df["atr_pct"]     = (atr / c).clip(0, 0.1)

    # Normalised RSI — centred at 0
    rsi               = df["rsi"] if "rsi" in df.columns else pd.Series(50, index=df.index)
    df["rsi_norm"]    = (rsi - 50) / 50   # bounded -1 to +1

    # Volume z-score — stationary
    vol_mean          = v.rolling(20).mean().replace(0, 1e-9)
    df["vol_zscore"]  = ((v - vol_mean) / (v.rolling(20).std() + 1e-9)).clip(-5, 5)

    # Trend alignment — categorical
    ema20             = c.ewm(span=20).mean()
    ema50             = c.ewm(span=50).mean()
    df["trend_align"] = np.sign(ema20 - ema50).astype(int)

    return df


def add_order_flow(df: pd.DataFrame) -> pd.DataFrame:
    """
    Order flow features — best short-term predictor of price direction.
    """
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"].fillna(0)

    bar_range         = (h - l).replace(0, 1e-9)
    buy_frac          = (c - l) / bar_range
    buy_vol           = v * buy_frac
    sell_vol          = v * (1 - buy_frac)

    # Cumulative delta — running buy/sell imbalance
    delta             = buy_vol - sell_vol
    cum_delta         = delta.rolling(20).sum()
    df["delta_z"]     = (
        (cum_delta - cum_delta.rolling(50).mean()) /
        (cum_delta.rolling(50).std() + 1e-9)
    ).clip(-5, 5)

    # Buy/sell ratio
    df["bs_ratio"]    = (
        buy_vol.rolling(10).sum() /
        (sell_vol.rolling(10).sum() + 1e-9)
    ).clip(0, 5)

    # Tick direction consistency
    df["tick_consist"] = np.sign(c.diff()).rolling(10).mean()

    return df


def add_session_overlaps(df: pd.DataFrame) -> pd.DataFrame:
    """Mark high-liquidity session overlap periods."""
    h = df.index.hour

    # London/NY overlap (13:00–17:00 UTC) — highest liquidity
    df["london_ny_overlap"] = ((h >= 13) & (h < 17)).astype(int)

    # London open (07:00–09:00 UTC) — second highest
    df["london_open"] = ((h >= 7) & (h < 9)).astype(int)

    # NY open (14:30–16:00 UTC) — high volatility
    df["ny_open"] = ((h >= 14) & (h < 16)).astype(int)

    # Asian session (00:00–07:00 UTC) — low liquidity, avoid
    df["asian_session"] = ((h >= 0) & (h < 7)).astype(int)

    return df


def add_labels_col(df: pd.DataFrame, forward_bars: int = 12,
                   sl_mult: float = 1.5, tp_mult: float = 2.0) -> pd.DataFrame:
    """
    Triple barrier labels — matches exactly how trades are executed.
    +1 = TP hit first
    -1 = SL hit first
     0 = Time barrier (neither hit in forward_bars)
    """
    if "atr" not in df.columns:
        df["atr"] = df["close"].rolling(14).std()

    labels = pd.Series(0, index=df.index)
    close  = df["close"].values
    atr    = df["atr"].values
    n      = len(df)

    for i in range(n - forward_bars):
        entry  = close[i]
        tp     = entry + tp_mult * atr[i]
        sl     = entry - sl_mult * atr[i]
        future = close[i+1 : i+forward_bars+1]

        tp_idx = next((j for j, p in enumerate(future) if p >= tp), None)
        sl_idx = next((j for j, p in enumerate(future) if p <= sl), None)

        if tp_idx is not None and sl_idx is not None:
            labels.iloc[i] = 1 if tp_idx < sl_idx else -1
        elif tp_idx is not None:
            labels.iloc[i] = 1
        elif sl_idx is not None:
            labels.iloc[i] = -1
        # else 0 — time barrier

    df["label"] = labels
    return df


def add_htf_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extended HTF context — our strongest predictive signal.
    """
    try:
        c = df["close"]

        # Existing: 15min and 1h trends
        c_15m    = c.resample("15min").last().ffill()
        ema_15m  = c_15m.ewm(span=20).mean()
        trend_15m = pd.Series(
            np.where(c_15m > ema_15m, 1, -1),
            index=c_15m.index
        ).reindex(df.index, method="ffill").fillna(0)

        c_1h     = c.resample("1h").last().ffill()
        ema_1h   = c_1h.ewm(span=20).mean()
        trend_1h = pd.Series(
            np.where(c_1h > ema_1h, 1, -1),
            index=c_1h.index
        ).reindex(df.index, method="ffill").fillna(0)

        # NEW: 4h trend — medium term direction
        c_4h     = c.resample("4h").last().ffill()
        ema_4h   = c_4h.ewm(span=20).mean()
        trend_4h = pd.Series(
            np.where(c_4h > ema_4h, 1, -1),
            index=c_4h.index
        ).reindex(df.index, method="ffill").fillna(0)

        # NEW: HTF momentum — how strongly is the trend moving
        c_1h_ret = c_1h.pct_change(5).reindex(df.index, method="ffill").fillna(0)

        # NEW: Trend consistency — are all timeframes aligned
        trend_consistency = (
            (trend_15m == trend_1h).astype(int) +
            (trend_1h  == trend_4h).astype(int)
        ) / 2   # 0, 0.5, or 1.0

        df["trend_15m"]          = trend_15m
        df["trend_1h"]           = trend_1h
        df["trend_4h"]           = trend_4h   # new
        df["htf_momentum"]       = c_1h_ret.clip(-0.05, 0.05)   # new
        df["trend_consistency"]  = trend_consistency   # new
        df["htf_alignment"]      = (
            (trend_15m == trend_1h) & (trend_1h == trend_4h)
        ).astype(int) * trend_1h   # +1, -1, or 0

        return df
    except Exception:
        for col in ["trend_15m", "trend_1h", "trend_4h",
                    "htf_momentum", "trend_consistency", "htf_alignment"]:
            df[col] = 0
        return df


def add_macro_context(
    df:  pd.DataFrame,
    vix: pd.Series = None,
    dxy: pd.Series = None,
) -> pd.DataFrame:
    """
    Add VIX and DXY macro context features.
    Both are daily series reindexed and forward-filled to match df's index.
    """
    if vix is not None:
        v = vix.reindex(df.index, method="ffill")
        df["vix_level"]   = v
        df["vix_z"]       = (
            (v - v.rolling(20).mean()) /
            (v.rolling(20).std() + 1e-9)
        )
        df["fear_regime"] = (v > 25).astype(int)
    else:
        df["vix_level"]   = 0.0
        df["vix_z"]       = 0.0
        df["fear_regime"] = 0

    if dxy is not None:
        d = dxy.reindex(df.index, method="ffill")
        df["dxy_trend"] = (
            (d > d.ewm(span=20).mean()).astype(int) * 2 - 1
        )
    else:
        df["dxy_trend"] = 0.0

    return df


# ── Feature column registry ───────────────────────────────────

FEATURE_COLS = [
    # HTF Context — our strongest signal (corr 0.10-0.21)
    "trend_1h", "trend_15m", "trend_4h",
    "htf_alignment", "htf_momentum", "trend_consistency",

    # Trend (medium correlation 0.04-0.07)
    "adx", "plus_di", "minus_di", "di_diff",
    "macd", "macd_hist",

    # Momentum (medium correlation)
    "rsi", "roc_6", "roc_1", "roc_12",
    "williams_r", "stoch_k",

    # Volatility (useful for sizing context)
    "atr_norm", "bb_width", "bb_pos",
    "atr_pct",   # new stationary version

    # Price action (some correlation)
    "body_ratio", "efficiency_ratio",
    "close_pos", "range_pos",   # stationary versions

    # Session/Time (corr 0.07-0.13)
    "hour_sin", "hour_cos",
    "london_open", "london_ny_overlap",
    "is_friday", "is_monday",
    "is_london",

    # Returns — stationary
    "ret_1", "ret_5", "ret_20",

    # Macro context
    "vix_z", "fear_regime", "dxy_trend",

    # Normalised indicators
    "rsi_norm",   # stationary RSI
    "trend_align",  # categorical trend

    # Volume (only for crypto where volume is real)
    "vol_ratio",   # keep but will be 0 for forex — model handles this

    # Alternative sentiment
    "fg_norm", "fg_contrarian",
]


def build_features(
    df:           pd.DataFrame,
    add_labels:   bool = True,
    drop_na:      bool = True,
    vix:          pd.Series = None,
    dxy:          pd.Series = None,
    fg_norm:      float = None,
    fg_contrarian: float = None,
    sl_mult:      float = 1.5,
    tp_mult:      float = 2.0,
    forward_bars: int   = 12,
    swing:        bool  = False,
) -> pd.DataFrame:
    """
    Full feature pipeline. Returns enriched DataFrame.

    Parameters
    ----------
    df         : raw OHLCV DataFrame
    add_labels : add forward-return target column
    horizon    : bars ahead for label
    drop_na    : drop rows with NaN in feature columns
    vix        : VIX daily series
    dxy        : DXY daily series
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
    df = add_microstructure(df)
    df = add_stationary_features(df)
    df = add_order_flow(df)
    df = add_session_overlaps(df)
    df = add_htf_context(df)
    df = add_macro_context(df, vix=vix, dxy=dxy)

    # ── Alternative data: Fear & Greed ───────────────────────
    # If caller supplies live scalars, broadcast them as constant columns.
    # During backtesting these default to 0 (neutral) — no lookahead.
    df["fg_norm"]      = float(fg_norm)      if fg_norm      is not None else 0.0
    df["fg_contrarian"] = float(fg_contrarian) if fg_contrarian is not None else 0.0

    if swing:
        df = add_swing_features(df)

    if add_labels:
        df = add_labels_col(df, forward_bars=forward_bars,
                        sl_mult=sl_mult, tp_mult=tp_mult)

    if drop_na:
        active_cols = SWING_FEATURE_COLS if swing else FEATURE_COLS
        cols = active_cols + (["label"] if add_labels else [])
        df   = df.dropna(subset=cols)


    # ── Clip extreme feature values (no-lookahead) ───────────
    # Uses expanding-window mean/std so each bar's clip bounds
    # are computed from past data only — no future leakage.
    for col in FEATURE_COLS:
        if col in df.columns:
            roll_mean = df[col].expanding(min_periods=50).mean()
            roll_std  = df[col].expanding(min_periods=50).std().fillna(1.0) + 1e-9
            df[col]   = df[col].clip(roll_mean - 5 * roll_std,
                                     roll_mean + 5 * roll_std)

    return df


def get_X_y(
    df:           pd.DataFrame,
    sl_mult:      float = 1.5,
    tp_mult:      float = 2.0,
    forward_bars: int   = 12,
    vix:          pd.Series = None,
    dxy:          pd.Series = None,
    swing:        bool  = False,
) -> tuple:
    """
    Build feature matrix and labels.
    - Removes class 0 (time barrier) — noise
    - Balances classes using SMOTE
    - Returns binary +1/-1 labels only
    - swing=True adds multi-day features and uses SWING_FEATURE_COLS
    """
    df_feat = build_features(
        df, add_labels=True,
        sl_mult=sl_mult, tp_mult=tp_mult,
        forward_bars=forward_bars,
        vix=vix, dxy=dxy,
        swing=swing,
    )
    if df_feat.empty:
        return np.array([]), np.array([]), df_feat

    active_cols = SWING_FEATURE_COLS if swing else FEATURE_COLS
    X = df_feat[active_cols].values
    y = df_feat["label"].values

    # ── Remove class 0 (time barrier — noise) ────────────────
    mask = y != 0
    X    = X[mask]
    y    = y[mask]
    df_feat = df_feat[mask]

    if len(X) < 100:
        return np.array([]), np.array([]), df_feat

    # NOTE: class balancing (SMOTE) is intentionally NOT applied here.
    # Applying SMOTE before cross-validation would let synthetic samples
    # generated from validation-period data leak into training folds.
    # SMOTE is applied inside each CV fold in StackedEnsemble.train().

    return X, y, df_feat


# ================================================================
# SWING-SPECIFIC FEATURES
# ================================================================

def add_swing_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features specifically designed for 1h swing trading.
    Focus on multi-day momentum and mean reversion signals.
    """
    c = df["close"]
    h = df["high"]
    l = df["low"]

    # Multi-day returns
    df["ret_5d"]  = c.pct_change(120).clip(-0.3, 0.3)
    df["ret_10d"] = c.pct_change(240).clip(-0.5, 0.5)
    df["ret_20d"] = c.pct_change(480).clip(-0.7, 0.7)

    # Position in weekly/monthly range
    weekly_high            = h.rolling(120).max()
    weekly_low             = l.rolling(120).min()
    df["weekly_range_pos"] = (c - weekly_low) / (weekly_high - weekly_low + 1e-9)

    monthly_high            = h.rolling(480).max()
    monthly_low             = l.rolling(480).min()
    df["monthly_range_pos"] = (c - monthly_low) / (monthly_high - monthly_low + 1e-9)

    # Multi-timeframe trend
    ema_20h  = c.ewm(span=20).mean()
    ema_120h = c.ewm(span=120).mean()
    ema_480h = c.ewm(span=480).mean()

    df["trend_1d"]  = (c > ema_20h).astype(int)  * 2 - 1
    df["trend_1w"]  = (c > ema_120h).astype(int) * 2 - 1
    df["trend_1m"]  = (c > ema_480h).astype(int) * 2 - 1
    df["mtf_align"] = (df["trend_1d"] + df["trend_1w"] + df["trend_1m"]) / 3

    # Volatility regime
    atr_20          = (h - l).rolling(20).mean()
    atr_120         = (h - l).rolling(120).mean()
    df["vol_regime"] = (atr_20 / (atr_120 + 1e-9)).clip(0, 5)

    return df


# ================================================================
# SWING FEATURE COLUMN REGISTRY
# ================================================================

SWING_FEATURE_COLS = FEATURE_COLS + [
    "ret_5d", "ret_10d", "ret_20d",
    "weekly_range_pos", "monthly_range_pos",
    "trend_1d", "trend_1w", "trend_1m",
    "mtf_align", "vol_regime",
]


