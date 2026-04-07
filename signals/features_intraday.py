"""
================================================================
  signals/features_intraday.py  — Intraday 5m Feature Pipeline
================================================================
  DO NOT MODIFY signals/features.py — this is a separate file.
================================================================
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("trading_firm.features_intraday")


def _normalize_index(idx: pd.Index, target: pd.Index) -> pd.Index:
    """Align idx dtype/tz to match target so that reindex() succeeds.

    Handles the common mismatch where QuestDB returns datetime64[s, UTC]
    and yfinance returns datetime64[us, UTC] (or naive equivalents).
    """
    # Step 1: strip tz from both, normalize to same tz-awareness
    if hasattr(target, 'tz') and target.tz is not None:
        # target is tz-aware — make idx tz-aware UTC too
        if hasattr(idx, 'tz') and idx.tz is None:
            idx = idx.tz_localize('UTC')
        elif hasattr(idx, 'tz') and idx.tz is not None:
            idx = idx.tz_convert('UTC')
    else:
        # target is tz-naive — strip tz from idx
        if hasattr(idx, 'tz') and idx.tz is not None:
            idx = idx.tz_localize(None)
    # Step 2: match resolution (ns/us/s)
    try:
        idx = idx.astype(target.dtype)
    except Exception:
        pass
    return idx


# ── Feature column registry ────────────────────────────────────
INTRADAY_FEATURE_COLS = [
    "bid_ask_proxy", "order_flow", "volume_surge",
    "price_accel", "tick_dir",
    "mins_london", "mins_ny", "is_overlap",
    "session_vol_ratio", "pre_london", "post_ny",
    "zscore_20", "vwap_dev", "bb_pct",
    "rsi_5", "rsi_14", "rsi_norm",
    "roc_3", "roc_6", "roc_12", "roc_24",
    "ema_3", "ema_8", "ema_cross", "macd_fast",
    "htf_trend", "htf_rsi", "htf_adx",
    "htf_alignment", "atr_1h",
    "atr_5m", "atr_norm", "rvol_10",
    "rvol_ratio", "bb_width", "adx_5m",
]

INTRADAY_HTF_COLS = [
    # 1H context (6)
    "h1_trend", "h1_rsi", "h1_bb_pct", "h1_ema_cross", "h1_momentum", "h1_vol_regime",
    # 4H context (4)
    "h4_trend", "h4_rsi", "h4_ema_cross", "h4_momentum",
    # Cross-TF (4)
    "tf_alignment", "htf_rsi_div", "volatility_regime", "trend_consistency",
]


# ── RSI helper ─────────────────────────────────────────────────
def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


# ── ADX helper ─────────────────────────────────────────────────
def _adx(high: pd.Series, low: pd.Series,
         close: pd.Series, period: int = 14) -> pd.Series:
    tr   = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    dm_plus  = high.diff().clip(lower=0).where(
        high.diff() > (-low.diff()).clip(lower=0), 0)
    dm_minus = (-low.diff()).clip(lower=0).where(
        (-low.diff()).clip(lower=0) > high.diff().clip(lower=0), 0)
    atr14     = tr.ewm(com=period - 1, min_periods=period).mean()
    di_plus   = 100 * dm_plus.ewm(com=period-1, min_periods=period).mean() / (atr14 + 1e-9)
    di_minus  = 100 * dm_minus.ewm(com=period-1, min_periods=period).mean() / (atr14 + 1e-9)
    dx        = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-9)
    adx       = dx.ewm(com=period - 1, min_periods=period).mean()
    return adx


# ── Intraday HTF feature builder ───────────────────────────────
def build_intraday_htf_features(df_5m: pd.DataFrame, df_1h: pd.DataFrame = None, df_4h: pd.DataFrame = None) -> pd.DataFrame:
    """Compute 14 intraday HTF feature columns aligned to 5m index.

    Uses merge_asof (backward fill) — no lookahead bias.
    """
    import pandas as pd
    idx = df_5m.index
    result = pd.DataFrame(0.0, index=idx, columns=INTRADAY_HTF_COLS)

    def _rsi_series(series, period=14):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-9)
        return 100 - 100 / (1 + rs)

    def _merge_htf(df_base_idx, df_htf, cols):
        """Merge HTF features onto 5m index via merge_asof backward fill."""
        # Normalise both indexes to UTC-naive int64 ns to avoid dtype mismatch
        # between yfinance (datetime64[s/us, UTC]) and QuestDB (datetime64[us/s, UTC])
        def _to_ns(idx):
            if hasattr(idx, 'tz') and idx.tz is not None:
                idx = idx.tz_localize(None)
            return pd.DatetimeIndex(idx).as_unit('ns')

        left_idx = _to_ns(df_base_idx)
        left     = pd.DataFrame(index=left_idx)

        # Normalize right AFTER dropna so lengths stay consistent
        right = df_htf[cols].dropna().copy()
        right.index = _to_ns(right.index)

        merged = pd.merge_asof(
            left.reset_index().rename(columns={left.index.name or "index": "ts"}),
            right.reset_index().rename(columns={right.index.name or "index": "ts"}),
            on="ts", direction="backward"
        ).set_index("ts")
        return merged

    # ── 1H context features ──────────────────────────────────────────────
    if df_1h is not None and not df_1h.empty and len(df_1h) >= 20:
        h1 = df_1h.copy()
        ema8  = h1["close"].ewm(span=8,  adjust=False).mean()
        ema21 = h1["close"].ewm(span=21, adjust=False).mean()
        ema20 = h1["close"].ewm(span=20, adjust=False).mean()

        h1["h1_trend"]      = (ema20 - ema20.shift(5)) / ema20.shift(5).replace(0, 1e-9)
        h1["h1_rsi"]        = _rsi_series(h1["close"], 14)
        sma20 = h1["close"].rolling(20).mean()
        std20 = h1["close"].rolling(20).std()
        h1["h1_bb_pct"]     = (h1["close"] - (sma20 - 2 * std20)) / (4 * std20.replace(0, 1e-9))
        h1["h1_ema_cross"]  = (ema8 - ema21) / h1["close"].replace(0, 1e-9)
        h1["h1_momentum"]   = h1["close"].pct_change(10)
        if "volume" in h1.columns:
            vol_avg = h1["volume"].rolling(20).mean().replace(0, 1e-9)
            h1["h1_vol_regime"] = (h1["volume"] / vol_avg).clip(0, 5)
        else:
            h1["h1_vol_regime"] = 1.0

        h1_cols = ["h1_trend", "h1_rsi", "h1_bb_pct", "h1_ema_cross", "h1_momentum", "h1_vol_regime"]
        merged_1h = _merge_htf(idx, h1, h1_cols)
        for col in h1_cols:
            if col in merged_1h.columns:
                result[col] = merged_1h[col].values

    # ── 4H context features ──────────────────────────────────────────────
    if df_4h is not None and not df_4h.empty and len(df_4h) >= 10:
        h4 = df_4h.copy()
        ema8  = h4["close"].ewm(span=8,  adjust=False).mean()
        ema21 = h4["close"].ewm(span=21, adjust=False).mean()
        ema20 = h4["close"].ewm(span=20, adjust=False).mean()

        h4["h4_trend"]     = (ema20 - ema20.shift(5)) / ema20.shift(5).replace(0, 1e-9)
        h4["h4_rsi"]       = _rsi_series(h4["close"], 14)
        h4["h4_ema_cross"] = (ema8 - ema21) / h4["close"].replace(0, 1e-9)
        h4["h4_momentum"]  = h4["close"].pct_change(10)

        h4_cols = ["h4_trend", "h4_rsi", "h4_ema_cross", "h4_momentum"]
        merged_4h = _merge_htf(idx, h4, h4_cols)
        for col in h4_cols:
            if col in merged_4h.columns:
                result[col] = merged_4h[col].values

    # ── Cross-TF features ────────────────────────────────────────────────
    if df_1h is not None and not df_1h.empty:
        ema20_1h = df_1h["close"].ewm(span=20, adjust=False).mean()
        trend_1h_df = pd.DataFrame({"trend_1h_bull": (ema20_1h > ema20_1h.shift(5)).astype(float)},
                                    index=df_1h.index)
        merged_cross = _merge_htf(idx, trend_1h_df, ["trend_1h_bull"])
        trend_1h_arr = merged_cross["trend_1h_bull"].fillna(0.5).values if "trend_1h_bull" in merged_cross.columns else 0.5

        trend_4h_arr = (result["h4_trend"].fillna(0) > 0).astype(float).values if "h4_trend" in result.columns else 0.5
        result["tf_alignment"] = (trend_1h_arr + trend_4h_arr) / 2.0

        rsi_5m = _rsi_series(df_5m["close"], 14)
        result["htf_rsi_div"] = (result["h1_rsi"].fillna(50).values - rsi_5m.values) / 100.0

        high_5m  = df_5m["high"]
        low_5m   = df_5m["low"]
        close_5m = df_5m["close"]
        tr_5m = pd.concat(
            [high_5m - low_5m,
             (high_5m - close_5m.shift()).abs(),
             (low_5m  - close_5m.shift()).abs()], axis=1
        ).max(axis=1)
        atr_5m = tr_5m.rolling(14).mean()
        result["volatility_regime"] = (atr_5m / close_5m.replace(0, 1e-9)).fillna(0)

        ema20_5m = df_5m["close"].ewm(span=20, adjust=False).mean()
        result["trend_consistency"] = (
            (df_5m["close"] > ema20_5m).rolling(20).mean().fillna(0.5)
        )

    return result.fillna(0.0)


# ── Main feature builder ───────────────────────────────────────
def build_features_intraday(
    df:            pd.DataFrame,
    ticker:        str   = "",
    add_labels:    bool  = True,
    drop_na:       bool  = True,
    sl_mult:       float = 1.0,
    tp_mult:       float = 1.0,
    forward_bars:  int   = 12,
    df_1h:         pd.DataFrame = None,
    df_4h:         pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Build all 35 intraday features on whatever subset is passed.
    Caller controls memory by passing a pre-sliced DataFrame.
    Returns DataFrame with feature columns (and 'label' if add_labels=True).
    """
    if len(df) < 50:
        return pd.DataFrame()

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # Ensure required OHLCV columns exist
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            logger.warning(f"Missing column: {col}")
            return pd.DataFrame()
    if "volume" not in df.columns:
        df["volume"] = 1.0

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    opn    = df["open"]
    volume = df["volume"].clip(lower=1e-9)

    # ── ATR 5m (needed early by other features) ─────────────
    tr    = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr_5m"]  = tr.rolling(14).mean()
    df["atr_norm"] = df["atr_5m"] / close.clip(1e-9)

    # ── Microstructure ───────────────────────────────────────
    df["bid_ask_proxy"] = (high - low) / close.clip(1e-9)
    df["order_flow"]    = (close - opn) / (high - low + 1e-9)
    df["volume_surge"]  = volume / volume.rolling(20).mean().clip(1e-9)
    df["price_accel"]   = close.pct_change().diff()
    df["tick_dir"]      = np.sign(close.diff())

    # ── Session features (UTC index) ─────────────────────────
    if hasattr(df.index, "hour"):
        hour_arr   = pd.Series(df.index.hour,   index=df.index)
        minute_arr = pd.Series(df.index.minute, index=df.index)
    else:
        hour_arr   = pd.Series(0, index=df.index)
        minute_arr = pd.Series(0, index=df.index)

    mins_of_day = hour_arr * 60 + minute_arr
    df["mins_london"]      = (mins_of_day - 420).clip(lower=0, upper=660)
    df["mins_ny"]          = (mins_of_day - 780).clip(lower=0, upper=480)
    df["is_overlap"]       = ((hour_arr >= 13) & (hour_arr < 16)).astype(float)
    df["pre_london"]       = ((hour_arr >= 5)  & (hour_arr < 7)).astype(float)
    df["post_ny"]          = ((hour_arr >= 21) & (hour_arr < 23)).astype(float)
    df["session_vol_ratio"] = volume / volume.rolling(78).mean().clip(1e-9)

    # ── Mean reversion ───────────────────────────────────────
    roll20_mean = close.rolling(20).mean()
    roll20_std  = close.rolling(20).std().clip(1e-9)
    df["zscore_20"] = (close - roll20_mean) / roll20_std

    # VWAP (daily cumulative; fallback to rolling 78-bar mean)
    try:
        tp   = (high + low + close) / 3.0
        pv   = tp * volume
        date_grp = df.index.normalize() if hasattr(df.index, "normalize") else None
        if date_grp is not None:
            cum_pv  = pv.groupby(date_grp).cumsum()
            cum_vol = volume.groupby(date_grp).cumsum()
            vwap    = cum_pv / cum_vol.clip(1e-9)
        else:
            raise ValueError("no normalize")
    except Exception:
        vwap = close.rolling(78).mean()
    df["vwap_dev"] = (close - vwap) / (df["atr_5m"].clip(1e-9))

    # Bollinger Bands (20, 2)
    bb_mid   = roll20_mean
    bb_std   = roll20_std
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_range = (bb_upper - bb_lower).clip(1e-9)
    df["bb_pct"]   = (close - bb_lower) / bb_range
    df["bb_width"] = bb_range / close.clip(1e-9)

    df["rsi_5"]    = _rsi(close, 5)
    df["rsi_14"]   = _rsi(close, 14)
    df["rsi_norm"] = (df["rsi_14"] - 50) / 50

    # ── Momentum ─────────────────────────────────────────────
    df["roc_3"]   = close.pct_change(3)
    df["roc_6"]   = close.pct_change(6)
    df["roc_12"]  = close.pct_change(12)
    df["roc_24"]  = close.pct_change(24)

    ema3  = close.ewm(span=3,  adjust=False).mean()
    ema8  = close.ewm(span=8,  adjust=False).mean()
    ema5  = close.ewm(span=5,  adjust=False).mean()
    ema13 = close.ewm(span=13, adjust=False).mean()
    df["ema_3"]     = ema3
    df["ema_8"]     = ema8
    df["ema_cross"] = ema3 - ema8
    df["macd_fast"] = ema5 - ema13

    # ── Volatility ───────────────────────────────────────────
    log_ret        = np.log(close / close.shift(1).clip(1e-9))
    rvol10         = log_ret.rolling(10).std()
    rvol10_ma      = rvol10.rolling(20).mean().clip(1e-9)
    df["rvol_10"]   = rvol10
    df["rvol_ratio"] = rvol10 / rvol10_ma
    df["adx_5m"]    = _adx(high, low, close, 14)

    # ── HTF context (use passed df_1h/df_4h; resample 5m only as fallback) ─────
    try:
        ohlcv_cols = {"open": "first", "high": "max",
                      "low": "min", "close": "last",
                      "volume": "sum"}
        # Use the externally-provided 1h DataFrame when available.
        # NEVER shadow the parameter — resampled 5m gives ~3-5× smaller ATR.
        if df_1h is not None and not df_1h.empty and len(df_1h) >= 10:
            _df_1h = df_1h.copy()
            # Normalise index to match df.index dtype exactly (handles tz + resolution mismatch)
            _df_1h.index = _normalize_index(_df_1h.index, df.index)
            logger.debug(f"HTF ({ticker}): using passed df_1h ({len(_df_1h)} bars)")
        else:
            _df_1h = df[["open","high","low","close","volume"]].resample("1h").agg(ohlcv_cols).dropna()
            logger.debug(f"HTF ({ticker}): resampling 5m→1h ({len(_df_1h)} bars, no df_1h passed)")
        if len(_df_1h) < 10:
            raise ValueError("too few 1h bars")

        ema50_1h  = _df_1h["close"].ewm(span=50,  adjust=False).mean()
        ema200_1h = _df_1h["close"].ewm(span=200, adjust=False).mean()
        htf_trend_1h = np.sign(ema50_1h - ema200_1h)
        htf_rsi_1h   = _rsi(_df_1h["close"], 14)
        htf_adx_1h   = _adx(_df_1h["high"], _df_1h["low"], _df_1h["close"], 14)

        # 1h ATR (true range on 1h bars, 14-bar rolling mean)
        tr_1h   = pd.concat([
            _df_1h["high"] - _df_1h["low"],
            (_df_1h["high"] - _df_1h["close"].shift(1)).abs(),
            (_df_1h["low"]  - _df_1h["close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr_1h_s = tr_1h.rolling(14).mean().reindex(df.index, method="ffill")

        # 4h for alignment — use passed df_4h if available
        if df_4h is not None and not df_4h.empty and len(df_4h) >= 5:
            _df_4h = df_4h.copy()
            _df_4h.index = _normalize_index(_df_4h.index, df.index)
        else:
            _df_4h = df[["open","high","low","close","volume"]].resample("4h").agg(ohlcv_cols).dropna()
        if len(_df_4h) < 5:
            raise ValueError("too few 4h bars")
        ema50_4h  = _df_4h["close"].ewm(span=50,  adjust=False).mean()
        ema200_4h = _df_4h["close"].ewm(span=200, adjust=False).mean()
        htf_align_4h = np.sign(ema50_4h - ema200_4h)

        # Reindex back to 5m (forward-fill)
        htf_trend_s  = htf_trend_1h.reindex(df.index, method="ffill")
        htf_rsi_s    = htf_rsi_1h.reindex(df.index, method="ffill")
        htf_adx_s    = htf_adx_1h.reindex(df.index, method="ffill")
        htf_align_s  = htf_align_4h.reindex(df.index, method="ffill")

        df["htf_trend"]    = htf_trend_s.fillna(0)
        df["htf_rsi"]      = htf_rsi_s.fillna(50)
        df["htf_adx"]      = htf_adx_s.fillna(0)
        df["htf_alignment"] = (df["htf_trend"] * htf_align_s.fillna(0))
        df["atr_1h"]       = atr_1h_s.fillna(df["atr_5m"] * 3)
    except Exception as e:
        logger.debug(f"HTF fallback ({ticker}): {e}")
        df["htf_trend"]    = 0.0
        df["htf_rsi"]      = 50.0
        df["htf_adx"]      = 0.0
        df["htf_alignment"] = 0.0
        df["atr_1h"]       = df["atr_5m"] * 3  # fallback: 3× 5m ATR

    # ── Triple-barrier labels (two-sided, matches backtest) ──
    # label=+1: going LONG (TP=+tp_mult*ATR, SL=-sl_mult*ATR)
    #           would win within forward_bars
    # label=-1: going SHORT (TP=-tp_mult*ATR, SL=+sl_mult*ATR)
    #           would win within forward_bars
    # label= 0: neither trade wins → filtered in get_X_y_intraday
    if add_labels:
        labels = np.zeros(len(df), dtype=np.int8)
        # Use atr_1h for barriers to match engine_intraday (viable cost structure)
        atr_arr   = df["atr_1h"].values if "atr_1h" in df.columns else df["atr_5m"].values * 3
        close_arr = close.values
        low_arr   = df["low"].values
        hi_arr    = df["high"].values
        n = len(df)
        for i in range(n - forward_bars):
            atr_i = atr_arr[i]
            if np.isnan(atr_i) or atr_i <= 0:
                continue
            entry    = close_arr[i]
            long_tp  = entry + tp_mult * atr_i
            long_sl  = entry - sl_mult * atr_i
            short_tp = entry - tp_mult * atr_i
            short_sl = entry + sl_mult * atr_i
            horizon  = min(i + forward_bars + 1, n)

            # Scan long trade outcome
            long_outcome = 0   # 0=timeout, 1=win, -1=lose
            for j in range(i + 1, horizon):
                if hi_arr[j] >= long_tp:
                    long_outcome = 1; break
                if low_arr[j] <= long_sl:
                    long_outcome = -1; break

            # Scan short trade outcome
            short_outcome = 0
            for j in range(i + 1, horizon):
                if low_arr[j] <= short_tp:
                    short_outcome = 1; break
                if hi_arr[j] >= short_sl:
                    short_outcome = -1; break

            if long_outcome == 1 and short_outcome != 1:
                labels[i] = 1
            elif short_outcome == 1 and long_outcome != 1:
                labels[i] = -1
            # both win or both lose/timeout → label stays 0
        df["label"] = labels

    # Append intraday HTF features if higher-TF data provided
    if df_1h is not None or df_4h is not None:
        htf = build_intraday_htf_features(df, df_1h=df_1h, df_4h=df_4h)
        for col in htf.columns:
            df[col] = htf[col].values

    if drop_na:
        cols_to_check = INTRADAY_FEATURE_COLS + (["label"] if add_labels else [])
        present = [c for c in cols_to_check if c in df.columns]
        df = df.dropna(subset=present)

    return df


# ── X/y extractor ─────────────────────────────────────────────
def get_X_y_intraday(
    df:           pd.DataFrame,
    sl_mult:      float = 1.0,
    tp_mult:      float = 1.0,
    forward_bars: int   = 12,
):
    df_feat = build_features_intraday(
        df, add_labels=True, drop_na=True,
        sl_mult=sl_mult, tp_mult=tp_mult,
        forward_bars=forward_bars,
    )
    if df_feat.empty:
        return np.empty((0, len(INTRADAY_FEATURE_COLS))), np.empty(0), df_feat

    mask    = df_feat["label"] != 0
    df_feat = df_feat[mask]
    X = df_feat[INTRADAY_FEATURE_COLS].values
    y = df_feat["label"].values
    return X, y, df_feat
