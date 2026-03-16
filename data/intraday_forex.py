"""
================================================================
  data/intraday_forex.py
  Intraday forex data pipeline using Histdata.com M1 CSV files.
  Resamples M1 → 5-minute bars for training and live signal gen.

  Pairs supported (add more by creating folders):
    USDCHF, NZDUSD, AUDUSD (and any other Histdata pair)

  Usage:
    from data.intraday_forex import IntradayForexPipeline
    pipeline = IntradayForexPipeline()
    pipeline.load_all()
    df = pipeline.get("USDCHF")
================================================================
"""

import os
import glob
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger("trading_firm.intraday_forex")

# ── Config ────────────────────────────────────────────────────
HISTDATA_ROOT  = "data/histdata"       # root folder
RESAMPLE_TO    = "5min"                # target timeframe
CACHE_DIR      = "data/cache/intraday" # parquet cache

# Map our ticker names → histdata folder names
PAIR_MAP = {
    "USDCHF": "USDCHF",
    #"NZDUSD": "NZDUSD",
    "AUDUSD": "AUDUSD",
    "EURUSD": "EURUSD",
    "GBPUSD": "GBPUSD",
}

# Trading session filters (UTC) — only keep liquid hours
# Forex is 24/5 but liquidity drops during Asian dead zone
SESSION_START = 6   # 6am UTC  (London pre-open)
SESSION_END   = 20  # 8pm UTC  (NY close)
"""
def add_htf_context(df_5m: pd.DataFrame) -> pd.DataFrame:
   
    Add 15-minute and 1-hour trend context to 5-minute bars.
    A 5-min signal aligned with the 1h trend is much stronger.
    
    # 15-minute close and trend direction
    df_15m = df_5m["close"].resample("15min").last().ffill()
    ema_15  = df_15m.ewm(span=20).mean()
    trend_15 = (df_15m > ema_15).astype(int) * 2 - 1  # +1 or -1

    # 1-hour close and trend direction
    df_1h  = df_5m["close"].resample("1h").last().ffill()
    ema_1h  = df_1h.ewm(span=20).mean()
    trend_1h = (df_1h > ema_1h).astype(int) * 2 - 1

    # Map back to 5-minute index
    df_5m["trend_15m"] = trend_15.reindex(df_5m.index, method="ffill").fillna(0)
    df_5m["trend_1h"]  = trend_1h.reindex(df_5m.index, method="ffill").fillna(0)

    # Alignment score: +2 if both agree, 0 if mixed, -2 if both oppose
    df_5m["htf_alignment"] = df_5m["trend_15m"] + df_5m["trend_1h"]

    return df_5m
"""

def load_pair(
    pair:        str,
    resample_to: str  = RESAMPLE_TO,
    session_filter: bool = True,
    use_cache:   bool = True,
) -> pd.DataFrame:
    """
    Load all Histdata M1 CSV files for a pair, resample, and return
    a clean OHLCV DataFrame indexed by UTC datetime.

    Parameters
    ----------
    pair          : e.g. "USDCHF"
    resample_to   : pandas resample string e.g. "5min", "15min", "1h"
    session_filter: if True, drop bars outside SESSION_START–SESSION_END UTC
    use_cache     : load from parquet cache if available and fresh
    """
    folder = os.path.join(HISTDATA_ROOT, PAIR_MAP.get(pair, pair))

    if not os.path.exists(folder):
        raise FileNotFoundError(
            f"No data folder found for {pair} at {folder}\n"
            f"Download from https://www.histdata.com and place CSVs in {folder}/"
        )

    # ── Cache check ───────────────────────────────────────────
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{pair}_{resample_to}.parquet")

    if use_cache and os.path.exists(cache_path):
        # Cache valid if newer than all source CSV files
        cache_mtime = os.path.getmtime(cache_path)
        csv_files   = sorted(glob.glob(os.path.join(folder, "*.csv")))
        newest_csv  = max(os.path.getmtime(f) for f in csv_files) if csv_files else 0
        if cache_mtime > newest_csv:
            df = pd.read_parquet(cache_path)
            logger.info(f"{pair}: loaded from cache ({len(df)} {resample_to} bars)")
            return df

    # ── Load CSVs ─────────────────────────────────────────────
    csv_files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {folder}")

    dfs = []
    for f in csv_files:
        try:
            tmp = pd.read_csv(
                f, sep=";", header=None,
                names=["datetime", "open", "high", "low", "close", "volume"],
                dtype={
                    "datetime": str,
                    "open":  float, "high":  float,
                    "low":   float, "close": float,
                    "volume": float,
                }
            )
            tmp.index = pd.to_datetime(tmp["datetime"], format="%Y%m%d %H%M%S")
            tmp = tmp.drop(columns=["datetime"])
            dfs.append(tmp)
            logger.debug(f"  {os.path.basename(f)}: {len(tmp)} M1 bars")
        except Exception as e:
            logger.warning(f"Could not load {f}: {e}")

    if not dfs:
        raise ValueError(f"No data loaded for {pair}")

    # ── Combine and clean ─────────────────────────────────────
    df = pd.concat(dfs).sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # Remove clearly bad ticks (price = 0)
    df = df[(df["open"] > 0) & (df["close"] > 0)]

    # ── Resample M1 → target timeframe ────────────────────────
    df = df.resample(resample_to).agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna(subset=["open", "close"])

    # ── Session filter ────────────────────────────────────────
    if session_filter:
        df = df[
            (df.index.hour >= SESSION_START) &
            (df.index.hour <  SESSION_END)
        ]
        # Drop weekends
        df = df[df.index.dayofweek < 5]

    # ── Cache result ──────────────────────────────────────────
    df.to_parquet(cache_path)

    logger.info(
        f"{pair}: loaded {len(df)} {resample_to} bars | "
        f"{df.index[0].date()} → {df.index[-1].date()}"
    )
    
    #df = add_htf_context(df)

    return df


class IntradayForexPipeline:
    """
    Manages intraday forex data for all configured pairs.
    Acts as a drop-in complement to DataPipeline for histdata pairs.
    """

    def __init__(
        self,
        pairs:       list = None,
        resample_to: str  = RESAMPLE_TO,
        session_filter: bool = True,
    ):
        self.pairs          = pairs or list(PAIR_MAP.keys())
        self.resample_to    = resample_to
        self.session_filter = session_filter
        self._store: dict   = {}

    def load_all(self, use_cache: bool = True) -> dict:
        """Load all pairs. Returns dict of {pair: DataFrame}."""
        failed = []
        for pair in self.pairs:
            try:
                df = load_pair(
                    pair,
                    resample_to    = self.resample_to,
                    session_filter = self.session_filter,
                    use_cache      = use_cache,
                )
                self._store[pair] = df
            except FileNotFoundError as e:
                logger.warning(str(e))
                failed.append(pair)
            except Exception as e:
                logger.error(f"{pair}: failed to load — {e}")
                failed.append(pair)

        if failed:
            logger.warning(f"Could not load: {failed}")

        logger.info(
            f"IntradayForexPipeline ready: "
            f"{len(self._store)} pairs loaded "
            f"({len(failed)} failed)"
        )
        return self._store

    def get(self, pair: str) -> pd.DataFrame:
        """Return DataFrame for a pair."""
        return self._store.get(pair, pd.DataFrame())

    def get_latest_price(self, pair: str) -> float:
        df = self.get(pair)
        return float(df["close"].iloc[-1]) if not df.empty else 0.0

    def available(self) -> list:
        return [p for p, df in self._store.items() if not df.empty]

    def summary(self):
        """Print a summary of loaded data."""
        print(f"\n{'─'*60}")
        print(f"  Intraday Forex Pipeline — {self.resample_to} bars")
        print(f"{'─'*60}")
        for pair, df in self._store.items():
            if df.empty:
                print(f"  {pair:<10} NO DATA")
            else:
                print(
                    f"  {pair:<10} {len(df):>7} bars | "
                    f"{df.index[0].date()} → {df.index[-1].date()}"
                )
        print(f"{'─'*60}\n")

    @property
    def store(self) -> dict:
        return self._store