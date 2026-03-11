"""
================================================================
  data/pipeline.py
  Multi-source data pipeline:
    - Primary:   yfinance (Forex, Equities, Commodities)
    - Secondary: Histdata CSV loader (Forex M1 backup)
    - Caching:   local parquet cache to avoid re-downloading
    - Validation: spike detection, gap filling, quality checks
================================================================
"""

import os
import hashlib
import logging
import warnings
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.settings import (
    GRANULARITY, LOOKBACK_PERIOD, DAILY_PERIOD,
    MIN_BARS, DATA_CACHE_DIR,
    FOREX_PAIRS, EQUITY_TICKERS, COMMODITY_TICKERS,
)

logger = logging.getLogger("trading_firm.data")

# ── Asset class mapping ───────────────────────────────────────
ASSET_CLASS = {}
for t in FOREX_PAIRS:       ASSET_CLASS[t] = "forex"
for t in EQUITY_TICKERS:    ASSET_CLASS[t] = "equity"
for t in COMMODITY_TICKERS: ASSET_CLASS[t] = "commodity"


# ── Cache helpers ─────────────────────────────────────────────

def _cache_path(ticker: str, interval: str, period: str) -> str:
    key  = hashlib.md5(f"{ticker}{interval}{period}".encode()).hexdigest()[:8]
    name = ticker.replace("=", "").replace("/", "")
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    return os.path.join(DATA_CACHE_DIR, f"{name}_{interval}_{key}.parquet")


def _cache_valid(path: str, max_age_minutes: int = 15) -> bool:
    if not os.path.exists(path):
        return False
    age = datetime.now().timestamp() - os.path.getmtime(path)
    return age < max_age_minutes * 60


# ── Data quality ──────────────────────────────────────────────

def _validate_and_clean(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Run quality checks on raw OHLCV data:
      1. Remove duplicate timestamps
      2. Forward-fill small gaps (≤ 3 bars)
      3. Remove price spikes (> 5σ from rolling mean)
      4. Ensure OHLCV logic (high ≥ low, close within high-low)
      5. Drop rows with zero volume (for equities)
    """
    if df.empty:
        return df

    # 1. Dedup
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()

    # 2. Forward-fill gaps ≤ 3 bars
    df = df.ffill(limit=3)

    # 3. Spike detection on close price
    roll_mean = df["close"].rolling(20, min_periods=5).mean()
    roll_std  = df["close"].rolling(20, min_periods=5).std()
    z_score   = (df["close"] - roll_mean).abs() / (roll_std + 1e-9)
    spikes    = z_score > 5
    if spikes.sum() > 0:
        logger.warning(f"{ticker}: removed {spikes.sum()} spike bars")
        df = df[~spikes]

    # 4. OHLCV logic validation
    invalid = (df["high"] < df["low"]) | (df["close"] > df["high"] * 1.01)
    if invalid.sum() > 0:
        logger.warning(f"{ticker}: removed {invalid.sum()} invalid OHLC bars")
        df = df[~invalid]

    # 5. Drop zero-volume rows (equity market close bars)
    asset = ASSET_CLASS.get(ticker, "equity")
    if asset == "equity":
        df = df[df["volume"] > 0]

    return df.dropna(subset=["open", "high", "low", "close"])


# ── Primary fetcher: yfinance ─────────────────────────────────

def fetch_yfinance(
    ticker:   str,
    interval: str = GRANULARITY,
    period:   str = LOOKBACK_PERIOD,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Yahoo Finance with local caching.

    Parameters
    ----------
    ticker   : Yahoo Finance symbol e.g. "EURUSD=X", "AAPL", "GC=F"
    interval : "1m","5m","15m","30m","60m","1d"
    period   : "7d","60d","1y","2y" etc.
    """
    cache = _cache_path(ticker, interval, period)

    if use_cache and _cache_valid(cache):
        df = pd.read_parquet(cache)
        logger.debug(f"{ticker}: loaded from cache ({len(df)} bars)")
        return df

    try:
        raw = yf.download(ticker, interval=interval, period=period,
                          progress=False, auto_adjust=True, multi_level_index=False)
        if raw.empty:
            raise ValueError(f"No data returned for {ticker}")

        raw.columns = [c.lower() for c in raw.columns]
        df = raw[["open", "high", "low", "close", "volume"]].copy()
        df.index.name = "time"
        df.index = pd.to_datetime(df.index)

        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df = _validate_and_clean(df, ticker)

        if len(df) < MIN_BARS:
            logger.warning(f"{ticker}: only {len(df)} bars after cleaning "
                           f"(min={MIN_BARS})")

        df.to_parquet(cache)
        logger.info(f"{ticker}: fetched {len(df)} bars @ {interval}")
        return df

    except Exception as e:
        logger.error(f"{ticker}: yfinance fetch failed — {e}")
        if os.path.exists(cache):
            logger.warning(f"{ticker}: falling back to stale cache")
            return pd.read_parquet(cache)
        return pd.DataFrame()


# ── Secondary fetcher: Histdata CSV ──────────────────────────

def load_histdata_csv(
    path_pattern: str,
    resample_to:  str = "5min",
) -> pd.DataFrame:
    """
    Load one or more Histdata.com M1 CSV files.
    Supports glob patterns: e.g. "data/DAT_MT_EURUSD_M1_*.csv"

    Parameters
    ----------
    path_pattern : file path or glob pattern
    resample_to  : target timeframe e.g. "5min","15min","1h"
    """
    import glob as _glob

    files = sorted(_glob.glob(path_pattern))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found matching: {path_pattern}\n"
            f"Download from https://www.histdata.com"
        )

    dfs = []
    for f in files:
        try:
            tmp = pd.read_csv(
                f, sep="\t", header=None,
                names=["date", "time", "open", "high", "low", "close", "volume"],
            )
            tmp.index = pd.to_datetime(
                tmp["date"] + " " + tmp["time"],
                format="%Y.%m.%d %H:%M"
            )
            tmp = tmp.drop(columns=["date", "time"]).astype(float)
            dfs.append(tmp)
            logger.debug(f"Loaded histdata: {f} ({len(tmp)} bars)")
        except Exception as e:
            logger.warning(f"Could not load {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs).sort_index()
    df = df[~df.index.duplicated(keep="first")]

    if resample_to and resample_to != "1min":
        df = df.resample(resample_to).agg({
            "open":   "first",
            "high":   "max",
            "low":    "min",
            "close":  "last",
            "volume": "sum",
        }).dropna()

    logger.info(
        f"Histdata loaded: {len(df)} {resample_to} bars "
        f"| {df.index[0].date()} → {df.index[-1].date()}"
    )
    return df


# ── Bulk fetcher ──────────────────────────────────────────────

class DataPipeline:
    """
    Manages data fetching for all instruments in the universe.
    Maintains an in-memory store of latest OHLCV DataFrames.
    """

    def __init__(self):
        self._store: dict[str, pd.DataFrame] = {}

    def refresh_all(
        self,
        tickers:  list = None,
        interval: str  = GRANULARITY,
        period:   str  = LOOKBACK_PERIOD,
    ) -> dict:
        """
        Fetch/refresh data for all (or given) tickers.
        Returns dict of {ticker: DataFrame}.
        """
        tickers = tickers or (FOREX_PAIRS + EQUITY_TICKERS + COMMODITY_TICKERS)
        failed  = []

        for ticker in tickers:
            df = fetch_yfinance(ticker, interval=interval, period=period)
            if df.empty:
                failed.append(ticker)
                logger.warning(f"No data for {ticker} — skipping")
            else:
                self._store[ticker] = df

        if failed:
            logger.warning(f"Failed to fetch: {failed}")

        logger.info(
            f"DataPipeline refreshed: {len(self._store)} instruments "
            f"({len(failed)} failed)"
        )
        return self._store

    def get(self, ticker: str) -> pd.DataFrame:
        """Return cached DataFrame for a ticker."""
        return self._store.get(ticker, pd.DataFrame())

    def get_latest_price(self, ticker: str) -> float:
        """Return the most recent close price."""
        df = self.get(ticker)
        if df.empty:
            return 0.0
        return float(df["close"].iloc[-1])

    def available(self) -> list:
        """Return list of tickers with valid data."""
        return [t for t, df in self._store.items() if not df.empty]

    @property
    def store(self) -> dict:
        return self._store
