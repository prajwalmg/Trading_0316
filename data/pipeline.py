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
import time
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
    ASSET_CLASS_MAP, GRANULARITY, LOOKBACK_PERIOD, DAILY_PERIOD,
    MIN_BARS, DATA_CACHE_DIR,
    FOREX_PAIRS, EQUITY_TICKERS, COMMODITY_TICKERS, CRYPTO_TICKERS,
)

logger = logging.getLogger("trading_firm.data")

# ── Asset class mapping ───────────────────────────────────────
ASSET_CLASS = {}
for t in FOREX_PAIRS:       ASSET_CLASS[t] = "forex"
for t in EQUITY_TICKERS:    ASSET_CLASS[t] = "equity"
for t in COMMODITY_TICKERS: ASSET_CLASS[t] = "commodity"
for t in CRYPTO_TICKERS:    ASSET_CLASS[t] = "crypto"

# ── Cache helpers ─────────────────────────────────────────────

def _cache_path(ticker: str, interval: str, period: str) -> str:
    key  = hashlib.md5(f"{ticker}{interval}{period}".encode()).hexdigest()[:8]
    name = ticker.replace("=", "").replace("/", "")
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    return os.path.join(DATA_CACHE_DIR, f"{name}_{interval}_{key}.parquet")


def _cache_valid(path: str, max_age_minutes: int = 90) -> bool:
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
        import time as _time
        raw = pd.DataFrame()
        for attempt in range(3):
            try:
                raw = yf.download(ticker, interval=interval, period=period,
                          progress=False, auto_adjust=True, multi_level_index=False)
                if not raw.empty:
                    break
                _time.sleep(2 ** attempt)   # 1s, 2s, 4s backoff
            except Exception as _e:
                logger.warning(f"{ticker}: attempt {attempt+1} failed — {_e}")
                _time.sleep(2 ** attempt)
        if raw.empty:
            raise ValueError(f"No data returned for {ticker}")

        #raw.columns = [c.lower() for c in raw.columns]
        #df = raw[["open", "high", "low", "close", "volume"]].copy()
        
        # Flatten MultiIndex if present, then lowercase
        if hasattr(raw.columns, "levels"):
            raw.columns = raw.columns.get_level_values(0)
        raw.columns = [str(c).lower().strip() for c in raw.columns]

        # Rename any variant column names
        raw = raw.rename(columns={
            "adj close": "close",
            "adjclose":  "close",
        })
        df = raw[[c for c in ["open", "high", "low", "close", "volume"] if c in raw.columns]].copy()

        # Add volume column with zeros if missing (forex has no volume)
        if "volume" not in df.columns:
            df["volume"] = 0

        
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

def fetch_macro_context(use_cache: bool = True) -> dict:
    """
    Fetch VIX and DXY daily bars.
    Returns dict with keys 'vix' and 'dxy' as pd.Series of close prices.
    """
    macro = {}
    tickers = {"vix": "^VIX", "dxy": "DX-Y.NYB"}
    
    for name, ticker in tickers.items():
        try:
            df = fetch_yfinance(
                ticker, interval="1d", period="730d",
                use_cache=use_cache
            )
            if not df.empty:
                macro[name] = df["close"]
                logger.info(f"Macro {name}: {len(df)} daily bars loaded")
            else:
                macro[name] = None
                logger.warning(f"Macro {name}: no data returned")
        except Exception as e:
            macro[name] = None
            logger.warning(f"Macro {name}: fetch failed — {e}")
    
    return macro

def update_intraday_cache(pair: str, yf_ticker: str, is_crypto: bool = False) -> pd.DataFrame:
    """
    Fetch latest 5-min bars and append to local parquet cache.
    Run daily to build up history beyond yfinance's 60-day limit.
    Call this at the start of each paper trading cycle.
    """
    cache_path = f"data/cache/intraday/{pair}_5min.parquet"
    os.makedirs("data/cache/intraday", exist_ok=True)

    # Load existing cache
    if os.path.exists(cache_path):
        existing  = pd.read_parquet(cache_path)
        last_date = existing.index[-1]
        logger.info(f"{pair}: cache loaded — {len(existing)} bars, last={last_date.date()}")
    else:
        existing  = pd.DataFrame()
        last_date = None

    # Fetch latest 7 days from yfinance
    for attempt in range(3):
        try:
            df_new = yf.download(
                yf_ticker, interval="5m", period="7d",
                progress=False, multi_level_index=False
            )
            if not df_new.empty:
                break
        except Exception as e:
            logger.warning(f"{pair}: cache update attempt {attempt+1} failed — {e}")
            time.sleep(2 ** attempt)
    else:
        logger.warning(f"{pair}: cache update failed — using existing cache")
        return existing if not existing.empty else pd.DataFrame()

    df_new.columns = [c.lower() for c in df_new.columns]
    df_new.index   = df_new.index.tz_localize(None)

    # Session filter for forex
    if not is_crypto:
        df_new = df_new[
            (df_new.index.weekday < 5) &
            (df_new.index.hour >= 7)   &
            (df_new.index.hour < 21)
        ]

    # Append only new bars
    if last_date is not None:
        df_new = df_new[df_new.index > last_date]

    if df_new.empty:
        logger.info(f"{pair}: cache already up to date")
        return existing

    combined = pd.concat([existing, df_new])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.to_parquet(cache_path)

    logger.info(
        f"{pair}: cache updated — added {len(df_new)} bars | "
        f"total={len(combined)} | now → {combined.index[-1].date()}"
    )
    return combined


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
        use_cache: bool = True,
    ) -> dict:
        """
        Fetch/refresh data for all (or given) tickers.
        Returns dict of {ticker: DataFrame}.
        """
        tickers = tickers or (FOREX_PAIRS + EQUITY_TICKERS + COMMODITY_TICKERS + CRYPTO_TICKERS)
        failed  = []

        for ticker in tickers:
            from config.settings import DATA_PERIODS, ASSET_CLASS_MAP
            asset_class  = ASSET_CLASS_MAP.get(ticker, "equity")
            data_period  = DATA_PERIODS.get(asset_class, period)
            df = fetch_yfinance(ticker, interval=interval, period=data_period, use_cache=use_cache)

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
