"""data/crypto_data.py — Crypto OHLCV via CCXT (Binance) with yfinance fallback."""
import os, hashlib, logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger("trading_firm.crypto_data")
CACHE_DIR = "data/cache/crypto"

# yfinance ticker → CCXT/Binance symbol
YF_TO_CCXT = {
    "BTC-USD":   "BTC/USDT",
    "ETH-USD":   "ETH/USDT",
    "SOL-USD":   "SOL/USDT",
    "BNB-USD":   "BNB/USDT",
    "XRP-USD":   "XRP/USDT",
    "ADA-USD":   "ADA/USDT",
    "AVAX-USD":  "AVAX/USDT",
    "MATIC-USD": "MATIC/USDT",
}

CCXT_TF = {"1h": "1h", "4h": "4h", "1d": "1d", "5m": "5m", "15m": "15m"}


def _cache_path(ticker: str, interval: str, years: int) -> Path:
    key = hashlib.md5(f"{ticker}_{interval}_{years}".encode()).hexdigest()
    return Path(CACHE_DIR) / f"{key}.parquet"


def _fetch_ccxt(ccxt_sym: str, interval: str = "1h", years: int = 2) -> pd.DataFrame:
    """Fetch OHLCV from Binance via CCXT with pagination."""
    try:
        import ccxt
        exchange = ccxt.binance({"enableRateLimit": True})
        tf = CCXT_TF.get(interval, "1h")

        # Estimate start timestamp
        limit_per_call = 1000
        since_dt = datetime.now(timezone.utc) - pd.Timedelta(days=365 * years + 1)
        since_ms  = int(since_dt.timestamp() * 1000)

        all_rows = []
        while True:
            batch = exchange.fetch_ohlcv(ccxt_sym, timeframe=tf, since=since_ms, limit=limit_per_call)
            if not batch:
                break
            all_rows.extend(batch)
            if len(batch) < limit_per_call:
                break
            since_ms = batch[-1][0] + 1

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()
        df.index = df.index.tz_convert(None)
        df = df[~df.index.duplicated(keep="last")]
        return df

    except Exception as e:
        logger.warning(f"CCXT {ccxt_sym}: {e}")
        return pd.DataFrame()


def _fetch_yfinance(ticker: str, interval: str = "1h", years: int = 2) -> pd.DataFrame:
    """Fallback: fetch crypto OHLCV from yfinance."""
    try:
        import yfinance as yf
        df = yf.download(
            ticker, period=f"{years * 365}d", interval=interval,
            progress=False, auto_adjust=True,
        )
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df[["open", "high", "low", "close", "volume"]].dropna()
    except Exception as e:
        logger.warning(f"yfinance {ticker}: {e}")
        return pd.DataFrame()


def fetch_crypto_ohlcv(ticker: str, interval: str = "1h", years: int = 2) -> pd.DataFrame:
    """Fetch crypto OHLCV: try CCXT/Binance, fall back to yfinance.

    Args:
        ticker:   yfinance-style ticker, e.g. 'BTC-USD'
        interval: '1h', '4h', '1d', '5m', '15m'
        years:    history depth in years

    Returns:
        DataFrame with columns [open, high, low, close, volume], DatetimeIndex (UTC-naive)
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = _cache_path(ticker, interval, years)

    # Return cache if < 4 hours old
    if cache_file.exists():
        age_h = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).total_seconds() / 3600
        if age_h < 4:
            try:
                return pd.read_parquet(cache_file)
            except Exception:
                pass

    ccxt_sym = YF_TO_CCXT.get(ticker)
    df = _fetch_ccxt(ccxt_sym, interval, years) if ccxt_sym else pd.DataFrame()

    if df.empty:
        logger.info(f"{ticker}: CCXT failed or unavailable — falling back to yfinance")
        df = _fetch_yfinance(ticker, interval, years)

    if not df.empty:
        try:
            df.to_parquet(cache_file)
        except Exception:
            pass
        logger.info(f"{ticker}: {len(df)} {interval} bars fetched")
    else:
        logger.warning(f"{ticker}: no data from any source")

    return df


class CryptoDataClient:
    """Class wrapper around module-level functions for compatibility."""

    def fetch_ohlcv(self, symbol: str, interval: str = "1h", years: int = 2) -> pd.DataFrame:
        """symbol is yfinance-style e.g. 'BTC-USD'"""
        return fetch_crypto_ohlcv(symbol, interval, years)

    def get_cached(self, symbol: str, interval: str = "1h") -> pd.DataFrame:
        """Return cached data if available and < 4h old, else empty DataFrame."""
        import os, hashlib
        from pathlib import Path
        cache_file = _cache_path(symbol, interval, 2)  # 2 years default
        if cache_file.exists():
            age_h = (datetime.now() - datetime.fromtimestamp(
                cache_file.stat().st_mtime)).total_seconds() / 3600
            if age_h < 4:
                try:
                    return pd.read_parquet(cache_file)
                except Exception:
                    pass
        return pd.DataFrame()
