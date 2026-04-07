"""data/massive.py — Polygon.io (massive v2.4.0) OHLCV client with yfinance fallback."""
import hashlib
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger("trading_firm.massive")

CACHE_DIR = Path("data/cache/massive")

# Polygon crypto tickers (X:BTCUSD) ↔ yfinance style (BTC-USD)
YF_TO_POLYGON = {
    "BTC-USD":   "X:BTCUSD",
    "ETH-USD":   "X:ETHUSD",
    "SOL-USD":   "X:SOLUSD",
    "BNB-USD":   "X:BNBUSD",
    "XRP-USD":   "X:XRPUSD",
    "ADA-USD":   "X:ADAUSD",
    "AVAX-USD":  "X:AVAXUSD",
    "MATIC-USD": "X:MATICUSD",
}
POLYGON_TO_YF = {v: k for k, v in YF_TO_POLYGON.items()}

# yfinance interval strings for fallback
_YF_INTERVAL = {
    ("hour",   1):  "1h",
    ("hour",   4):  "4h",
    ("minute", 1):  "1m",
    ("minute", 5):  "5m",
    ("minute", 15): "15m",
    ("minute", 30): "30m",
    ("day",    1):  "1d",
    ("week",   1):  "1wk",
    ("month",  1):  "1mo",
}

# Commodity futures — Polygon uses C: prefix (e.g. C:GC for Gold continuous)
_COMMODITY_SUFFIXES = ("=F",)
_COMMODITY_TO_POLYGON = {
    "GC=F": "C:GC",   # Gold continuous
    "CL=F": "C:CL",   # Crude Oil WTI continuous
    "SI=F": "C:SI",   # Silver continuous
    "NG=F": "C:NG",   # Natural Gas continuous
    "ZC=F": "C:ZC",   # Corn continuous
    "ZW=F": "C:ZW",   # Wheat continuous
}


def _is_commodity(ticker: str) -> bool:
    return any(ticker.endswith(s) for s in _COMMODITY_SUFFIXES)


def _cache_path(ticker: str, granularity: str, multiplier: int, years: int) -> Path:
    key = hashlib.md5(f"{ticker}_{granularity}_{multiplier}_{years}".encode()).hexdigest()
    return CACHE_DIR / f"{key}.parquet"


def _polygon_ticker(ticker: str) -> str:
    """Convert yfinance-style ticker to Polygon format (crypto only)."""
    return YF_TO_POLYGON.get(ticker, ticker)


def _yf_ticker(polygon_ticker: str) -> str:
    """Convert Polygon ticker to yfinance format."""
    return POLYGON_TO_YF.get(polygon_ticker, polygon_ticker)


def _yf_interval(granularity: str, multiplier: int) -> str:
    """Map (granularity, multiplier) to a yfinance interval string."""
    return _YF_INTERVAL.get((granularity, multiplier), f"{multiplier}{granularity[0]}")


class MassiveClient:
    """Polygon.io REST client (massive v2.4.0) with caching and yfinance fallback."""

    def __init__(self):
        from config import settings
        from massive import RESTClient  # noqa: PLC0415

        self._api_key = settings.MASSIVE_API_KEY
        self._client = RESTClient(api_key=self._api_key)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_bars(
        self,
        ticker: str,
        granularity: str,
        multiplier: int,
        years: int = 2,
    ) -> pd.DataFrame:
        """Fetch OHLCV bars.

        Parameters
        ----------
        ticker:      yfinance-style symbol, e.g. 'AAPL', 'BTC-USD', 'GC=F'
        granularity: 'minute', 'hour', 'day', 'week', 'month'
        multiplier:  bar size multiplier, e.g. 1, 5, 15
        years:       how many years of history to fetch

        Returns
        -------
        DataFrame with columns [open, high, low, close, volume],
        DatetimeIndex UTC-naive.  Empty DataFrame on total failure.
        """
        cache_file = _cache_path(ticker, granularity, multiplier, years)

        # Serve from cache if < 4 hours old
        if cache_file.exists():
            age_h = (
                datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            ).total_seconds() / 3600
            if age_h < 4:
                try:
                    logger.debug(f"Cache hit for {ticker} ({multiplier}{granularity})")
                    return pd.read_parquet(cache_file)
                except Exception:
                    pass

        df = pd.DataFrame()

        # Commodities: try Polygon (C: prefix) first, fall back to yfinance
        if _is_commodity(ticker):
            poly_ticker = _COMMODITY_TO_POLYGON.get(ticker)
            if poly_ticker:
                logger.info(f"{ticker}: commodity — trying Polygon ({poly_ticker})")
                df = self._fetch_polygon(poly_ticker, granularity, multiplier, years)
            if df.empty:
                logger.info(f"{ticker}: Polygon failed — using yfinance")
                df = self._fetch_yfinance(ticker, granularity, multiplier, years)
        else:
            poly_ticker = _polygon_ticker(ticker)
            df = self._fetch_polygon(poly_ticker, granularity, multiplier, years)
            if df.empty:
                logger.info(
                    f"{ticker}: Polygon failed — falling back to yfinance"
                )
                yf_sym = _yf_ticker(poly_ticker) if poly_ticker.startswith("X:") else ticker
                df = self._fetch_yfinance(yf_sym, granularity, multiplier, years)

        if not df.empty:
            try:
                df.to_parquet(cache_file)
            except Exception as exc:
                logger.warning(f"Cache write failed for {ticker}: {exc}")
            logger.info(f"{ticker}: {len(df)} {multiplier}{granularity} bars fetched")
        else:
            logger.warning(f"{ticker}: no data from any source")

        return df

    def get_snapshot(self, ticker: str) -> dict:
        """Return latest price snapshot for a ticker.

        Uses Polygon's snapshot endpoint.  Returns empty dict on any failure.
        """
        try:
            # Determine asset type and use appropriate snapshot endpoint
            poly_ticker = _polygon_ticker(ticker)
            if poly_ticker.startswith("X:"):
                snap = self._client.get_snapshot_ticker("crypto", poly_ticker)
            else:
                snap = self._client.get_snapshot_ticker("stocks", poly_ticker)

            if snap is None:
                return {}

            # Normalise to a plain dict
            result = {}
            if hasattr(snap, "__dict__"):
                result = {k: v for k, v in vars(snap).items() if not k.startswith("_")}
            elif isinstance(snap, dict):
                result = snap
            return result
        except Exception as exc:
            logger.warning(f"get_snapshot({ticker}): {exc}")
            return {}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_polygon(
        self,
        poly_ticker: str,
        granularity: str,
        multiplier: int,
        years: int,
    ) -> pd.DataFrame:
        """Fetch bars from Polygon.io via the massive RESTClient."""
        try:
            to_date = datetime.utcnow().date()
            from_date = (datetime.utcnow() - timedelta(days=365 * years + 1)).date()

            aggs = self._client.get_aggs(
                poly_ticker,
                multiplier,
                granularity,
                str(from_date),
                str(to_date),
                limit=50000,
                adjusted=True,
            )

            rows = []
            for bar in aggs:
                rows.append(
                    {
                        "timestamp": bar.timestamp,
                        "open":      bar.open,
                        "high":      bar.high,
                        "low":       bar.low,
                        "close":     bar.close,
                        "volume":    bar.volume,
                    }
                )

            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.set_index("timestamp").sort_index()
            df.index = df.index.tz_convert(None)   # UTC-naive
            df = df[~df.index.duplicated(keep="last")]
            return df[["open", "high", "low", "close", "volume"]].dropna()

        except Exception as exc:
            logger.warning(f"Polygon {poly_ticker} ({multiplier}{granularity}): {exc}")
            return pd.DataFrame()

    def _fetch_yfinance(
        self,
        ticker: str,
        granularity: str,
        multiplier: int,
        years: int,
    ) -> pd.DataFrame:
        """Fallback: fetch OHLCV from yfinance."""
        try:
            import yfinance as yf  # noqa: PLC0415

            interval = _yf_interval(granularity, multiplier)
            df = yf.download(
                ticker,
                period=f"{years * 365}d",
                interval=interval,
                progress=False,
                auto_adjust=True,
            )
            if df.empty:
                return pd.DataFrame()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() for c in df.columns]
            df.index = pd.to_datetime(df.index).tz_localize(None)
            return df[["open", "high", "low", "close", "volume"]].dropna()
        except Exception as exc:
            logger.warning(f"yfinance {ticker} ({multiplier}{granularity}): {exc}")
            return pd.DataFrame()
