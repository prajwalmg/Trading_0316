"""
================================================================
  data/twelvedata.py — Twelve Data REST API client

  Rate limits (enforced internally):
    Free plan  : 8 req/min,    800 credits/day
    Grow plan  : 60 req/min, 50 000 credits/day
    Pro plan   : 120 req/min, unlimited credits/day

  Usage:
    client = TwelveDataClient()                    # reads env key
    client = TwelveDataClient(api_key="abc123")    # explicit key

    df = client.fetch_ohlcv("EUR/USD", interval="1h", outputsize=5000)
    cal = client.fetch_economic_calendar()

  outputsize:
    Free : max ~5000 bars per request (~7 months of 1h)
    Paid : up to 50 000 bars (~5 years of 1h)

  Symbol map covers all system instruments — pass internal
  tickers (e.g. "EURUSD=X") or Twelve Data symbols ("EUR/USD").
================================================================
"""
import os
import time
import logging
import threading
from datetime import datetime, timezone
from typing import Optional

import numpy  as np
import pandas as pd

logger = logging.getLogger("trading_firm.twelvedata")

# ── Symbol mapping: internal ticker → Twelve Data symbol ──────
SYMBOL_MAP: dict = {
    # Forex majors
    "EURUSD=X": "EUR/USD",
    "GBPUSD=X": "GBP/USD",
    "USDJPY=X": "USD/JPY",
    "USDCHF=X": "USD/CHF",
    "AUDUSD=X": "AUD/USD",
    "NZDUSD=X": "NZD/USD",
    "USDCAD=X": "USD/CAD",
    # Forex crosses
    "GBPJPY=X": "GBP/JPY",
    "EURJPY=X": "EUR/JPY",
    "EURGBP=X": "EUR/GBP",
    "EURCHF=X": "EUR/CHF",
    "EURAUD=X": "EUR/AUD",
    "EURCAD=X": "EUR/CAD",
    "GBPCHF=X": "GBP/CHF",
    "GBPAUD=X": "GBP/AUD",
    "GBPCAD=X": "GBP/CAD",
    "AUDJPY=X": "AUD/JPY",
    "AUDCHF=X": "AUD/CHF",
    "AUDCAD=X": "AUD/CAD",
    "CADJPY=X": "CAD/JPY",
    "CHFJPY=X": "CHF/JPY",
    "NZDJPY=X": "NZD/JPY",
    "AUDNZD=X": "AUD/NZD",
    "NZDCAD=X": "NZD/CAD",
    # Commodities
    "GC=F":    "XAU/USD",
    "SI=F":    "XAG/USD",
    "CL=F":    "WTI/USD",
    # Crypto
    "BTC-USD": "BTC/USD",
    "ETH-USD": "ETH/USD",
    "SOL-USD": "SOL/USD",
    "XRP-USD": "XRP/USD",
    # Equities (pass symbol directly — no mapping needed)
    "AAPL":    "AAPL",
    "NVDA":    "NVDA",
    "TSLA":    "TSLA",
    "GS":      "GS",
    "JPM":     "JPM",
    "MSFT":    "MSFT",
    "AMZN":    "AMZN",
}

# Reverse map: Twelve Data symbol → internal ticker
_REVERSE_MAP: dict = {v: k for k, v in SYMBOL_MAP.items()}

# ── Interval translation ───────────────────────────────────────
# Internal granularity strings → Twelve Data interval strings
INTERVAL_MAP: dict = {
    "1m":  "1min",
    "5m":  "5min",
    "15m": "15min",
    "30m": "30min",
    "1h":  "1h",
    "4h":  "4h",
    "1d":  "1day",
    "1w":  "1week",
}

# ── Cache TTLs (seconds) ───────────────────────────────────────
_CACHE_TTL_TRAIN = 86_400   # 24 h for training data
_CACHE_TTL_LIVE  = 14_400   # 4 h for live prices (swing cycle = 1h, no need to re-fetch every 5min)

_BASE_URL = "https://api.twelvedata.com"


class _RateLimiter:
    """
    Token-bucket rate limiter.
    Enforces max_per_minute requests per 60-second sliding window.
    """

    def __init__(self, max_per_minute: int = 8):
        self._max      = max_per_minute
        self._window   = 60.0
        self._calls    = []
        self._lock     = threading.Lock()

    def wait(self):
        with self._lock:
            now = time.monotonic()
            # Drop calls older than the window
            self._calls = [t for t in self._calls if now - t < self._window]
            if len(self._calls) >= self._max:
                sleep_for = self._window - (now - self._calls[0]) + 0.05
                logger.debug(f"TwelveData rate-limit: sleeping {sleep_for:.2f}s")
                time.sleep(sleep_for)
                now = time.monotonic()
                self._calls = [t for t in self._calls if now - t < self._window]
            self._calls.append(time.monotonic())


class TwelveDataClient:
    """
    Twelve Data REST API client with:
      - Automatic rate limiting (8 req/min free, configurable)
      - 24h parquet cache
      - Clean OHLCV output matching the rest of the pipeline
      - Graceful fallback: returns empty DataFrame on any error
    """

    def __init__(
        self,
        api_key:        str = "",
        max_per_minute: int = 8,
        cache_dir:      str = "data/cache/twelvedata",
    ):
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        try:
            from config.settings import TWELVE_DATA_API_KEY
            self._api_key = api_key or TWELVE_DATA_API_KEY
        except ImportError:
            self._api_key = api_key or os.getenv("TWELVE_DATA_API_KEY", "")

        if not self._api_key:
            logger.warning(
                "TwelveDataClient: no API key set. "
                "Set TWELVE_DATA_API_KEY env var or pass api_key= argument."
            )

        self._rate_limiter = _RateLimiter(max_per_minute)
        self._cache_dir    = cache_dir
        os.makedirs(self._cache_dir, exist_ok=True)

    # ── Internal helpers ───────────────────────────────────────

    def _resolve_symbol(self, symbol: str) -> str:
        """Convert internal ticker or raw symbol → Twelve Data symbol."""
        return SYMBOL_MAP.get(symbol, symbol)

    def _cache_path(self, symbol: str, interval: str) -> str:
        safe = symbol.replace("/", "_").replace("=", "_")
        return os.path.join(self._cache_dir, f"{safe}_{interval}.parquet")

    def _cache_valid(self, path: str, ttl: int) -> bool:
        if not os.path.exists(path):
            return False
        age = time.time() - os.path.getmtime(path)
        return age < ttl

    def _get(self, endpoint: str, params: dict) -> dict:
        """Raw GET with rate limiting. Returns parsed JSON dict."""
        try:
            import requests
        except ImportError:
            logger.error("requests not installed — pip install requests")
            return {}

        self._rate_limiter.wait()
        params["apikey"] = self._api_key
        url = f"{_BASE_URL}/{endpoint}"
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") == "error":
                logger.error(
                    f"TwelveData API error [{data.get('code')}]: "
                    f"{data.get('message')}"
                )
                return {}
            return data
        except Exception as e:
            logger.error(f"TwelveData request failed: {e}")
            return {}

    def _to_dataframe(self, values: list) -> pd.DataFrame:
        """
        Convert Twelve Data 'values' list to a clean OHLCV DataFrame.
        The API returns newest-first; we reverse to oldest-first.
        """
        if not values:
            return pd.DataFrame()

        rows = []
        for v in reversed(values):   # oldest → newest
            try:
                rows.append({
                    "time":   pd.to_datetime(v["datetime"]),
                    "open":   float(v["open"]),
                    "high":   float(v["high"]),
                    "low":    float(v["low"]),
                    "close":  float(v["close"]),
                    "volume": float(v.get("volume", 0)),
                })
            except (KeyError, ValueError):
                continue

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("time")
        df.index.name = "time"
        df = df[~df.index.duplicated(keep="first")].sort_index()
        # Drop rows with any zero/NaN OHLC
        df = df.replace(0, np.nan).dropna(subset=["open", "high", "low", "close"])
        return df

    # ── Public API ─────────────────────────────────────────────

    def fetch_ohlcv(
        self,
        symbol:        str,
        interval:      str  = "1h",
        outputsize:    int  = 5_000,
        max_bars:      int  = 25_000,
        use_cache:     bool = True,
        training_mode: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars from Twelve Data with automatic pagination.

        On the free plan each request returns up to 5 000 bars
        (~7 months of 1h data). This method chains multiple requests
        walking backwards in time to build up to `max_bars` of history,
        staying inside the 800 credits/day free-plan budget.

        Parameters
        ----------
        symbol        : internal ticker (e.g. "EURUSD=X") or
                        Twelve Data symbol (e.g. "EUR/USD")
        interval      : "1m","5m","15m","30m","1h","4h","1d"
        outputsize    : bars per page (max 5 000 free, 50 000 paid)
        max_bars      : total bars to accumulate across pages
                        (default 25 000 ≈ 5 years of 1h forex)
        use_cache     : serve from parquet cache if fresh
        training_mode : use 24h cache TTL (vs 5min for live)

        Returns
        -------
        pd.DataFrame  columns: open, high, low, close, volume
                      index:   DatetimeIndex named "time", oldest first
        """
        td_symbol   = self._resolve_symbol(symbol)
        td_interval = INTERVAL_MAP.get(interval, interval)
        cache_path  = self._cache_path(td_symbol, td_interval)
        ttl         = _CACHE_TTL_TRAIN if training_mode else _CACHE_TTL_LIVE

        if use_cache and self._cache_valid(cache_path, ttl):
            try:
                df = pd.read_parquet(cache_path)
                logger.info(
                    f"TwelveData cache hit: {symbol} {interval} "
                    f"({len(df)} bars)"
                )
                return df
            except Exception:
                pass

        if not self._api_key:
            logger.warning(f"TwelveData: no API key — skipping {symbol}")
            return pd.DataFrame()

        # ── Paginated fetch ─────────────────────────────────────
        all_chunks: list[pd.DataFrame] = []
        end_date: str = ""           # empty = "now" for the first page
        page = 0
        max_pages = max(1, (max_bars + outputsize - 1) // outputsize)

        while page < max_pages:
            params: dict = {
                "symbol":     td_symbol,
                "interval":   td_interval,
                "outputsize": outputsize,
                "order":      "desc",   # newest-first per page
            }
            if end_date:
                params["end_date"] = end_date

            logger.info(
                f"TwelveData fetch p{page+1}/{max_pages}: "
                f"{td_symbol} {td_interval}"
                + (f" end={end_date}" if end_date else "")
            )
            data   = self._get("time_series", params)
            values = data.get("values", [])
            if not values:
                break

            chunk = self._to_dataframe(values)
            if chunk.empty:
                break

            all_chunks.append(chunk)

            # Next page: end just before the oldest bar in this chunk
            oldest      = chunk.index[0]   # oldest-first after _to_dataframe
            # Subtract one interval-worth to avoid overlap
            end_date    = oldest.strftime("%Y-%m-%d %H:%M:%S")
            page       += 1

            # Stop if we got fewer bars than requested (hit start of history)
            if len(chunk) < outputsize:
                break

        if not all_chunks:
            logger.warning(f"TwelveData: empty response for {symbol}")
            return pd.DataFrame()

        # Concatenate all pages, deduplicate, sort oldest→newest
        df = pd.concat(all_chunks)
        df = df[~df.index.duplicated(keep="first")].sort_index()
        df = df.iloc[-max_bars:]   # trim to max_bars if over

        # Save to cache
        try:
            df.to_parquet(cache_path)
        except Exception as e:
            logger.warning(f"TwelveData cache write failed: {e}")

        logger.info(
            f"TwelveData: {symbol} {interval} → {len(df):,} bars  "
            f"({df.index[0].date()} → {df.index[-1].date()})  "
            f"[{page} page(s)]"
        )
        return df

    def fetch_economic_calendar(
        self,
        start_date: str = "",
        end_date:   str = "",
    ) -> pd.DataFrame:
        """
        Fetch high-impact economic calendar events from Twelve Data.

        Returns a DataFrame with columns:
          event, country, date, impact, actual, forecast, previous

        Falls back to an empty DataFrame on any error.
        """
        params: dict = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        data = self._get("economic_calendar", params)
        events = data.get("result", {}).get("list", [])
        if not events:
            logger.warning("TwelveData economic calendar: empty response")
            return pd.DataFrame()

        rows = []
        for e in events:
            impact = str(e.get("importance", "")).lower()
            if impact not in ("high", "medium"):
                continue
            rows.append({
                "event":    e.get("event", ""),
                "country":  e.get("country", ""),
                "date":     pd.to_datetime(e.get("date", ""), errors="coerce"),
                "impact":   impact,
                "actual":   e.get("actual",   ""),
                "forecast": e.get("forecast", ""),
                "previous": e.get("previous", ""),
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).dropna(subset=["date"]).sort_values("date")
        logger.info(f"TwelveData calendar: {len(df)} medium/high-impact events")
        return df

    def available_symbols(self) -> pd.DataFrame:
        """List all symbols available on the account's plan."""
        data = self._get("stocks", {"exchange": "NASDAQ"})
        return pd.DataFrame(data.get("data", []))
