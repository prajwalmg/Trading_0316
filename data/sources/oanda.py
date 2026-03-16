"""
================================================================
  data/sources/oanda.py
  Fetch OHLCV data from the OANDA v20 REST API.

  Why OANDA for forex:
    - Real bid/ask mid prices — not adjusted/synthetic like yfinance
    - Free with a practice account (same key you use for execution)
    - Up to 5,000 candles per request, paginated for full history
    - Covers all major and minor forex pairs
    - Sub-minute granularity available (S5, S10, S30, M1 ... H4, D)

  Setup:
    1. Open a free OANDA practice account at oanda.com
    2. Generate an API key under My Account → API Access
    3. Set OANDA_API_KEY and OANDA_ACCOUNT_ID in config/settings.py

  Instrument name mapping:
    yfinance "EURUSD=X"  →  OANDA "EUR_USD"
================================================================
"""

import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests
import pandas as pd
import numpy as np

logger = logging.getLogger("trading_firm.data.oanda")

# ── Instrument name map ───────────────────────────────────────
# Maps yfinance-style tickers to OANDA instrument names
OANDA_INSTRUMENT_MAP = {
    "EURUSD=X": "EUR_USD",
    "GBPUSD=X": "GBP_USD",
    "EURGBP=X": "EUR_GBP",
    "EURJPY=X": "EUR_JPY",
    "USDCHF=X": "USD_CHF",
    "EURCHF=X": "EUR_CHF",
    "USDJPY=X": "USD_JPY",
    "AUDUSD=X": "AUD_USD",
    "NZDUSD=X": "NZD_USD",
    "USDCAD=X": "USD_CAD",
}

# ── Granularity map ───────────────────────────────────────────
OANDA_GRANULARITY_MAP = {
    "1m":  "M1",
    "5m":  "M5",
    "15m": "M15",
    "30m": "M30",
    "1h":  "H1",
    "4h":  "H4",
    "1d":  "D",
}

MAX_CANDLES_PER_REQUEST = 5000
BASE_URLS = {
    "practice": "https://api-fxpractice.oanda.com",
    "live":     "https://api-fxtrade.oanda.com",
}


def fetch_oanda(
    ticker:      str,
    interval:    str = "1h",
    days:        int = 730,
    api_key:     Optional[str] = None,
    env:         Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV mid-price candles from OANDA.

    Parameters
    ----------
    ticker   : yfinance-style ticker e.g. "EURUSD=X"
    interval : "1m","5m","15m","30m","1h","4h","1d"
    days     : how many calendar days of history to fetch
    api_key  : OANDA API key (falls back to settings.OANDA_API_KEY)
    env      : "practice" or "live" (falls back to settings.OANDA_ENV)

    Returns
    -------
    pd.DataFrame with columns [open, high, low, close, volume]
    index = timezone-naive UTC datetime
    """
    try:
        from config.settings import OANDA_API_KEY, OANDA_ENV
        api_key = api_key or OANDA_API_KEY
        env     = env     or OANDA_ENV
    except ImportError:
        pass

    if not api_key or api_key == "YOUR_KEY":
        logger.warning(
            "OANDA API key not set — forex will fall back to yfinance.\n"
            "Add OANDA_API_KEY = 'your_key' to config/settings.py"
        )
        return pd.DataFrame()

    instrument = OANDA_INSTRUMENT_MAP.get(ticker)
    if not instrument:
        logger.debug(f"No OANDA instrument mapping for {ticker}")
        return pd.DataFrame()

    granularity = OANDA_GRANULARITY_MAP.get(interval, "H1")
    base_url    = BASE_URLS.get(env or "practice")

    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
        "Accept-Datetime-Format": "RFC3339",
    })

    # Paginate forward from start_dt to now in chunks of MAX_CANDLES
    start_dt     = datetime.now(timezone.utc) - timedelta(days=days)
    end_dt       = datetime.now(timezone.utc)
    current_from = start_dt
    all_rows: list = []

    while current_from < end_dt:
        params = {
            "granularity": granularity,
            "count":       MAX_CANDLES_PER_REQUEST,
            "from":        current_from.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
            "price":       "M",   # mid prices
        }

        try:
            r = session.get(
                f"{base_url}/v3/instruments/{instrument}/candles",
                params=params,
                timeout=30,
            )
            r.raise_for_status()
            candles = r.json().get("candles", [])
        except requests.HTTPError as e:
            logger.error(f"OANDA HTTP error ({ticker}): {e.response.status_code} — {e}")
            break
        except Exception as e:
            logger.error(f"OANDA fetch error ({ticker}): {e}")
            break

        if not candles:
            break

        for c in candles:
            if not c.get("complete", True):
                continue
            mid = c.get("mid", {})
            try:
                all_rows.append({
                    "time":   pd.to_datetime(c["time"]),
                    "open":   float(mid["o"]),
                    "high":   float(mid["h"]),
                    "low":    float(mid["l"]),
                    "close":  float(mid["c"]),
                    "volume": int(c.get("volume", 0)),
                })
            except (KeyError, ValueError):
                continue

        # Advance window to just after the last candle returned
        last_time = pd.to_datetime(candles[-1]["time"])
        if last_time.tzinfo is None:
            last_time = last_time.tz_localize(timezone.utc)
        current_from = last_time + timedelta(seconds=1)

        # Respect OANDA rate limit (120 requests/min)
        time.sleep(0.05)

    if not all_rows:
        logger.warning(f"OANDA returned no data for {ticker}")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows).set_index("time")
    df.index = pd.to_datetime(df.index).tz_localize(None)   # naive UTC
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # Trim to requested range
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    df     = df[df.index >= cutoff]

    logger.info(f"OANDA {ticker}: {len(df)} {interval} bars  "
                f"({df.index[0].date()} → {df.index[-1].date()})")
    return df
