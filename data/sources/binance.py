"""
================================================================
  data/sources/binance.py
  Fetch OHLCV data from the Binance public REST API.

  Why Binance for crypto:
    - NO API KEY required for historical OHLCV
    - Free, unlimited, professional-grade data
    - 1-minute bars going back years (BTC from 2017, alts from listing)
    - Real trade-based OHLCV (not synthetic)
    - 1,000 candles per request, paginated automatically

  Binance kline endpoint:
    GET https://api.binance.com/api/v3/klines

  No account, no registration required.

  Ticker mapping:
    yfinance "XRP-USD"  →  Binance "XRPUSDT"
    yfinance "BNB-USD"  →  Binance "BNBUSDT"
================================================================
"""

import logging
import time
from datetime import datetime, timezone, timedelta

import requests
import pandas as pd

logger = logging.getLogger("trading_firm.data.binance")

BASE_URL = "https://api.binance.com/api/v3/klines"

# ── Symbol map ────────────────────────────────────────────────
BINANCE_SYMBOL_MAP = {
    "XRP-USD":  "XRPUSDT",
    "BNB-USD":  "BNBUSDT",
    "BTC-USD":  "BTCUSDT",
    "ETH-USD":  "ETHUSDT",
    "SOL-USD":  "SOLUSDT",
    "ADA-USD":  "ADAUSDT",
    "DOGE-USD": "DOGEUSDT",
}

# ── Interval map ──────────────────────────────────────────────
BINANCE_INTERVAL_MAP = {
    "1m":  "1m",
    "5m":  "5m",
    "15m": "15m",
    "30m": "30m",
    "1h":  "1h",
    "4h":  "4h",
    "1d":  "1d",
}

MAX_CANDLES_PER_REQUEST = 1000


def fetch_binance(
    ticker:   str,
    interval: str = "1h",
    days:     int = 730,
) -> pd.DataFrame:
    """
    Fetch OHLCV from Binance public REST API. No key required.

    Paginates forward from (now - days) to now in chunks of 1,000
    candles to build the full history.

    Parameters
    ----------
    ticker   : yfinance-style ticker e.g. "XRP-USD", "BNB-USD"
    interval : "1m","5m","15m","30m","1h","4h","1d"
    days     : how many calendar days of history to fetch

    Returns
    -------
    pd.DataFrame with columns [open, high, low, close, volume]
    index = timezone-naive UTC datetime
    """
    symbol = BINANCE_SYMBOL_MAP.get(ticker)
    if not symbol:
        logger.debug(f"No Binance symbol mapping for {ticker}")
        return pd.DataFrame()

    b_interval  = BINANCE_INTERVAL_MAP.get(interval, "1h")
    start_dt    = datetime.now(timezone.utc) - timedelta(days=days)
    end_dt      = datetime.now(timezone.utc)
    start_ms    = int(start_dt.timestamp() * 1000)
    end_ms      = int(end_dt.timestamp()   * 1000)

    all_rows: list = []
    current_start_ms = start_ms
    session = requests.Session()

    while current_start_ms < end_ms:
        params = {
            "symbol":    symbol,
            "interval":  b_interval,
            "startTime": current_start_ms,
            "endTime":   end_ms,
            "limit":     MAX_CANDLES_PER_REQUEST,
        }

        try:
            r = session.get(BASE_URL, params=params, timeout=30)
            r.raise_for_status()
            candles = r.json()
        except requests.HTTPError as e:
            logger.error(f"Binance HTTP error ({ticker}): {e.response.status_code}")
            break
        except Exception as e:
            logger.error(f"Binance fetch error ({ticker}): {e}")
            break

        if not candles:
            break

        all_rows.extend(candles)

        # Advance start to just after the last candle returned
        # Binance kline: index 0 = open_time_ms, index 6 = close_time_ms
        last_close_ms  = int(candles[-1][6])
        current_start_ms = last_close_ms + 1

        if len(candles) < MAX_CANDLES_PER_REQUEST:
            break   # got fewer than max → reached end of available data

        time.sleep(0.05)   # stay well within Binance rate limit (1200/min)

    if not all_rows:
        logger.warning(f"Binance returned no data for {ticker}")
        return pd.DataFrame()

    # Binance kline columns:
    # [0]open_time [1]open [2]high [3]low [4]close [5]volume
    # [6]close_time [7]quote_vol [8]n_trades [9]taker_buy_base
    # [10]taker_buy_quote [11]ignore
    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "n_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])

    df["time"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.set_index("time")[["open", "high", "low", "close", "volume"]]
    df = df.astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # Trim to requested window
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    df = df[df.index >= cutoff]

    logger.info(f"Binance {ticker}: {len(df)} {interval} bars  "
                f"({df.index[0].date()} → {df.index[-1].date()})")
    return df
