"""
================================================================
  data/sources/alpaca.py
  Fetch OHLCV data from the Alpaca Markets historical data API.

  Why Alpaca for equities:
    - Free tier: unlimited historical bars for all US equities
    - Real exchange data (SIP feed), not synthetic Yahoo adjustments
    - Minute bars available with correct split/dividend adjustments
    - Same API key you already use for paper trading execution
    - Crypto also available 24/7

  Free tier limits:
    - 200 requests/min
    - Data delayed 15 min on free plan for live; historical is instant
    - Up to 10,000 bars per request

  Setup:
    1. Sign up free at alpaca.markets
    2. Generate API keys in the dashboard
    3. Set ALPACA_API_KEY and ALPACA_SECRET in config/settings.py

  Data endpoint is separate from trading endpoint:
    Trading: paper-api.alpaca.markets  (or live-api.alpaca.markets)
    Data:    data.alpaca.markets        ← this module uses this
================================================================
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger("trading_firm.data.alpaca")

# ── Timeframe map ─────────────────────────────────────────────
ALPACA_TIMEFRAME_MAP = {
    "1m":  "1Min",
    "5m":  "5Min",
    "15m": "15Min",
    "30m": "30Min",
    "1h":  "1Hour",
    "4h":  "4Hour",
    "1d":  "1Day",
}


def fetch_alpaca(
    ticker:     str,
    interval:   str = "1h",
    days:       int = 730,
    api_key:    Optional[str] = None,
    secret_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV from Alpaca v2 historical bars API.

    Works for US equities (NVDA, AAPL, etc.) and crypto
    (BTC/USD, ETH/USD) without a paid subscription.

    Parameters
    ----------
    ticker     : exchange symbol e.g. "NVDA", "AAPL"
    interval   : "1m","5m","15m","30m","1h","4h","1d"
    days       : how many calendar days of history
    api_key    : Alpaca key ID (falls back to settings.ALPACA_API_KEY)
    secret_key : Alpaca secret (falls back to settings.ALPACA_SECRET)

    Returns
    -------
    pd.DataFrame with columns [open, high, low, close, volume]
    index = timezone-naive UTC datetime
    """
    try:
        from config.settings import ALPACA_API_KEY, ALPACA_SECRET
        api_key    = api_key    or ALPACA_API_KEY
        secret_key = secret_key or ALPACA_SECRET
    except ImportError:
        pass

    if not api_key or api_key == "YOUR_KEY":
        logger.warning(
            "Alpaca API key not set — equities will fall back to yfinance.\n"
            "Sign up free at alpaca.markets and add keys to config/settings.py"
        )
        return pd.DataFrame()

    try:
        import alpaca_trade_api as tradeapi
    except ImportError:
        logger.error(
            "alpaca-trade-api not installed.\n"
            "Run: pip install alpaca-trade-api"
        )
        return pd.DataFrame()

    timeframe = ALPACA_TIMEFRAME_MAP.get(interval, "1Hour")
    end       = datetime.now(timezone.utc)
    start     = end - timedelta(days=days)

    # Data API lives at data.alpaca.markets — different from trading URL
    data_api = tradeapi.REST(
        api_key,
        secret_key,
        base_url="https://data.alpaca.markets",
        api_version="v2",
    )

    try:
        bars_gen = data_api.get_bars(
            ticker,
            timeframe,
            start=start.isoformat(),
            end=end.isoformat(),
            limit=10_000,
            adjustment="all",    # apply splits and dividends
            feed="iex",          # IEX = free; use "sip" if you have paid plan
        )
        bars = bars_gen.df
    except Exception as e:
        logger.error(f"Alpaca fetch error ({ticker}): {e}")
        return pd.DataFrame()

    if bars is None or bars.empty:
        logger.warning(f"Alpaca returned no data for {ticker}")
        return pd.DataFrame()

    # Normalise column names — Alpaca returns lowercase already
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in bars.columns]
    bars = bars[keep].copy()

    if "volume" not in bars.columns:
        bars["volume"] = 0

    bars.index = pd.to_datetime(bars.index).tz_localize(None)
    bars.index.name = "time"
    bars = bars[~bars.index.duplicated(keep="last")].sort_index()

    logger.info(f"Alpaca {ticker}: {len(bars)} {interval} bars  "
                f"({bars.index[0].date()} → {bars.index[-1].date()})")
    return bars
