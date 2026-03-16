"""
================================================================
  data/sources/fred.py
  Fetch macro context data from FRED (Federal Reserve Economic Data).

  Why FRED for macro:
    - Official Fed data — the authoritative source for US macro
    - Free API key, no cost ever
    - VIX, yield curve, dollar index, fed funds rate
    - Daily observations, goes back decades
    - More reliable than yfinance for macro series

  Free API key:
    1. Go to: https://fred.stlouisfed.org/docs/api/api_key.html
    2. Sign up (takes < 2 minutes)
    3. Add FRED_API_KEY = "your_key" to config/settings.py

  Series used:
    VIXCLS    → CBOE Volatility Index (VIX)
    DTWEXBGS  → Broad Dollar Index (US Dollar strength, proxy for DXY)
    DGS10     → 10-Year Treasury Yield
    DGS2      → 2-Year Treasury Yield
    DFF       → Fed Funds Rate (overnight interest rate)
================================================================
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import requests
import pandas as pd

logger = logging.getLogger("trading_firm.data.fred")

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# ── Series registry ───────────────────────────────────────────
FRED_SERIES = {
    "vix":          "VIXCLS",     # CBOE VIX — fear gauge
    "dxy":          "DTWEXBGS",   # Broad Dollar Index (proxy for DXY)
    "us10y":        "DGS10",      # 10-Year Treasury Yield
    "us2y":         "DGS2",       # 2-Year Treasury Yield
    "fed_funds":    "DFF",        # Fed Funds Rate
}


def fetch_fred(
    series:  str,
    days:    int = 730,
    api_key: Optional[str] = None,
) -> pd.Series:
    """
    Fetch a FRED macro series as a daily pd.Series.

    Parameters
    ----------
    series  : key from FRED_SERIES (e.g. "vix", "dxy", "us10y")
    days    : how many calendar days of history
    api_key : FRED API key (falls back to settings.FRED_API_KEY)

    Returns
    -------
    pd.Series indexed by date, forward-filled.
    Returns empty Series if key not set (graceful degradation).
    """
    if api_key is None:
        try:
            from config.settings import FRED_API_KEY
            api_key = FRED_API_KEY
        except (ImportError, AttributeError):
            pass

    if not api_key or api_key == "YOUR_FRED_KEY":
        logger.info(
            f"FRED API key not set — '{series}' will be zeros.\n"
            "Get a free key in < 2 min: "
            "https://fred.stlouisfed.org/docs/api/api_key.html\n"
            "Then add:  FRED_API_KEY = 'your_key'  to config/settings.py"
        )
        return pd.Series(dtype=float)

    fred_id = FRED_SERIES.get(series)
    if not fred_id:
        logger.warning(f"Unknown FRED series key: '{series}'. "
                       f"Available: {list(FRED_SERIES)}")
        return pd.Series(dtype=float)

    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    end   = datetime.now().strftime("%Y-%m-%d")

    params = {
        "series_id":         fred_id,
        "api_key":           api_key,
        "file_type":         "json",
        "observation_start": start,
        "observation_end":   end,
        "sort_order":        "asc",
    }

    try:
        r = requests.get(FRED_BASE_URL, params=params, timeout=30)
        r.raise_for_status()
        observations = r.json().get("observations", [])
    except requests.HTTPError as e:
        logger.error(f"FRED HTTP error ({series}): {e.response.status_code} — {e}")
        return pd.Series(dtype=float)
    except Exception as e:
        logger.error(f"FRED fetch error ({series}): {e}")
        return pd.Series(dtype=float)

    records = []
    for obs in observations:
        try:
            # FRED uses "." for missing values — skip them
            val = float(obs["value"])
            records.append({"date": pd.to_datetime(obs["date"]), "value": val})
        except (ValueError, KeyError):
            continue

    if not records:
        logger.warning(f"FRED ({series}): no valid observations returned")
        return pd.Series(dtype=float)

    s = (pd.DataFrame(records)
           .set_index("date")["value"]
           .sort_index()
           .ffill())

    logger.info(f"FRED {series} ({fred_id}): {len(s)} daily observations  "
                f"({s.index[0].date()} → {s.index[-1].date()})")
    return s


def fetch_yield_spread(
    days:    int = 730,
    api_key: Optional[str] = None,
) -> pd.Series:
    """
    10Y − 2Y Treasury yield spread (basis points).

    A negative spread (yield curve inversion) historically precedes
    recessions by 6–18 months. Use as a macro regime feature.

    Returns pd.Series of spread values, or empty Series on failure.
    """
    us10y = fetch_fred("us10y", days=days, api_key=api_key)
    us2y  = fetch_fred("us2y",  days=days, api_key=api_key)

    if us10y.empty or us2y.empty:
        return pd.Series(dtype=float)

    combined = pd.concat([us10y.rename("us10y"),
                          us2y.rename("us2y")], axis=1).ffill().dropna()
    spread = combined["us10y"] - combined["us2y"]
    logger.info(f"Yield spread computed: {len(spread)} observations")
    return spread


def fetch_all_macro(
    days:    int = 730,
    api_key: Optional[str] = None,
) -> dict:
    """
    Convenience wrapper — fetch all macro series at once.

    Returns
    -------
    dict with keys: "vix", "dxy", "yield_spread", "fed_funds"
    Each value is a pd.Series (empty if key not configured).
    """
    return {
        "vix":          fetch_fred("vix",       days=days, api_key=api_key),
        "dxy":          fetch_fred("dxy",       days=days, api_key=api_key),
        "yield_spread": fetch_yield_spread(     days=days, api_key=api_key),
        "fed_funds":    fetch_fred("fed_funds", days=days, api_key=api_key),
    }
