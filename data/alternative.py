"""
================================================================
  data/alternative.py
  Alternative sentiment data — Fear & Greed index.

  Sources:
    - CNN Business Fear & Greed (market-wide, equities/forex)
    - Alternative.me Crypto Fear & Greed (crypto specific)

  Used as additional ML features:
    fg_norm       — current reading normalised to [-1, +1]
                    (-1 = extreme fear, +1 = extreme greed)
    fg_contrarian — inverted: extreme fear → buy signal,
                    extreme greed → sell signal

  Both endpoints are free, no API key required.
  Cached for 60 minutes to avoid rate limits.
================================================================
"""
import logging
import requests
from datetime import datetime, timezone, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger("trading_firm.alternative")

_CACHE_TTL_MINUTES = 60

# ── Cache ─────────────────────────────────────────────────────
_cache: dict = {
    "market_fg":  {"value": None, "ts": None},
    "crypto_fg":  {"value": None, "ts": None},
}


def _is_stale(key: str) -> bool:
    ts = _cache[key]["ts"]
    if ts is None:
        return True
    return (datetime.now(timezone.utc) - ts).seconds > _CACHE_TTL_MINUTES * 60


# ── CNN Fear & Greed (market-wide) ────────────────────────────

def fetch_market_fear_greed() -> Optional[float]:
    """
    Fetch CNN Business Fear & Greed index (0–100).
    Uses the public API endpoint (no key required).
    Returns None on failure.
    """
    if not _is_stale("market_fg"):
        return _cache["market_fg"]["value"]

    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.cnn.com/"},
            timeout=8,
        )
        data  = resp.json()
        score = float(data["fear_and_greed"]["score"])
        _cache["market_fg"]["value"] = score
        _cache["market_fg"]["ts"]    = datetime.now(timezone.utc)
        logger.debug(f"CNN Fear & Greed: {score:.1f}")
        return score
    except Exception as e:
        logger.warning(f"CNN Fear & Greed fetch failed: {e}")
        return None


# ── Alternative.me Crypto Fear & Greed ───────────────────────

def fetch_crypto_fear_greed() -> Optional[float]:
    """
    Fetch Alternative.me Crypto Fear & Greed index (0–100).
    Returns None on failure.
    """
    if not _is_stale("crypto_fg"):
        return _cache["crypto_fg"]["value"]

    try:
        url  = "https://api.alternative.me/fng/?limit=1&format=json"
        resp = requests.get(url, timeout=8)
        data = resp.json()
        score = float(data["data"][0]["value"])
        _cache["crypto_fg"]["value"] = score
        _cache["crypto_fg"]["ts"]    = datetime.now(timezone.utc)
        logger.debug(f"Crypto Fear & Greed: {score:.1f}")
        return score
    except Exception as e:
        logger.warning(f"Crypto Fear & Greed fetch failed: {e}")
        return None


# ── Public helpers ────────────────────────────────────────────

def get_fear_greed(asset_class: str = "market") -> dict:
    """
    Return Fear & Greed features for the given asset class.

    Parameters
    ----------
    asset_class : "crypto" | anything else → market (equities/forex)

    Returns
    -------
    dict with keys:
      raw         — 0–100 raw score (50 = neutral, None if unavailable)
      fg_norm     — normalised to [-1, +1]  (-1=extreme fear, +1=extreme greed)
      fg_contrarian — inverted [-1, +1]      (-1=extreme greed, +1=extreme fear)
    """
    if asset_class == "crypto":
        raw = fetch_crypto_fear_greed()
    else:
        raw = fetch_market_fear_greed()

    if raw is None:
        return {"raw": None, "fg_norm": 0.0, "fg_contrarian": 0.0}

    fg_norm      = (raw - 50.0) / 50.0          # -1 … +1
    fg_contrarian = -fg_norm                     # invert: fear → buy

    return {
        "raw":          raw,
        "fg_norm":      round(fg_norm, 4),
        "fg_contrarian": round(fg_contrarian, 4),
    }


def get_fear_greed_series(asset_class: str = "market",
                          periods: int = 1) -> pd.DataFrame:
    """
    Return a single-row DataFrame with fg_norm / fg_contrarian
    compatible with feature pipeline broadcasting.
    """
    fg = get_fear_greed(asset_class)
    return pd.DataFrame(
        {"fg_norm": [fg["fg_norm"]], "fg_contrarian": [fg["fg_contrarian"]]}
    )
