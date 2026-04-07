"""data/macro_calendar.py — Macro event gate via Finnhub economic calendar.

Returns whether the current day has a high-impact event for major currencies.
Results are cached for 6 hours to avoid hammering the API.

Usage:
    from data.macro_calendar import MacroCalendar
    cal = MacroCalendar()
    is_high, event_name = cal.is_high_impact_day()
    size_mult = cal.position_size_multiplier()   # 0.5 on high-impact, 1.0 otherwise
"""

import logging
import time
from datetime import datetime, timezone
from typing import Tuple

logger = logging.getLogger("trading_firm.macro_calendar")

# High-impact Finnhub impact codes and keyword filters
_HIGH_IMPACT_CODES = {"high"}
_HIGH_IMPACT_KEYWORDS = {
    "nfp", "non-farm", "nonfarm", "fed", "fomc", "cpi", "gdp",
    "unemployment", "interest rate", "rate decision", "payroll",
    "inflation", "ecb", "boe", "boj", "rba", "rbnz", "snb",
}

# Currencies we care about (the 6 majors)
_WATCHED_CURRENCIES = {"USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF"}

_CACHE_TTL = 6 * 3600   # 6 hours


class MacroCalendar:
    """
    Queries Finnhub /calendar/economic for today's high-impact events.
    Results are cached for _CACHE_TTL seconds.

    On any API failure, falls back to non-blocking defaults:
      is_high_impact_day() → (False, "")
      position_size_multiplier() → 1.0
    """

    def __init__(self, api_key: str | None = None):
        if api_key is None:
            from config.settings import FINNHUB_API_KEY
            api_key = FINNHUB_API_KEY
        self._api_key  = api_key
        self._cache    = None           # (timestamp, result)
        self._cache_ts = 0.0

    # ── Internal ─────────────────────────────────────────────────────────────

    def _fetch(self) -> list:
        """Fetch today's economic calendar from Finnhub. Returns list of events."""
        import urllib.request
        import json
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        url = (
            f"https://finnhub.io/api/v1/calendar/economic"
            f"?from={today}&to={today}&token={self._api_key}"
        )
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
        return data.get("economicCalendar", [])

    def _is_high(self, event: dict) -> bool:
        """Return True if event is high-impact for a watched currency."""
        impact   = (event.get("impact") or "").lower()
        currency = (event.get("country") or event.get("currency") or "").upper()
        event_name = (event.get("event") or "").lower()

        if currency not in _WATCHED_CURRENCIES:
            return False
        if impact in _HIGH_IMPACT_CODES:
            return True
        for kw in _HIGH_IMPACT_KEYWORDS:
            if kw in event_name:
                return True
        return False

    def _get_cached(self) -> Tuple[bool, str]:
        now = time.monotonic()
        if self._cache is not None and (now - self._cache_ts) < _CACHE_TTL:
            return self._cache

        try:
            events = self._fetch()
            high_events = [e for e in events if self._is_high(e)]
            if high_events:
                name = high_events[0].get("event", "high-impact event")
                result = (True, name)
            else:
                result = (False, "")
            self._cache    = result
            self._cache_ts = now
            logger.debug(f"MacroCalendar: {len(events)} events today, high={result[0]}")
        except Exception as exc:
            logger.debug(f"MacroCalendar: fetch failed ({exc}) — defaulting to no event")
            result = (False, "")
            # Don't cache failures — retry next call

        return result

    # ── Public API ───────────────────────────────────────────────────────────

    def is_high_impact_day(self) -> Tuple[bool, str]:
        """
        Returns (True, event_name) if today has a high-impact macro event
        for a major currency, otherwise (False, "").

        Never raises — returns (False, "") on any error.
        """
        try:
            return self._get_cached()
        except Exception:
            return False, ""

    def position_size_multiplier(self) -> float:
        """
        Returns 0.5 on high-impact days, 1.0 otherwise.
        Multiply your intended unit count by this before sending the order.
        """
        is_high, _ = self.is_high_impact_day()
        return 0.5 if is_high else 1.0
