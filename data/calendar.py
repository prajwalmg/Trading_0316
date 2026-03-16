"""
================================================================
  data/calendar.py
  Economic calendar — fetches high-impact events from
  ForexFactory (free, no API key required).

  Used to avoid trading during news events which cause
  unpredictable spreads and adverse fills.

  Events filtered: NFP, CPI, FOMC, ECB, GDP, PMI (High impact)
  Blackout window: 30 minutes before → 60 minutes after event
================================================================
"""
import logging
import requests
from datetime import datetime, timezone, timedelta
from typing import List, Dict

logger = logging.getLogger("trading_firm.calendar")

# Blackout window around each high-impact event
BLACKOUT_BEFORE_MINUTES = 30
BLACKOUT_AFTER_MINUTES  = 60

# Currency → forex pairs affected
CURRENCY_PAIRS = {
    "USD": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X",
            "AUDUSD=X", "NZDUSD=X", "USDCAD=X", "USDMXN=X",
            "USDZAR=X", "USDNOK=X", "USDSEK=X"],
    "EUR": ["EURUSD=X", "EURGBP=X", "EURJPY=X", "EURCHF=X",
            "EURAUD=X", "EURCAD=X", "EURNZD=X", "EURNOK=X", "EURSEK=X"],
    "GBP": ["GBPUSD=X", "EURGBP=X", "GBPJPY=X", "GBPCHF=X",
            "GBPAUD=X", "GBPCAD=X", "GBPNZD=X"],
    "JPY": ["USDJPY=X", "EURJPY=X", "GBPJPY=X", "AUDJPY=X",
            "NZDJPY=X", "CADJPY=X", "CHFJPY=X"],
    "AUD": ["AUDUSD=X", "EURAUD=X", "GBPAUD=X", "AUDJPY=X",
            "AUDCHF=X", "AUDCAD=X", "AUDNZD=X"],
    "CAD": ["USDCAD=X", "EURCAD=X", "GBPCAD=X", "CADJPY=X",
            "CADCHF=X", "AUDCAD=X", "NZDCAD=X"],
    "CHF": ["USDCHF=X", "EURCHF=X", "GBPCHF=X", "CHFJPY=X",
            "AUDCHF=X", "NZDCHF=X", "CADCHF=X"],
    "NZD": ["NZDUSD=X", "EURNZD=X", "GBPNZD=X", "NZDJPY=X",
            "NZDCHF=X", "NZDCAD=X", "AUDNZD=X"],
}

_cached_events: List[Dict] = []
_cache_time: datetime      = None
_CACHE_TTL_MINUTES         = 60


def _fetch_events() -> List[Dict]:
    """Fetch this week's events from ForexFactory. Cached for 60 minutes."""
    global _cached_events, _cache_time
    now = datetime.now(timezone.utc)

    if _cache_time and (now - _cache_time).seconds < _CACHE_TTL_MINUTES * 60:
        return _cached_events

    try:
        response = requests.get(
            "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=8,
        )
        events        = response.json()
        _cached_events = events
        _cache_time    = now
        logger.debug(f"Calendar: fetched {len(events)} events")
        return events
    except Exception as e:
        logger.warning(f"Calendar fetch failed: {e} — news filter disabled")
        return []


def get_upcoming_high_impact(hours_ahead: int = 24) -> List[Dict]:
    """
    Returns list of upcoming high-impact events within the next N hours.
    Each entry: {title, country, currency, time, impact}
    """
    events = _fetch_events()
    now    = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=hours_ahead)
    result = []

    for e in events:
        if e.get("impact") != "High":
            continue
        try:
            event_time = datetime.fromisoformat(
                e["date"].replace("Z", "+00:00")
            )
            if event_time.tzinfo is None:
                event_time = event_time.replace(tzinfo=timezone.utc)
            if now <= event_time <= cutoff:
                result.append({
                    "title":    e.get("title", "Unknown"),
                    "country":  e.get("country", ""),
                    "currency": e.get("currency", ""),
                    "time":     event_time,
                    "impact":   "High",
                })
        except Exception:
            continue

    return result


def is_news_blackout(instrument: str = None) -> tuple:
    """
    Check if we are currently in a news blackout window.

    Parameters
    ----------
    instrument : optional ticker to check (e.g. "EURUSD=X")
                 if None, checks all currencies

    Returns
    -------
    (bool, str) — (is_blackout, reason_string)
    """
    events = _fetch_events()
    now    = datetime.now(timezone.utc)

    for e in events:
        if e.get("impact") != "High":
            continue
        try:
            event_time = datetime.fromisoformat(
                e["date"].replace("Z", "+00:00")
            )
            if event_time.tzinfo is None:
                event_time = event_time.replace(tzinfo=timezone.utc)

            window_start = event_time - timedelta(minutes=BLACKOUT_BEFORE_MINUTES)
            window_end   = event_time + timedelta(minutes=BLACKOUT_AFTER_MINUTES)

            if not (window_start <= now <= window_end):
                continue

            # If no instrument specified, any high-impact event triggers blackout
            if instrument is None:
                return True, f"News blackout: {e.get('title','?')} ({e.get('country','')})"

            # Check if the event currency affects this instrument
            event_currency = e.get("currency", "").upper()
            affected_pairs = CURRENCY_PAIRS.get(event_currency, [])
            if instrument in affected_pairs:
                return (True,
                        f"News blackout: {e.get('title','?')} "
                        f"({event_currency}, affects {instrument})")
        except Exception:
            continue

    return False, ""


def log_upcoming_events():
    """Log upcoming high-impact events — call at startup."""
    events = get_upcoming_high_impact(hours_ahead=24)
    if not events:
        logger.info("Calendar: no high-impact events in next 24h")
        return
    logger.info(f"Calendar: {len(events)} high-impact events in next 24h:")
    for e in events:
        logger.info(f"  {e['time'].strftime('%H:%M UTC')} "
                    f"{e['currency']:<4} {e['title']}")
