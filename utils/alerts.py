"""
================================================================
  utils/alerts.py
  Telegram Bot alert system for real-time trade notifications.

  Setup:
    1. Message @BotFather on Telegram to create a bot → get BOT_TOKEN
    2. Send any message to your bot, then visit:
         https://api.telegram.org/bot<TOKEN>/getUpdates
       to find your CHAT_ID
    3. Set environment variables (or .env file):
         TELEGRAM_BOT_TOKEN=<token>
         TELEGRAM_CHAT_ID=<chat_id>

  Graceful degradation:
    If token/chat_id are not set, all functions log a warning
    and return False — the rest of the system keeps running.
================================================================
"""
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("trading_firm.alerts")

# ── Config from environment ────────────────────────────────────
_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID",   "")
_ENABLED   = bool(_BOT_TOKEN and _CHAT_ID)

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


# ── Core sender ───────────────────────────────────────────────

def send_alert(message: str, parse_mode: str = "HTML") -> bool:
    """
    Send a Telegram message.  Returns True on success.
    Silent no-op (returns False) if credentials are not configured.
    """
    if not _ENABLED:
        logger.debug("Telegram not configured — alert suppressed")
        return False

    try:
        import urllib.request
        import urllib.parse
        import json

        url     = _TELEGRAM_API.format(token=_BOT_TOKEN)
        payload = json.dumps({
            "chat_id":    _CHAT_ID,
            "text":       message,
            "parse_mode": parse_mode,
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data    = payload,
            headers = {"Content-Type": "application/json"},
            method  = "POST",
        )

        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read())
            if not body.get("ok"):
                logger.warning(f"Telegram API error: {body}")
                return False
        return True

    except Exception as e:
        logger.warning(f"Telegram send_alert failed: {e}")
        return False


# ── Typed alert helpers ────────────────────────────────────────

def alert_startup(version: str = "1.0") -> bool:
    """Sent when the trading loop starts."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    msg = (
        "🚀 <b>Trading System Started</b>\n"
        f"Version : {version}\n"
        f"Time    : {now}"
    )
    return send_alert(msg)


def alert_trade_opened(
    ticker:    str,
    direction: int,
    units:     int,
    entry:     float,
    sl:        float,
    tp:        float,
    confidence: float,
    strategy:  str = "",
) -> bool:
    """Sent immediately after a market order is filled."""
    side  = "🟢 LONG" if direction == 1 else "🔴 SHORT"
    now   = datetime.now(timezone.utc).strftime("%H:%M UTC")
    strat = f"\nStrategy : {strategy}" if strategy else ""
    msg = (
        f"{side} <b>{ticker}</b> opened\n"
        f"Units     : {units:,}\n"
        f"Entry     : {entry:.5f}\n"
        f"SL        : {sl:.5f}\n"
        f"TP        : {tp:.5f}\n"
        f"Confidence: {confidence:.1%}{strat}\n"
        f"Time      : {now}"
    )
    return send_alert(msg)


def alert_trade_closed(
    ticker:    str,
    direction: int,
    units:     int,
    entry:     float,
    close:     float,
    pnl:       float,
    pnl_pct:   float,
) -> bool:
    """Sent when a position is closed."""
    side   = "🟢 LONG" if direction == 1 else "🔴 SHORT"
    emoji  = "✅" if pnl >= 0 else "❌"
    now    = datetime.now(timezone.utc).strftime("%H:%M UTC")
    msg = (
        f"{emoji} {side} <b>{ticker}</b> closed\n"
        f"Units  : {units:,}\n"
        f"Entry  : {entry:.5f} → Exit: {close:.5f}\n"
        f"P&L    : {pnl:+.2f} ({pnl_pct:+.2%})\n"
        f"Time   : {now}"
    )
    return send_alert(msg)


def alert_daily_summary(
    nav:          float,
    daily_pnl:    float,
    daily_pnl_pct: float,
    n_trades:     int,
    win_rate:     float,
    max_dd:       float,
) -> bool:
    """End-of-day summary alert."""
    emoji  = "📈" if daily_pnl >= 0 else "📉"
    now    = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    msg = (
        f"{emoji} <b>Daily Summary — {now}</b>\n"
        f"NAV       : ${nav:,.2f}\n"
        f"Daily P&L : {daily_pnl:+.2f} ({daily_pnl_pct:+.2%})\n"
        f"Trades    : {n_trades}\n"
        f"Win Rate  : {win_rate:.1%}\n"
        f"Max DD    : {max_dd:.2%}"
    )
    return send_alert(msg)


def alert_circuit_breaker(
    reason:    str,
    nav:       float,
    daily_pnl: float,
) -> bool:
    """Sent when the risk engine halts trading for the day."""
    now = datetime.now(timezone.utc).strftime("%H:%M UTC")
    msg = (
        "🚨 <b>CIRCUIT BREAKER TRIGGERED</b>\n"
        f"Reason    : {reason}\n"
        f"NAV       : ${nav:,.2f}\n"
        f"Daily P&L : {daily_pnl:+.2f}\n"
        f"Time      : {now}\n"
        "Trading halted for the rest of the session."
    )
    return send_alert(msg)


def alert_error(error_msg: str) -> bool:
    """Sent on unhandled exceptions in the main loop."""
    now = datetime.now(timezone.utc).strftime("%H:%M UTC")
    msg = (
        f"⚠️ <b>System Error</b>\n"
        f"Time  : {now}\n"
        f"Error : {error_msg[:400]}"
    )
    return send_alert(msg)
