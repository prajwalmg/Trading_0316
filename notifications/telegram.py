import requests, logging
from datetime import datetime
logger = logging.getLogger("trading_firm.telegram")

SYSTEM_EMOJI = {
    'swing':     '📊',
    'intraday':  '⚡',
    'multi':     '🌐',
    'unknown':   '❓',
}

ASSET_EMOJI = {
    'forex':     '💱',
    'crypto':    '🪙',
    'equity':    '📈',
    'commodity': '🥇',
    'unknown':   '📌',
}

DIRECTION_EMOJI = {
    1:  '🟢 LONG',
    -1: '🔴 SHORT',
}


def _send(message):
    try:
        from config.settings import (
            TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        if not TELEGRAM_BOT_TOKEN: return False
        r = requests.post(
            f"https://api.telegram.org/"
            f"bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id":TELEGRAM_CHAT_ID,
                  "text":message,
                  "parse_mode":"HTML"},
            timeout=5)
        return r.status_code == 200
    except Exception as e:
        logger.warning(f"Telegram: {e}")
        return False


def trade_opened(
    ticker:      str,
    direction:   int,
    entry_price: float,
    sl_price:    float,
    tp_price:    float,
    lot_size:    float,
    confidence:  float,
    regime:      str,
    system:      str = 'swing',
    asset_class: str = 'forex',
    atr:         float = None,
    risk_pct:    float = None,
    risk_amount: float = None,
    nav:         float = None,
    timeframe:   str = '1h',
):
    sys_emoji   = SYSTEM_EMOJI.get(system, '❓')
    sys_label   = system.upper()
    asset_emoji = ASSET_EMOJI.get(asset_class, '📌')
    dir_label   = DIRECTION_EMOJI.get(direction, '❓')

    dist_tp = abs(tp_price - entry_price)
    dist_sl = abs(entry_price - sl_price)
    rr = (dist_tp / dist_sl if dist_sl > 0 else 0)

    if asset_class == 'forex':
        fmt = '.5f'
    elif asset_class == 'crypto':
        fmt = '.2f' if entry_price > 100 else '.4f'
    else:
        fmt = '.2f'

    lines = [
        f"{sys_emoji} <b>{sys_label} — TRADE OPENED</b>",
        "──────────────────────",
        f"{asset_emoji} <b>{ticker}</b>  |  {dir_label}",
        "",
        f"📍 Entry:        <b>{entry_price:{fmt}}</b>",
        f"🎯 Take Profit:  <b>{tp_price:{fmt}}</b>  (+{dist_tp:{fmt}})",
        f"🛑 Stop Loss:    <b>{sl_price:{fmt}}</b>  (-{dist_sl:{fmt}})",
        f"📐 RR Ratio:     {rr:.2f} : 1",
        "",
        f"📦 Lot Size:     <b>{lot_size:.4f}</b>",
    ]

    if risk_pct is not None:
        lines.append(
            f"⚠️  Risk:         {risk_pct:.2f}%"
            + (f"  (€{risk_amount:.2f})" if risk_amount else ""))

    lines += [
        f"📊 Confidence:   {confidence:.1%}",
        f"🌊 Regime:       {regime}",
    ]

    if atr is not None:
        lines.append(f"📏 ATR:          {atr:{fmt}}")

    lines.append(f"⏱ Timeframe:    {timeframe}")

    if nav is not None:
        lines.append("")
        lines.append(f"💼 NAV:          €{nav:,.2f}")

    lines += [
        "",
        f"🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
    ]

    _send('\n'.join(lines))


def trade_closed(
    ticker:      str,
    direction:   int,
    entry_price: float,
    exit_price:  float,
    sl_price:    float,
    tp_price:    float,
    lot_size:    float,
    pnl:         float,
    pnl_pct:     float,
    reason:      str,
    system:      str = 'swing',
    asset_class: str = 'forex',
    hold_hours:  float = None,
    confidence:  float = None,
    nav:         float = None,
    total_pnl:   float = None,
    win_streak:  int = None,
    timeframe:   str = '1h',
):
    sys_emoji   = SYSTEM_EMOJI.get(system, '❓')
    sys_label   = system.upper()
    asset_emoji = ASSET_EMOJI.get(asset_class, '📌')
    dir_label   = DIRECTION_EMOJI.get(direction, '❓')

    win         = pnl >= 0
    result_icon = '✅ WIN' if win else '❌ LOSS'
    pnl_sign    = '+' if win else ''

    reason_map = {
        'tp':       '→ TP HIT 🎯',
        'sl':       '→ SL HIT 🛑',
        'timeout':  '→ TIME EXPIRY ⏰',
        'reversal': '→ SIGNAL REVERSED 🔄',
        'manual':   '→ MANUAL CLOSE ✋',
        'eod':      '→ END OF DAY 📅',
    }
    reason_label = reason_map.get(reason.lower(), f'→ {reason.upper()}')

    if asset_class == 'forex':
        fmt = '.5f'
    elif asset_class == 'crypto':
        fmt = '.2f' if entry_price > 100 else '.4f'
    else:
        fmt = '.2f'

    lines = [
        f"{sys_emoji} <b>{sys_label} — TRADE CLOSED  {result_icon}</b>",
        "──────────────────────────────",
        f"{asset_emoji} <b>{ticker}</b>  |  {dir_label}",
        "",
        f"📍 Entry:        {entry_price:{fmt}}",
        f"🏁 Exit:         <b>{exit_price:{fmt}}</b>  {reason_label}",
        f"🎯 Target was:   {tp_price:{fmt}}",
        f"🛑 Stop was:     {sl_price:{fmt}}",
        "",
        f"{'💰' if win else '💸'} PnL:         <b>{pnl_sign}€{pnl:.2f}  ({pnl_sign}{pnl_pct:.2f}%)</b>",
        f"📦 Lot Size:     {lot_size:.4f}",
    ]

    if hold_hours is not None:
        lines.append(f"⏱ Hold Time:     {hold_hours:.1f} hours")

    if confidence is not None:
        lines.append(f"📊 Entry Conf:   {confidence:.1%}")

    lines.append(f"⏱ Timeframe:    {timeframe}")

    if nav is not None or total_pnl is not None:
        lines.append("")

    if nav is not None:
        lines.append(f"💼 NAV:          €{nav:,.2f}")

    if total_pnl is not None:
        sign = '+' if total_pnl >= 0 else ''
        lines.append(f"📈 Total PnL:    {sign}€{total_pnl:.2f} since start")

    if win_streak and win_streak > 1:
        lines.append(f"🔥 Win Streak:   {win_streak}")

    lines += [
        "",
        f"🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
    ]

    _send('\n'.join(lines))


def daily_report(
    nav:          float,
    daily_pnl:    float,
    total_trades: int,
    win_rate:     float,
    total_pnl:    float,
    open_pos:     int,
    system_stats: dict = None,
):
    emoji      = '📈' if daily_pnl >= 0 else '📉'
    sign       = '+' if daily_pnl >= 0 else ''
    total_sign = '+' if total_pnl >= 0 else ''

    lines = [
        f"{emoji} <b>Daily Report</b>",
        f"{'─'*28}",
        f"💼 NAV:         €{nav:,.2f}",
        f"📅 Today:       {sign}€{daily_pnl:.2f}",
        f"📈 Total PnL:   {total_sign}€{total_pnl:.2f}",
        f"🔢 Trades:      {total_trades}",
        f"🎯 Win Rate:    {win_rate:.1%}",
        f"📂 Open:        {open_pos}",
    ]

    if system_stats:
        lines.append(f"{'─'*28}")
        lines.append("<b>By System:</b>")
        for sys_name, stats in system_stats.items():
            sys_emoji = SYSTEM_EMOJI.get(sys_name, '❓')
            s_sign = '+' if stats['pnl'] >= 0 else ''
            lines.append(
                f"{sys_emoji} {sys_name:<10} "
                f"{s_sign}€{stats['pnl']:>8.2f}"
                f"  {stats['trades']}t"
                f"  {stats['wr']:.0%}")

    lines += [
        f"{'─'*28}",
        f"📆 {datetime.utcnow().strftime('%d %b %Y')}",
    ]

    _send('\n'.join(lines))


def drawdown_alert(dd, pair=None):
    where = f" on {pair}" if pair else ""
    _send(f"⚠️ <b>DRAWDOWN ALERT</b>\n"
          f"DD{where}: {dd:.1%}\n"
          f"Time: {datetime.now().strftime('%H:%M:%S')}")


def system_started(n_pairs, capital):
    _send(f"🚀 <b>FIRM OS Started</b>\n"
          f"Pairs:   {n_pairs} instruments\n"
          f"Capital: €{capital:,.2f}\n"
          f"Mode:    Paper Trading\n"
          f"Time:    {datetime.now().strftime('%H:%M:%S')}")


def system_error(msg):
    _send(f"🚨 <b>SYSTEM ERROR</b>\n"
          f"{msg[:200]}\n"
          f"Time: {datetime.now().strftime('%H:%M:%S')}")
