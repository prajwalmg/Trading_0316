"""
================================================================
  dashboard/cli.py
  Real-time terminal dashboard — updates in-place using
  ANSI escape codes. No external UI library required.

  Displays:
    ┌─ Account Overview (NAV, P&L, drawdown)
    ├─ Open Positions (instrument, direction, P&L)
    ├─ Risk Status (circuit breakers, limits)
    ├─ Performance Metrics (Sharpe, Sortino, win rate)
    ├─ Regime Map (per-instrument regime)
    └─ Recent Trades (last 5)
================================================================
"""

import os
import sys
import time
import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.settings import (
    MAX_DAILY_LOSS_PCT, MAX_DRAWDOWN_PCT,
    MAX_OPEN_POSITIONS, INITIAL_CAPITAL,
)


# ── ANSI helpers ──────────────────────────────────────────────

def clr(text, code):  return f"\033[{code}m{text}\033[0m"
def bold(t):          return clr(t, "1")
def green(t):         return clr(t, "92")
def red(t):           return clr(t, "91")
def yellow(t):        return clr(t, "93")
def cyan(t):          return clr(t, "96")
def blue(t):          return clr(t, "94")
def grey(t):          return clr(t, "90")
def magenta(t):       return clr(t, "95")

def pnl_colour(val):
    """Colour a P&L value green/red based on sign."""
    s = f"{val:+.2f}"
    return green(s) if val >= 0 else red(s)

def pct_colour(val, good_positive=True):
    s = f"{val:+.2%}"
    if good_positive:
        return green(s) if val >= 0 else red(s)
    else:
        return red(s) if val >= 0 else green(s)

def regime_colour(regime: str) -> str:
    colours = {
        "trending_up":    green,
        "trending_down":  red,
        "ranging":        yellow,
        "high_volatility": magenta,
    }
    return colours.get(regime, grey)(regime.replace("_", " ").title())


# ── Metrics computation ───────────────────────────────────────

def compute_metrics(trade_df: pd.DataFrame) -> dict:
    if trade_df.empty or "pnl" not in trade_df.columns:
        return {
            "win_rate": 0, "profit_factor": 0, "expectancy": 0,
            "sharpe": 0, "sortino": 0, "total_trades": 0,
            "avg_win": 0, "avg_loss": 0,
        }

    pnls    = trade_df["pnl"]
    winners = pnls[pnls > 0]
    losers  = pnls[pnls < 0]

    win_rate      = len(winners) / len(pnls) if len(pnls) > 0 else 0
    avg_win       = winners.mean() if len(winners) > 0 else 0
    avg_loss      = abs(losers.mean()) if len(losers) > 0 else 1
    profit_factor = (winners.sum() / abs(losers.sum())) if losers.sum() != 0 else 0
    expectancy    = win_rate * avg_win - (1 - win_rate) * avg_loss

    # Sharpe (simplified, daily)
    if len(pnls) > 2:
        sharpe  = (pnls.mean() / (pnls.std() + 1e-9)) * np.sqrt(252)
        neg     = pnls[pnls < 0]
        sortino = (pnls.mean() / (neg.std() + 1e-9)) * np.sqrt(252) if len(neg) > 1 else 0
    else:
        sharpe = sortino = 0

    return {
        "win_rate":      win_rate,
        "profit_factor": profit_factor,
        "expectancy":    expectancy,
        "sharpe":        sharpe,
        "sortino":       sortino,
        "total_trades":  len(pnls),
        "avg_win":       avg_win,
        "avg_loss":      avg_loss,
    }


# ── Dashboard renderer ────────────────────────────────────────

class Dashboard:
    """
    In-place terminal dashboard. Clears and redraws on each update.
    Call .render() once per trading loop cycle.
    """

    def __init__(self):
        self.start_time = datetime.now(timezone.utc)
        self._width     = min(shutil.get_terminal_size().columns, 100)

    def _line(self, char="─"):
        return grey(char * self._width)

    def _header(self, title: str) -> str:
        pad = self._width - len(title) - 4
        return cyan(f"┌── {bold(title)} " + "─" * max(0, pad) + "┐")

    def _row(self, label: str, value: str, width: int = 28) -> str:
        label_fmt = grey(f"  {label:<{width}}")
        return f"{label_fmt}{value}"

    def render(
        self,
        account:    dict,
        risk_report: dict,
        open_positions: list,
        trade_history:  pd.DataFrame,
        regime_map:     dict,
        portfolio_summary: dict,
        cycle:      int = 0,
        regime_tracker = None,   # RegimeTracker instance for confidence/vol/transition
    ):
        """
        Render the full dashboard. Clears terminal first.
        """
        os.system("clear" if os.name != "nt" else "cls")

        nav        = account.get("nav",      INITIAL_CAPITAL)
        balance    = account.get("balance",  INITIAL_CAPITAL)
        unreal_pnl = account.get("unrealised_pnl", 0.0)
        daily_pnl  = risk_report.get("daily_pnl",  0.0)
        total_pnl  = risk_report.get("total_pnl",  0.0)
        drawdown   = risk_report.get("drawdown",   "0.00%")
        uptime     = str(datetime.now(timezone.utc) - self.start_time).split(".")[0]

        lines = []

        # ── Title bar ────────────────────────────────────────
        lines.append("")
        title = f"  ⚡  TRADING FIRM OS  |  Cycle #{cycle}  |  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  |  Uptime: {uptime}"
        lines.append(bold(cyan(title)))
        lines.append(self._line("═"))

        # ── Account Overview ─────────────────────────────────
        lines.append(self._header("ACCOUNT"))
        lines.append(self._row("NAV",             bold(f"${nav:,.2f}")))
        lines.append(self._row("Cash Balance",    f"${balance:,.2f}"))
        lines.append(self._row("Unrealised P&L",  pnl_colour(unreal_pnl)))
        lines.append(self._row("Today's P&L",     pnl_colour(daily_pnl) + grey(f"  ({daily_pnl/nav*100:+.2f}%)")))
        lines.append(self._row("All-Time P&L",    pnl_colour(total_pnl)))
        lines.append(self._row("Drawdown",        red(drawdown) if float(drawdown.strip("%")) > 5 else yellow(drawdown)))
        lines.append("")

        # ── Risk Status ──────────────────────────────────────
        lines.append(self._header("RISK STATUS"))

        def status_flag(halted, label):
            return (red(f"⛔ HALTED — {label}") if halted else green(f"✓ Active — {label}"))

        lines.append(self._row("Daily Limit",     status_flag(risk_report.get("daily_halted"),    f"limit={MAX_DAILY_LOSS_PCT:.0%}/day")))
        lines.append(self._row("Weekly Limit",    status_flag(risk_report.get("weekly_halted"),   "limit=6%/week")))
        lines.append(self._row("Drawdown Guard",  status_flag(risk_report.get("drawdown_halted"), f"limit={MAX_DRAWDOWN_PCT:.0%}")))
        lines.append(self._row("Daily Used",      risk_report.get("daily_limit_used", "0%")))
        lines.append(self._row("Open Positions",  f"{len(open_positions)} / {MAX_OPEN_POSITIONS}"))
        lines.append("")

        # ── Open Positions ───────────────────────────────────
        lines.append(self._header("OPEN POSITIONS"))
        if not open_positions:
            lines.append(grey("  No open positions"))
        else:
            hdr = f"  {'Instrument':<14}{'Direction':<10}{'Units':<10}{'Entry':<12}{'Unreal P&L':<14}"
            lines.append(grey(hdr))
            lines.append(grey("  " + "─" * 58))
            for pos in open_positions:
                inst = pos.get("instrument", "?")
                dir_ = green("LONG") if pos.get("direction", 1) == 1 else red("SHORT")
                up   = pos.get("unrealised_pnl", 0)
                lines.append(
                    f"  {bold(inst):<14}{dir_:<20}"
                    f"{pos.get('units','?'):<10}"
                    f"{pos.get('entry', 0):<12.5f}"
                    f"{pnl_colour(up)}"
                )
        lines.append("")

        # ── Performance Metrics ──────────────────────────────
        lines.append(self._header("PERFORMANCE METRICS"))
        m = compute_metrics(trade_history)

        col1 = [
            ("Sharpe Ratio",    f"{m['sharpe']:.3f}"),
            ("Sortino Ratio",   f"{m['sortino']:.3f}"),
            ("Profit Factor",   f"{m['profit_factor']:.3f}"),
        ]
        col2 = [
            ("Win Rate",        f"{m['win_rate']:.1%}"),
            ("Expectancy",      f"${m['expectancy']:.2f}"),
            ("Total Trades",    str(m["total_trades"])),
        ]
        for (l1, v1), (l2, v2) in zip(col1, col2):
            lines.append(
                grey(f"  {l1:<18}") + cyan(f"{v1:<16}") +
                grey(f"  {l2:<18}") + cyan(v2)
            )
        lines.append(self._row("Avg Win / Avg Loss", f"${m['avg_win']:.2f} / ${m['avg_loss']:.2f}"))
        lines.append("")

        # ── Regime Map ───────────────────────────────────────
        lines.append(self._header("MARKET REGIMES"))
        dom = portfolio_summary.get("dominant_regime", "unknown")
        lines.append(self._row("Portfolio Dominant", regime_colour(dom)))

        if regime_tracker is not None and regime_map:
            # Extended HMM display: one row per instrument
            hdr = grey(f"  {'Ticker':<14}{'Regime':<18}{'Conf':>6}  {'Vol':>6}  {'Status'}")
            lines.append(hdr)
            lines.append(grey("  " + "─" * 60))
            for ticker, regime in list(regime_map.items())[:12]:
                conf_r   = regime_tracker.get_regime_confidence(ticker)
                vol_fct  = regime_tracker.get_vol_forecast(ticker)
                in_trans = regime_tracker.is_in_transition(ticker)
                conf_str = f"{conf_r:.0%}"
                vol_str  = f"{vol_fct:.4f}"
                flag     = yellow(" TRANSITION") if in_trans else ""
                lines.append(
                    f"  {bold(ticker):<14}"
                    f"{regime_colour(regime):<28}"
                    f"{grey(conf_str):>14}  "
                    f"{grey(vol_str):>14}"
                    f"{flag}"
                )
        else:
            # Compact fallback (no tracker available)
            items = list(regime_map.items())[:12]
            for i in range(0, len(items), 3):
                row_items = items[i:i+3]
                row = "  "
                for ticker, regime in row_items:
                    row += grey(f"{ticker:<14}") + regime_colour(regime) + "   "
                lines.append(row)
        lines.append("")

        # ── Strategy Weights ─────────────────────────────────
        lines.append(self._header("STRATEGY WEIGHTS & P&L"))
        weights = portfolio_summary.get("strategy_weights", {})
        pnls    = portfolio_summary.get("strategy_pnl", {})
        for strat, w in weights.items():
            bar  = "█" * int(w * 40)
            pnl  = pnls.get(strat, 0)
            lines.append(
                grey(f"  {strat:<18}") +
                cyan(f"{w:.0%}  ") +
                blue(f"{bar:<20}") +
                f"  {pnl_colour(pnl)}"
            )
        lines.append("")

        # ── Recent Trades ────────────────────────────────────
        lines.append(self._header("RECENT TRADES (last 5)"))
        if trade_history.empty:
            lines.append(grey("  No completed trades yet"))
        else:
            hdr = f"  {'Instrument':<14}{'Dir':<8}{'Entry':<12}{'Exit':<12}{'P&L':<12}{'Reason'}"
            lines.append(grey(hdr))
            recent = trade_history.tail(5).iloc[::-1]
            for _, t in recent.iterrows():
                dir_  = green("L") if t.get("direction", 1) == 1 else red("S")
                pnl   = t.get("pnl", 0)
                lines.append(
                    f"  {bold(str(t.get('instrument','?'))):<14}"
                    f"{dir_:<18}"
                    f"{t.get('entry', 0):<12.5f}"
                    f"{t.get('exit', 0):<12.5f}"
                    f"{pnl_colour(pnl):<20}"
                    f"{grey(str(t.get('reason','?')))}"
                )

        lines.append("")
        lines.append(self._line("═"))
        lines.append(grey(f"  Press Ctrl+C to stop  |  Refreshes every 60s  |  Logs → logs/system.log"))
        lines.append("")

        print("\n".join(lines))

    def render_startup(self):
        """Show startup banner while system initialises."""
        os.system("clear" if os.name != "nt" else "cls")
        print(bold(cyan("""
  ████████╗██████╗  █████╗ ██████╗ ██╗███╗   ██╗ ██████╗
     ██╔══╝██╔══██╗██╔══██╗██╔══██╗██║████╗  ██║██╔════╝
     ██║   ██████╔╝███████║██║  ██║██║██╔██╗ ██║██║  ███╗
     ██║   ██╔══██╗██╔══██║██║  ██║██║██║╚██╗██║██║   ██║
     ██║   ██║  ██║██║  ██║██████╔╝██║██║ ╚████║╚██████╔╝
     ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚═╝╚═╝  ╚═══╝ ╚═════╝
        """)))
        print(bold("       FIRM OS  —  Professional Algorithmic Trading System"))
        print(grey("       Forex  |  Equities  |  Commodities  |  Python 3.11"))
        print()
        print(yellow("  ⚙  Initialising components..."))
        print()
