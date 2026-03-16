"""
================================================================
  utils/report.py
  Auto-generated HTML daily performance report.

  Output:  logs/reports/report_YYYYMMDD_HHMMSS.html

  Sections:
    1. Account snapshot (NAV, drawdown, P&L)
    2. Open positions table
    3. Trade history table (last 50 trades)
    4. Risk metrics (VaR, CVaR, daily/weekly limits)
    5. Per-instrument win-rate summary
================================================================
"""
import logging
import os
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger("trading_firm.report")

_REPORT_DIR = "logs/reports"

_STYLE = """
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0d1117; color: #c9d1d9;
    font-family: 'Courier New', monospace; font-size: 13px;
    padding: 24px;
  }
  h1 { color: #58a6ff; font-size: 20px; margin-bottom: 8px; }
  h2 { color: #3fb950; font-size: 15px; margin: 24px 0 8px; border-bottom: 1px solid #30363d; padding-bottom: 4px; }
  .meta { color: #8b949e; font-size: 11px; margin-bottom: 20px; }
  .grid { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 16px; }
  .card {
    background: #161b22; border: 1px solid #30363d; border-radius: 6px;
    padding: 12px 18px; min-width: 160px;
  }
  .card .label { color: #8b949e; font-size: 11px; text-transform: uppercase; }
  .card .value { font-size: 18px; font-weight: bold; margin-top: 2px; }
  .pos  { color: #3fb950; }
  .neg  { color: #f85149; }
  .neu  { color: #c9d1d9; }
  table {
    width: 100%; border-collapse: collapse;
    margin-bottom: 16px; background: #161b22;
  }
  th {
    background: #21262d; color: #8b949e; text-align: left;
    padding: 6px 10px; font-size: 11px; text-transform: uppercase;
    border-bottom: 1px solid #30363d;
  }
  td { padding: 5px 10px; border-bottom: 1px solid #21262d; }
  tr:hover td { background: #1c2128; }
  .win  { color: #3fb950; }
  .loss { color: #f85149; }
  .flat { color: #8b949e; }
</style>
"""


def _fmt_pnl(v: float) -> str:
    cls = "win" if v > 0 else ("loss" if v < 0 else "flat")
    return f'<span class="{cls}">{v:+,.2f}</span>'


def _fmt_pct(v: float) -> str:
    cls = "win" if v > 0 else ("loss" if v < 0 else "flat")
    return f'<span class="{cls}">{v:+.2%}</span>'


def generate_daily_report(
    risk_engine,
    trade_history: pd.DataFrame,
    open_positions: list,
    instruments:   list = None,
) -> str:
    """
    Generate a dark-themed HTML report and write it to
    logs/reports/report_YYYYMMDD_HHMMSS.html.

    Parameters
    ----------
    risk_engine    : RiskEngine instance
    trade_history  : pd.DataFrame of closed trades
    open_positions : list of open position dicts
    instruments    : list of ticker strings (optional)

    Returns
    -------
    str — full path to the generated HTML file
    """
    os.makedirs(_REPORT_DIR, exist_ok=True)
    now      = datetime.now(timezone.utc)
    ts_str   = now.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(_REPORT_DIR, f"report_{ts_str}.html")

    rpt = risk_engine.report()
    nav       = rpt.get("nav",       0)
    peak_nav  = rpt.get("peak_nav",  nav)
    daily_pnl = rpt.get("daily_pnl", 0)
    total_pnl = rpt.get("total_pnl", 0)
    dd_str    = rpt.get("drawdown",  "0.00%")
    n_trades  = rpt.get("total_trades", 0)

    # Win rate from trade history
    win_rate = 0.0
    avg_win  = 0.0
    avg_loss = 0.0
    if not trade_history.empty and "pnl" in trade_history.columns:
        winners  = trade_history[trade_history["pnl"] > 0]["pnl"]
        losers   = trade_history[trade_history["pnl"] <= 0]["pnl"]
        win_rate = len(winners) / max(len(trade_history), 1)
        avg_win  = winners.mean() if len(winners) else 0.0
        avg_loss = losers.mean()  if len(losers)  else 0.0

    # VaR / CVaR
    var_data = risk_engine.var_cvar_report() if hasattr(risk_engine, "var_cvar_report") else {}

    # ── Account snapshot ─────────────────────────────────────────
    nav_cls   = "pos" if nav >= risk_engine.initial_capital else "neg"
    dpnl_cls  = "pos" if daily_pnl >= 0 else "neg"
    tpnl_cls  = "pos" if total_pnl >= 0 else "neg"
    wr_cls    = "pos" if win_rate >= 0.50 else "neg"

    cards_html = f"""
    <div class="grid">
      <div class="card"><div class="label">NAV</div>
        <div class="value {nav_cls}">${nav:,.2f}</div></div>
      <div class="card"><div class="label">Daily P&amp;L</div>
        <div class="value {dpnl_cls}">{daily_pnl:+,.2f}</div></div>
      <div class="card"><div class="label">Total P&amp;L</div>
        <div class="value {tpnl_cls}">{total_pnl:+,.2f}</div></div>
      <div class="card"><div class="label">Drawdown</div>
        <div class="value neg">{dd_str}</div></div>
      <div class="card"><div class="label">Win Rate</div>
        <div class="value {wr_cls}">{win_rate:.1%}</div></div>
      <div class="card"><div class="label">Total Trades</div>
        <div class="value neu">{n_trades}</div></div>
    </div>
    """

    # ── Risk metrics ─────────────────────────────────────────────
    if var_data:
        var_html = f"""
        <h2>Risk Metrics</h2>
        <div class="grid">
          <div class="card"><div class="label">VaR 95%</div>
            <div class="value neg">{var_data.get('var_pct',0):.2%}
            (${var_data.get('var_usd',0):,.0f})</div></div>
          <div class="card"><div class="label">CVaR 95%</div>
            <div class="value neg">{var_data.get('cvar_pct',0):.2%}
            (${var_data.get('cvar_usd',0):,.0f})</div></div>
          <div class="card"><div class="label">Observations</div>
            <div class="value neu">{var_data.get('n_obs',0)}</div></div>
        </div>
        """
    else:
        var_html = "<h2>Risk Metrics</h2><p class='meta'>Need ≥20 trades for VaR/CVaR.</p>"

    # ── Open positions ────────────────────────────────────────────
    if open_positions:
        rows = ""
        for p in open_positions:
            inst   = p.get("instrument", "")
            side   = "LONG" if p.get("direction", 1) == 1 else "SHORT"
            units  = p.get("units", 0)
            entry  = p.get("entry", 0)
            sl     = p.get("sl", 0)
            tp     = p.get("tp", 0)
            rows += (
                f"<tr><td>{inst}</td><td>{side}</td>"
                f"<td>{units:,}</td><td>{entry:.5f}</td>"
                f"<td>{sl:.5f}</td><td>{tp:.5f}</td></tr>"
            )
        open_pos_html = f"""
        <h2>Open Positions ({len(open_positions)})</h2>
        <table>
          <tr><th>Instrument</th><th>Side</th><th>Units</th>
              <th>Entry</th><th>SL</th><th>TP</th></tr>
          {rows}
        </table>
        """
    else:
        open_pos_html = "<h2>Open Positions</h2><p class='meta'>None</p>"

    # ── Trade history ─────────────────────────────────────────────
    if not trade_history.empty:
        last50 = trade_history.tail(50)
        rows   = ""
        for _, r in last50.iterrows():
            inst   = r.get("instrument", r.get("ticker", ""))
            side   = "LONG" if r.get("direction", 1) == 1 else "SHORT"
            entry  = r.get("entry",  0)
            exit_  = r.get("exit",   r.get("close", 0))
            pnl    = r.get("pnl",    0)
            t      = r.get("time",   r.get("close_time", ""))
            rows  += (
                f"<tr><td>{inst}</td><td>{side}</td>"
                f"<td>{entry:.5f}</td><td>{exit_:.5f}</td>"
                f"<td>{_fmt_pnl(pnl)}</td><td>{str(t)[:16]}</td></tr>"
            )
        hist_html = f"""
        <h2>Trade History (last 50)</h2>
        <table>
          <tr><th>Instrument</th><th>Side</th><th>Entry</th>
              <th>Exit</th><th>P&amp;L</th><th>Time</th></tr>
          {rows}
        </table>
        """
    else:
        hist_html = "<h2>Trade History</h2><p class='meta'>No trades yet.</p>"

    # ── Assemble ──────────────────────────────────────────────────
    ts_display = now.strftime("%Y-%m-%d %H:%M UTC")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Trading Firm — Daily Report {ts_display}</title>
  {_STYLE}
</head>
<body>
  <h1>Trading Firm OS — Daily Performance Report</h1>
  <p class="meta">Generated: {ts_display} &nbsp;|&nbsp;
     Instruments: {len(instruments) if instruments else '?'} &nbsp;|&nbsp;
     Peak NAV: ${peak_nav:,.2f}</p>

  {cards_html}
  {var_html}
  {open_pos_html}
  {hist_html}
</body>
</html>
"""

    with open(filename, "w", encoding="utf-8") as fh:
        fh.write(html)

    logger.info(f"Daily report written: {filename}")
    return filename
