#!/usr/bin/env python3
"""Save current paper broker state to JSON so it can be restored after restart.

Usage:
    venv/bin/python scripts/save_state.py

Reads logs/paper_trades.csv to reconstruct NAV.
Reads logs/system.log to find open positions.
Writes logs/broker_state.json.
"""
import sys, os, json, re
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STATE_FILE = "logs/broker_state.json"

def parse_open_positions_from_log():
    """Scan system.log for open/close events to reconstruct open positions."""
    log_path = Path("logs/system.log")
    if not log_path.exists():
        return []

    open_re  = re.compile(r"Position opened: (\S+) \| (LONG|SHORT) (\d+) \| entry=([\d.]+) \| sl=([\d.]+) \| tp=([\d.]+)")
    close_re = re.compile(r"closed.*?(\S+=X|\S+USD|\S+=F|\S+-USD)")

    opened  = {}  # ticker → list of open events
    closed_count = {}

    with open(log_path) as f:
        for line in f:
            m = open_re.search(line)
            if m:
                ticker, direction, units, entry, sl, tp = m.groups()
                opened.setdefault(ticker, []).append({
                    "instrument": ticker,
                    "direction":  1 if direction == "LONG" else -1,
                    "units":      int(units),
                    "entry":      float(entry),
                    "sl":         float(sl),
                    "tp":         float(tp),
                    "time":       line[:23],
                })

    # Count closed trades per ticker from paper_trades.csv
    import pandas as pd
    try:
        df = pd.read_csv("logs/paper_trades.csv")
        for ticker, count in df["instrument"].value_counts().items() if "instrument" in df.columns else []:
            closed_count[ticker] = count
    except Exception:
        pass

    # Open = opened - closed
    open_positions = []
    for ticker, events in opened.items():
        n_closed = closed_count.get(ticker, 0)
        remaining = events[n_closed:]  # latest events assumed open
        open_positions.extend(remaining)

    return open_positions


def compute_nav():
    """Compute current NAV from trade history."""
    try:
        import pandas as pd
        from config.settings import INITIAL_CAPITAL
        df = pd.read_csv("logs/paper_trades.csv")
        if df.empty or "pnl" not in df.columns:
            return INITIAL_CAPITAL
        return INITIAL_CAPITAL + df["pnl"].sum()
    except Exception:
        return 10000.0


def main():
    os.makedirs("logs", exist_ok=True)
    nav = compute_nav()
    open_pos = parse_open_positions_from_log()

    state = {
        "saved_at":      datetime.now().isoformat(),
        "nav":           nav,
        "open_positions": open_pos,
    }

    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

    print(f"State saved to {STATE_FILE}")
    print(f"  NAV: {nav:.2f}")
    print(f"  Open positions: {len(open_pos)}")
    for p in open_pos:
        print(f"    {p['instrument']} {'Long' if p['direction']==1 else 'Short'} "
              f"entry={p['entry']:.5f} sl={p['sl']:.5f} tp={p['tp']:.5f}")


if __name__ == "__main__":
    main()
