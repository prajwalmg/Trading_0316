import sqlite3
import pandas as pd
import os
from datetime import datetime

DB_PATH = 'data/trades.db'

def init_db():
    os.makedirs('data', exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id          INTEGER PRIMARY KEY,
            instrument  TEXT NOT NULL,
            direction   TEXT NOT NULL,
            entry       REAL,
            exit_price  REAL,
            units       REAL,
            pnl         REAL,
            reason      TEXT,
            entry_time  TEXT,
            exit_time   TEXT,
            confidence  REAL,
            regime      TEXT,
            system      TEXT DEFAULT 'swing',
            created_at  TEXT DEFAULT (datetime('now'))
        )
    ''')
    # Add system column to existing DB if missing (migration)
    try:
        conn.execute("ALTER TABLE trades ADD COLUMN system TEXT DEFAULT 'swing'")
        conn.commit()
    except Exception:
        pass  # Column already exists
    conn.execute('''
        CREATE TABLE IF NOT EXISTS equity_curve (
            id          INTEGER PRIMARY KEY,
            timestamp   TEXT,
            nav         REAL,
            daily_pnl   REAL
        )
    ''')
    conn.commit()
    conn.close()

def log_trade(instrument, direction, entry, exit_price, units, pnl, reason,
              entry_time, exit_time, confidence=None, regime=None, system='swing'):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        INSERT INTO trades
        (instrument, direction, entry, exit_price, units, pnl, reason,
         entry_time, exit_time, confidence, regime, system)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (instrument, direction, entry, exit_price, units, pnl, reason,
          str(entry_time), str(exit_time), confidence, regime, system))
    conn.commit()
    conn.close()

def get_trades(days=None) -> pd.DataFrame:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    if days:
        query = f"""
            SELECT * FROM trades
            WHERE exit_time >= datetime('now', '-{days} days')
            ORDER BY exit_time DESC
        """
    else:
        query = "SELECT * FROM trades ORDER BY exit_time DESC"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_stats() -> dict:
    df = get_trades()
    if df.empty:
        return {}
    import numpy as np
    total_pnl = df['pnl'].sum()
    wins = (df['pnl'] > 0).sum()
    equity = 10000 + df['pnl'].cumsum()
    peak = equity.expanding().max()
    max_dd = ((equity - peak) / peak).min()
    return {
        'total_trades': len(df),
        'total_pnl':    round(total_pnl, 2),
        'win_rate':     round(wins / len(df), 4),
        'max_drawdown': round(max_dd, 4),
        'nav':          round(10000 + total_pnl, 2),
    }

def get_system_stats(days: int = 1) -> dict:
    """Return per-system trade stats for daily_report."""
    df = get_trades(days=days)
    if df.empty or 'system' not in df.columns:
        return {}
    stats = {}
    for sys_name in df['system'].dropna().unique():
        sys_df = df[df['system'] == sys_name]
        stats[sys_name] = {
            'trades': len(sys_df),
            'pnl':    round(float(sys_df['pnl'].sum()), 2),
            'wr':     float((sys_df['pnl'] > 0).mean()) if len(sys_df) > 0 else 0.0,
        }
    return stats
