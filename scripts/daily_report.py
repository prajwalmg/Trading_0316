#!/usr/bin/env python3
"""
Daily performance report.
Run at market close or schedule via cron.
Sends summary to Telegram.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from notifications.telegram import _send

def generate_report():
    today = datetime.now().strftime('%d %b %Y')
    report_lines = [
        f"📊 <b>Daily Report — {today}</b>",
        "─────────────────────"
    ]

    try:
        df = pd.read_csv('logs/paper_trades.csv')
        df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')

        if df.empty:
            _send("📊 Daily Report\nNo trades yet.")
            return

        today_dt = pd.Timestamp.now(tz='UTC').normalize()
        today_trades = df[df['time'] >= today_dt]

        total_pnl    = df['pnl'].sum()
        total_trades = len(df)
        wins         = (df['pnl'] > 0).sum()
        win_rate     = wins / total_trades if total_trades > 0 else 0
        nav          = 10000 + total_pnl

        equity = 10000 + df['pnl'].cumsum()
        peak   = equity.expanding().max()
        dd     = ((equity - peak) / peak).min()

        df['date']  = df['time'].dt.date
        daily_pnl   = df.groupby('date')['pnl'].sum()
        daily_ret   = daily_pnl / 10000
        sharpe      = (daily_ret.mean() / (daily_ret.std() + 1e-9) * np.sqrt(252)) if len(daily_ret) > 5 else 0

        report_lines += [
            f"NAV:          €{nav:,.2f}",
            f"Total PnL:    €{total_pnl:+,.2f}",
            f"Sharpe:       {sharpe:.2f}",
            f"Max DD:       {dd:.1%}",
            f"Win Rate:     {win_rate:.1%}",
            f"Total Trades: {total_trades}",
            "─────────────────────",
        ]

        if not today_trades.empty:
            t_pnl = today_trades['pnl'].sum()
            t_wins = (today_trades['pnl'] > 0).sum()
            t_wr = t_wins / len(today_trades)
            report_lines.append(
                f"Today: {len(today_trades)} trades | €{t_pnl:+.2f} | WR {t_wr:.0%}")
        else:
            report_lines.append("Today: No trades")

        report_lines.append("─────────────────────")
        pair_stats = df.groupby('instrument').agg(
            trades=('pnl', 'count'),
            pnl=('pnl', 'sum'),
            wr=('pnl', lambda x: (x > 0).mean())
        ).sort_values('pnl', ascending=False)

        for pair, row in pair_stats.iterrows():
            p = pair.replace('=X', '')
            emoji = '✅' if row['pnl'] > 0 else '❌'
            report_lines.append(
                f"{emoji} {p:<8} €{row['pnl']:>+7.2f} WR:{row['wr']:.0%} ({row['trades']:.0f}t)")

        if dd < -0.03:
            report_lines.append(f"\n⚠️ DRAWDOWN ALERT: {dd:.1%}")

        _send('\n'.join(report_lines))
        print("Report sent successfully")

    except FileNotFoundError:
        _send(f"📊 Daily Report — {today}\nNo trade log found yet.")
    except Exception as e:
        _send(f"⚠️ Report error: {e}")
        raise

if __name__ == '__main__':
    generate_report()
