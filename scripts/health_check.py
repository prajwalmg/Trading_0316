#!/usr/bin/env python3
"""
Interactive health check — prints a full status report to stdout.
Run with: venv/bin/python scripts/health_check.py
"""
import sys, os, sqlite3, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH   = os.path.join(BASE_DIR, 'data', 'trades.db')
LOG_FILE  = os.path.join(BASE_DIR, 'logs', 'system.log')
PID_FILES = {
    'intraday': os.path.join(BASE_DIR, 'logs', 'intraday.pid'),
    'swing':    os.path.join(BASE_DIR, 'logs', 'swing.pid'),
}
INITIAL_CAPITAL = 10_000.0
QUESTDB_URL = "http://localhost:9000"

SEP = "=" * 60


def _read_pid(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except Exception:
        return None


def _pid_alive(pid):
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


# ── 1. Process status ────────────────────────────────────────────
section("1. PROCESS STATUS")

ps_result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
ps_lines  = ps_result.stdout.splitlines()

for mode, pid_file in PID_FILES.items():
    pid    = _read_pid(pid_file)
    alive  = _pid_alive(pid)

    # Also check ps output for running processes
    running_in_ps = any(f'--mode {mode}' in line for line in ps_lines)
    running_in_ps = running_in_ps or any(
        ('main.py' in line and mode in line) for line in ps_lines
    )

    pid_str = str(pid) if pid else "N/A"
    if alive:
        status = f"RUNNING  (pid={pid_str})"
    elif running_in_ps:
        status = f"RUNNING  (found in ps, pid file={pid_str})"
    else:
        status = f"STOPPED  (last pid={pid_str})"

    print(f"  {mode:<12}: {status}")

# Check for multi-asset as well
multi_pid_file = os.path.join(BASE_DIR, 'logs', 'multi_asset.pid')
multi_pid   = _read_pid(multi_pid_file)
multi_alive = _pid_alive(multi_pid)
multi_ps    = any('--mode multi' in line for line in ps_lines)
if multi_alive or multi_ps:
    pid_str = str(multi_pid) if multi_pid else "N/A"
    print(f"  {'multi':<12}: RUNNING  (pid={pid_str})")


# ── 2. Last 5 trades ─────────────────────────────────────────────
section("2. LAST 5 TRADES")

try:
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("""
        SELECT id, instrument, direction, entry, exit_price, units, pnl,
               reason, entry_time, exit_time, confidence, regime, system
        FROM trades
        ORDER BY id DESC
        LIMIT 5
    """)
    rows = cur.fetchall()
    conn.close()

    if rows:
        header = f"  {'ID':>5}  {'Instrument':<12}  {'Dir':<5}  {'Entry':>9}  {'Exit':>9}  {'Units':>8}  {'PnL':>8}  {'System':<10}  {'ExitTime'}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for r in rows:
            id_, instr, dire, entry, exit_p, units, pnl, reason, entry_t, exit_t, conf, regime, sys_name = r
            pnl_str = f"{pnl:+.4f}" if pnl is not None else "   open"
            exit_p_str = f"{exit_p:.5f}" if exit_p else "  open "
            entry_str  = f"{entry:.5f}"  if entry  else "  ----  "
            units_str  = f"{units:.0f}"  if units  else "   --"
            sys_str    = sys_name or "?"
            exit_t_str = str(exit_t or "open")[:19]
            print(f"  {id_:>5}  {instr:<12}  {dire:<5}  {entry_str:>9}  {exit_p_str:>9}  {units_str:>8}  {pnl_str:>8}  {sys_str:<10}  {exit_t_str}")
    else:
        print("  No trades in database.")
except Exception as e:
    print(f"  ERROR reading trades: {e}")


# ── 3. Win rate & total PnL ──────────────────────────────────────
section("3. PERFORMANCE SUMMARY")

try:
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    cur.execute("SELECT count(*) FROM trades WHERE pnl IS NOT NULL")
    total = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM trades WHERE pnl > 0")
    wins = cur.fetchone()[0]

    cur.execute("SELECT sum(pnl) FROM trades WHERE pnl IS NOT NULL")
    total_pnl = cur.fetchone()[0] or 0.0

    cur.execute("SELECT count(*) FROM trades WHERE exit_price IS NULL OR pnl IS NULL")
    open_trades = cur.fetchone()[0]

    conn.close()

    win_rate = wins / total if total > 0 else 0.0
    nav      = INITIAL_CAPITAL + total_pnl

    print(f"  Total closed trades : {total}")
    print(f"  Open/pending trades : {open_trades}")
    print(f"  Wins                : {wins}")
    print(f"  Win rate            : {win_rate:.1%}")
    print(f"  Total PnL           : {total_pnl:+.4f}")
    print(f"  Estimated NAV       : {nav:,.2f}  (base={INITIAL_CAPITAL:,.0f})")

    # Per-system breakdown
    cur2 = sqlite3.connect(DB_PATH).cursor()
    cur2.execute("""
        SELECT system, count(*), sum(pnl),
               sum(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)
        FROM trades
        WHERE pnl IS NOT NULL
        GROUP BY system
    """)
    rows = cur2.fetchall()
    if rows:
        print(f"\n  Per-system breakdown:")
        for sys_name, cnt, spnl, swins in rows:
            wr = swins / cnt if cnt else 0
            print(f"    {str(sys_name):<12}  trades={cnt:>4}  pnl={spnl:+.4f}  wr={wr:.1%}")
except Exception as e:
    print(f"  ERROR computing stats: {e}")


# ── 4. QuestDB reachability ──────────────────────────────────────
section("4. QUESTDB STATUS")

try:
    import urllib.request
    url = f"{QUESTDB_URL}/exec?query=SELECT+count()+FROM+trades"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=3) as resp:
        import json
        data = json.loads(resp.read().decode())
        count = data.get('dataset', [[0]])[0][0]
        print(f"  QuestDB : REACHABLE  ({QUESTDB_URL})")
        print(f"  Row count in QuestDB trades: {count}")
except Exception as e:
    print(f"  QuestDB : UNREACHABLE — {e}")
    print(f"  (QuestDB may not be running; SQLite is the primary store)")


# ── 5. Recent ERRORs in log ──────────────────────────────────────
section("5. RECENT LOG ERRORS (last 200 lines)")

try:
    with open(LOG_FILE, 'r') as f:
        all_lines = f.readlines()
    last_lines = all_lines[-200:]
    errors = [l.rstrip() for l in last_lines if '[ERROR]' in l or '[CRITICAL]' in l]
    if errors:
        print(f"  Found {len(errors)} ERROR/CRITICAL lines:")
        for e in errors[-10:]:   # show at most 10
            print(f"    {e}")
        if len(errors) > 10:
            print(f"    ... ({len(errors) - 10} more not shown)")
    else:
        print("  No ERROR lines in last 200 log lines.")
except FileNotFoundError:
    # Try today's dated log
    today = datetime.now().strftime('%Y%m%d')
    alt_log = os.path.join(BASE_DIR, 'logs', f'paper_trading_{today}.log')
    try:
        with open(alt_log, 'r') as f:
            last_lines = f.readlines()[-200:]
        errors = [l.rstrip() for l in last_lines if 'ERROR' in l or 'CRITICAL' in l]
        if errors:
            print(f"  Found {len(errors)} ERROR lines in {alt_log}:")
            for e in errors[-10:]:
                print(f"    {e}")
        else:
            print(f"  No ERROR lines in last 200 lines of {alt_log}.")
    except Exception as e2:
        print(f"  Could not read log: {e2}")
except Exception as e:
    print(f"  ERROR reading log: {e}")


# ── 6. NAV estimate ─────────────────────────────────────────────
section("6. NAV ESTIMATE")

try:
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("SELECT sum(pnl) FROM trades WHERE pnl IS NOT NULL")
    total_pnl = cur.fetchone()[0] or 0.0

    # Check for open positions (no exit_price)
    cur.execute("SELECT count(*) FROM trades WHERE exit_price IS NULL")
    open_cnt = cur.fetchone()[0]
    conn.close()

    nav = INITIAL_CAPITAL + total_pnl
    print(f"  Initial capital     : {INITIAL_CAPITAL:>12,.2f}")
    print(f"  Realised PnL        : {total_pnl:>+12.4f}")
    print(f"  NAV (realised)      : {nav:>12,.2f}")
    print(f"  Open positions      : {open_cnt}  (unrealised PnL not included)")
    pct = (nav / INITIAL_CAPITAL - 1) * 100
    print(f"  Return              : {pct:>+12.2f}%")
except Exception as e:
    print(f"  ERROR estimating NAV: {e}")

print(f"\n{SEP}")
print(f"  Health check completed at {datetime.now():%Y-%m-%d %H:%M:%S}")
print(SEP + "\n")
