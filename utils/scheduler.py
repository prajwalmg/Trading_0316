"""
Walk-forward retraining scheduler.
Run in a third terminal alongside paper trading.

Usage:
    python utils/scheduler.py swing
    python utils/scheduler.py intraday
"""
import os, json
import sys
import time
import subprocess
import logging
from datetime import datetime, timezone

logger = logging.getLogger("trading_firm.scheduler")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

RETRAIN_INTERVAL_DAYS = 5
LAST_TRAIN_FILE       = "models/saved/last_train_{mode}.txt"

def should_retrain(mode: str) -> bool:
    # Check scheduled retrain
    path = LAST_TRAIN_FILE.format(mode=mode)
    if not os.path.exists(path):
        return True
    with open(path) as f:
        last = datetime.fromisoformat(f.read().strip())
    days_since = (datetime.now(timezone.utc) - last).days
    if days_since >= RETRAIN_INTERVAL_DAYS:
        logger.info(f"Scheduled retrain due — {days_since} days since last")
        return True

    # Check performance-triggered retrain
    import glob
    wins, total = 0, 0
    for path in glob.glob("logs/feedback/*_outcomes.jsonl"):
        lines = open(path).readlines()[-20:]   # last 20 trades
        for line in lines:
            try:
                r = json.loads(line)
                total += 1
                if r.get("won"):
                    wins += 1
            except Exception:
                continue

    if total >= 20:
        win_rate = wins / total
        if win_rate < 0.40:
            logger.info(
                f"Performance retrain triggered — "
                f"win rate {win_rate:.1%} below 40% over last {total} trades"
            )
            return True

    return False


def mark_retrained(mode: str):
    os.makedirs("models/saved", exist_ok=True)
    path = LAST_TRAIN_FILE.format(mode=mode)
    with open(path, "w") as f:
        f.write(datetime.now(timezone.utc).isoformat())
    logger.info(f"Marked {mode} retrain timestamp")

def run_retrain(mode: str):
    script = "main.py" if mode == "swing" else "intraday_forex.py"
    logger.info(f"Starting {mode} retrain — {script}")
    result = subprocess.run(
        ["python", script, "--mode", "train"],
        capture_output=False,
    )
    if result.returncode == 0:
        mark_retrained(mode)
        logger.info(f"{mode} retrain complete")
    else:
        logger.error(f"{mode} retrain failed — returncode {result.returncode}")

def run_scheduler(mode: str = "swing"):
    logger.info(f"Scheduler started | mode={mode} | interval={RETRAIN_INTERVAL_DAYS} days")
    
    # Check immediately on startup
    if should_retrain(mode):
        logger.info(f"Retrain needed on startup")
        run_retrain(mode)

    # Check calibration weekly
    now = datetime.now(timezone.utc)
    if now.weekday() == 6 and now.hour == 21:   # Sunday noon
        from execution.broker import PaperBroker
        b      = PaperBroker(initial_capital=1000)
        report = b.calibration_report()
        logger.info("=== Confidence Calibration Report ===")
        for bucket, data in report.items():
            logger.info(
                f"  {bucket}: WR={data['win_rate']:.1%} | "
                f"Trades={data['trades']} | {data['verdict']}"
            )

    while True:
        now = datetime.now(timezone.utc)
        # Check once per day at midnight UTC
        if now.hour == 0 and now.minute < 5:
            if should_retrain(mode):
                run_retrain(mode)
        time.sleep(300)  # check every 5 minutes


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "swing"
    if mode not in ("swing", "intraday"):
        print("Usage: python utils/scheduler.py [swing|intraday]")
        sys.exit(1)
    run_scheduler(mode)
