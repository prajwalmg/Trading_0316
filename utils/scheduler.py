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

def run_calibration_check(min_trades_per_bucket: int = 10) -> dict:
    """
    Bucket closed trades by confidence, compute win-rate per bucket,
    and auto-adjust MIN_CONFIDENCE in config/runtime_overrides.json.

    Buckets (confidence):
      low    : [0.00, 0.55)
      medium : [0.55, 0.70)
      high   : [0.70, 0.85)
      very_high : [0.85, 1.00]

    Logic:
      - If the lowest-confidence bucket (low) has WR < 40% and
        has enough trades, raise MIN_CONFIDENCE to 0.55.
      - If medium bucket also weak, raise to 0.70.
      - If high bucket is strong (WR ≥ 55%), lower MIN_CONFIDENCE
        back toward 0.55 (loosen).
      - Always clamp between 0.45 and 0.80.

    Returns dict with calibration results per bucket.
    """
    import glob as _glob

    BUCKETS = [
        ("low",       0.00, 0.55),
        ("medium",    0.55, 0.70),
        ("high",      0.70, 0.85),
        ("very_high", 0.85, 1.01),
    ]

    bucket_stats = {name: {"wins": 0, "total": 0} for name, _, _ in BUCKETS}

    for fpath in _glob.glob("logs/feedback/*_outcomes.jsonl"):
        try:
            with open(fpath) as fh:
                for line in fh:
                    try:
                        r    = json.loads(line)
                        conf = float(r.get("confidence", 0.0))
                        won  = bool(r.get("won", False))
                        for name, lo, hi in BUCKETS:
                            if lo <= conf < hi:
                                bucket_stats[name]["total"] += 1
                                if won:
                                    bucket_stats[name]["wins"] += 1
                                break
                    except Exception:
                        pass
        except Exception:
            pass

    results = {}
    for name, lo, hi in BUCKETS:
        total = bucket_stats[name]["total"]
        wins  = bucket_stats[name]["wins"]
        wr    = wins / total if total > 0 else None
        results[name] = {
            "range":   f"[{lo:.2f}, {hi:.2f})",
            "total":   total,
            "wins":    wins,
            "win_rate": wr,
            "sufficient": total >= min_trades_per_bucket,
        }
        if wr is not None:
            logger.info(
                f"Calibration bucket '{name}' "
                f"conf=[{lo:.2f},{hi:.2f}): "
                f"WR={wr:.1%} ({wins}/{total})"
            )

    # ── Determine new MIN_CONFIDENCE ──────────────────────────────
    overrides_path = "config/runtime_overrides.json"
    try:
        with open(overrides_path) as fh:
            overrides = json.load(fh)
    except Exception:
        overrides = {}

    from config.settings import MIN_CONFIDENCE as DEFAULT_MIN_CONF
    current_min = float(overrides.get("MIN_CONFIDENCE", DEFAULT_MIN_CONF))
    new_min     = current_min

    low_bucket  = results["low"]
    med_bucket  = results["medium"]
    high_bucket = results["high"]

    # Raise threshold if low-confidence trades are losing money
    if low_bucket["sufficient"] and low_bucket["win_rate"] is not None:
        if low_bucket["win_rate"] < 0.40:
            new_min = max(new_min, 0.55)
            logger.info("Calibration: low bucket WR<40% → MIN_CONFIDENCE ≥ 0.55")

    if med_bucket["sufficient"] and med_bucket["win_rate"] is not None:
        if med_bucket["win_rate"] < 0.40:
            new_min = max(new_min, 0.70)
            logger.info("Calibration: medium bucket WR<40% → MIN_CONFIDENCE ≥ 0.70")

    # Loosen threshold if high-confidence trades perform well
    if high_bucket["sufficient"] and high_bucket["win_rate"] is not None:
        if high_bucket["win_rate"] >= 0.55 and new_min > 0.55:
            new_min = max(0.55, new_min - 0.05)
            logger.info("Calibration: high bucket WR≥55% → relax MIN_CONFIDENCE")

    # Clamp
    new_min = round(float(max(0.45, min(new_min, 0.80))), 4)

    if abs(new_min - current_min) >= 0.01:
        overrides["MIN_CONFIDENCE"] = new_min
        os.makedirs("config", exist_ok=True)
        with open(overrides_path, "w") as fh:
            json.dump(overrides, fh, indent=2)
        logger.info(
            f"Calibration: MIN_CONFIDENCE updated "
            f"{current_min:.2f} → {new_min:.2f} "
            f"(saved to {overrides_path})"
        )
    else:
        logger.info(
            f"Calibration: MIN_CONFIDENCE unchanged at {current_min:.2f}"
        )

    results["new_min_confidence"] = new_min
    results["prev_min_confidence"] = current_min
    return results


def run_scheduler(mode: str = "swing"):
    logger.info(f"Scheduler started | mode={mode} | interval={RETRAIN_INTERVAL_DAYS} days")
    
    # Check immediately on startup
    if should_retrain(mode):
        logger.info(f"Retrain needed on startup")
        run_retrain(mode)

    # Check calibration weekly (Sunday ~21:00 UTC)
    now = datetime.now(timezone.utc)
    if now.weekday() == 6 and now.hour == 21:
        logger.info("=== Running Confidence Calibration Check ===")
        cal = run_calibration_check()
        logger.info(
            f"Calibration complete — "
            f"MIN_CONFIDENCE: {cal.get('prev_min_confidence')} → "
            f"{cal.get('new_min_confidence')}"
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
