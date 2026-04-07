#!/usr/bin/env python3
"""
scripts/refresh_dukascopy_5m.py
Incrementally updates Dukascopy 5m parquet files by fetching only
the missing hours since the last cached bar, then appending.

Safe to re-run — skips pairs where cache is already up to date.
"""
import os, sys, glob, logging
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("logs/dukascopy_refresh.log")],
)
logger = logging.getLogger("dukascopy.refresh")
os.makedirs("logs", exist_ok=True)

from data.dukascopy import fetch_ohlcv, SYMBOL_MAP, CACHE_DIR

INTRADAY_PAIRS = [
    "USDJPY=X", "GBPUSD=X", "EURJPY=X", "EURUSD=X",
    "GBPJPY=X", "AUDUSD=X", "USDCAD=X", "EURGBP=X",
]

def incremental_update(ticker: str, interval: str = "5m"):
    sym = SYMBOL_MAP.get(ticker)
    if sym is None:
        logger.warning(f"{ticker}: no SYMBOL_MAP entry — skip")
        return

    pattern = f"{CACHE_DIR}/{sym}_{interval}_*.parquet"
    files   = sorted(glob.glob(pattern))
    if not files:
        logger.warning(f"{ticker}: no cached file found — run full fetch first")
        return

    # Load the most complete existing file
    existing_file = max(files, key=os.path.getsize)
    df_existing   = pd.read_parquet(existing_file)
    if df_existing.empty:
        logger.warning(f"{ticker}: existing file empty")
        return

    last_ts = df_existing.index[-1]
    if hasattr(last_ts, "to_pydatetime"):
        last_ts = last_ts.to_pydatetime()
    last_ts = last_ts.replace(tzinfo=None)

    now = datetime.utcnow()
    gap_hours = (now - last_ts).total_seconds() / 3600

    logger.info(f"{ticker}: last bar = {last_ts} | gap = {gap_hours:.1f}h")

    if gap_hours < 2:
        logger.info(f"{ticker}: already up to date — skip")
        return

    # Fetch missing slice
    fetch_start = last_ts.replace(minute=0, second=0, microsecond=0)
    fetch_end   = now

    logger.info(f"{ticker}: fetching {fetch_start} → {fetch_end} ({gap_hours:.0f}h)")
    df_new = fetch_ohlcv(ticker, fetch_start, fetch_end, interval)

    if df_new.empty:
        logger.warning(f"{ticker}: no new data returned")
        return

    # Merge and deduplicate
    df_merged = pd.concat([df_existing, df_new])
    df_merged = df_merged[~df_merged.index.duplicated(keep="last")].sort_index()

    # Write with updated end-date filename
    orig_start = df_existing.index[0]
    if hasattr(orig_start, "to_pydatetime"):
        orig_start = orig_start.to_pydatetime().replace(tzinfo=None)
    new_key    = (f"{sym}_{interval}_"
                  f"{orig_start.strftime('%Y%m%d')}_"
                  f"{now.strftime('%Y%m%d')}")
    new_cache  = f"{CACHE_DIR}/{new_key}.parquet"
    df_merged.to_parquet(new_cache)

    new_rows = len(df_merged) - len(df_existing)
    logger.info(f"{ticker}: saved {new_cache} "
                f"| +{new_rows} new bars | total={len(df_merged)}")


if __name__ == "__main__":
    logger.info("Starting incremental Dukascopy 5m refresh...")
    for pair in INTRADAY_PAIRS:
        try:
            incremental_update(pair, "5m")
        except Exception as e:
            logger.error(f"{pair}: {e}")
    logger.info("Done.")
