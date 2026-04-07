"""data/unified_pipeline.py — Unified data pipeline with QuestDB as primary cache.

Routes by asset class:
  forex      → Dukascopy (tick/OHLCV) with yfinance fallback
  crypto     → CCXT/Binance with yfinance fallback
  equity     → Polygon.io (massive) with yfinance fallback
  commodity  → yfinance (futures) with FMP fallback

Incremental fetch: reads max(ts) from QuestDB, fetches only the delta.
Multi-timeframe: 1h base, 4H and 1D resampled on-the-fly.
"""
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger("trading_firm.unified_pipeline")

# ---------------------------------------------------------------------------
# Instrument registry — 27 swing + 10 intraday
# ---------------------------------------------------------------------------
INSTRUMENT_REGISTRY = {
    # Forex swing (17)
    "EURUSD=X":  {"asset_class": "forex",     "source": "dukascopy", "base_tf": "1h",  "swing": True,  "intraday": False},
    "GBPUSD=X":  {"asset_class": "forex",     "source": "dukascopy", "base_tf": "1h",  "swing": True,  "intraday": False},
    "USDJPY=X":  {"asset_class": "forex",     "source": "dukascopy", "base_tf": "1h",  "swing": True,  "intraday": False},
    "AUDUSD=X":  {"asset_class": "forex",     "source": "dukascopy", "base_tf": "1h",  "swing": True,  "intraday": False},
    "USDCAD=X":  {"asset_class": "forex",     "source": "dukascopy", "base_tf": "1h",  "swing": True,  "intraday": False},
    "NZDUSD=X":  {"asset_class": "forex",     "source": "dukascopy", "base_tf": "1h",  "swing": True,  "intraday": False},
    "USDCHF=X":  {"asset_class": "forex",     "source": "dukascopy", "base_tf": "1h",  "swing": True,  "intraday": False},
    "EURGBP=X":  {"asset_class": "forex",     "source": "dukascopy", "base_tf": "1h",  "swing": True,  "intraday": False},
    "EURJPY=X":  {"asset_class": "forex",     "source": "dukascopy", "base_tf": "1h",  "swing": True,  "intraday": False},
    "GBPJPY=X":  {"asset_class": "forex",     "source": "dukascopy", "base_tf": "1h",  "swing": True,  "intraday": False},
    "AUDJPY=X":  {"asset_class": "forex",     "source": "dukascopy", "base_tf": "1h",  "swing": True,  "intraday": False},
    "EURAUD=X":  {"asset_class": "forex",     "source": "dukascopy", "base_tf": "1h",  "swing": True,  "intraday": False},
    "GBPAUD=X":  {"asset_class": "forex",     "source": "dukascopy", "base_tf": "1h",  "swing": True,  "intraday": False},
    "CADJPY=X":  {"asset_class": "forex",     "source": "dukascopy", "base_tf": "1h",  "swing": True,  "intraday": False},
    "CHFJPY=X":  {"asset_class": "forex",     "source": "dukascopy", "base_tf": "1h",  "swing": True,  "intraday": False},
    "EURCAD=X":  {"asset_class": "forex",     "source": "dukascopy", "base_tf": "1h",  "swing": True,  "intraday": False},
    "GBPCAD=X":  {"asset_class": "forex",     "source": "dukascopy", "base_tf": "1h",  "swing": True,  "intraday": False},
    # Equity swing (5)
    "AAPL":      {"asset_class": "equity",    "source": "polygon",   "base_tf": "1h",  "swing": True,  "intraday": False},
    "MSFT":      {"asset_class": "equity",    "source": "polygon",   "base_tf": "1h",  "swing": True,  "intraday": False},
    "NVDA":      {"asset_class": "equity",    "source": "polygon",   "base_tf": "1h",  "swing": True,  "intraday": False},
    "TSLA":      {"asset_class": "equity",    "source": "polygon",   "base_tf": "1h",  "swing": True,  "intraday": False},
    "SPY":       {"asset_class": "equity",    "source": "polygon",   "base_tf": "1h",  "swing": True,  "intraday": False},
    # Crypto swing (3) — Polygon (X:BTCUSD etc) with CCXT fallback
    "BTC-USD":   {"asset_class": "crypto",    "source": "polygon",   "base_tf": "1h",  "swing": True,  "intraday": False},
    "ETH-USD":   {"asset_class": "crypto",    "source": "polygon",   "base_tf": "1h",  "swing": True,  "intraday": False},
    "SOL-USD":   {"asset_class": "crypto",    "source": "polygon",   "base_tf": "1h",  "swing": True,  "intraday": False},
    # Commodity swing (2) — Polygon (C:GC, C:CL) with yfinance fallback
    "GC=F":      {"asset_class": "commodity", "source": "polygon",   "base_tf": "1h",  "swing": True,  "intraday": False},
    "CL=F":      {"asset_class": "commodity", "source": "polygon",   "base_tf": "1h",  "swing": True,  "intraday": False},
    # Forex intraday (10)
    "EURUSDX_5m": {"asset_class": "forex",    "source": "dukascopy", "base_tf": "5m",  "swing": False, "intraday": True,  "base_pair": "EURUSD=X"},
    "GBPUSDX_5m": {"asset_class": "forex",    "source": "dukascopy", "base_tf": "5m",  "swing": False, "intraday": True,  "base_pair": "GBPUSD=X"},
    "USDJPYX_5m": {"asset_class": "forex",    "source": "dukascopy", "base_tf": "5m",  "swing": False, "intraday": True,  "base_pair": "USDJPY=X"},
    "AUDUSDX_5m": {"asset_class": "forex",    "source": "dukascopy", "base_tf": "5m",  "swing": False, "intraday": True,  "base_pair": "AUDUSD=X"},
    "USDCADX_5m": {"asset_class": "forex",    "source": "dukascopy", "base_tf": "5m",  "swing": False, "intraday": True,  "base_pair": "USDCAD=X"},
    "NZDUSDX_5m": {"asset_class": "forex",    "source": "dukascopy", "base_tf": "5m",  "swing": False, "intraday": True,  "base_pair": "NZDUSD=X"},
    "USDCHFX_5m": {"asset_class": "forex",    "source": "dukascopy", "base_tf": "5m",  "swing": False, "intraday": True,  "base_pair": "USDCHF=X"},
    "EURGBPX_5m": {"asset_class": "forex",    "source": "dukascopy", "base_tf": "5m",  "swing": False, "intraday": True,  "base_pair": "EURGBP=X"},
    "EURJPYX_5m": {"asset_class": "forex",    "source": "dukascopy", "base_tf": "5m",  "swing": False, "intraday": True,  "base_pair": "EURJPY=X"},
    "GBPJPYX_5m": {"asset_class": "forex",    "source": "dukascopy", "base_tf": "5m",  "swing": False, "intraday": True,  "base_pair": "GBPJPY=X"},
}

QUESTDB_URL = "http://localhost:9000"

# ---------------------------------------------------------------------------
# QuestDB helpers
# ---------------------------------------------------------------------------

def _qdb_exec(sql: str) -> dict:
    """Execute SQL on QuestDB REST API, return parsed JSON."""
    try:
        r = requests.get(f"{QUESTDB_URL}/exec", params={"query": sql}, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"QuestDB exec failed: {e}")
        return {}


def _ilp_escape_tag(val: str) -> str:
    """Escape ILP tag value — =, comma, space must be backslash-escaped."""
    return val.replace(",", "\\,").replace("=", "\\=").replace(" ", "\\ ")


def _qdb_ilp_write(df: pd.DataFrame, ticker: str, timeframe: str):
    """Write OHLCV DataFrame to QuestDB ohlcv table via ILP (line protocol).

    Writes in chunks of 1000 rows to avoid HTTP request size limits.
    """
    if df.empty:
        return 0
    t_esc  = _ilp_escape_tag(ticker)
    tf_esc = _ilp_escape_tag(timeframe)
    lines = []
    for ts, row in df.iterrows():
        ts_ns = int(pd.Timestamp(ts).value)  # nanoseconds
        line = (
            f"ohlcv,ticker={t_esc},timeframe={tf_esc} "
            f"open={float(row['open'])},high={float(row['high'])},"
            f"low={float(row['low'])},close={float(row['close'])},"
            f"volume={float(row.get('volume', 0.0))} "
            f"{ts_ns}"
        )
        lines.append(line)

    written = 0
    chunk_size = 1000
    for i in range(0, len(lines), chunk_size):
        chunk = "\n".join(lines[i:i + chunk_size])
        try:
            r = requests.post(f"{QUESTDB_URL}/write", data=chunk.encode(), timeout=60)
            r.raise_for_status()
            written += len(lines[i:i + chunk_size])
        except Exception as e:
            logger.warning(f"QuestDB ILP write {ticker}/{timeframe} chunk {i//chunk_size}: {e}")
    return written


def _qdb_get_max_ts(ticker: str, timeframe: str) -> Optional[datetime]:
    """Return max(ts) for ticker+timeframe in ohlcv table, or None."""
    sql = f"SELECT max(ts) FROM ohlcv WHERE ticker='{ticker}' AND timeframe='{timeframe}'"
    result = _qdb_exec(sql)
    try:
        val = result["dataset"][0][0]
        if val is None:
            return None
        return pd.Timestamp(val, tz="UTC").to_pydatetime().replace(tzinfo=None)
    except Exception:
        return None


def _qdb_read(ticker: str, timeframe: str, since: Optional[datetime] = None, limit: int = 5000) -> pd.DataFrame:
    """Read OHLCV rows from QuestDB into DataFrame."""
    where = f"ticker='{ticker}' AND timeframe='{timeframe}'"
    if since:
        since_str = since.strftime("%Y-%m-%dT%H:%M:%S")
        where += f" AND ts >= '{since_str}'"
    sql = f"SELECT ts, open, high, low, close, volume FROM ohlcv WHERE {where} ORDER BY ts LIMIT {limit}"
    result = _qdb_exec(sql)
    rows = result.get("dataset", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.set_index("ts").sort_index()
    return df


def _log_fetch(ticker: str, timeframe: str, rows: int, source: str, status: str):
    """Write a fetch log entry to QuestDB."""
    ts_ns = int(pd.Timestamp.utcnow().value)
    line = (
        f"fetch_log,ticker={ticker},timeframe={timeframe},source={source},status={status} "
        f"rows_fetched={rows}i {ts_ns}"
    )
    try:
        requests.post(f"{QUESTDB_URL}/write", data=line.encode(), timeout=10)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Source fetchers
# ---------------------------------------------------------------------------

def _fetch_dukascopy(pair: str, timeframe: str, since: datetime, years: int = 2) -> pd.DataFrame:
    """Fetch forex OHLCV from Dukascopy with yfinance fallback."""
    try:
        from data.dukascopy import fetch_ohlcv as duka_fetch, fetch_5yr
        end_dt = datetime.utcnow()
        if since is not None:
            start_dt = since
            df = duka_fetch(pair, start_dt, end_dt, interval=timeframe)
        else:
            df = fetch_5yr(pair, interval=timeframe)
        if not df.empty:
            return df
    except Exception as e:
        logger.debug(f"Dukascopy {pair}: {e}")

    # Fallback: yfinance
    return _fetch_yfinance(pair, timeframe, since, years)


def _fetch_yfinance(ticker: str, timeframe: str, since: datetime, years: int = 2) -> pd.DataFrame:
    """Fetch OHLCV from yfinance."""
    try:
        import yfinance as yf
        # Map timeframe to yfinance interval
        tf_map = {"1h": "1h", "4h": "1h", "1d": "1d", "5m": "5m", "15m": "15m"}
        yf_interval = tf_map.get(timeframe, "1h")
        period = f"{min(years * 365, 729)}d" if yf_interval in ("1h", "5m", "15m") else f"{years * 365}d"
        df = yf.download(ticker, period=period, interval=yf_interval, progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        if since is not None:
            df = df[df.index > since]
        return df
    except Exception as e:
        logger.warning(f"yfinance {ticker}/{timeframe}: {e}")
        return pd.DataFrame()


def _fetch_ccxt(ticker: str, timeframe: str, since: datetime, years: int = 2) -> pd.DataFrame:
    """Fetch crypto OHLCV from CCXT/Binance with yfinance fallback."""
    try:
        from data.crypto_data import fetch_crypto_ohlcv
        df = fetch_crypto_ohlcv(ticker, timeframe, years)
        if not df.empty and since is not None:
            df = df[df.index > since]
        return df
    except Exception as e:
        logger.debug(f"CCXT {ticker}: {e}")
    return _fetch_yfinance(ticker, timeframe, since, years)


def _fetch_polygon(ticker: str, timeframe: str, since: datetime, years: int = 2) -> pd.DataFrame:
    """Fetch equity OHLCV from Polygon.io (massive) with yfinance fallback."""
    try:
        from data.massive import MassiveClient
        client = MassiveClient()
        # Map timeframe
        gran_map = {"1h": "hour", "4h": "hour", "1d": "day", "5m": "minute", "15m": "minute"}
        mult_map = {"1h": 1, "4h": 4, "1d": 1, "5m": 5, "15m": 15}
        granularity = gran_map.get(timeframe, "hour")
        multiplier = mult_map.get(timeframe, 1)
        df = client.fetch_bars(ticker, granularity, multiplier, years=years)
        if not df.empty and since is not None:
            df = df[df.index > since]
        return df
    except Exception as e:
        logger.debug(f"Polygon {ticker}: {e}")
    return _fetch_yfinance(ticker, timeframe, since, years)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def _resample_to_htf(df_1h: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """Resample 1h OHLCV to 4H or 1D. No lookahead — uses closed='left', label='left'."""
    if df_1h.empty:
        return pd.DataFrame()
    rule_map = {"4h": "4h", "4H": "4h", "1d": "1D", "1D": "1D"}
    rule = rule_map.get(target_tf, "4h")
    ohlc = df_1h.resample(rule, closed="left", label="left").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna(subset=["close"])
    return ohlc


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class UnifiedDataPipeline:
    """Single interface to all OHLCV data with QuestDB as primary cache.

    Usage:
        udp = UnifiedDataPipeline()
        df_1h = udp.get("EURUSD=X", "1h")
        df_4h, df_1d = udp.get_multi_timeframe("EURUSD=X")
        udp.initialise_all(swing=True, intraday=True)  # background fetch
    """

    def __init__(self):
        self._qdb_ok = self._check_qdb()

    def _check_qdb(self) -> bool:
        result = _qdb_exec("SELECT 1")
        ok = "dataset" in result
        if not ok:
            logger.warning("QuestDB not reachable — will use parquet fallback")
        return ok

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, ticker: str, timeframe: str = "1h", years: int = 2,
            force_refresh: bool = False) -> pd.DataFrame:
        """Get OHLCV for ticker/timeframe.

        1. Try QuestDB (incremental refresh if stale).
        2. If QuestDB empty/unavailable, fetch from source and store.
        Returns DataFrame with [open, high, low, close, volume].
        """
        info = INSTRUMENT_REGISTRY.get(ticker, {})
        asset_class = info.get("asset_class", "equity")

        # Determine staleness threshold
        is_crypto = asset_class == "crypto"
        stale_hours = 2 if is_crypto else 4

        if self._qdb_ok and not force_refresh:
            max_ts = _qdb_get_max_ts(ticker, timeframe)
            if max_ts is not None:
                age_h = (datetime.utcnow() - max_ts).total_seconds() / 3600
                if age_h < stale_hours:
                    # Fresh enough — read from QuestDB
                    cutoff = datetime.utcnow() - timedelta(days=years * 365)
                    df = _qdb_read(ticker, timeframe, since=cutoff, limit=100_000)
                    if not df.empty:
                        return df
                # Stale — incremental fetch from max_ts
                df_new = self._fetch_from_source(ticker, timeframe, since=max_ts, years=years)
                if not df_new.empty:
                    written = _qdb_ilp_write(df_new, ticker, timeframe)
                    _log_fetch(ticker, timeframe, written, info.get("source", "unknown"), "incremental")
                    logger.info(f"{ticker}/{timeframe}: wrote {written} new bars to QuestDB")
                # Return full history from QuestDB
                cutoff = datetime.utcnow() - timedelta(days=years * 365)
                return _qdb_read(ticker, timeframe, since=cutoff, limit=100_000)

        # QuestDB empty or unavailable — full fetch
        df = self._fetch_from_source(ticker, timeframe, since=None, years=years)
        if not df.empty and self._qdb_ok:
            written = _qdb_ilp_write(df, ticker, timeframe)
            _log_fetch(ticker, timeframe, written, info.get("source", "unknown"), "full")
            logger.info(f"{ticker}/{timeframe}: initial load {written} bars")
        return df

    def get_multi_timeframe(self, ticker: str, years: int = 2):
        """Return (df_4h, df_1d) for ticker based on 1h data.

        Uses QuestDB-cached 1h data, resamples to 4H and 1D.
        Returns (df_4h, df_1d) — both may be empty DataFrames.
        """
        df_1h = self.get(ticker, "1h", years=years)
        if df_1h.empty:
            return pd.DataFrame(), pd.DataFrame()
        df_4h = _resample_to_htf(df_1h, "4h")
        df_1d = _resample_to_htf(df_1h, "1d")
        return df_4h, df_1d

    def get_latest_in_db(self, ticker: str, timeframe: str) -> Optional[datetime]:
        """Return timestamp of most recent bar in QuestDB, or None."""
        return _qdb_get_max_ts(ticker, timeframe)

    def fetch_and_store(self, ticker: str, timeframe: str = "1h", years: int = 2) -> int:
        """Force-fetch from source and write to QuestDB. Returns bars written."""
        info = INSTRUMENT_REGISTRY.get(ticker, {})
        since = _qdb_get_max_ts(ticker, timeframe)  # incremental if possible
        df = self._fetch_from_source(ticker, timeframe, since=since, years=years)
        if df.empty:
            return 0
        written = _qdb_ilp_write(df, ticker, timeframe)
        _log_fetch(ticker, timeframe, written, info.get("source", "unknown"), "manual")
        return written

    def initialise_all(self, swing: bool = True, intraday: bool = True, years: int = 2):
        """Background-fetch all instruments into QuestDB.

        Swing instruments: fetch 1h.
        Intraday instruments: fetch 5m (and 1h for HTF context).
        This is a blocking call — run in a thread or background process.
        """
        tasks = []
        for ticker, info in INSTRUMENT_REGISTRY.items():
            if swing and info.get("swing"):
                tasks.append((ticker, "1h", years))
            if intraday and info.get("intraday"):
                tasks.append((ticker, "5m", 1))  # 5m: 1 year max
                # Also fetch 1h for the base pair (HTF context)
                base = info.get("base_pair")
                if base and base in INSTRUMENT_REGISTRY:
                    tasks.append((base, "1h", years))

        logger.info(f"UnifiedDataPipeline: initialising {len(tasks)} fetch tasks")
        for ticker, tf, yrs in tasks:
            try:
                n = self.fetch_and_store(ticker, tf, yrs)
                logger.info(f"  {ticker}/{tf}: {n} bars stored")
            except Exception as e:
                logger.warning(f"  {ticker}/{tf}: fetch failed — {e}")
            time.sleep(0.5)  # gentle rate limiting

    # ------------------------------------------------------------------
    # Internal routing
    # ------------------------------------------------------------------

    def _fetch_from_source(self, ticker: str, timeframe: str,
                           since: Optional[datetime], years: int = 2) -> pd.DataFrame:
        """Route fetch request to correct data source."""
        info = INSTRUMENT_REGISTRY.get(ticker, {})
        asset_class = info.get("asset_class", "equity")
        source = info.get("source", "yfinance")

        try:
            if asset_class == "forex":
                # For 5m intraday instruments with _5m suffix, use base pair
                actual_pair = info.get("base_pair", ticker)
                return _fetch_dukascopy(actual_pair, timeframe, since, years)
            elif asset_class == "crypto":
                # CCXT (Binance) has 4+ years of 1h history — use it
                return _fetch_ccxt(ticker, timeframe, since, years)
            elif asset_class == "equity":
                return _fetch_polygon(ticker, timeframe, since, years)
            elif asset_class == "commodity":
                # yfinance: 1h limited to last 730 days, cap years at 2
                return _fetch_yfinance(ticker, timeframe, since, min(years, 2))
            else:
                return _fetch_yfinance(ticker, timeframe, since, years)
        except Exception as e:
            logger.error(f"_fetch_from_source {ticker}/{timeframe}: {e}")
            return pd.DataFrame()
