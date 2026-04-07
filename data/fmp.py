"""data/fmp.py — Financial Modeling Prep data client."""
import os, time, json, hashlib, logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import requests

logger = logging.getLogger("trading_firm.fmp")

class FMPClient:
    BASE = "https://financialmodelingprep.com/api"
    CACHE_DIR = "data/cache/fmp"

    def __init__(self):
        from config.settings import FMP_API_KEY
        self.key = FMP_API_KEY
        os.makedirs(self.CACHE_DIR, exist_ok=True)

    def _cache_path(self, endpoint, params):
        key = hashlib.md5(
            (endpoint + json.dumps(params, sort_keys=True)).encode()
        ).hexdigest()
        return Path(self.CACHE_DIR) / f"{key}.json"

    def _get(self, endpoint, params=None, cache_hours=24):
        params = params or {}
        cache_file = self._cache_path(endpoint, params)
        if cache_file.exists():
            age = time.time() - cache_file.stat().st_mtime
            if age < cache_hours * 3600:
                with open(cache_file) as f:
                    return json.load(f)
        params["apikey"] = self.key
        url = f"{self.BASE}{endpoint}"
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            with open(cache_file, "w") as f:
                json.dump(data, f)
            time.sleep(0.1)
            return data
        except Exception as e:
            logger.warning(f"FMP {endpoint}: {e}")
            return []

    def get_ohlcv(self, ticker: str, interval: str = "1hour", years: int = 5) -> pd.DataFrame:
        """Get OHLCV bars. intervals: 1min,5min,15min,30min,1hour,4hour,1day"""
        end = datetime.now()
        start = end - timedelta(days=365 * years)
        if interval == "1day":
            endpoint = f"/v3/historical-price-full/{ticker}"
            data = self._get(endpoint, {"from": start.strftime("%Y-%m-%d"), "to": end.strftime("%Y-%m-%d")}, cache_hours=4)
            if isinstance(data, dict):
                rows = data.get("historical", [])
            else:
                rows = []
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            df = df.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"})
            return df[["open", "high", "low", "close", "volume"]].dropna()
        else:
            endpoint = f"/v3/historical-chart/{interval}/{ticker}"
            data = self._get(endpoint, {"from": start.strftime("%Y-%m-%d"), "to": end.strftime("%Y-%m-%d")}, cache_hours=4)
            if not data or not isinstance(data, list):
                return pd.DataFrame()
            df = pd.DataFrame(data)
            if df.empty or "date" not in df.columns:
                return pd.DataFrame()
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
            return df[cols].dropna()

    def get_income_statement(self, ticker: str, period: str = "quarter", limit: int = 20) -> pd.DataFrame:
        data = self._get(f"/v3/income-statement/{ticker}", {"period": period, "limit": limit}, cache_hours=24)
        if not data or not isinstance(data, list):
            return pd.DataFrame()
        return pd.DataFrame(data)

    def get_balance_sheet(self, ticker: str, period: str = "quarter", limit: int = 20) -> pd.DataFrame:
        data = self._get(f"/v3/balance-sheet-statement/{ticker}", {"period": period, "limit": limit}, cache_hours=24)
        if not data or not isinstance(data, list):
            return pd.DataFrame()
        return pd.DataFrame(data)

    def get_cash_flow(self, ticker: str, period: str = "quarter", limit: int = 20) -> pd.DataFrame:
        data = self._get(f"/v3/cash-flow-statement/{ticker}", {"period": period, "limit": limit}, cache_hours=24)
        if not data or not isinstance(data, list):
            return pd.DataFrame()
        return pd.DataFrame(data)

    def get_analyst_estimates(self, ticker: str, period: str = "quarter") -> pd.DataFrame:
        data = self._get(f"/v3/analyst-estimates/{ticker}", {"period": period}, cache_hours=12)
        if not data or not isinstance(data, list):
            return pd.DataFrame()
        return pd.DataFrame(data)

    def get_earnings_surprises(self, ticker: str, limit: int = 20) -> pd.DataFrame:
        data = self._get(f"/v3/earnings-surprises/{ticker}", {"limit": limit}, cache_hours=24)
        if not data or not isinstance(data, list):
            return pd.DataFrame()
        df = pd.DataFrame(data)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df

    def get_insider_trading(self, ticker: str, limit: int = 100) -> pd.DataFrame:
        data = self._get("/v4/insider-trading", {"symbol": ticker, "limit": limit}, cache_hours=24)
        if not data or not isinstance(data, list):
            return pd.DataFrame()
        return pd.DataFrame(data)

    def get_earnings_calendar(self, from_date: str, to_date: str) -> pd.DataFrame:
        data = self._get("/v3/earning_calendar", {"from": from_date, "to": to_date}, cache_hours=6)
        if not data or not isinstance(data, list):
            return pd.DataFrame()
        return pd.DataFrame(data)

    def get_price_target(self, ticker: str) -> dict:
        data = self._get("/v4/price-target-consensus", {"symbol": ticker}, cache_hours=12)
        if not data or not isinstance(data, list):
            return {}
        return data[0] if data else {}
