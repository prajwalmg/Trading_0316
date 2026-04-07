import os, logging, requests
import numpy as np
import pandas as pd
from datetime import datetime

logger = logging.getLogger("trading_firm.macro")

FRED_SERIES = {
    "fed_funds_rate": "DFF",
    "yield_curve":    "T10Y2Y",
    "credit_spread":  "BAA10Y",
    "vix":            "VIXCLS",
    "unemployment":   "UNRATE",
    "cpi":            "CPIAUCSL",
}
CACHE_DIR       = "data/cache/macro"
CACHE_TTL_HOURS = 24


def _cache_valid(path: str) -> bool:
    if not os.path.exists(path):
        return False
    age = (datetime.now().timestamp() - os.path.getmtime(path)) / 3600
    return age < CACHE_TTL_HOURS


def load_macro() -> pd.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)
    dfs = {}
    for name, sid in FRED_SERIES.items():
        cache = f"{CACHE_DIR}/{sid}.parquet"
        if _cache_valid(cache):
            try:
                dfs[name] = pd.read_parquet(cache)["value"]
                continue
            except Exception:
                pass
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            import io
            df = pd.read_csv(io.StringIO(resp.text), index_col=0, parse_dates=True)
            df.columns = ["value"]
            df = df.replace(".", np.nan).astype(float).ffill()
            df.to_parquet(cache)
            dfs[name] = df["value"]
            logger.info(f"FRED {sid}: {len(df)} observations")
        except Exception as e:
            logger.warning(f"FRED {sid} failed: {e}")
    if not dfs:
        return pd.DataFrame()
    macro = pd.DataFrame(dfs).ffill()
    return macro


def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 7 FRED macro z-score features to an OHLCV feature DataFrame."""
    feature_cols = [
        "macro_fed_funds_z", "macro_yield_curve_z",
        "macro_credit_z",    "macro_vix_z",
        "macro_unemployment_z", "macro_cpi_yoy_z",
        "macro_regime_score",
    ]
    try:
        macro = load_macro()
    except Exception:
        macro = pd.DataFrame()

    if macro.empty:
        for col in feature_cols:
            df[col] = 0.0
        return df

    # Build 252-day rolling z-scores
    for col in macro.columns:
        roll = macro[col].rolling(252)
        macro[f"{col}_z"] = (macro[col] - roll.mean()) / (roll.std() + 1e-9)

    macro["cpi_yoy_z"] = macro["cpi"].pct_change(252)

    vix_z  = macro.get("vix_z",          pd.Series(0.0, index=macro.index))
    crd_z  = macro.get("credit_spread_z", pd.Series(0.0, index=macro.index))
    yc_z   = macro.get("yield_curve_z",   pd.Series(0.0, index=macro.index))
    risk_on = (-vix_z - crd_z + yc_z) / 3
    macro["macro_regime_score"] = risk_on.clip(-1, 1)

    # Resample daily → hourly then align to df index
    try:
        macro_h = macro.resample("1h").ffill()
    except Exception:
        macro_h = macro
    macro_a = macro_h.reindex(df.index, method="ffill")

    col_map = {
        "fed_funds_rate_z":   "macro_fed_funds_z",
        "yield_curve_z":      "macro_yield_curve_z",
        "credit_spread_z":    "macro_credit_z",
        "vix_z":              "macro_vix_z",
        "unemployment_z":     "macro_unemployment_z",
        "cpi_yoy_z":          "macro_cpi_yoy_z",
        "macro_regime_score": "macro_regime_score",
    }
    for src, dst in col_map.items():
        if src in macro_a.columns:
            df[dst] = macro_a[src].values
        else:
            df[dst] = 0.0
    return df


def get_macro_regime_score() -> float:
    """Return current macro regime score in [-1, 1]. Negative = risk-off."""
    try:
        macro = load_macro()
        if macro.empty:
            return 0.0

        def z(s: pd.Series) -> float:
            return float(
                ((s - s.rolling(252).mean()) /
                 (s.rolling(252).std() + 1e-9)).iloc[-1]
            )

        score = (-z(macro["vix"]) - z(macro["credit_spread"]) + z(macro["yield_curve"])) / 3
        return float(np.clip(score, -1, 1))
    except Exception:
        return 0.0


_singleton = None


def get_macro() -> pd.DataFrame:
    global _singleton
    if _singleton is None:
        _singleton = load_macro()
    return _singleton
