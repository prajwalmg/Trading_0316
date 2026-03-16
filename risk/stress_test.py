"""
================================================================
  risk/stress_test.py
  Historical VaR, CVaR and stress-test scenarios.

  VaR / CVaR:
    Uses historical simulation (no parametric assumptions).
    Default confidence = 95%.

  Stress scenarios:
    Four historical drawdown periods mapped to approximate
    percentage shocks per asset class:

    2008 GFC          — equity -55%, forex -15%, commodity -40%, crypto N/A
    2020 COVID crash  — equity -35%, forex  -8%, commodity -25%, crypto -50%
    2022 bear market  — equity -25%, forex  -5%, commodity  +15%, crypto -75%
    2018 crypto winter — crypto -85%, all others -5%
================================================================
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict

logger = logging.getLogger("trading_firm.stress_test")

# ── VaR / CVaR ────────────────────────────────────────────────

def compute_var_cvar(
    returns: pd.Series,
    confidence: float = 0.95,
    capital: float = 1.0,
) -> Dict[str, float]:
    """
    Historical simulation VaR and CVaR.

    Parameters
    ----------
    returns    : daily/bar P&L returns as a fraction of capital
    confidence : confidence level (default 0.95 → 95% VaR)
    capital    : portfolio NAV — output is in currency units

    Returns
    -------
    dict with:
      var_pct   — VaR as % of capital (positive = loss)
      cvar_pct  — CVaR (Expected Shortfall) as % of capital
      var_usd   — VaR in currency units
      cvar_usd  — CVaR in currency units
      n_obs     — number of return observations used
    """
    if returns is None or len(returns) < 20:
        return {
            "var_pct": 0.0, "cvar_pct": 0.0,
            "var_usd": 0.0, "cvar_usd": 0.0,
            "n_obs": 0,
        }

    r = np.asarray(returns.dropna())

    # VaR: the (1-confidence) quantile of losses (negative returns)
    var_pct  = float(-np.percentile(r, (1 - confidence) * 100))

    # CVaR: mean of all returns worse than VaR
    tail     = r[r <= -var_pct]
    cvar_pct = float(-tail.mean()) if len(tail) > 0 else var_pct

    return {
        "var_pct":  round(var_pct,  6),
        "cvar_pct": round(cvar_pct, 6),
        "var_usd":  round(var_pct  * capital, 2),
        "cvar_usd": round(cvar_pct * capital, 2),
        "n_obs":    len(r),
    }


# ── Historical stress scenarios ───────────────────────────────

_SCENARIOS = {
    "2008_gfc": {
        "label":   "2008 Global Financial Crisis",
        "shocks":  {
            "equity":    -0.55,
            "forex":     -0.15,
            "commodity": -0.40,
            "crypto":    -0.50,   # proxy — crypto didn't exist; use conservative
        },
    },
    "2020_covid": {
        "label":   "2020 COVID-19 Crash (Feb–Mar)",
        "shocks":  {
            "equity":    -0.35,
            "forex":     -0.08,
            "commodity": -0.25,
            "crypto":    -0.50,
        },
    },
    "2022_bear": {
        "label":   "2022 Bear Market",
        "shocks":  {
            "equity":    -0.25,
            "forex":     -0.05,
            "commodity":  0.15,   # energy/commodities rallied
            "crypto":    -0.75,
        },
    },
    "2018_crypto_winter": {
        "label":   "2018 Crypto Winter",
        "shocks":  {
            "equity":    -0.05,
            "forex":     -0.05,
            "commodity": -0.05,
            "crypto":    -0.85,
        },
    },
}


def run_stress_test(
    positions: list,
    asset_class_map: dict,
    capital: float,
) -> Dict[str, dict]:
    """
    Apply historical shock scenarios to current open positions.

    Parameters
    ----------
    positions       : list of position dicts from broker.get_open_positions()
                      Each must have keys: instrument, units, entry, direction
    asset_class_map : {ticker → asset_class}  from config.settings
    capital         : current portfolio NAV

    Returns
    -------
    dict keyed by scenario name, each containing:
      label         — human-readable scenario name
      pnl           — total stressed P&L across all positions
      pnl_pct       — stressed P&L as % of capital
      position_pnl  — per-instrument stressed P&L dict
    """
    results = {}

    for scenario_key, scenario in _SCENARIOS.items():
        shocks      = scenario["shocks"]
        total_pnl   = 0.0
        pos_pnl     = {}

        for pos in positions:
            instrument = pos.get("instrument", "")
            units      = abs(pos.get("units", 0))
            entry      = pos.get("entry", 0.0)
            direction  = pos.get("direction", 1)

            if units == 0 or entry == 0:
                continue

            ac    = asset_class_map.get(instrument, "equity")
            shock = shocks.get(ac, -0.10)

            # Stressed price — shock applied against entry in the
            # direction of the position (long suffers a drop, short
            # suffers a rise, but here we assume worst-case for all).
            stressed_price = entry * (1 + shock)
            pnl_per_unit   = (stressed_price - entry) * direction
            pnl            = pnl_per_unit * units

            pos_pnl[instrument] = round(pnl, 2)
            total_pnl          += pnl

        results[scenario_key] = {
            "label":        scenario["label"],
            "pnl":          round(total_pnl, 2),
            "pnl_pct":      round(total_pnl / capital, 6) if capital else 0.0,
            "position_pnl": pos_pnl,
        }

    return results
