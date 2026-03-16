"""
================================================================
  risk/portfolio_optimizer.py
  Mean-variance portfolio optimisation using cvxpy.

  Features:
    - Ledoit-Wolf-inspired covariance shrinkage (no sklearn needed)
    - Long-only or long/short weight constraints
    - Max single-instrument weight cap (default 40%)
    - Gross exposure limit (default 100%)
    - Falls back to equal-weight if cvxpy not installed or solve fails

  Usage:
    weights = optimise_weights(
        tickers        = ["EUR_USD", "GBP_USD", "SPY"],
        expected_returns = np.array([0.001, 0.0008, 0.0012]),
        cov_matrix     = cov,          # (n, n) annualised covariance
        risk_aversion  = 1.0,          # λ — higher = more risk-averse
        max_weight     = 0.40,
        gross_limit    = 1.0,
    )
    # returns dict {ticker: weight}
================================================================
"""
import logging
import numpy as np
from typing import List, Optional, Dict

logger = logging.getLogger("trading_firm.portfolio_optimizer")


# ── Covariance estimation ──────────────────────────────────────

def estimate_covariance(
    returns_df,
    shrinkage: float = 0.1,
) -> np.ndarray:
    """
    Ledoit-Wolf-inspired analytical shrinkage estimator.

    Parameters
    ----------
    returns_df : pd.DataFrame  shape (T, n_assets) of period returns
    shrinkage  : float         shrinkage intensity toward diagonal target
                               0 = sample cov, 1 = diagonal target

    Returns
    -------
    Shrunk covariance matrix  (n, n) as np.ndarray
    """
    import pandas as pd

    if returns_df is None or returns_df.empty:
        return np.eye(1)

    R = np.asarray(returns_df.dropna(), dtype=float)
    if R.ndim == 1:
        R = R.reshape(-1, 1)

    T, n = R.shape
    if T < n + 2:
        # Not enough observations — return identity
        return np.eye(n)

    sample_cov = np.cov(R, rowvar=False)
    if n == 1:
        return sample_cov.reshape(1, 1)

    # Shrinkage target: diagonal of sample covariance (variances only)
    target = np.diag(np.diag(sample_cov))

    cov_shrunk = (1.0 - shrinkage) * sample_cov + shrinkage * target

    # Ensure positive semi-definite
    eigvals = np.linalg.eigvalsh(cov_shrunk)
    if eigvals.min() < 1e-8:
        cov_shrunk += (abs(eigvals.min()) + 1e-6) * np.eye(n)

    return cov_shrunk


# ── Portfolio optimisation ─────────────────────────────────────

def optimise_weights(
    tickers:          List[str],
    expected_returns: np.ndarray,
    cov_matrix:       np.ndarray,
    risk_aversion:    float = 1.0,
    max_weight:       float = 0.40,
    min_weight:       float = 0.0,
    gross_limit:      float = 1.0,
    allow_short:      bool  = False,
) -> Dict[str, float]:
    """
    Mean-variance optimisation:
      maximise  μᵀw - (λ/2) wᵀΣw
      subject to:
        min_weight ≤ wᵢ ≤ max_weight
        Σ|wᵢ| ≤ gross_limit   (gross exposure)
        Σwᵢ = 1               (fully invested)

    Falls back to equal-weight allocation on any failure.

    Parameters
    ----------
    tickers          : asset identifiers (for output dict)
    expected_returns : (n,) array of expected returns per bar
    cov_matrix       : (n, n) covariance matrix
    risk_aversion    : λ — controls risk/return trade-off
    max_weight       : max weight per instrument (default 0.40)
    min_weight       : min weight per instrument (default 0.0)
    gross_limit      : sum of absolute weights (default 1.0)
    allow_short      : if True, weights can be negative

    Returns
    -------
    dict {ticker: weight}   weights sum to 1
    """
    n = len(tickers)
    equal_w = {t: round(1.0 / n, 6) for t in tickers}

    if n == 0:
        return {}
    if n == 1:
        return {tickers[0]: 1.0}

    mu  = np.asarray(expected_returns, dtype=float)
    Sig = np.asarray(cov_matrix,       dtype=float)

    if mu.shape[0] != n or Sig.shape != (n, n):
        logger.warning("optimise_weights: shape mismatch — returning equal weight")
        return equal_w

    try:
        import cvxpy as cp

        w = cp.Variable(n)

        portfolio_return = mu @ w
        portfolio_risk   = cp.quad_form(w, Sig)

        objective = cp.Maximize(portfolio_return - (risk_aversion / 2) * portfolio_risk)

        constraints = [
            cp.sum(w) == 1,
        ]

        if allow_short:
            constraints += [
                w >= min_weight - max_weight,
                w <= max_weight,
                cp.norm1(w) <= gross_limit,
            ]
        else:
            constraints += [
                w >= max(0.0, min_weight),
                w <= max_weight,
            ]

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS, warm_start=True)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            logger.warning(f"cvxpy status={prob.status} — falling back to equal weight")
            return equal_w

        raw_w = w.value
        if raw_w is None:
            return equal_w

        # Clip small negatives from numerical noise
        raw_w = np.clip(raw_w, 0.0, None) if not allow_short else raw_w
        total = raw_w.sum()
        if abs(total) < 1e-8:
            return equal_w
        raw_w /= total  # renormalise

        result = {tickers[i]: round(float(raw_w[i]), 6) for i in range(n)}
        logger.debug(f"Optimised weights: {result}")
        return result

    except ImportError:
        logger.info("cvxpy not installed — using equal-weight allocation")
        return equal_w
    except Exception as e:
        logger.warning(f"optimise_weights failed ({e}) — returning equal weight")
        return equal_w


# ── Convenience: rolling expected return estimate ──────────────

def estimate_expected_returns(
    returns_df,
    lookback: int = 60,
) -> np.ndarray:
    """
    Simple rolling mean of the last `lookback` bars as expected return.

    Parameters
    ----------
    returns_df : pd.DataFrame  shape (T, n_assets)
    lookback   : int           number of bars to average

    Returns
    -------
    (n,) array of expected returns
    """
    import pandas as pd

    if returns_df is None or returns_df.empty:
        return np.zeros(returns_df.shape[1] if returns_df is not None else 1)

    tail = returns_df.tail(lookback).dropna()
    return np.asarray(tail.mean(), dtype=float)
