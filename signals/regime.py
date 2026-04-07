"""
================================================================
  signals/regime.py
  Gaussian Mixture HMM regime detection.

  4 hidden states × 3 Gaussians per state (GMMHMM).
  Observable features per bar (5):
    1. log_return        = log(close/close.shift(1))
    2. realised_vol_20   = log_return.rolling(20).std()
    3. range_ratio       = (high - low) / close
    4. momentum_10       = close/close.shift(10) - 1
    5. volume_z          = standardised volume (0 for forex)

  States are auto-labelled after each fit:
    highest vol emission  → "high_volatility"
    of remaining 3, rank by mean log_return:
      highest  → "trending_up"
      lowest   → "trending_down"
      middle   → "ranging"

  Persistence filter: 3-bar majority vote prevents flip-flopping.

  Public API (backward-compatible with old regime.py callers):
    detect(df, ticker)         → str
    all_current()              → dict
    get_state_probs(df, ticker)→ np.array(4,)
    get_next_probs(ticker)     → np.array(4,)
    get_vol_forecast(ticker)   → float
    get_regime_confidence(t)   → float
    is_in_transition(ticker)   → bool
    get_strategy_weights(t)    → dict
    pretrain_all(tickers)      → None
    update(ticker, df)         → (for PortfolioManager compatibility)
    current(ticker)            → str
    dominant_regime()          → str
    summary()                  → pd.DataFrame

  Fallback: if GMMHMM fails, falls back to ADX-based detection.
================================================================
"""

import os
import time
import pickle
import logging
import threading
from collections import deque
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.settings import (
    ADX_TREND_THRESHOLD, RVOL_HIGH_THRESHOLD,
    REGIME_LOOKBACK, EMA_FAST, EMA_SLOW, MODEL_DIR,
)

logger = logging.getLogger("trading_firm.regime")

# ── Constants ────────────────────────────────────────────────
N_STATES       = 4
N_MIX          = 3
RETRAIN_HOURS  = 720       # retrain monthly
MIN_TRAIN_BARS = 500
TRANSITION_THR = 0.45
MIN_COVAR      = 1e-3
VOTE_WINDOW    = 3         # majority-vote bars for persistence filter

STATE_NAMES = ["trending_up", "trending_down", "ranging", "high_volatility"]

_STRATEGY_WEIGHTS = {
    "trending_up":     {"momentum": 1.5, "mean_reversion": 0.3,
                        "breakout": 1.2, "regime_adaptive": 1.0},
    "trending_down":   {"momentum": 1.5, "mean_reversion": 0.3,
                        "breakout": 1.2, "regime_adaptive": 1.0},
    "ranging":         {"momentum": 0.3, "mean_reversion": 1.5,
                        "breakout": 0.5, "regime_adaptive": 1.0},
    "high_volatility": {"momentum": 0.5, "mean_reversion": 0.5,
                        "breakout": 0.5, "regime_adaptive": 1.5},
}


# ── Feature builder ──────────────────────────────────────────

def _build_obs(df: pd.DataFrame) -> np.ndarray:
    """
    Build (N, 5) observation matrix from raw OHLCV DataFrame.
    Returns only rows where all 5 features are finite.
    """
    c = df["close"]
    lr   = np.log(c / c.shift(1))
    rvol = lr.rolling(20).std()
    rng  = (df["high"] - df["low"]) / c
    mom  = c / c.shift(10) - 1

    vol_raw = df.get("volume", pd.Series(0.0, index=df.index))
    vol_mean = vol_raw.rolling(20).mean()
    vol_std  = vol_raw.rolling(20).std().replace(0, np.nan)
    vol_z    = (vol_raw - vol_mean) / vol_std

    # Forex has zero volume — keep as zeros
    vol_z = vol_z.fillna(0.0)

    obs = pd.DataFrame({
        "lr":   lr,
        "rvol": rvol,
        "rng":  rng,
        "mom":  mom,
        "volz": vol_z,
    })
    obs = obs.replace([np.inf, -np.inf], np.nan).dropna()
    return obs.values.astype(np.float64), obs.index


# ── Simple fallback regime detector (ADX-based) ──────────────

def _fallback_regime(df: pd.DataFrame) -> str:
    """ADX + realised-vol fallback when GMMHMM cannot be trained."""
    try:
        c = df["close"]
        h, l = df["high"], df["low"]
        tr      = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        atr14   = tr.ewm(span=14, adjust=False).mean()
        pdm     = (h.diff()).clip(lower=0)
        mdm     = (-l.diff()).clip(lower=0)
        pdi     = 100 * pdm.ewm(span=14, adjust=False).mean() / (atr14 + 1e-9)
        mdi     = 100 * mdm.ewm(span=14, adjust=False).mean() / (atr14 + 1e-9)
        dx      = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-9)
        adx     = dx.ewm(span=14, adjust=False).mean().iloc[-1]

        ema_f   = c.ewm(span=EMA_FAST, adjust=False).mean()
        ema_s   = c.ewm(span=EMA_SLOW, adjust=False).mean()

        lr      = np.log(c / c.shift(1))
        rvol    = lr.rolling(10).std()
        rvol_m  = rvol.rolling(60).mean()
        rvol_r  = (rvol / (rvol_m + 1e-9)).iloc[-1]

        if rvol_r > 2.0:
            return "high_volatility"
        if adx > ADX_TREND_THRESHOLD:
            return "trending_up" if ema_f.iloc[-1] > ema_s.iloc[-1] else "trending_down"
        return "ranging"
    except Exception:
        return "ranging"


# ── Per-ticker HMM bundle ────────────────────────────────────

class _HMMBundle:
    """Stores a fitted GMMHMM plus its state-label mapping."""

    def __init__(self):
        self.model        = None      # fitted hmmlearn.hmm.GMMHMM
        self.label_map    = {}        # {raw_state_int: regime_str}
        self.trained_at   = None      # datetime (UTC)
        self.obs_index    = None      # pd.Index of training observations
        self.last_state   = None      # int: last decoded state
        self.state_hist   = deque(maxlen=VOTE_WINDOW)   # raw int states
        self.regime_hist  = deque(maxlen=VOTE_WINDOW)   # regime strings

    @property
    def age_hours(self) -> float:
        if self.trained_at is None:
            return float("inf")
        return (datetime.now(timezone.utc) - self.trained_at).total_seconds() / 3600

    def label_states(self):
        """
        Assign regime labels to the N_STATES raw states.
        Highest mean realised-vol state → "high_volatility".
        Of the rest, rank by mean log_return:
          max → "trending_up", min → "trending_down", mid → "ranging".
        """
        m = self.model
        means = np.array([
            np.mean([m.means_[s][k][0] for k in range(N_MIX)])   # mean lr
            for s in range(N_STATES)
        ])
        vols = np.array([
            np.mean([m.means_[s][k][1] for k in range(N_MIX)])   # mean rvol
            for s in range(N_STATES)
        ])

        hv_state   = int(np.argmax(vols))
        rest       = [s for s in range(N_STATES) if s != hv_state]
        rest_means = [(s, means[s]) for s in rest]
        rest_sorted = sorted(rest_means, key=lambda x: x[1])

        self.label_map = {
            rest_sorted[2][0]: "trending_up",
            rest_sorted[0][0]: "trending_down",
            rest_sorted[1][0]: "ranging",
            hv_state:          "high_volatility",
        }

    def regime_for(self, raw_state: int) -> str:
        return self.label_map.get(raw_state, "ranging")

    def voted_regime(self) -> str:
        """3-bar majority vote of recent raw states."""
        if not self.state_hist:
            return "ranging"
        # Map to regime strings, then take most common
        regimes = [self.regime_for(s) for s in self.state_hist]
        return max(set(regimes), key=regimes.count)


# ── RegimeTracker ─────────────────────────────────────────────

class RegimeTracker:
    """
    GMMHMM-based regime tracker.  One GMMHMM per instrument.

    Thread-safe via a per-ticker RLock — multiple strategies can
    call detect() concurrently without corrupting model state.
    """

    def __init__(self):
        self._bundles: dict[str, _HMMBundle] = {}
        self._locks:   dict[str, threading.RLock] = {}

        os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Internal helpers ─────────────────────────────────────

    def _lock(self, ticker: str) -> threading.RLock:
        if ticker not in self._locks:
            self._locks[ticker] = threading.RLock()
        return self._locks[ticker]

    def _model_path(self, ticker: str) -> str:
        safe = ticker.replace("/", "_").replace("=", "_").replace("-", "_")
        return os.path.join(MODEL_DIR, f"hmm_{safe}.pkl")

    def _load_bundle(self, ticker: str) -> _HMMBundle | None:
        path = self._model_path(ticker)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                bundle = pickle.load(f)
            logger.debug(f"HMM loaded: {ticker}  (age={bundle.age_hours:.1f}h)")
            return bundle
        except Exception as e:
            logger.warning(f"HMM load failed for {ticker}: {e}")
            return None

    def _save_bundle(self, ticker: str, bundle: _HMMBundle):
        path = self._model_path(ticker)
        try:
            with open(path, "wb") as f:
                pickle.dump(bundle, f)
        except Exception as e:
            logger.warning(f"HMM save failed for {ticker}: {e}")

    def _fit(self, ticker: str, df: pd.DataFrame) -> _HMMBundle | None:
        """
        Fit a new GMMHMM on df.  Returns _HMMBundle or None on failure.
        """
        try:
            from hmmlearn import hmm as hmmlib
        except ImportError:
            logger.error("hmmlearn not installed — pip install hmmlearn")
            return None

        obs, idx = _build_obs(df)
        if len(obs) < MIN_TRAIN_BARS:
            logger.debug(f"{ticker}: only {len(obs)} obs < {MIN_TRAIN_BARS} — skipping HMM fit")
            return None

        try:
            model = hmmlib.GMMHMM(
                n_components    = N_STATES,
                n_mix           = N_MIX,
                covariance_type = "full",
                n_iter          = 200,
                min_covar       = MIN_COVAR,
                random_state    = 42,
            )
            model.fit(obs)

            bundle = _HMMBundle()
            bundle.model      = model
            bundle.trained_at = datetime.now(timezone.utc)
            bundle.obs_index  = idx
            bundle.label_states()

            logger.info(f"HMM trained: {ticker} | {len(obs)} obs | "
                        f"label_map={bundle.label_map}")
            return bundle

        except Exception as e:
            logger.warning(f"GMMHMM training failed for {ticker}: {e}")
            return None

    def _get_bundle(self, ticker: str, df: pd.DataFrame) -> _HMMBundle | None:
        """
        Return a trained _HMMBundle for ticker, retraining if stale.
        Returns None if training is impossible.
        """
        with self._lock(ticker):
            bundle = self._bundles.get(ticker)

            # Try loading from disk if not in memory
            if bundle is None:
                bundle = self._load_bundle(ticker)
                if bundle is not None:
                    self._bundles[ticker] = bundle

            # Retrain if missing or stale
            if bundle is None or bundle.age_hours > RETRAIN_HOURS:
                new_bundle = self._fit(ticker, df)
                if new_bundle is not None:
                    bundle = new_bundle
                    self._bundles[ticker] = bundle
                    self._save_bundle(ticker, bundle)
                elif bundle is None:
                    return None   # no model at all

            return bundle

    # ── Public API ───────────────────────────────────────────

    def detect(self, df: pd.DataFrame, ticker: str) -> str:
        """
        Return current regime for ticker.
        Trains/loads model as needed; uses 3-bar majority vote.
        Falls back to ADX-based detection on any error.
        """
        if df is None or df.empty:
            return "ranging"

        try:
            bundle = self._get_bundle(ticker, df)
            if bundle is None:
                return _fallback_regime(df)

            obs, _ = _build_obs(df)
            if len(obs) == 0:
                return _fallback_regime(df)

            with self._lock(ticker):
                states = bundle.model.predict(obs)
                last_raw = int(states[-1])
                bundle.state_hist.append(last_raw)
                bundle.last_state = last_raw
                regime = bundle.voted_regime()
                bundle.regime_hist.append(regime)

            return regime

        except Exception as e:
            logger.warning(f"HMM detect failed for {ticker}: {e}")
            return _fallback_regime(df)

    def all_current(self) -> dict:
        """Return {ticker: regime_string} for all trained tickers."""
        result = {}
        for ticker, bundle in self._bundles.items():
            if bundle is not None:
                result[ticker] = bundle.voted_regime()
        return result

    def get_state_probs(self, df: pd.DataFrame, ticker: str) -> np.ndarray:
        """
        Return probability of being in each of the 4 states for
        the most recent bar.  Shape: (4,) ordered by STATE_NAMES.
        """
        try:
            bundle = self._get_bundle(ticker, df)
            if bundle is None:
                return np.full(N_STATES, 1.0 / N_STATES)

            obs, _ = _build_obs(df)
            if len(obs) == 0:
                return np.full(N_STATES, 1.0 / N_STATES)

            # Posterior marginals via forward algorithm
            log_prob, posteriors = bundle.model.score_samples(obs)
            last_post = posteriors[-1]   # shape (N_STATES,)

            # Reorder so that index matches STATE_NAMES order
            reordered = np.zeros(N_STATES)
            inv_map = {v: k for k, v in bundle.label_map.items()}
            for i, name in enumerate(STATE_NAMES):
                raw_idx = inv_map.get(name, 0)
                reordered[i] = last_post[raw_idx]
            return reordered

        except Exception as e:
            logger.debug(f"get_state_probs failed for {ticker}: {e}")
            return np.full(N_STATES, 1.0 / N_STATES)

    def get_next_probs(self, ticker: str) -> np.ndarray:
        """
        Return next-bar regime probabilities using the transition matrix
        and the current state posterior.  Shape: (4,).
        """
        try:
            bundle = self._bundles.get(ticker)
            if bundle is None or bundle.last_state is None:
                return np.full(N_STATES, 1.0 / N_STATES)

            trans = bundle.model.transmat_           # (N_STATES, N_STATES)
            cur   = bundle.last_state
            next_raw = trans[cur]                    # shape (N_STATES,)

            # Reorder to STATE_NAMES
            reordered = np.zeros(N_STATES)
            inv_map = {v: k for k, v in bundle.label_map.items()}
            for i, name in enumerate(STATE_NAMES):
                raw_idx = inv_map.get(name, 0)
                reordered[i] = next_raw[raw_idx]
            return reordered

        except Exception as e:
            logger.debug(f"get_next_probs failed for {ticker}: {e}")
            return np.full(N_STATES, 1.0 / N_STATES)

    def get_vol_forecast(self, ticker: str) -> float:
        """
        Return expected realised-vol for next bar based on current
        state's emission distribution (feature index 1 = rvol_20).
        """
        try:
            bundle = self._bundles.get(ticker)
            if bundle is None or bundle.last_state is None:
                return 0.0

            s = bundle.last_state
            # Weighted mean of the N_MIX Gaussian means for feature 1 (rvol)
            weights = bundle.model.weights_[s]      # (N_MIX,)
            means   = bundle.model.means_[s, :, 1]  # (N_MIX,) — feature 1
            vol_est = float(np.dot(weights, means))
            return max(vol_est, 0.0)

        except Exception as e:
            logger.debug(f"get_vol_forecast failed for {ticker}: {e}")
            return 0.0

    def get_regime_confidence(self, ticker: str) -> float:
        """
        Return max(state_probs) for the most recent bar.
        High = clearly in one regime; low = transition.
        """
        try:
            bundle = self._bundles.get(ticker)
            if bundle is None or bundle.last_state is None:
                return 1.0 / N_STATES

            # Use the last posterior if available via score_samples,
            # else fall back to state distribution from transmat
            last_state = bundle.last_state
            trans = bundle.model.transmat_
            # Approximate: probability of staying in current state
            # (the diagonal of the transition matrix gives persistence)
            return float(trans[last_state, last_state])
        except Exception:
            return 1.0 / N_STATES

    def is_in_transition(self, ticker: str) -> bool:
        """
        True if regime confidence is below TRANSITION_THR or if
        the regime changed in the last 3 bars.
        """
        try:
            if self.get_regime_confidence(ticker) < TRANSITION_THR:
                return True
            bundle = self._bundles.get(ticker)
            if bundle is None:
                return False
            if len(bundle.regime_hist) >= 2:
                return len(set(bundle.regime_hist)) > 1
            return False
        except Exception:
            return False

    def get_strategy_weights(self, ticker: str) -> dict:
        """Return position-size multipliers per strategy for this ticker's regime."""
        bundle = self._bundles.get(ticker)
        if bundle is None:
            return _STRATEGY_WEIGHTS["ranging"]
        regime = bundle.voted_regime()
        return _STRATEGY_WEIGHTS.get(regime, _STRATEGY_WEIGHTS["ranging"])

    def predict_states(self, df: pd.DataFrame, ticker: str) -> pd.Series:
        """
        Decode the most likely hidden-state sequence for df using the
        already-fitted model for ticker (Viterbi algorithm).
        Returns a pd.Series of regime strings indexed like df.
        Used by the backtest engine for per-bar regime labelling.
        """
        try:
            bundle = self._bundles.get(ticker)
            if bundle is None:
                # Fall back to ADX-based per-bar labelling
                return detect_regime(df)

            obs, idx = _build_obs(df)
            if len(obs) == 0:
                return pd.Series("ranging", index=df.index)

            raw_states = bundle.model.predict(obs)
            regimes    = pd.Series(
                [bundle.regime_for(int(s)) for s in raw_states],
                index=idx,
            )
            # Forward-fill to cover any rows dropped by _build_obs
            regimes = regimes.reindex(df.index).ffill().fillna("ranging")
            return regimes

        except Exception as e:
            logger.debug(f"predict_states failed for {ticker}: {e}")
            return detect_regime(df)

    def train_on(self, ticker: str, df: pd.DataFrame) -> bool:
        """
        Train a fresh HMM on df and store in memory (not saved to disk).
        Used by the walk-forward engine to train a per-fold model on
        training data only, avoiding any look-ahead into the test window.
        Returns True on success, False on fallback.
        """
        bundle = self._fit(ticker, df)
        if bundle is not None:
            with self._lock(ticker):
                self._bundles[ticker] = bundle
            return True
        return False

    def pretrain_all(self, tickers: list):
        """
        Train GMMHMM for every ticker in the list.
        Called once during --mode train.
        """
        from data.pipeline import DataPipeline
        pipeline = DataPipeline()

        fallback_tickers = []
        for ticker in tickers:
            logger.info(f"HMM pretrain: {ticker}")
            df = pipeline.get(ticker)
            if df.empty:
                pipeline.refresh_all([ticker], training_mode=True)
                df = pipeline.get(ticker)

            if df.empty or len(df) < MIN_TRAIN_BARS:
                logger.warning(f"{ticker}: not enough data for HMM ({len(df)} bars) — will fallback")
                fallback_tickers.append(ticker)
                continue

            bundle = self._fit(ticker, df)
            if bundle is None:
                logger.warning(f"{ticker}: GMMHMM failed — will fallback")
                fallback_tickers.append(ticker)
            else:
                self._bundles[ticker] = bundle
                self._save_bundle(ticker, bundle)

        if fallback_tickers:
            logger.warning(f"Fallback (ADX) tickers: {fallback_tickers}")
        logger.info(
            f"HMM pretrain complete: {len(tickers)-len(fallback_tickers)}/{len(tickers)} succeeded"
        )

    # ── Backward-compat methods (used by PortfolioManager) ──

    def update(self, ticker: str, df: pd.DataFrame):
        """Run detect() and cache result (PortfolioManager compatibility)."""
        self.detect(df, ticker)

    def current(self, ticker: str) -> str:
        """Return most recently cached regime for ticker."""
        bundle = self._bundles.get(ticker)
        if bundle is None:
            return "ranging"
        return bundle.voted_regime()

    def dominant_regime(self) -> str:
        """Most common regime across all trained tickers."""
        if not self._bundles:
            return "ranging"
        from collections import Counter
        regimes = [b.voted_regime() for b in self._bundles.values() if b is not None]
        if not regimes:
            return "ranging"
        return Counter(regimes).most_common(1)[0][0]

    def summary(self) -> pd.DataFrame:
        data = [
            {"ticker": t, "regime": b.voted_regime()}
            for t, b in self._bundles.items()
            if b is not None
        ]
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data).set_index("ticker")


# ── Module-level helpers (backward compat with old callers) ──

def detect_regime(df: pd.DataFrame) -> pd.Series:
    """
    Stateless fallback returning an ADX-based regime Series.
    Kept for any code that still calls the old module function.
    """
    c = df["close"]
    h, l = df["high"], df["low"]
    tr      = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr14   = tr.ewm(span=14, adjust=False).mean()
    pdm     = (h.diff()).clip(lower=0)
    mdm     = (-l.diff()).clip(lower=0)
    pdi     = 100 * pdm.ewm(span=14, adjust=False).mean() / (atr14 + 1e-9)
    mdi     = 100 * mdm.ewm(span=14, adjust=False).mean() / (atr14 + 1e-9)
    dx      = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-9)
    adx     = dx.ewm(span=14, adjust=False).mean()
    ema_f   = c.ewm(span=EMA_FAST, adjust=False).mean()
    ema_s   = c.ewm(span=EMA_SLOW, adjust=False).mean()
    lr      = c.pct_change()
    rvol    = lr.rolling(REGIME_LOOKBACK // 5).std()
    rvol_m  = rvol.rolling(REGIME_LOOKBACK).mean()
    rvol_r  = rvol / (rvol_m + 1e-9)

    regime = pd.Series("ranging", index=df.index)
    regime[(adx > ADX_TREND_THRESHOLD) & (ema_f > ema_s)] = "trending_up"
    regime[(adx > ADX_TREND_THRESHOLD) & (ema_f < ema_s)] = "trending_down"
    regime[rvol_r > RVOL_HIGH_THRESHOLD] = "high_volatility"
    return regime


def get_current_regime(df: pd.DataFrame) -> str:
    regimes = detect_regime(df)
    return regimes.iloc[-1] if not regimes.empty else "ranging"


def regime_position_scale(regime: str) -> float:
    return {"trending_up": 1.0, "trending_down": 1.0,
            "ranging": 0.75, "high_volatility": 0.30}.get(regime, 0.75)


def strategy_weights_by_regime(regime: str) -> dict:
    return _STRATEGY_WEIGHTS.get(regime, {
        "momentum": 0.35, "mean_reversion": 0.30,
        "breakout": 0.20, "regime_adaptive": 0.15,
    })
