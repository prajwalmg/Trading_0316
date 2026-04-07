"""
================================================================
  signals/intraday_model.py  — Lightweight Intraday Model
================================================================
  Fast LightGBM + XGBoost ensemble for 5m bars.
  Exposes same predict_with_confidence() interface as
  StackedEnsemble so engine_intraday.py works unchanged.
  Trains in ~5-30 seconds per pair (vs 10+ min for full stack).
================================================================
"""

import logging
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger("trading_firm.intraday_model")


class IntradayModel:
    """
    Two-model blend (LightGBM + XGBoost) with probability calibration.
    Binary classification: +1 (long) vs -1 (short).
    Confidence = max(P_long, P_short).
    """

    def __init__(self, instrument: str = "intraday"):
        self.instrument = instrument
        self.scaler     = RobustScaler()
        self._lgbm      = None
        self._xgb       = None
        self.is_trained = False
        self._classes   = np.array([-1, 1])

    # ── Training ──────────────────────────────────────────────
    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        from lightgbm import LGBMClassifier
        from xgboost  import XGBClassifier

        X = self.scaler.fit_transform(X)

        # Map [-1,1] → [0,1] for XGBoost
        y_enc = (y == 1).astype(int)

        lgbm_params = dict(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1, n_jobs=1,
        )
        xgb_params = dict(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=1,
        )

        # Calibrate probabilities with isotonic regression (3-fold CV)
        self._lgbm = CalibratedClassifierCV(
            LGBMClassifier(**lgbm_params), method="isotonic", cv=3
        )
        self._xgb = CalibratedClassifierCV(
            XGBClassifier(**xgb_params), method="isotonic", cv=3
        )

        self._lgbm.fit(X, y)       # [-1,1] directly
        self._xgb.fit(X, y_enc)    # [0,1]
        self.is_trained = True
        logger.info(f"[{self.instrument}] IntradayModel trained on {len(X)} samples")
        return {}

    # ── Inference ─────────────────────────────────────────────
    def predict_with_confidence(self, X: np.ndarray):
        """
        Returns (signals, confidences, _) matching StackedEnsemble interface.
        signal  ∈ {-1, 0, 1}
        confidence ∈ [0, 1]
        """
        n = len(X)
        if not self.is_trained:
            return np.zeros(n, dtype=int), np.full(n, 0.5), None

        X_sc = self.scaler.transform(X)

        # LGBM: classes_ = [-1, 1], so col0=P(-1), col1=P(1)
        p_lgbm = self._lgbm.predict_proba(X_sc)   # (n, 2)
        # XGB: classes_ = [0, 1] (encoded), so col0=P(-1), col1=P(1)
        p_xgb  = self._xgb.predict_proba(X_sc)    # (n, 2)

        # Blend equally
        p_blend = 0.5 * p_lgbm + 0.5 * p_xgb     # (n, 2)

        # col0 = P(short=-1), col1 = P(long=+1)
        p_short = p_blend[:, 0]
        p_long  = p_blend[:, 1]

        signals     = np.zeros(n, dtype=int)
        confidences = np.full(n, 0.5)

        long_mask  = p_long  > p_short
        short_mask = p_short > p_long

        signals[long_mask]      =  1
        confidences[long_mask]  = p_long[long_mask]
        signals[short_mask]     = -1
        confidences[short_mask] = p_short[short_mask]

        return signals, confidences, None

    # ── Persistence ───────────────────────────────────────────
    def save(self, path: str) -> None:
        import pickle, os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"IntradayModel saved → {path}")

    @classmethod
    def load(cls, path: str) -> "IntradayModel":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
