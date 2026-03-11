"""
================================================================
  signals/ensemble.py
  Stacked Ensemble ML Signal Engine:

  Base layer:
    1. XGBoost
    2. LightGBM
    3. Random Forest

  Meta-learner:
    Logistic Regression (learns which model to trust per regime)

  Output:
    signal       : -1 (sell), 0 (flat), 1 (buy)
    confidence   : 0.0 – 1.0 (ensemble probability)
    model_votes  : dict of per-model signals
================================================================
"""

import os
import pickle
import logging
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection    import TimeSeriesSplit, cross_val_predict
from sklearn.ensemble           import RandomForestClassifier
from sklearn.linear_model       import LogisticRegression
from sklearn.preprocessing      import StandardScaler
from sklearn.metrics            import accuracy_score, classification_report
from sklearn.calibration        import CalibratedClassifierCV
from xgboost                    import XGBClassifier
from lightgbm                   import LGBMClassifier

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.settings import (
    XGB_PARAMS, LGBM_PARAMS, RF_PARAMS,
    CV_SPLITS, MODEL_DIR, MIN_CONFIDENCE,
)
from signals.features import FEATURE_COLS, build_features, get_X_y

logger = logging.getLogger("trading_firm.ensemble")


class StackedEnsemble:
    """
    3-base-model stacked ensemble with logistic regression meta-learner.

    Training procedure:
      1. Walk-forward CV on XGB, LGBM, RF individually
      2. Generate out-of-fold probability predictions from each
      3. Stack OOF predictions as meta-features
      4. Train LogisticRegression meta-learner on meta-features
      5. Final base models retrained on full dataset

    Prediction:
      1. Each base model outputs class probabilities
      2. Probabilities stacked and fed to meta-learner
      3. Meta-learner outputs final signal + confidence
    """

    def __init__(self, instrument: str = "model"):
        self.instrument   = instrument
        self.base_models  = {}
        self.meta_learner = None
        self.scaler       = StandardScaler()
        self.classes_     = np.array([-1, 0, 1])
        self.is_trained   = False
        self._cv_metrics  = {}

    # ── Base model builders ──────────────────────────────────

    def _make_xgb(self) -> XGBClassifier:
        return XGBClassifier(**XGB_PARAMS)

    def _make_lgbm(self) -> LGBMClassifier:
        return LGBMClassifier(**LGBM_PARAMS)

    def _make_rf(self) -> RandomForestClassifier:
        return RandomForestClassifier(**RF_PARAMS)

    # ── Training ─────────────────────────────────────────────

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Full stacked ensemble training.

        Parameters
        ----------
        X : feature matrix (n_samples, n_features)
        y : labels [-1, 0, 1]

        Returns
        -------
        dict with per-model and ensemble CV metrics
        """
        n = len(X)
        logger.info(
            f"[{self.instrument}] Training stacked ensemble | "
            f"samples={n} | features={X.shape[1]}"
        )

        tscv = TimeSeriesSplit(n_splits=CV_SPLITS)

        # ── Step 1: Generate OOF predictions from each base model ──
        oof_xgb  = np.zeros((n, 3))
        oof_lgbm = np.zeros((n, 3))
        oof_rf   = np.zeros((n, 3))

        xgb_scores, lgbm_scores, rf_scores = [], [], []

        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            # XGBoost
            xgb = self._make_xgb()
            xgb.fit(X_tr, y_tr)
            oof_xgb[val_idx]  = xgb.predict_proba(X_val)
            xgb_scores.append(accuracy_score(y_val, xgb.predict(X_val)))

            # LightGBM
            lgbm = self._make_lgbm()
            lgbm.fit(X_tr, y_tr)
            oof_lgbm[val_idx] = lgbm.predict_proba(X_val)
            lgbm_scores.append(accuracy_score(y_val, lgbm.predict(X_val)))

            # Random Forest
            rf = self._make_rf()
            rf.fit(X_tr, y_tr)
            oof_rf[val_idx]   = rf.predict_proba(X_val)
            rf_scores.append(accuracy_score(y_val, rf.predict(X_val)))

            logger.info(
                f"  Fold {fold+1}/{CV_SPLITS}: "
                f"XGB={xgb_scores[-1]:.4f}  "
                f"LGBM={lgbm_scores[-1]:.4f}  "
                f"RF={rf_scores[-1]:.4f}"
            )

        # ── Step 2: Stack OOF probabilities as meta-features ──────
        meta_X = np.hstack([oof_xgb, oof_lgbm, oof_rf])   # shape (n, 9)
        meta_X = self.scaler.fit_transform(meta_X)

        # ── Step 3: Train meta-learner on OOF predictions ─────────
        self.meta_learner = LogisticRegression(
            C=1.0, max_iter=1000,
            multi_class="multinomial",
            random_state=42
        )
        self.meta_learner.fit(meta_X, y)
        meta_preds = self.meta_learner.predict(meta_X)
        meta_acc   = accuracy_score(y, meta_preds)

        # ── Step 4: Retrain base models on full dataset ────────────
        self.base_models["xgb"]  = self._make_xgb();   self.base_models["xgb"].fit(X, y)
        self.base_models["lgbm"] = self._make_lgbm();  self.base_models["lgbm"].fit(X, y)
        self.base_models["rf"]   = self._make_rf();    self.base_models["rf"].fit(X, y)

        self.is_trained = True

        self._cv_metrics = {
            "xgb":      {"mean": np.mean(xgb_scores),  "std": np.std(xgb_scores)},
            "lgbm":     {"mean": np.mean(lgbm_scores), "std": np.std(lgbm_scores)},
            "rf":       {"mean": np.mean(rf_scores),   "std": np.std(rf_scores)},
            "ensemble": {"meta_accuracy": meta_acc},
        }

        logger.info(
            f"[{self.instrument}] Training complete:\n"
            f"  XGB    : {np.mean(xgb_scores):.4f} ± {np.std(xgb_scores):.4f}\n"
            f"  LGBM   : {np.mean(lgbm_scores):.4f} ± {np.std(lgbm_scores):.4f}\n"
            f"  RF     : {np.mean(rf_scores):.4f} ± {np.std(rf_scores):.4f}\n"
            f"  Ensemble meta-accuracy: {meta_acc:.4f}"
        )
        return self._cv_metrics

    # ── Prediction ────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return ensemble probability matrix (n_samples, 3)."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call .train() first.")

        p_xgb  = self.base_models["xgb"].predict_proba(X)
        p_lgbm = self.base_models["lgbm"].predict_proba(X)
        p_rf   = self.base_models["rf"].predict_proba(X)

        meta_X = np.hstack([p_xgb, p_lgbm, p_rf])
        meta_X = self.scaler.transform(meta_X)
        return self.meta_learner.predict_proba(meta_X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        idx   = proba.argmax(axis=1)
        return self.classes_[idx]

    def predict_with_confidence(self, X: np.ndarray) -> tuple:
        """
        Returns (signal_array, confidence_array, votes_dict).

        signal     : array of [-1, 0, 1]
        confidence : max probability from meta-learner
        votes      : dict of per-model predictions for transparency
        """
        proba      = self.predict_proba(X)
        confidence = proba.max(axis=1)
        signal     = self.classes_[proba.argmax(axis=1)]

        votes = {
            "xgb":  self.base_models["xgb"].predict(X).tolist(),
            "lgbm": self.base_models["lgbm"].predict(X).tolist(),
            "rf":   self.base_models["rf"].predict(X).tolist(),
        }
        return signal, confidence, votes

    def signal_for_latest_bar(self, df: pd.DataFrame) -> dict:
        """
        High-level method: takes raw OHLCV, returns signal dict
        for the most recent bar. Used by live trading loop.
        """
        df_feat = build_features(df, add_labels=False, drop_na=True)
        if df_feat.empty or len(df_feat) < 1:
            return {"signal": 0, "confidence": 0.0, "reason": "Insufficient data"}

        latest = df_feat.iloc[[-1]]
        X      = latest[FEATURE_COLS].values

        try:
            signals, confs, votes = self.predict_with_confidence(X)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"signal": 0, "confidence": 0.0, "reason": str(e)}

        sig  = int(signals[0])
        conf = float(confs[0])

        return {
            "signal":      sig,
            "confidence":  conf,
            "votes":       {k: int(v[0]) for k, v in votes.items()},
            "atr":         float(latest["atr"].iloc[0]) if "atr" in latest else 0.0,
            "atr_norm":    float(latest["atr_norm"].iloc[0]) if "atr_norm" in latest else 1.0,
            "close":       float(latest["close"].iloc[0]),
            "bar_time":    df_feat.index[-1],
            "tradeable":   conf >= MIN_CONFIDENCE and sig != 0,
        }

    # ── Feature importance ────────────────────────────────────

    def feature_importance(self) -> pd.DataFrame:
        """Aggregated feature importance across all base models."""
        fi_xgb  = pd.Series(self.base_models["xgb"].feature_importances_,  index=FEATURE_COLS)
        fi_lgbm = pd.Series(self.base_models["lgbm"].feature_importances_, index=FEATURE_COLS)
        fi_rf   = pd.Series(self.base_models["rf"].feature_importances_,   index=FEATURE_COLS)

        df_fi = pd.DataFrame({
            "xgb":  fi_xgb,
            "lgbm": fi_lgbm,
            "rf":   fi_rf,
        })
        df_fi["ensemble_avg"] = df_fi.mean(axis=1)
        return df_fi.sort_values("ensemble_avg", ascending=False)

    # ── Persistence ───────────────────────────────────────────

    def save(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        path = os.path.join(MODEL_DIR, f"{self.instrument}_ensemble.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Ensemble saved → {path}")

    @classmethod
    def load(cls, instrument: str) -> "StackedEnsemble":
        path = os.path.join(MODEL_DIR, f"{instrument}_ensemble.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No saved model for '{instrument}' at {path}\n"
                f"Run:  python main.py --mode train --instrument {instrument}"
            )
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Ensemble loaded ← {path}")
        return model

    @property
    def cv_metrics(self) -> dict:
        return self._cv_metrics
