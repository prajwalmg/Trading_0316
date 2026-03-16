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

  Note on label encoding:
    XGBoost requires classes [0, 1, 2].
    We encode [-1, 0, 1] → [0, 1, 2] before training
    and decode back to [-1, 0, 1] after prediction automatically.
================================================================
"""

import os
import pickle
import logging
import warnings
import numpy as np
import pandas as pd

from sklearn.neural_network  import MLPClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble        import RandomForestClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import accuracy_score, classification_report
from xgboost                 import XGBClassifier
from lightgbm                import LGBMClassifier

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.settings import (
    XGB_PARAMS, LGBM_PARAMS, RF_PARAMS, MLPC_PARAMS,
    CV_SPLITS, MODEL_DIR, MIN_CONFIDENCE,
)
from signals.features import FEATURE_COLS, build_features, get_X_y

logger = logging.getLogger("trading_firm.ensemble")


class StackedEnsemble:
    """
    3-base-model stacked ensemble with logistic regression meta-learner.

    Label encoding:
      XGBoost requires integer classes starting at 0.
      Internally we map [-1, 0, 1] → [0, 1, 2] for XGBoost/LightGBM,
      and decode [0, 1, 2] → [-1, 0, 1] in all prediction outputs.
      RandomForest and LogisticRegression handle [-1, 0, 1] natively.
    """

    _ENCODE = {-1: 0,  0: 1,  1: 2}   # for XGBoost / LightGBM
    _DECODE = { 0: -1, 1: 0,  2: 1}   # back to trading signals
    _ENCODE_BINARY = {-1: 0, 1: 1}    # binary: no class 0
    _DECODE_BINARY = { 0: -1, 1: 1}   # binary back to signals

    def __init__(self, instrument: str = "model"):
        self.instrument   = instrument
        self.base_models  = {}
        self.meta_learner = None
        self.scaler       = StandardScaler()
        self.classes_     = np.array([-1, 0, 1])
        self.is_trained   = False
        self._cv_metrics  = {}
        self.selected_features_ = None
        self.feature_scaler_ = None

    # ── Label helpers ────────────────────────────────────────

    def _enc(self, y: np.ndarray) -> np.ndarray:
        """Auto-detects binary [-1,1] or 3-class [-1,0,1] and encodes."""
        unique = set(np.unique(y).tolist())
        if unique <= {-1, 1}:   # binary
            return np.vectorize(self._ENCODE_BINARY.get)(y)
        return np.vectorize(self._ENCODE.get)(y)

    def _dec(self, y: np.ndarray) -> np.ndarray:
        """Auto-detects binary [0,1] or 3-class [0,1,2] and decodes."""
        unique = set(np.unique(y).tolist())
        if unique <= {0, 1} and 2 not in unique:   # binary
            return np.vectorize(self._DECODE_BINARY.get)(y)
        return np.vectorize(self._DECODE.get)(y)


    # ── Base model builders ──────────────────────────────────

    def _make_xgb(self):
        return XGBClassifier(**XGB_PARAMS)

    def _make_lgbm(self):
        return LGBMClassifier(**LGBM_PARAMS)

    def _make_rf(self):
        return RandomForestClassifier(**RF_PARAMS)
    
    def _make_mlp(self):
        return MLPClassifier(**MLPC_PARAMS)

    # ── Training ─────────────────────────────────────────────

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Full stacked ensemble training with walk-forward CV.

        Parameters
        ----------
        X : feature matrix (n_samples, n_features)
        y : labels [-1, 0, 1]
        """
        n    = len(X)
        y_xg = self._enc(y)   # encoded labels for XGBoost/LightGBM [0,1,2]

        # ── Fit feature scaler on full training data ──────────────
        from sklearn.preprocessing import RobustScaler
        self.feature_scaler_ = RobustScaler()
        X = self.feature_scaler_.fit_transform(X)

        logger.info(
            f"[{self.instrument}] Training stacked ensemble | "
            f"samples={n} | features={X.shape[1]}"
        )

        tscv = TimeSeriesSplit(n_splits=CV_SPLITS)

        n_classes = len(np.unique(y))
        oof_xgb  = np.zeros((n, n_classes))
        oof_lgbm = np.zeros((n, n_classes))
        oof_rf   = np.zeros((n, n_classes))
        oof_mlp  = np.zeros((n, n_classes))

        xgb_scores, lgbm_scores, rf_scores, mlpc_scores = [], [], [], []

        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val   = X[tr_idx],    X[val_idx]
            y_tr_xg       = y_xg[tr_idx]  # encoded for XGB/LGBM
            y_tr_raw      = y[tr_idx]      # raw for RF
            y_val_raw     = y[val_idx]

            # XGBoost (needs encoded labels)
            xgb = self._make_xgb()
            xgb.fit(X_tr, y_tr_xg)
            oof_xgb[val_idx] = xgb.predict_proba(X_val)
            xgb_preds = self._dec(xgb.predict(X_val))
            xgb_scores.append(accuracy_score(y_val_raw, xgb_preds))

            # LightGBM (needs encoded labels)
            lgbm = self._make_lgbm()
            lgbm.fit(X_tr, y_tr_xg)
            oof_lgbm[val_idx] = lgbm.predict_proba(X_val)
            lgbm_preds = self._dec(lgbm.predict(X_val))
            lgbm_scores.append(accuracy_score(y_val_raw, lgbm_preds))

            # Random Forest (handles [-1,0,1] natively)
            rf = self._make_rf()
            rf.fit(X_tr, y_tr_raw)
            oof_rf[val_idx] = rf.predict_proba(X_val)
            rf_scores.append(accuracy_score(y_val_raw, rf.predict(X_val)))

            # MLP Classifier (needs encoded labels)
            mlp = self._make_mlp()
            mlp.fit(X_tr, y_tr_xg)
            oof_mlp[val_idx] = mlp.predict_proba(X_val)
            mlpc_scores.append(accuracy_score(y_val_raw, self._dec(mlp.predict(X_val))))

            logger.info(
                f"  Fold {fold+1}/{CV_SPLITS}: "
                f"XGB={xgb_scores[-1]:.4f}  "
                f"LGBM={lgbm_scores[-1]:.4f}  "
                f"RF={rf_scores[-1]:.4f}  "
                f"MLP={mlpc_scores[-1]:.4f}"
            )

        # ── Feature selection using XGBoost importances ───────
        # Train a quick XGBoost on full data to get importances
        _selector_xgb = self._make_xgb()
        _selector_xgb.fit(X, y_xg)
        importances = _selector_xgb.feature_importances_

        # Keep features above 0.5% importance OR top 40 — whichever is larger
        threshold     = 0.005
        mask_thresh   = importances >= threshold
        mask_top40    = np.zeros(len(importances), dtype=bool)
        top40_idx     = np.argsort(importances)[-40:]
        mask_top40[top40_idx] = True
        mask          = mask_thresh | mask_top40

        self.selected_features_ = np.where(mask)[0]
        n_selected = len(self.selected_features_)

        logger.info(
            f"[{self.instrument}] Feature selection: "
            f"{X.shape[1]} → {n_selected} features "
            f"(threshold={threshold}, top40 union)"
        )

        # Reduce X to selected features only
        X        = X[:, self.selected_features_]
        y_xg     = self._enc(y)  # re-encode since y unchanged

        # Rebuild OOF arrays with reduced feature set
        n_classes = len(np.unique(y))
        oof_xgb  = np.zeros((n, n_classes))
        oof_lgbm = np.zeros((n, n_classes))
        oof_rf   = np.zeros((n, n_classes))
        oof_mlp  = np.zeros((n, n_classes))     

        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr_xg     = y_xg[tr_idx]
            y_tr_raw    = y[tr_idx]

            _x = self._make_xgb();  _x.fit(X_tr, y_tr_xg)
            oof_xgb[val_idx]  = _x.predict_proba(X_val)

            _l = self._make_lgbm(); _l.fit(X_tr, y_tr_xg)
            oof_lgbm[val_idx] = _l.predict_proba(X_val)

            _r = self._make_rf();   _r.fit(X_tr, y_tr_raw)
            oof_rf[val_idx]   = _r.predict_proba(X_val)

            _m = self._make_mlp();  _m.fit(X_tr, y_tr_xg)
            oof_mlp[val_idx]  = _m.predict_proba(X_val)
        

        # ── Stack OOF probs as meta-features ─────────────────
        meta_X = np.hstack([oof_xgb, oof_lgbm, oof_rf, oof_mlp])   # (n, 12)
        meta_X = self.scaler.fit_transform(meta_X)

        # Meta-learner uses raw [-1,0,1] labels
        self.meta_learner = LogisticRegression(
            C=1.0, max_iter=1000,
            random_state=42
        )
        self.meta_learner.fit(meta_X, y)
        meta_acc = accuracy_score(y, self.meta_learner.predict(meta_X))

        # ── Retrain base models on full dataset ───────────────
        self.base_models["xgb"]  = self._make_xgb()
        self.base_models["xgb"].fit(X, y_xg)

        self.base_models["lgbm"] = self._make_lgbm()
        self.base_models["lgbm"].fit(X, y_xg)

        self.base_models["mlp"] = self._make_mlp()
        self.base_models["mlp"].fit(X, y_xg)

        self.base_models["rf"]   = self._make_rf()
        self.base_models["rf"].fit(X, y)         # RF uses raw labels

        self.is_trained = True

        self._cv_metrics = {
            "xgb":      {"mean": np.mean(xgb_scores),  "std": np.std(xgb_scores)},
            "lgbm":     {"mean": np.mean(lgbm_scores), "std": np.std(lgbm_scores)},
            "rf":       {"mean": np.mean(rf_scores),   "std": np.std(rf_scores)},
            "mlp":      {"mean": np.mean(mlpc_scores), "std": np.std(mlpc_scores)},
            "ensemble": {"meta_accuracy": meta_acc},
        }

        logger.info(
            f"[{self.instrument}] Training complete:\n"
            f"  XGB     : {np.mean(xgb_scores):.4f} ± {np.std(xgb_scores):.4f}\n"
            f"  LGBM    : {np.mean(lgbm_scores):.4f} ± {np.std(lgbm_scores):.4f}\n"
            f"  RF      : {np.mean(rf_scores):.4f} ± {np.std(rf_scores):.4f}\n"
            f"  MLP     : {np.mean(mlpc_scores):.4f} ± {np.std(mlpc_scores):.4f}\n"
            f"  Ensemble meta-accuracy: {meta_acc:.4f}"
        )
        return self._cv_metrics

    # ── Prediction ────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return ensemble probability matrix (n_samples, 3).
        Column order matches self.classes_ = [-1, 0, 1].
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call .train() first.")

        if self.feature_scaler_ is not None:
            X = self.feature_scaler_.transform(X)

        if self.selected_features_ is not None:
            X = X[:, self.selected_features_]
        
        # XGBoost/LGBM return probs for [0,1,2] — order is preserved
        p_xgb  = self.base_models["xgb"].predict_proba(X)
        p_lgbm = self.base_models["lgbm"].predict_proba(X)
        p_rf   = self.base_models["rf"].predict_proba(X)
        p_mlp  = self.base_models["mlp"].predict_proba(X)

        meta_X = np.hstack([p_xgb, p_lgbm, p_rf, p_mlp])   
        meta_X = self.scaler.transform(meta_X)
        return self.meta_learner.predict_proba(meta_X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        idx   = proba.argmax(axis=1)
        return self.meta_learner.classes_[idx]

    def predict_with_confidence(self, X: np.ndarray) -> tuple:
        """
        Returns (signal_array, confidence_array, votes_dict).
        All signals are in [-1, 0, 1].
        """
        # predict_proba handles feature selection internally
        proba      = self.predict_proba(X)
        confidence = proba.max(axis=1)
        signal     = self.meta_learner.classes_[proba.argmax(axis=1)]

        # Apply selection once for individual model votes
        X_sel = X[:, self.selected_features_] if self.selected_features_ is not None else X

        votes = {
            "xgb":  self._dec(self.base_models["xgb"].predict(X_sel)).tolist(),
            "lgbm": self._dec(self.base_models["lgbm"].predict(X_sel)).tolist(),
            "rf":   self.base_models["rf"].predict(X_sel).tolist(),
        }
        return signal, confidence, votes


    def signal_for_latest_bar(self, df: pd.DataFrame) -> dict:
        """
        Takes raw OHLCV DataFrame, returns signal dict for most recent bar.
        Used by the live trading loop.
        """
        df_feat = build_features(df, add_labels=False, drop_na=True)
        if df_feat.empty:
            return {"signal": 0, "confidence": 0.0, "reason": "Insufficient data", "tradeable": False}

        latest = df_feat.iloc[[-1]]
        X      = latest[FEATURE_COLS].values

        try:
            signals, confs, votes = self.predict_with_confidence(X)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"signal": 0, "confidence": 0.0, "reason": str(e), "tradeable": False}

        sig  = int(signals[0])
        conf = float(confs[0])

        return {
            "signal":     sig,
            "confidence": conf,
            "votes":      {k: int(v[0]) for k, v in votes.items()},
            "atr":        float(latest["atr"].iloc[0])      if "atr"      in latest.columns else 0.0,
            "atr_norm":   float(latest["atr_norm"].iloc[0]) if "atr_norm" in latest.columns else 1.0,
            "close":      float(latest["close"].iloc[0]),
            "bar_time":   df_feat.index[-1],
            "tradeable":  conf >= MIN_CONFIDENCE and sig != 0,
        }

    # ── Feature importance ────────────────────────────────────

    def feature_importance(self) -> pd.DataFrame:
        fi_xgb  = pd.Series(self.base_models["xgb"].feature_importances_,  index=FEATURE_COLS)
        fi_lgbm = pd.Series(self.base_models["lgbm"].feature_importances_, index=FEATURE_COLS)
        fi_rf   = pd.Series(self.base_models["rf"].feature_importances_,   index=FEATURE_COLS)
        df_fi   = pd.DataFrame({"xgb": fi_xgb, "lgbm": fi_lgbm, "rf": fi_rf})
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
                f"Run:  python main.py --mode train"
            )
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Ensemble loaded ← {path}")
        return model

    @property
    def cv_metrics(self) -> dict:
        return self._cv_metrics