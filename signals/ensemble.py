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

from sklearn.ensemble        import RandomForestClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import accuracy_score, classification_report
from xgboost                 import XGBClassifier
from lightgbm                import LGBMClassifier


# ── Purged Walk-Forward CV ────────────────────────────────────

class PurgedTimeSeriesSplit:
    """
    Walk-forward CV with a purge gap and embargo period.

    Purge gap  — removes training samples whose feature windows
                 overlap with the validation period.  Set to the
                 longest indicator lookback (default 200 bars for
                 EMA-200).

    Embargo    — removes the first N bars of validation to prevent
                 leakage from the last training bar's market impact.

    Without this, an EMA-200 feature at the last training bar uses
    the same prices as the first validation bar — information leaks
    both ways across the fold boundary.
    """

    def __init__(
        self,
        n_splits:     int = 5,
        purge_bars:   int = 200,   # longest feature lookback
        embargo_bars: int = 10,    # bars to drop at start of val
    ):
        self.n_splits     = n_splits
        self.purge_bars   = purge_bars
        self.embargo_bars = embargo_bars

    def split(self, X, y=None, groups=None):  # noqa: y/groups unused (sklearn API)
        n         = len(X)
        fold_size = n // (self.n_splits + 1)

        for i in range(1, self.n_splits + 1):
            train_end = i * fold_size

            # Purge: drop the last purge_bars of training so feature
            # windows don't overlap with the start of validation.
            purged_train_end = max(0, train_end - self.purge_bars)
            train_idx = np.arange(0, purged_train_end)

            # Validation starts right after the original train_end
            # (not the purged end) plus an embargo buffer.
            val_start = train_end + self.embargo_bars
            val_end   = val_start + fold_size
            if val_end > n or len(train_idx) < 50:
                break

            val_idx = np.arange(val_start, val_end)
            yield train_idx, val_idx

    def get_n_splits(self, X=None, y=None, groups=None):  # noqa: sklearn API
        return self.n_splits

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.settings import (
    XGB_PARAMS, LGBM_PARAMS, RF_PARAMS, MLPC_PARAMS,
    CV_SPLITS, MODEL_DIR, MIN_CONFIDENCE,
)
from signals.features import (FEATURE_COLS, SWING_FEATURE_COLS, build_features, get_X_y,
                               HTF_FEATURE_COLS)

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

    def __init__(self, instrument: str = "model", swing: bool = False):
        self.instrument   = instrument
        self.base_models  = {}
        self.meta_learner = None
        # Adaptive per-class confidence thresholds (set during train())
        self._long_threshold  = MIN_CONFIDENCE
        self._short_threshold = MIN_CONFIDENCE
        self.scaler       = StandardScaler()
        self.classes_     = np.array([-1, 0, 1])
        self.is_trained   = False
        self._cv_metrics  = {}
        self.selected_features_ = None
        self.feature_scaler_ = None
        # Record the feature mode used during training so inference always
        # uses the exact same column list, even after save/load.
        self._swing = swing
        self._intraday = False           # set True when trained on INTRADAY_FEATURE_COLS
        self._feature_cols: list = []   # populated by train()

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

    def _make_lstm(self, n_features: int, n_classes: int):
        from signals.lstm_model import LSTMClassifier
        return LSTMClassifier(
            n_features=n_features,
            n_classes=n_classes,
            random_state=42,
        )

    # ── Training ─────────────────────────────────────────────

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Full stacked ensemble training with walk-forward CV.

        Parameters
        ----------
        X : feature matrix (n_samples, n_features)
        y : labels [-1, 0, 1]
        """
        # Stamp the exact feature list used so signal_for_latest_bar() and
        # predict_proba() always apply the correct columns after save/load.
        # Auto-detect swing vs intraday from actual X.shape[1] so this is
        # robust regardless of how swing= was set on the constructor.
        import os as _os
        _os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

        _swing_htf = list(SWING_FEATURE_COLS) + list(HTF_FEATURE_COLS)
        try:
            from signals.features_intraday import INTRADAY_FEATURE_COLS as _IFC
        except ImportError:
            _IFC = []
        if X.shape[1] == len(_swing_htf):
            self._feature_cols = _swing_htf
            self._swing = True
            self._intraday = False
        elif X.shape[1] == len(SWING_FEATURE_COLS):
            self._feature_cols = list(SWING_FEATURE_COLS)
            self._swing = True
            self._intraday = False
        elif X.shape[1] == len(FEATURE_COLS):
            self._feature_cols = list(FEATURE_COLS)
            self._swing = False
            self._intraday = False
        elif _IFC and X.shape[1] == len(_IFC):
            self._feature_cols = list(_IFC)
            self._swing = False
            self._intraday = True
        else:
            # Fallback: generic names for arbitrary feature count
            _base = list(SWING_FEATURE_COLS if self._swing else FEATURE_COLS)
            if X.shape[1] > len(_base):
                _base = _base + [f"feat_{i}" for i in range(X.shape[1] - len(_base))]
            self._feature_cols = _base[:X.shape[1]]
            self._intraday = False
        self._full_feature_cols = list(self._feature_cols)  # save pre-selection list

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

        # PurgedTimeSeriesSplit: purge 200 bars (= longest feature lookback,
        # EMA-200) + 10-bar embargo so no feature window straddles the fold.
        tscv = PurgedTimeSeriesSplit(
            n_splits=CV_SPLITS, purge_bars=200, embargo_bars=10
        )

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

            # ── SMOTE inside fold (no leakage) ────────────────
            # Synthetic samples are generated only from this fold's
            # training split, so no validation-period information
            # can bleed into the oversampled training set.
            try:
                from imblearn.over_sampling import SMOTE
                k = min(5, min(np.bincount(y_tr_xg + 1)) - 1)
                if k >= 1:
                    smote         = SMOTE(random_state=42, k_neighbors=k)
                    X_tr, y_tr_xg = smote.fit_resample(X_tr, y_tr_xg)
                    y_tr_raw      = self._dec(y_tr_xg)
            except ImportError:
                pass   # imblearn not installed — train on imbalanced data
            except Exception as _e:
                logger.debug(f"SMOTE skipped fold {fold}: {_e}")

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

        # ── LSTM OOF (separate loop — uses selected features) ────
        # LSTM is run after feature selection so its sequence inputs
        # are the same reduced feature set used by the meta-learner.
        oof_lstm = np.zeros((n, n_classes))

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
        self._feature_cols = [self._feature_cols[i]
                              for i in self.selected_features_]
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

        lstm_scores = []
        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr_xg     = y_xg[tr_idx]
            y_tr_raw    = y[tr_idx]
            y_val_raw   = y[val_idx]

            _x = self._make_xgb();  _x.fit(X_tr, y_tr_xg)
            oof_xgb[val_idx]  = _x.predict_proba(X_val)[:, :n_classes]

            _l = self._make_lgbm(); _l.fit(X_tr, y_tr_xg)
            oof_lgbm[val_idx] = _l.predict_proba(X_val)[:, :n_classes]

            _r = self._make_rf();   _r.fit(X_tr, y_tr_raw)
            oof_rf[val_idx]   = _r.predict_proba(X_val)[:, :n_classes]

            _m = self._make_mlp();  _m.fit(X_tr, y_tr_xg)
            oof_mlp[val_idx]  = _m.predict_proba(X_val)[:, :n_classes]

            # LSTM — trained on selected features with encoded labels
            try:
                _lstm = self._make_lstm(n_features=X_tr.shape[1], n_classes=n_classes)
                _lstm.fit(X_tr, y_tr_xg)
                lstm_p = _lstm.predict_proba(X_val)[:, :n_classes]
                oof_lstm[val_idx] = lstm_p
                lstm_preds = self._dec(_lstm.predict(X_val))
                lstm_scores.append(accuracy_score(y_val_raw, lstm_preds))
            except Exception as _e:
                logger.debug(f"LSTM fold {fold} skipped: {_e}")
                oof_lstm[val_idx] = 1.0 / n_classes   # neutral fallback

        # ── Stack OOF probs as meta-features ─────────────────
        meta_X = np.hstack([oof_xgb, oof_lgbm, oof_rf, oof_mlp, oof_lstm])  # (n, 10)
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

        # LSTM full-data retrain (on selected features)
        try:
            self.base_models["lstm"] = self._make_lstm(
                n_features=X.shape[1], n_classes=n_classes
            )
            self.base_models["lstm"].fit(X, y_xg)
        except Exception as _e:
            logger.warning(f"LSTM full retrain failed: {_e} — using neutral proba")
            self.base_models["lstm"] = None

        self.is_trained = True

        # ── Compute signal bias on full training set ──────────
        self._compute_signal_bias(X)

        lstm_mean = np.mean(lstm_scores) if lstm_scores else 0.0
        lstm_std  = np.std(lstm_scores)  if lstm_scores else 0.0

        self._cv_metrics = {
            "xgb":      {"mean": np.mean(xgb_scores),  "std": np.std(xgb_scores)},
            "lgbm":     {"mean": np.mean(lgbm_scores), "std": np.std(lgbm_scores)},
            "rf":       {"mean": np.mean(rf_scores),   "std": np.std(rf_scores)},
            "mlp":      {"mean": np.mean(mlpc_scores), "std": np.std(mlpc_scores)},
            "lstm":     {"mean": lstm_mean,             "std": lstm_std},
            "ensemble": {"meta_accuracy": meta_acc},
        }

        logger.info(
            f"[{self.instrument}] Training complete:\n"
            f"  XGB     : {np.mean(xgb_scores):.4f} ± {np.std(xgb_scores):.4f}\n"
            f"  LGBM    : {np.mean(lgbm_scores):.4f} ± {np.std(lgbm_scores):.4f}\n"
            f"  RF      : {np.mean(rf_scores):.4f} ± {np.std(rf_scores):.4f}\n"
            f"  MLP     : {np.mean(mlpc_scores):.4f} ± {np.std(mlpc_scores):.4f}\n"
            f"  LSTM    : {lstm_mean:.4f} ± {lstm_std:.4f}\n"
            f"  Ensemble meta-accuracy: {meta_acc:.4f}"
        )
        return self._cv_metrics

    # ── Signal bias correction ────────────────────────────────

    def _compute_signal_bias(self, X_context: np.ndarray):
        """
        Compute rolling long/short base rate over last 500 training bars.
        If long rate > 75% of all directional signals at conf>0.60,
        raise the long threshold (+0.05) to filter extreme long bias.
        The short threshold is never lowered — counter-trend shorts in
        a bull-regime training window would generate regressions.

        Thresholds are stored as attributes and used in
        predict_with_confidence() and signal_for_latest_bar().
        """
        sample = X_context[-500:] if len(X_context) > 500 else X_context
        try:
            proba = self.predict_proba(sample)
            classes = list(self.meta_learner.classes_)
            if 1 not in classes or -1 not in classes:
                return
            long_idx  = classes.index(1)
            short_idx = classes.index(-1)
            long_rate  = float((proba[:, long_idx]  > 0.60).mean())
            short_rate = float((proba[:, short_idx] > 0.60).mean())
            total_rate = long_rate + short_rate

            if total_rate > 0 and long_rate / (total_rate + 1e-9) > 0.75:
                # Extreme long bias (>75%) — raise long bar only.
                # Do NOT lower the short threshold: counter-trend shorts
                # in a bull-regime training window cause regressions.
                self._long_threshold  = MIN_CONFIDENCE + 0.05
                self._short_threshold = MIN_CONFIDENCE
                logger.info(
                    f"[{self.instrument}] Signal bias correction: "
                    f"long_rate={long_rate:.1%} → "
                    f"long_thr={self._long_threshold:.2f} "
                    f"(short_thr unchanged at {self._short_threshold:.2f})"
                )
            else:
                self._long_threshold  = MIN_CONFIDENCE
                self._short_threshold = MIN_CONFIDENCE
        except Exception as e:
            logger.debug(f"_compute_signal_bias failed: {e}")
            self._long_threshold  = MIN_CONFIDENCE
            self._short_threshold = MIN_CONFIDENCE

    # ── Prediction ────────────────────────────────────────────

    def _pin_inference_threads(self):
        """
        Force n_jobs=1 on every base model that uses parallel workers.

        Background: XGBoost, LightGBM, and RandomForest are all trained
        with n_jobs=-1 (all cores).  That setting is pickled into the
        model object.  On macOS (Darwin) PyTorch initialises its own
        OpenMP/MKL thread pool during the first LSTM.predict_proba call.
        When the *next* ticker's RF / LGBM inference tries to spawn new
        loky/OpenMP worker processes via os.fork(), the child inherits
        PyTorch's locked pthread mutexes and deadlocks — the process
        hangs silently forever.

        Setting n_jobs=1 tells joblib to run inference in the calling
        thread (no fork), which is safe regardless of what PyTorch has
        done.  Inference speed is negligible (<1 s for 5 000 rows) so
        there is no practical cost to single-threading here.

        This method is idempotent — safe to call multiple times.
        """
        for name, m in self.base_models.items():
            if m is None:
                continue
            # RandomForest, ExtraTrees — sklearn n_jobs attribute
            if hasattr(m, "n_jobs"):
                m.n_jobs = 1
            # LightGBM — stored in _other_params as well as top-level
            if hasattr(m, "_other_params") and "n_jobs" in getattr(m, "_other_params", {}):
                m._other_params["n_jobs"] = 1
            # XGBoost — nthread / n_jobs
            if hasattr(m, "get_params"):
                try:
                    params = m.get_params()
                    if "n_jobs" in params:
                        m.set_params(n_jobs=1)
                    if "nthread" in params:
                        m.set_params(nthread=1)
                except Exception:
                    pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return ensemble probability matrix (n_samples, 3).
        Column order matches self.classes_ = [-1, 0, 1].
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call .train() first.")

        # Guard against fork-after-torch deadlock on macOS:
        # pin all parallel base models to n_jobs=1 before any inference.
        self._pin_inference_threads()

        if self.feature_scaler_ is not None:
            X = self.feature_scaler_.transform(X)

        if self.selected_features_ is not None:
            X = X[:, self.selected_features_]

        # XGBoost/LGBM return probs for [0,1,2] — order is preserved
        p_xgb  = self.base_models["xgb"].predict_proba(X)
        p_lgbm = self.base_models["lgbm"].predict_proba(X)
        p_rf   = self.base_models["rf"].predict_proba(X)
        p_mlp  = self.base_models["mlp"].predict_proba(X)

        # LSTM — graceful fallback to neutral if unavailable
        lstm_model = self.base_models.get("lstm")
        if lstm_model is not None:
            try:
                p_lstm = lstm_model.predict_proba(X)[:, :p_xgb.shape[1]]
            except Exception:
                p_lstm = np.full_like(p_xgb, 1.0 / p_xgb.shape[1])
        else:
            p_lstm = np.full_like(p_xgb, 1.0 / p_xgb.shape[1])

        meta_X = np.hstack([p_xgb, p_lgbm, p_rf, p_mlp, p_lstm])
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

        Applies per-class adaptive thresholds (set during train()) to
        correct directional bias: if the model over-produces long signals,
        it requires higher confidence for longs and accepts lower for shorts.
        """
        # predict_proba handles feature selection internally
        proba      = self.predict_proba(X)
        confidence = proba.max(axis=1)
        raw_signal = self.meta_learner.classes_[proba.argmax(axis=1)]

        # Retrieve adaptive thresholds (fall back to MIN_CONFIDENCE for old models)
        long_thr  = getattr(self, "_long_threshold",  MIN_CONFIDENCE)
        short_thr = getattr(self, "_short_threshold", MIN_CONFIDENCE)
        classes   = list(self.meta_learner.classes_)

        # Apply class-specific threshold: zero out signal if below its threshold
        if long_thr != MIN_CONFIDENCE or short_thr != MIN_CONFIDENCE:
            long_idx  = classes.index(1)  if 1  in classes else None
            short_idx = classes.index(-1) if -1 in classes else None
            signal = np.zeros(len(raw_signal), dtype=int)
            for i, (sig, conf) in enumerate(zip(raw_signal, confidence)):
                if sig == 1 and long_idx is not None and proba[i, long_idx] >= long_thr:
                    signal[i] = 1
                elif sig == -1 and short_idx is not None and proba[i, short_idx] >= short_thr:
                    signal[i] = -1
                # else: signal stays 0 (filtered out by bias correction)
        else:
            signal = raw_signal

        # Apply the same scaler+selection used inside predict_proba so the
        # individual model votes are computed on correctly preprocessed features.
        X_sc  = self.feature_scaler_.transform(X) if self.feature_scaler_ is not None else X
        X_sel = X_sc[:, self.selected_features_] if self.selected_features_ is not None else X_sc

        votes = {
            "xgb":  self._dec(self.base_models["xgb"].predict(X_sel)).tolist(),
            "lgbm": self._dec(self.base_models["lgbm"].predict(X_sel)).tolist(),
            "rf":   self.base_models["rf"].predict(X_sel).tolist(),
        }
        return signal, confidence, votes


    def signal_for_latest_bar(self, df: pd.DataFrame) -> dict:
        """
        Takes a feature-enriched (or raw) DataFrame, returns signal dict for
        the most recent bar.  Used by the live trading loop.

        If df already has the feature columns (built by build_features() in the
        caller with fg_norm/fg_contrarian etc.), those values are used as-is.
        If df is raw OHLCV, build_features() is called internally (no F&G).

        The correct feature column list (_feature_cols, set during train()) is
        always used so swing-trained models get SWING_FEATURE_COLS and intraday
        models get FEATURE_COLS without any manual flag at the call site.
        """
        # predict_proba applies feature_scaler_ (fit on full feature set) then
        # selects down to selected_features_.  Always pass the full pre-selection
        # feature set here; _feature_cols (selected subset) is used only to
        # verify the dataframe contains the required columns.
        # Use _full_feature_cols (pre-selection list saved during train) when available
        # so HTF-trained models correctly pass all feature columns at inference.
        full_cols = (getattr(self, "_full_feature_cols", None)
                     or (SWING_FEATURE_COLS if self._swing else FEATURE_COLS))
        active_cols = self._feature_cols if self._feature_cols else full_cols

        # If the caller already built features (all full_cols present), use
        # the dataframe directly.  Otherwise build features from raw OHLCV.
        if all(c in df.columns for c in full_cols):
            df_feat = df
        else:
            df_feat = build_features(df, add_labels=False, drop_na=True,
                                     swing=self._swing)
        if df_feat.empty:
            return {"signal": 0, "confidence": 0.0, "reason": "Insufficient data", "tradeable": False}

        latest = df_feat.iloc[[-1]]
        # Extract only the columns present; missing HTF cols default to 0
        avail = [c for c in full_cols if c in df_feat.columns]
        row   = latest.reindex(columns=full_cols, fill_value=0.0)
        X     = row.values

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
            "tradeable":  sig != 0,   # threshold already applied in predict_with_confidence
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