"""Confidence sweep for EURUSD=X — Task 2."""
import sys, os, warnings, logging
# Force single-threaded to avoid joblib worker hangs on macOS background processes
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import config.settings as _cfg
_cfg.CV_SPLITS = 3
# Override n_jobs to avoid multiprocessing in background context
_cfg.RF_PARAMS   = dict(_cfg.RF_PARAMS,   n_jobs=1)
_cfg.XGB_PARAMS  = dict(_cfg.XGB_PARAMS,  n_jobs=1)
_cfg.LGBM_PARAMS = dict(_cfg.LGBM_PARAMS, n_jobs=1)
# Reduce LSTM epochs for sweep speed — patch the class default directly
import signals.lstm_model as _lstm_mod
from signals.lstm_model import LSTMClassifier as _LC
# Replace the default for 'epochs' in __init__ (index 6 in defaults tuple)
_defs = list(_LC.__init__.__defaults__)
_epochs_idx = list(_LC.__init__.__code__.co_varnames).index('epochs') - 1  # -1 for self
_defs[_epochs_idx] = 5
_LC.__init__.__defaults__ = tuple(_defs)

from data.pipeline import fetch_ohlcv
from signals.features import get_X_y, FEATURE_COLS
from signals.ensemble import StackedEnsemble

print(f"FEATURE_COLS: {len(FEATURE_COLS)} features")
print("Fetching EURUSD=X data...")
df_raw = fetch_ohlcv("EURUSD=X", interval="1h", days=730, use_cache=True, training_mode=True)
n_total = len(df_raw)
split = int(n_total * 0.80)
df_tr = df_raw.iloc[:split]
df_te = df_raw.iloc[split:]
X_tr, y_tr, _ = get_X_y(df_tr, swing=False)
X_te, y_te, _ = get_X_y(df_te, swing=False)
print(f"EUR: {n_total} bars | train: {len(X_tr)} | test: {len(X_te)}")
unique, counts = np.unique(y_tr, return_counts=True)
print(f"Class dist: {dict(zip(unique.tolist(), counts.tolist()))}")

print("Training ensemble...")
model = StackedEnsemble(instrument="EUR_sweep")
m = model.train(X_tr, y_tr)
print(f"OOF accuracy: {m.get('oof_accuracy', 'N/A')}")

proba = model.predict_proba(X_te)
classes = list(model.meta_learner.classes_)

print()
print(f"  {'Thr':>5}  {'Trades':>7}  {'WinRate':>9}  {'PF':>7}  {'Cover%':>8}")
print(f"  {'─'*5}  {'─'*7}  {'─'*9}  {'─'*7}  {'─'*8}")

best_thr = 0.60
for thr in [0.50, 0.52, 0.55, 0.58, 0.60, 0.63, 0.65, 0.70]:
    conf = proba.max(axis=1)
    sig  = np.array(classes)[proba.argmax(axis=1)]
    mask = (conf >= thr) & (sig != 0)
    if mask.sum() < 10:
        print(f"  {thr:>5.2f}  {mask.sum():>7}  {'N/A':>9}  {'N/A':>7}  {mask.mean():>8.1%}")
        continue
    sf = sig[mask]; yf = y_te[mask]
    wins = (sf == yf).sum(); losses = (sf != yf).sum()
    wr = wins / len(sf); pf = wins / max(losses, 1)
    if wr >= 0.52 and mask.sum() >= 200:
        best_thr = thr
    flag = " <-- best" if (wr >= 0.52 and mask.sum() >= 200 and thr == best_thr) else ""
    print(f"  {thr:>5.2f}  {mask.sum():>7}  {wr:>9.1%}  {pf:>7.3f}  {mask.mean():>8.1%}{flag}")

print(f"\nRecommended MIN_CONFIDENCE: {best_thr}")
