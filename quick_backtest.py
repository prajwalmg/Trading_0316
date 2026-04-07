"""
Quick backtest script for improvement testing.
70/30 split, EURUSD=X only, no walk-forward.
"""
import os, sys
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["LOKY_MAX_CPU_COUNT"]   = "1"

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from data.pipeline     import DataPipeline
from signals.features  import get_X_y
from signals.ensemble  import StackedEnsemble
from backtest.engine   import run_backtest_single
from main              import get_label_params, get_min_confidence

TICKER = "EURUSD=X"

print(f"Fetching data for {TICKER}...")
pipeline = DataPipeline()
pipeline.refresh_all([TICKER], training_mode=True)
df = pipeline.get(TICKER)
print(f"Total bars: {len(df)}")

split    = int(len(df) * 0.70)
df_train = df.iloc[:split]
df_test  = df.iloc[split:]

print(f"Train: {len(df_train)} bars | Test: {len(df_test)} bars")

params = get_label_params(TICKER)
X_tr, y_tr, _ = get_X_y(df_train, **params, swing=True)
print(f"Training samples: {len(X_tr)} | classes: {dict(zip(*np.unique(y_tr, return_counts=True)))}")

print("Training model...")
model = StackedEnsemble(instrument=TICKER)
model.train(X_tr, y_tr)
print(f"_feature_cols: {len(model._feature_cols)} | selected: {len(model.selected_features_)}")

print("Running backtest...")
result = run_backtest_single(
    df_test, model, 10000,
    swing=True, ticker=TICKER,
    min_conf=get_min_confidence(TICKER),
    use_circuit_breaker=False,
)
m = result["metrics"]
print("\n" + "="*50)
print(f"  {TICKER} Backtest Results")
print("="*50)
for k, v in m.items():
    print(f"  {k:<25} {v}")
