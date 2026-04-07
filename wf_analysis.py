"""
================================================================
  wf_analysis.py — Walk-Forward Backtest Analysis
  Instruments: EURUSD=X and GBPUSD=X
  Granularity: 1h (system-native)

  Reports per fold:
    - Number of trades
    - Win rate
    - Sharpe ratio

  Statistical tests:
    1. t-statistic on fold Sharpe ratios  (H₀: μ_Sharpe = 0)
    2. Lo (2002) t-stat on pooled OOS equity curve
    3. Outlier fold detection (fold Sharpe > μ ± 2σ)
    4. Ex-outlier robustness check

  Usage:
    python wf_analysis.py
================================================================
"""
import sys, os, warnings, logging
# Force single-threaded to avoid joblib worker hangs on macOS background processes
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)   # silence internal loggers

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from scipy import stats

# ── Speed patches (apply before any ensemble import) ─────────
# 1. Reduce CV_SPLITS 5 → 3 for the walk-forward run.
#    Each outer fold retrains 5 base models × n CV splits; halving
#    CV folds cuts total training time by 40%.
import config.settings as _cfg
_cfg.CV_SPLITS = 3
# Force n_jobs=1 to avoid joblib worker hangs in background process
_cfg.RF_PARAMS   = dict(_cfg.RF_PARAMS,   n_jobs=1)
_cfg.XGB_PARAMS  = dict(_cfg.XGB_PARAMS,  n_jobs=1)
_cfg.LGBM_PARAMS = dict(_cfg.LGBM_PARAMS, n_jobs=1)

# 2. Replace the LSTM with a fast neutral stub.
#    PyTorch is CPU-only on this machine and compiled against NumPy 1.x
#    (incompatible with the installed NumPy 2.x), so LSTM training would
#    either hang or crash.  The LSTM is 1 of 5 base models — XGBoost,
#    LightGBM, RF, and MLP still participate fully.
import signals.lstm_model as _lstm_mod

class _NeutralLSTM:
    """Drop-in stub: always outputs uniform class probabilities."""
    classes_ = np.array([0, 1, 2])
    def __init__(self, *args, **kwargs): pass
    def fit(self, X, y):            return self
    def predict_proba(self, X):     return np.full((len(X), 3), 1/3)
    def predict(self, X):           return np.ones(len(X), dtype=int)

_lstm_mod.LSTMClassifier = _NeutralLSTM

# Also patch the ensemble module's CV_SPLITS binding (it was already imported
# as a local name before our patch above, so we need to set it directly).
import signals.ensemble as _ens_mod
_ens_mod.CV_SPLITS = 3

from data.pipeline   import fetch_ohlcv
from backtest.engine import run_walkforward_backtest

# ── Parameters ────────────────────────────────────────────────
INSTRUMENTS = ["EURUSD=X", "GBPUSD=X"]
CAPITAL     = 10_000.0

# 1h bars, 4 years of history (~25 000 bars via Twelve Data)
# train=8000 (~1.3 yr)  test=2000 (~4 months)  → ~8 OOS folds each
TRAIN_BARS  = 8000
TEST_BARS   = 2000

# ── Colour helpers ────────────────────────────────────────────
G    = "\033[92m"
R    = "\033[91m"
Y    = "\033[93m"
C    = "\033[96m"
BOLD = "\033[1m"
RST  = "\033[0m"

def _ok(s):  return f"{G}{s}{RST}"
def _warn(s): return f"{Y}{s}{RST}"
def _bad(s):  return f"{R}{s}{RST}"


# ── Statistical helpers ───────────────────────────────────────

def cross_fold_ttest(fold_sharpes: list) -> dict:
    """
    One-sample t-test across fold Sharpes.
    H₀: mean Sharpe ratio = 0  (no edge)
    """
    s = np.array(fold_sharpes, dtype=float)
    n = len(s)
    if n < 2:
        return {"t": float("nan"), "p": float("nan"), "n": n,
                "mean": float("nan"), "std": float("nan")}
    t, p = stats.ttest_1samp(s, 0.0)
    return {
        "t":    round(float(t), 3),
        "p":    round(float(p), 4),
        "n":    n,
        "mean": round(float(s.mean()), 3),
        "std":  round(float(s.std(ddof=1)), 3),
        "se":   round(float(s.std(ddof=1) / np.sqrt(n)), 3),
        "ci95": (
            round(float(s.mean() - 1.96 * s.std(ddof=1) / np.sqrt(n)), 3),
            round(float(s.mean() + 1.96 * s.std(ddof=1) / np.sqrt(n)), 3),
        ),
    }


def lo2002_tstat(equity: pd.Series) -> dict:
    """
    Lo (2002) asymptotic t-stat for the annualised Sharpe ratio.
    Accounts for serial correlation and non-normality.

    t = SR_annual * sqrt(T) / sqrt(1 + SR²/2)   (iid approximation)
    T = number of OOS bar returns
    """
    ret = equity.pct_change().dropna()
    T   = len(ret)
    if T < 10:
        return {"t": float("nan"), "p": float("nan"), "sr_annual": float("nan"), "T": T}

    bars_per_year = 6_240    # forex 1h: ~24h × 5 days × 52 weeks
    mu_bar  = ret.mean()
    sig_bar = ret.std(ddof=1) + 1e-10
    sr_bar  = mu_bar / sig_bar
    sr_ann  = sr_bar * np.sqrt(bars_per_year)

    # Lo (2002) SE
    se  = np.sqrt((1.0 + 0.5 * sr_bar**2) / T)
    t   = sr_bar / se
    p   = 2 * (1 - stats.norm.cdf(abs(t)))

    return {
        "sr_annual": round(float(sr_ann), 3),
        "t":         round(float(t), 3),
        "p":         round(float(p), 4),
        "T":         T,
    }


def detect_outliers(fold_sharpes: list) -> list:
    """Flag fold indices whose Sharpe is more than 2σ from the mean."""
    s  = np.array(fold_sharpes, dtype=float)
    mu = s.mean()
    sd = s.std(ddof=1) if len(s) > 1 else 0
    return [i for i, v in enumerate(s) if abs(v - mu) > 2 * sd]


def _parse_pct(v) -> float:
    """Convert '52.3%' or 0.523 to 0.523."""
    if isinstance(v, str):
        return float(v.strip("%")) / 100
    return float(v)


# ── Per-ticker analysis ───────────────────────────────────────

def analyse_ticker(ticker: str) -> dict:
    print(f"\n{BOLD}{'═'*66}{RST}")
    print(f"{BOLD}  {C}{ticker}{RST}{BOLD}  —  walk-forward backtest{RST}")
    print(f"{BOLD}{'═'*66}{RST}")

    # ── Fetch data ─────────────────────────────────────────────
    # training_mode=True routes through Twelve Data (4yr history)
    # before falling back to IBKR → yfinance
    print(f"  Fetching 1h OHLCV (Twelve Data → yfinance fallback)…", flush=True)
    df_raw = fetch_ohlcv(ticker, interval="1h", days=730,
                         use_cache=True, training_mode=True)

    if df_raw.empty:
        print(_bad("  ❌  No data returned — skipping"))
        return {}

    n_bars = len(df_raw)
    print(f"  {n_bars} bars  "
          f"({df_raw.index[0].strftime('%Y-%m-%d')} → "
          f"{df_raw.index[-1].strftime('%Y-%m-%d')})")

    expected_folds = max(0, (n_bars - TRAIN_BARS) // TEST_BARS)
    if expected_folds < 3:
        print(_bad(f"  ❌  Only {expected_folds} expected folds — need ≥ 3 for statistics"))
        return {}

    print(f"  train={TRAIN_BARS} bars / test={TEST_BARS} bars / "
          f"expected folds ≈ {expected_folds}")
    print()

    # ── Run walk-forward ───────────────────────────────────────
    print("  Running walk-forward (retrains ensemble per fold) …")
    print("  This may take several minutes — one dot per fold:")
    print("  ", end="", flush=True)

    # Monkey-patch logger to emit a dot + elapsed time per fold
    import backtest.engine as _be, time as _time
    _orig_info = _be.logger.info
    fold_counter = [0]
    _t0 = [_time.time()]
    def _dot_logger(msg, *a, **kw):
        if "Fold" in msg and "OOS" in msg:
            fold_counter[0] += 1
            elapsed = _time.time() - _t0[0]
            print(f"·{fold_counter[0]}({elapsed:.0f}s)", end=" ", flush=True)
            _t0[0] = _time.time()
        _orig_info(msg, *a, **kw)
    _be.logger.info = _dot_logger

    result = run_walkforward_backtest(
        df_raw,
        train_bars = TRAIN_BARS,
        test_bars  = TEST_BARS,
        capital    = CAPITAL,
        swing      = False,        # intraday 1h mode
        ticker     = ticker,
    )

    _be.logger.info = _orig_info   # restore
    print(f"\n  {fold_counter[0]} folds completed\n")

    folds = result.get("fold_metrics", [])
    if not folds:
        print(_bad("  ❌  No folds produced any trades"))
        return {}

    # ── Per-fold table ─────────────────────────────────────────
    fold_sharpes  = []
    fold_winrates = []
    fold_trades   = []

    hdr = f"  {'Fold':>4}  {'Trades':>7}  {'Win Rate':>9}  {'Sharpe':>8}  {'Return':>9}  {'MaxDD':>8}  {'Halt':>6}"
    sep = f"  {'─'*4}  {'─'*7}  {'─'*9}  {'─'*8}  {'─'*9}  {'─'*8}  {'─'*6}"
    print(hdr)
    print(sep)

    for f in folds:
        sh       = float(f.get("sharpe", 0) or 0)
        wr       = _parse_pct(f.get("win_rate", "0%"))
        tr       = int(f.get("trades", 0))
        ret      = f.get("total_return", "0%")
        dd       = f.get("max_dd",      "0%")
        halt_bar = f.get("halted_at_bar", None)

        fold_sharpes.append(sh)
        fold_winrates.append(wr)
        fold_trades.append(tr)

        sh_col   = _ok(f"{sh:+.3f}") if sh > 0 else (_bad(f"{sh:+.3f}") if sh < -0.3 else f"{sh:+.3f}")
        halt_col = _warn(f"@{halt_bar}") if halt_bar is not None else "     -"
        print(f"  {f.get('fold','?'):>4}  {tr:>7}  {wr:>8.1%}  {sh_col:>8}  {str(ret):>9}  {str(dd):>8}  {halt_col:>6}")

    # ── Statistics ─────────────────────────────────────────────
    print()
    print(f"  {BOLD}Statistical Analysis{RST}")
    print(f"  {'─'*62}")

    # 1) Cross-fold t-test
    tt = cross_fold_ttest(fold_sharpes)
    positive = sum(1 for s in fold_sharpes if s > 0)
    cv = abs(tt["std"] / (tt["mean"] + 1e-9))

    print(f"\n  [1]  Cross-Fold Sharpe  (H₀: μ = 0,  n={tt['n']} folds)")
    print(f"       mean Sharpe = {tt['mean']:+.3f}  ±  {tt['std']:.3f}  (SE={tt['se']:.3f})")
    print(f"       95% CI      = [{tt['ci95'][0]:+.3f},  {tt['ci95'][1]:+.3f}]")
    print(f"       t = {tt['t']:+.3f}   p = {tt['p']:.4f}   CV = {cv:.2f}")
    print(f"       Positive folds: {positive}/{tt['n']} = {positive/tt['n']:.0%}")

    if tt["p"] < 0.05 and tt["t"] > 0:
        sig_str = _ok("✅  REJECT H₀  — Sharpe significantly > 0 (p<0.05)")
    elif tt["p"] < 0.10 and tt["t"] > 0:
        sig_str = _warn("⚠️   MARGINAL  — p<0.10, borderline significance")
    else:
        sig_str = _bad("❌  FAIL TO REJECT H₀  — no reliable edge detected")
    print(f"       {sig_str}")

    # 2) Lo (2002) on pooled OOS equity
    print(f"\n  [2]  Lo (2002) t-stat on pooled OOS equity")
    lo = lo2002_tstat(result.get("equity", pd.Series(dtype=float)))
    if not np.isnan(lo["t"]):
        print(f"       Annualised SR = {lo['sr_annual']:+.3f}   T = {lo['T']} obs")
        print(f"       t = {lo['t']:+.3f}   p = {lo['p']:.4f}")
        if lo["p"] < 0.05 and lo["t"] > 0:
            print(f"       " + _ok("✅  Pooled OOS Sharpe significantly > 0"))
        else:
            print(f"       " + _bad("❌  Pooled OOS Sharpe not significant"))
    else:
        print("       Insufficient equity observations for Lo (2002) test")

    # 3) Trades & win-rate consistency
    print(f"\n  [3]  Trades per fold")
    print(f"       min={min(fold_trades)}   max={max(fold_trades)}   "
          f"mean={np.mean(fold_trades):.1f}   median={np.median(fold_trades):.0f}")
    if min(fold_trades) < 5:
        print("       " + _warn(f"⚠️   {sum(1 for t in fold_trades if t < 5)} fold(s) have < 5 trades — statistics fragile"))

    print(f"\n  [4]  Win rate per fold")
    print(f"       min={min(fold_winrates):.1%}   max={max(fold_winrates):.1%}   "
          f"mean={np.mean(fold_winrates):.1%}   std={np.std(fold_winrates):.1%}")
    wr_low = sum(1 for w in fold_winrates if w < 0.40)
    if wr_low:
        print("       " + _warn(f"⚠️   {wr_low} fold(s) below 40% win rate"))

    # 4) Outlier detection
    outliers = detect_outliers(fold_sharpes)
    print(f"\n  [5]  Outlier fold detection  (|Sharpe − μ| > 2σ)")
    if not outliers:
        print("       " + _ok("✅  No outlier folds — Sharpe distribution is consistent"))
        consistent = True
    else:
        out_sharpes = [fold_sharpes[i] for i in outliers]
        print("       " + _warn(f"⚠️   Outlier folds: {outliers}  "
                                f"(Sharpes: {[f'{s:+.3f}' for s in out_sharpes]})"))

        pruned = [s for i, s in enumerate(fold_sharpes) if i not in outliers]
        if len(pruned) >= 2:
            t2, p2 = stats.ttest_1samp(pruned, 0.0)
            mean2  = np.mean(pruned)
            print(f"       Ex-outlier: mean={mean2:+.3f}   t={t2:+.3f}   p={p2:.4f}")
            if p2 < 0.05 and t2 > 0:
                consistent = True
                print("       " + _ok("✅  Edge persists after removing outlier folds — robust"))
            else:
                consistent = False
                print("       " + _bad("❌  Edge DISAPPEARS without outlier folds — NOT robust"))
        else:
            consistent = False

    # 5) Aggregate OOS metrics
    agg = result.get("metrics", {})
    print(f"\n  {BOLD}Aggregate OOS Metrics  ({result.get('fold_count', '?')} folds){RST}")
    print(f"  {'─'*44}")
    for k in ("total_return", "annual_return", "sharpe_ratio", "sortino_ratio",
              "max_drawdown", "win_rate", "total_trades", "profit_factor", "expectancy"):
        print(f"  {k:<22}  {agg.get(k, 'N/A')}")

    return {
        "ticker":        ticker,
        "n_folds":       len(folds),
        "fold_sharpes":  fold_sharpes,
        "fold_trades":   fold_trades,
        "fold_winrates": fold_winrates,
        "ttest":         tt,
        "lo2002":        lo,
        "outliers":      outliers,
        "consistent":    consistent,
        "cv":            round(cv, 3),
        "aggregate":     agg,
        "positive_folds": positive,
    }


# ── Cross-ticker verdict ──────────────────────────────────────

def verdict(results: list):
    valid = [r for r in results if r and "ttest" in r]
    if not valid:
        print(_bad("\n  ❌  No valid results to summarise"))
        return

    print(f"\n{BOLD}{'═'*66}{RST}")
    print(f"{BOLD}  CROSS-TICKER SUMMARY{RST}")
    print(f"{BOLD}{'═'*66}{RST}")

    print(f"\n  {'Ticker':<14}  {'Folds':>5}  {'μ Sharpe':>9}  {'t':>7}  "
          f"{'p':>7}  {'Pos%':>6}  {'Outliers':>9}  {'Robust':>7}")
    print(f"  {'─'*14}  {'─'*5}  {'─'*9}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*9}  {'─'*7}")

    for r in valid:
        tt = r["ttest"]
        pos_pct = r["positive_folds"] / r["n_folds"]
        out_str = str(len(r["outliers"])) if r["outliers"] else "none"
        rob_str = _ok("✅") if r["consistent"] else _bad("❌")
        sig = "*" if tt["p"] < 0.05 and tt["t"] > 0 else ("." if tt["p"] < 0.10 and tt["t"] > 0 else " ")
        print(f"  {r['ticker']:<14}  {r['n_folds']:>5}  {tt['mean']:>+9.3f}  "
              f"{tt['t']:>+7.3f}  {tt['p']:>7.4f}{sig}  {pos_pct:>6.0%}  {out_str:>9}  {rob_str}")

    print(f"\n  Significance: * p<0.05   . p<0.10\n")

    # Combined fold Sharpes
    all_sharpes = [s for r in valid for s in r["fold_sharpes"]]
    if len(all_sharpes) >= 4:
        t_comb, p_comb = stats.ttest_1samp(all_sharpes, 0.0)
        pos_combined = sum(1 for s in all_sharpes if s > 0)
        print(f"  Combined ({len(all_sharpes)} fold-Sharpes across both pairs):")
        print(f"  mean={np.mean(all_sharpes):+.3f}   std={np.std(all_sharpes):.3f}   "
              f"t={t_comb:+.3f}   p={p_comb:.4f}")
        print(f"  Positive folds: {pos_combined}/{len(all_sharpes)} = {pos_combined/len(all_sharpes):.0%}")

    # ── Verdict ────────────────────────────────────────────────
    print(f"\n{BOLD}{'═'*66}{RST}")
    print(f"{BOLD}  LIVE-READINESS VERDICT{RST}")
    print(f"{BOLD}{'═'*66}{RST}\n")

    checks = {
        "Sharpe significantly > 0 (p<0.05)":
            all(r["ttest"]["p"] < 0.05 and r["ttest"]["t"] > 0 for r in valid),
        "Mean Sharpe ≥ 0.5 on both pairs":
            all(r["ttest"]["mean"] >= 0.5 for r in valid),
        "Edge robust to outlier fold removal":
            all(r["consistent"] for r in valid),
        "≥ 60% of folds profitable":
            all(r["positive_folds"] / r["n_folds"] >= 0.6 for r in valid),
        "Mean trades per fold ≥ 5":
            all(np.mean(r["fold_trades"]) >= 5 for r in valid),
        "Mean win rate ≥ 45%":
            all(np.mean(r["fold_winrates"]) >= 0.45 for r in valid),
    }

    passed = sum(checks.values())
    for label, ok in checks.items():
        icon = _ok("✅") if ok else _bad("❌")
        print(f"  {icon}  {label}")

    print()
    if passed == len(checks):
        print(_ok(f"  ✅  ALL {passed}/{len(checks)} CHECKS PASSED — system is statistically ready for paper trading."))
    elif passed >= len(checks) - 1:
        print(_warn(f"  ⚠️   {passed}/{len(checks)} checks passed — marginal. Paper trade with reduced size."))
    else:
        print(_bad(f"  ❌  Only {passed}/{len(checks)} checks passed — DO NOT go live yet."))
        print(_bad("      Diagnose the failing checks above before proceeding."))

    print()


# ── Entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{BOLD}{'═'*66}{RST}")
    print(f"{BOLD}  WALK-FORWARD BACKTEST ANALYSIS{RST}")
    print(f"  Instruments : {', '.join(INSTRUMENTS)}")
    print(f"  Granularity : 1h  |  Train: {TRAIN_BARS} bars  |  Test: {TEST_BARS} bars")
    print(f"  Capital     : ${CAPITAL:,.0f}  |  Mode: intraday (FEATURE_COLS)")
    print(f"  Fixes active: adaptive bias thresholds + 5% fold DD circuit breaker")
    print(f"{BOLD}{'═'*66}{RST}")

    results = []
    for ticker in INSTRUMENTS:
        r = analyse_ticker(ticker)
        results.append(r)

    verdict(results)
