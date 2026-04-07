# Trading Firm OS

A professional multi-asset ML trading system for Forex, Equities, Commodities and Crypto.
Runs two independent systems in parallel: a **swing system** (1h bars) and an **intraday system** (5m bars), both with full circuit-breaker protection.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Requirements](#2-requirements)
3. [Configuration](#3-configuration)
4. [Architecture](#4-architecture)
5. [Running the Systems](#5-running-the-systems)
6. [Circuit Breakers & Risk Controls](#6-circuit-breakers--risk-controls)
7. [Data Sources](#7-data-sources)
8. [ML Models & Features](#8-ml-models--features)
9. [Monitoring & Status](#9-monitoring--status)
10. [Model Training & Backtesting](#10-model-training--backtesting)

---

## 1. Quick Start

```bash
# Clone and set up
git clone <repo-url>
cd trading_firm
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Set API keys (see Section 3)
export TWELVE_DATA_API_KEY="your_key"
export TELEGRAM_BOT_TOKEN="your_token"
export TELEGRAM_CHAT_ID="your_chat_id"

# Train models (first run only)
python main.py --mode train
python main.py --mode intraday  # loads pre-trained intraday models from models/saved/

# Start both systems
bash scripts/run_paper.sh &      # swing system  — logs to logs/paper_trading_YYYYMMDD.log
bash scripts/run_intraday.sh &   # intraday system — logs to logs/intraday_trading_YYYYMMDD.log

# Check status
bash scripts/status.sh
```

---

## 2. Requirements

**Python 3.11 recommended** (LightGBM requires 3.11 on macOS arm64).

```bash
pip install -r requirements.txt
```

| Package | Version | Purpose |
|---|---|---|
| `pandas` | ≥ 2.0.0 | DataFrames, time-series |
| `numpy` | ≥ 1.24.0 | Numerical operations |
| `pyarrow` | ≥ 14.0.0 | Parquet cache |
| `yfinance` | ≥ 0.2.28 | Fallback data + equities/commodities |
| `scikit-learn` | ≥ 1.3.0 | Random Forest, MLP, preprocessing |
| `xgboost` | ≥ 2.0.0 | Gradient boosted trees (base model 1) |
| `lightgbm` | ≥ 4.1.0 | LightGBM (base model 2) |
| `imbalanced-learn` | ≥ 0.11.0 | SMOTE class balancing |
| `requests` | ≥ 2.31.0 | HTTP data fetchers (FMP, Dukascopy, FRED) |
| `alpaca-trade-api` | ≥ 3.0.0 | Alpaca broker API |
| `cvxpy` | ≥ 1.4.0 | Mean-variance portfolio optimisation |
| `python-dateutil` | ≥ 2.8.0 | Date parsing |
| `pytz` | ≥ 2023.3 | Timezone handling |
| `tqdm` | ≥ 4.66.0 | Progress bars |

**Optional — PyTorch LSTM (5th base model):**
```bash
# CPU (macOS / Linux without GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# CUDA (Linux with NVIDIA GPU)
pip install torch torchvision
```
If PyTorch is not installed, the LSTM base model is silently skipped and the ensemble runs on 4 models.

**Optional — QuestDB (high-speed trade logging):**
- Install QuestDB locally: [questdb.io/docs/get-started](https://questdb.io/docs/get-started/homebrew/)
- Default: `localhost:8812` (ILP ingress), `localhost:9000` (REST API)
- If not running, trade logging falls back to SQLite only.

---

## 3. Configuration

All settings live in [config/settings.py](config/settings.py). Key values:

```python
ENV             = "paper"       # "paper" | "live"
BASE_CURRENCY   = "EUR"
INITIAL_CAPITAL = 10_000.0
```

### API Keys

Set in `config/settings.py` or via environment variables:

| Service | Setting | Where to get |
|---|---|---|
| Twelve Data | `TWELVE_DATA_API_KEY` | twelvedata.com (free: 800 credits/day) |
| FMP | `FMP_API_KEY` | financialmodelingprep.com |
| FRED | `FRED_API_KEY` | fred.stlouisfed.org (free) |
| Finnhub | `FINNHUB_API_KEY` | finnhub.io (free tier) |
| Telegram | `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` | @BotFather on Telegram |

### Broker (IBKR)

```python
IBKR_HOST      = "127.0.0.1"
IBKR_PORT      = 7497   # 7497 = paper trading, 7496 = live
IBKR_CLIENT_ID = 10
```

Requires TWS or IB Gateway running locally. Paper trading works without IBKR — the system falls back to `PaperBroker` automatically.

### Risk Parameters

```python
MAX_RISK_PCT          = 0.01    # 1% of NAV risked per trade
MAX_DAILY_LOSS_PCT    = 0.05    # 5% daily loss → swing halt
MAX_DRAWDOWN_PCT      = 0.20    # 20% drawdown → full halt
MAX_OPEN_POSITIONS    = 3
SL_ATR_MULT           = 1.5     # Stop loss = entry ± 1.5 × ATR
TP_ATR_MULT           = 1.5     # Take profit = entry ± 1.5 × ATR
KELLY_FRACTION        = 0.5     # Quarter-Kelly sizing
```

---

## 4. Architecture

```
trading_firm/
├── config/
│   └── settings.py              # All configuration: API keys, instruments, risk params
│
├── data/
│   ├── pipeline.py              # Primary data pipeline (cache → IBKR → FMP → yfinance)
│   ├── unified_pipeline.py      # Multi-source orchestrator + QuestDB writer
│   ├── dukascopy.py             # Free tick data for Forex/Gold/Oil (5m bars)
│   ├── fmp.py                   # Financial Modeling Prep (1h forex bars)
│   ├── twelvedata.py            # Twelve Data API client
│   ├── macro_calendar.py        # Finnhub economic calendar — macro event gate
│   ├── alternative.py           # CNN Fear & Greed sentiment
│   ├── cot.py                   # CFTC Commitment of Traders
│   └── trade_db.py              # SQLite trade persistence
│
├── signals/
│   ├── features.py              # 52 swing features (EMA, RSI, MACD, ATR, regime, HTF)
│   ├── features_intraday.py     # 5m intraday features + 1h/4h HTF context
│   ├── ensemble.py              # Stacked ensemble: XGB + LGBM + RF + MLP meta-learner
│   ├── intraday_model.py        # Intraday ensemble training pipeline
│   ├── regime.py                # HMM regime detection (trending/ranging/volatile)
│   └── lstm_model.py            # Optional PyTorch LSTM (5th base model)
│
├── execution/
│   ├── broker.py                # PaperBroker + IBKRBroker, SL/TP, trade logging
│   └── risk_manager.py          # SessionRiskManager — intraday circuit breakers
│
├── portfolio/
│   └── manager.py               # Signal aggregation: momentum, mean-reversion,
│                                #   breakout, regime-adaptive + portfolio optimiser
│
├── risk/
│   └── engine.py                # RiskEngine: position sizing, ATR stops, Kelly,
│                                #   drawdown halt, correlation guard
│
├── backtest/
│   ├── engine.py                # Swing backtest (Sharpe, Sortino, Calmar, MaxDD)
│   └── engine_intraday.py       # Intraday backtest with realistic spread model
│
├── notifications/
│   └── telegram.py              # Trade open/close, daily reports, drawdown alerts
│
├── scripts/
│   ├── run_paper.sh             # Persistent swing trading (auto-restart)
│   ├── run_intraday.sh          # Persistent intraday trading (auto-restart)
│   ├── status.sh                # Quick health check
│   ├── health_check.py          # Detailed system health report
│   ├── train_swing_models.py    # Batch swing model retraining
│   └── train_intraday_models.py # Batch intraday model retraining
│
├── dashboard/
│   └── cli.py                   # Terminal dashboard (live NAV, positions, stats)
│
├── models/saved/                # Trained ensemble .pkl files per instrument
├── logs/                        # Trade logs, feedback outcomes, daily reports
├── data/cache/                  # Parquet bar cache per instrument
├── data/trades.db               # SQLite trade history
│
├── main.py                      # Swing entry point (1h bars, all asset classes)
└── intraday_forex.py            # Intraday entry point (5m bars, Forex majors)
```

---

## 5. Running the Systems

### Recommended: persistent shell scripts (auto-restart on crash)

```bash
# Swing system — Forex majors + equities + commodities
bash scripts/run_paper.sh &

# Intraday system — Forex majors at 5m resolution
bash scripts/run_intraday.sh &
```

Both scripts restart automatically after a crash (60s delay). They stop cleanly on `Ctrl-C` (SIGINT) or `kill` (SIGTERM).

### Manual: direct Python commands

```bash
# ── Swing system (1h bars) ────────────────────────────────────────────────
# Train models
python main.py --mode train

# Walk-forward backtest
python main.py --mode backtest

# Paper trading (all default instruments, poll every 3600s)
python main.py --mode paper --poll 3600

# Paper trading — custom instrument list
python main.py --mode paper \
  --instruments EURUSD=X GBPUSD=X USDJPY=X AUDUSD=X USDCAD=X \
  --capital 10000 --poll 3600

# ── Intraday system (5m bars) ─────────────────────────────────────────────
# Paper trading
python main.py --mode intraday --poll 300

# With explicit capital
OMP_NUM_THREADS=1 python main.py --mode intraday --capital 10000 --poll 300

# ── Background (nohup) ───────────────────────────────────────────────────
OMP_NUM_THREADS=1 nohup python main.py --mode paper \
  --instruments EURUSD=X GBPUSD=X USDJPY=X AUDUSD=X USDCAD=X GBPJPY=X EURJPY=X EURGBP=X \
  >> logs/paper_$(date +%Y%m%d_%H%M).log 2>&1 &

OMP_NUM_THREADS=1 nohup python main.py --mode intraday \
  >> logs/intraday_$(date +%Y%m%d_%H%M).log 2>&1 &
```

### Available `--mode` values

| Mode | System | Description |
|---|---|---|
| `paper` / `swing` | Swing | Live paper trading on 1h bars |
| `train` | Swing | Train ensemble models from historical data |
| `backtest` | Swing | Walk-forward backtest, prints Sharpe/MaxDD/PF |
| `walkforward` | Swing | Explicit train/test split validation |
| `intraday` | Intraday | Live paper trading on 5m bars with HTF context |
| `multi` | Multi | Multi-asset paper trading (all instruments) |

---

## 6. Circuit Breakers & Risk Controls

### Intraday — `SessionRiskManager` (`execution/risk_manager.py`)

Fires before every new entry. Resets automatically at midnight UTC.

| Gate | Threshold | Action |
|---|---|---|
| Daily loss limit | −3% of session NAV | Halt all new entries until midnight |
| Per-instrument cap | 3 trades/day | Suspend that instrument for the session |
| Consecutive losses | 2 in a row | Suspend that instrument for the session |
| Total daily cap | 10 trades across all pairs | Block all new entries |
| NAV floor | 50% of starting capital ($5,000) | Shut down intraday process |
| Macro event gate | High-impact day (NFP/CPI/FOMC/ECB/GDP) | Halve position size |

### Swing — `RiskEngine` + `SessionRiskManager` (`risk/engine.py`)

| Gate | Threshold | Action |
|---|---|---|
| Daily loss limit | −5% of session NAV | Halt all new entries until next cycle |
| Per-instrument cap | 2 trades/day | Suspend instrument for session |
| Consecutive losses | 2 in a row | Suspend instrument for session |
| Total daily cap | 8 trades | Block all new entries |
| Max drawdown | −20% from peak | Full system halt |
| Max open positions | 3 | Block new entries |
| Correlation guard | 0.70 correlation cap | Reject correlated trades |

### Position Sizing

```
risk_amount = NAV × 1%
units       = risk_amount / (entry - stop_loss)
max_units   = NAV × 30 / entry   ← 30× leverage safety rail
```

Stop loss and take profit are ATR-based:
```
SL = entry ± 1.5 × ATR(1h)
TP = entry ± 1.5 × ATR(1h)
```

---

## 7. Data Sources

### Hierarchy (live / paper trading)

```
IBKR Gateway (real-time)
    ↓ (if unavailable)
FMP — forex 1h bars (paid plan)
    ↓ (if 403 / unavailable)
yfinance — equities, commodities, crypto, forex fallback
    ↓
Parquet cache — data/cache/  (avoids redundant API calls)
```

### Historical / training data

| Source | Instrument types | Granularity | Notes |
|---|---|---|---|
| Twelve Data | Forex, equities, crypto | 5m, 1h, daily | Primary; 800 credits/day free |
| Dukascopy | Forex, Gold, Oil | Tick → 5m/1h | Free, no login required |
| yfinance | All | 1m–daily | Fallback and commodities |
| FRED | Macro (VIX, Fed rate, CPI) | Daily | Free API key required |
| FMP | Forex 1h | 1h | Paid plan; 403 on free tier |

### QuestDB (optional time-series database)

When QuestDB is running on `localhost:8812`, all bar data and closed trades are written via ILP (line protocol) for fast querying. Falls back to SQLite + Parquet if unavailable.

```bash
# Install QuestDB (macOS)
brew install questdb
brew services start questdb
# Web console: http://localhost:9000
```

---

## 8. ML Models & Features

### Stacked Ensemble

Each instrument gets its own ensemble stored in `models/saved/`:

```
Base models:
  1. XGBoost       (400 estimators, max_depth=4)
  2. LightGBM      (500 estimators, num_leaves=31)
  3. Random Forest (200 estimators, max_depth=6)
  4. MLP           (128→64→32, ReLU, early stopping)
  5. LSTM          (optional, requires PyTorch)

Meta-learner:
  Logistic Regression on base model probability outputs
```

Signal classes: `−1` (short), `0` (neutral/flat), `+1` (long).  
Minimum confidence threshold: `0.55–0.62` (per-instrument, tuned by walk-forward sweep).

### Swing Features (52 total)

| Group | Count | Examples |
|---|---|---|
| Trend | 6 | EMA 9/21 cross, EMA 50/200, golden/death cross |
| Momentum | 8 | RSI-14, MACD, ROC-6/20, Stochastic K/D, Williams %R |
| Volatility | 5 | ATR-14, Bollinger bandwidth, Keltner channel, historical vol |
| Candle | 6 | Body size, wicks, engulfing, hammer, doji, pin bar |
| Session/Time | 4 | Hour UTC, day of week, London session, NY session |
| Volume | 3 | OBV, volume trend, MFI |
| Multi-timeframe | 4–10 | 4h/daily regime features, HTF EMA alignment |
| Sentiment | 2 | Fear & Greed normalised, contrarian signal |

### Regime Detection

HMM (Hidden Markov Model) with 4 states: `trending_up`, `trending_down`, `ranging`, `volatile`.  
Strategy weights shift dynamically with regime:
- Trending → momentum overweighted
- Ranging → mean-reversion overweighted
- Volatile → all positions halved, tighter stops

---

## 9. Monitoring & Status

```bash
# Quick status: PIDs, last 5 trades, NAV, recent logs
bash scripts/status.sh

# Detailed health report: win rate, drawdown, QuestDB, errors
python scripts/health_check.py
```

### Log files

| File | Contents |
|---|---|
| `logs/paper_trading_YYYYMMDD.log` | Swing system — signals, fills, errors |
| `logs/intraday_trading_YYYYMMDD.log` | Intraday system — signals, fills, errors |
| `logs/system.log` | Combined system log |
| `logs/paper_trades.csv` | Swing trade history (CSV) |
| `logs/intraday_trades.csv` | Intraday trade history (CSV) |
| `data/trades.db` | SQLite — all trades, queryable |

### Stopping the systems

```bash
# Graceful stop (shell scripts will not restart)
kill -TERM $(pgrep -f "mode paper")
kill -TERM $(pgrep -f "mode intraday")

# Or by PID
kill <PID>
```

---

## 10. Model Training & Backtesting

### Train swing models (all default instruments)

```bash
python main.py --mode train
# Models saved to models/saved/<ticker>_ensemble.pkl
```

### Train intraday models

```bash
python scripts/train_intraday_models.py
# Models saved to models/saved/intraday_<ticker>_ensemble.pkl
```

### Walk-forward backtest

```bash
# Swing
python main.py --mode backtest

# Intraday — quick backtest with pre-built engine
python quick_backtest.py

# Confidence sweep — find optimal per-instrument thresholds
python run_sweep.py
```

### Retrain periodically

```bash
# Monthly retrain (runs train → validate → restart)
bash scripts/monthly_retrain.sh
```

### Backtest metrics reported

- Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Max Drawdown (% and absolute)
- Win Rate, Profit Factor, Expectancy
- Per-trade log with entry/exit/SL/TP/PnL/reason

---

## Instrument Universe

**Forex (46 pairs):** USD majors, EUR/GBP/AUD/NZD/CAD crosses, exotics  
**Equities (12):** NVDA, AAPL, MSFT, GOOGL, META, AMZN, TSLA, GS, JPM, SPY, EWG, SAP  
**Commodities (6):** GC=F (Gold), SI=F (Silver), CL=F (Crude Oil), NG=F (Nat Gas), HG=F (Copper), ZW=F (Wheat)  
**Crypto (8):** BTC, ETH, SOL, XRP, BNB, ADA, AVAX, MATIC

Instruments excluded from swing trading (no valid walk-forward threshold): `NZDUSD=X`, `USDCHF=X`
