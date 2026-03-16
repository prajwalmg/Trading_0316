"""
================================================================
  TradingFirm OS — config/settings.py
  Master configuration for all modules.
  Edit this file to customise the entire system.
================================================================
"""

# ── ENVIRONMENT ──────────────────────────────────────────────
ENV              = "paper"          # "paper" | "live"
BASE_CURRENCY    = "USD"
INITIAL_CAPITAL  = 1_000.0         # Works from €10 upward

# ── ASSET UNIVERSE ───────────────────────────────────────────
FOREX_PAIRS = [
    "EURUSD=X",
    "GBPUSD=X",
    "EURGBP=X",
    "EURJPY=X",
    "USDCHF=X",
    "EURCHF=X",
]

EQUITY_TICKERS = [
    "NVDA",
    "AAPL",
    "GS",
    "TSLA",
    "JPM",
    "EWG",
    "SAP",
]

COMMODITY_TICKERS = [
    "GC=F",
    "SI=F",
    "NG=F",
    "HG=F",
]

CRYPTO_TICKERS = [
    "XRP-USD",
    "BNB-USD",
]

ASSET_CLASS_MAP = {}
for t in FOREX_PAIRS:
    ASSET_CLASS_MAP[t] = "forex"
for t in EQUITY_TICKERS:
    ASSET_CLASS_MAP[t] = "equity"
for t in COMMODITY_TICKERS:
    ASSET_CLASS_MAP[t] = "commodity"
for t in CRYPTO_TICKERS:
    ASSET_CLASS_MAP[t] = "crypto"

# Add after existing ASSET_CLASS_MAP loops:
CRYPTO_SHORT_NAMES = {
    "BTC": "crypto", "ETH": "crypto", "SOL": "crypto",
    "BNB": "crypto", "ADA": "crypto", "XRP": "crypto",
}
ASSET_CLASS_MAP.update(CRYPTO_SHORT_NAMES)


# Add crypto risk multipliers:
ASSET_RISK_PCT = {
    "forex":     0.005,
    "equity":    0.015,
    "commodity": 0.015,
    "crypto":    0.002,   # 0.2% — reduced from 0.5% for safety
}


ASSET_ATR_MULTIPLIER = {
    "forex":     2.5,
    "equity":    2.0,
    "commodity": 2.0,
    "crypto":    5.0,       #3.5,
}

DATA_PERIODS = {
    "forex":     "730d",
    "equity":    "max",
    "commodity": "max",
    "crypto":    "730d",
}


ALL_INSTRUMENTS = FOREX_PAIRS + EQUITY_TICKERS + CRYPTO_TICKERS + COMMODITY_TICKERS

# ── DATA ─────────────────────────────────────────────────────
GRANULARITY      = "1h"             # yfinance: 1m,2m,5m,15m,30m,60m,1d
LOOKBACK_PERIOD  = "730d"            # yfinance period for intraday
DAILY_PERIOD     = "2y"             # for daily-bar features
MIN_BARS         = 50              # minimum bars needed to compute features
DATA_CACHE_DIR   = "data/cache"

# ── FEATURES ─────────────────────────────────────────────────
FEATURE_HORIZON  = 6                # bars ahead to predict
SIGNAL_THRESHOLD = 0.0002           # min return to label up/down

# Indicator periods
EMA_FAST, EMA_SLOW   = 9,  21
MACD_FAST, MACD_SLOW, MACD_SIG = 12, 26, 9
RSI_PERIOD           = 14
ATR_PERIOD           = 14
BB_PERIOD, BB_STD    = 20, 2.0
STOCH_K, STOCH_D     = 14, 3

# ── REGIME DETECTION ─────────────────────────────────────────
REGIME_LOOKBACK      = 50           # bars for regime calculation
ADX_TREND_THRESHOLD  = 25           # ADX > 25 = trending
RVOL_HIGH_THRESHOLD  = 1.5          # realised vol ratio > 1.5 = high vol

# Regimes: "trending_up", "trending_down", "ranging", "high_volatility"

# ── ML MODELS ────────────────────────────────────────────────
CV_SPLITS        = 5
TRAIN_DAYS       = 45
TEST_DAYS        = 10
MODEL_DIR        = "models/saved"

XGB_PARAMS = {
    "n_estimators":      400,
    "max_depth":         4,
    "learning_rate":     0.04,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "use_label_encoder": False,
    "eval_metric":       "mlogloss",
    "random_state":      42,
    "n_jobs":            -1,
}

LGBM_PARAMS = {
    "n_estimators":  500,
    "max_depth":     5,
    "learning_rate": 0.03,
    "num_leaves":    31,
    "subsample":     0.8,
    "random_state":  42,
    "verbose":       -1,
    "n_jobs":        -1,
}

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth":    6,
    "random_state": 42,
    "n_jobs":       -1,
}

MLPC_PARAMS = {
    "hidden_layer_sizes": (128, 64, 32),
    "activation": "relu",
    "max_iter": 500,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "random_state": 42,
    "verbose": False,
}

# Minimum ensemble confidence to generate a signal
MIN_CONFIDENCE   =  0.60        #0.58  #0.48
MIN_BARS_BETWEEN_TRADES = 0

# ── RISK MANAGEMENT ──────────────────────────────────────────
# Scales automatically with account size (works from €10)
MAX_RISK_PCT          = 0.03        # 3% account risk per trade
MAX_DAILY_LOSS_PCT    = 0.03        # 3% daily loss → stop trading
MAX_WEEKLY_LOSS_PCT   = 0.06        # 6% weekly loss → stop trading
MAX_DRAWDOWN_PCT      = 0.10        # 10% drawdown → halt all trading
MAX_OPEN_POSITIONS    = 20           # across all instruments
MAX_CORRELATED_TRADES = 10           # max trades in same asset class at once
MAX_SINGLE_EXPOSURE   = 0.20        # max 20% of capital in one instrument
MAX_HOLD_BARS         = 24

SL_ATR_MULT      = 1.5
TP_ATR_MULT      = 2.5
TRAILING_STOP    = True
TRAIL_ATR_MULT   = 1.0

# Kelly criterion cap (never bet more than half-Kelly)
KELLY_FRACTION   = 0.5

# ── EXECUTION ────────────────────────────────────────────────
BROKER           = "alpaca"         # "alpaca" | "oanda" | "paper"
ALPACA_API_KEY   = "YOUR_KEY"
ALPACA_SECRET    = "YOUR_SECRET"
ALPACA_BASE_URL  = "https://paper-api.alpaca.markets"   # paper

OANDA_API_KEY    = "YOUR_KEY"
OANDA_ACCOUNT_ID = "YOUR_ACCOUNT"
OANDA_ENV        = "practice"

# ── DATA SOURCES ─────────────────────────────────────────────
# FRED API key — free, takes < 2 min to get:
#   https://fred.stlouisfed.org/docs/api/api_key.html
# Used for: VIX, yield curve (10Y-2Y), Fed Funds Rate, Dollar Index
FRED_API_KEY     = "YOUR_FRED_KEY"

# Data source routing — which source to try first per asset class.
# "auto"  = use the best available source (recommended)
# "yfinance" = force yfinance for everything (original behaviour)
DATA_SOURCE_MODE = "auto"

SLIPPAGE_BPS     = 2                # 2 basis points slippage model
COMMISSION_BPS   = 1                # 1 basis point commission

# ── PORTFOLIO ────────────────────────────────────────────────
# Strategy weights — how to allocate capital across strategies
STRATEGY_WEIGHTS = {
    "momentum":        0.35,
    "mean_reversion":  0.30,
    "breakout":        0.20,
    "regime_adaptive": 0.15,
}

REBALANCE_HOURS  = 24               # rebalance portfolio weights every N hours
CORRELATION_CAP  = 0.70             # max correlation between held positions

# ── LOGGING ──────────────────────────────────────────────────
LOG_LEVEL        = "INFO"
LOG_FILE         = "logs/system.log"
TRADE_LOG_FILE   = "logs/trades.log"
