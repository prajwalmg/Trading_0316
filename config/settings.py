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
INITIAL_CAPITAL  = 10_000.0         # Works from €10 upward

# ── ASSET UNIVERSE ───────────────────────────────────────────
FOREX_PAIRS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X",
    "AUDUSD=X", "USDCAD=X", "USDCHF=X",
    "NZDUSD=X", "EURGBP=X",
]

EQUITY_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN",
    "NVDA", "META", "TSLA", "SPY",
]

COMMODITY_TICKERS = [
    "GC=F",    # Gold
    "CL=F",    # Crude Oil WTI
    "SI=F",    # Silver
    "NG=F",    # Natural Gas
]

ALL_INSTRUMENTS = FOREX_PAIRS + EQUITY_TICKERS + COMMODITY_TICKERS

# ── DATA ─────────────────────────────────────────────────────
GRANULARITY      = "5m"             # yfinance: 1m,2m,5m,15m,30m,60m,1d
LOOKBACK_PERIOD  = "60d"            # yfinance period for intraday
DAILY_PERIOD     = "2y"             # for daily-bar features
MIN_BARS         = 200              # minimum bars needed to compute features
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

# Minimum ensemble confidence to generate a signal
MIN_CONFIDENCE   = 0.58

# ── RISK MANAGEMENT ──────────────────────────────────────────
# Scales automatically with account size (works from €10)
MAX_RISK_PCT          = 0.01        # 1% account risk per trade
MAX_DAILY_LOSS_PCT    = 0.03        # 3% daily loss → stop trading
MAX_WEEKLY_LOSS_PCT   = 0.06        # 6% weekly loss → stop trading
MAX_DRAWDOWN_PCT      = 0.10        # 10% drawdown → halt all trading
MAX_OPEN_POSITIONS    = 5           # across all instruments
MAX_CORRELATED_TRADES = 2           # max trades in same asset class at once
MAX_SINGLE_EXPOSURE   = 0.20        # max 20% of capital in one instrument

SL_ATR_MULT      = 1.5
TP_ATR_MULT      = 2.5
TRAILING_STOP    = True
TRAIL_ATR_MULT   = 1.0

# Kelly criterion cap (never bet more than half-Kelly)
KELLY_FRACTION   = 0.25

# ── EXECUTION ────────────────────────────────────────────────
BROKER           = "alpaca"         # "alpaca" | "oanda" | "paper"
ALPACA_API_KEY   = "YOUR_KEY"
ALPACA_SECRET    = "YOUR_SECRET"
ALPACA_BASE_URL  = "https://paper-api.alpaca.markets"   # paper

OANDA_API_KEY    = "YOUR_KEY"
OANDA_ACCOUNT_ID = "YOUR_ACCOUNT"
OANDA_ENV        = "practice"

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
