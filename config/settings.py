"""
================================================================
  config/settings.py  — FULL UNIVERSE EDITION
================================================================
"""

import os

ENV             = "paper"
BASE_CURRENCY   = "EUR"
INITIAL_CAPITAL = 10_000.0

# ── Twelve Data ───────────────────────────────────────────────
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "")
TWELVE_DATA_PLAN    = os.getenv("TWELVE_DATA_PLAN", "free")
# free plan  : 8 req/min,     800 credits/day, ~5 000 bars/request
# grow plan  : 60 req/min, 50 000 credits/day, ~50 000 bars/request
# pro plan   : 120 req/min, unlimited credits

#FRED_API_KEY = "aab3e4e47326cc5548e1614b135643ab"
FMP_API_KEY        = "iPsIBee0pikODSBhCGdprLW6xRGjpk9B"
MASSIVE_API_KEY    = "QL2Glfc5RNbCfeapS9jyokG4tH8T45hj"
FINNHUB_API_KEY    = "d75bmqpr01qk56kc5s10d75bmqpr01qk56kc5s1g"
FRED_API_KEY       = "aab3e4e47326cc5548e1614b135643ab"
EDGAR_IDENTITY     = "Prajwal M mgprajwal13@gmail.com"
QUESTDB_HOST       = "localhost"
QUESTDB_PORT       = 8812

# ── IBKR ─────────────────────────────────────────────────────
IBKR_HOST      = "127.0.0.1"
IBKR_PORT      = 7497       # 7497=paper | 7496=live
IBKR_CLIENT_ID = 10

# Telegram (for trade alerts and system notifications)
TELEGRAM_BOT_TOKEN = "8735513793:AAHHPDvtFYqRWf53MgMew0lGLROZp4jOp0g"
TELEGRAM_CHAT_ID   = "601840484"

# ══════════════════════════════════════════════════════════════
#  FOREX  (46 pairs — all free on IBKR IDEALPRO)
# ══════════════════════════════════════════════════════════════

FOREX_PAIRS = [
    # USD Majors (USDCHF and NZDUSD excluded — no valid confidence threshold found in sweep)
    "EURUSD=X", "GBPUSD=X", "USDJPY=X",
    "AUDUSD=X", "USDCAD=X",

    # USD Minors / Exotics
    "USDNOK=X", "USDSEK=X", "USDDKK=X", "USDSGD=X",
    "USDHKD=X", "USDMXN=X", "USDTRY=X", "USDZAR=X",
    "USDPLN=X", "USDCZK=X", "USDHUF=X", "USDCNH=X",

    # EUR Crosses
    "EURGBP=X", "EURJPY=X", "EURCHF=X", "EURAUD=X",
    "EURCAD=X", "EURNZD=X", "EURNOK=X", "EURSEK=X",
    "EURPLN=X", "EURHUF=X", "EURTRY=X",

    # GBP Crosses
    "GBPJPY=X", "GBPCHF=X", "GBPAUD=X", "GBPCAD=X", "GBPNZD=X",

    # AUD / NZD Crosses
    "AUDJPY=X", "AUDCHF=X", "AUDCAD=X", "AUDNZD=X",
    "NZDJPY=X", "NZDCHF=X", "NZDCAD=X",

    # CAD / CHF Crosses
    "CADJPY=X", "CADCHF=X", "CHFJPY=X",
]

# ══════════════════════════════════════════════════════════════
#  CRYPTO  (8 pairs — free on IBKR PAXOS, no subscription)
#  Requires: Client Portal → Settings → Trading Permissions → Crypto
# ══════════════════════════════════════════════════════════════

CRYPTO_TICKERS = [
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "XRP-USD",
    "BNB-USD",
    "ADA-USD",
    "AVAX-USD",
    "MATIC-USD",
]

# ══════════════════════════════════════════════════════════════
#  EQUITIES  (yfinance is primary; IBKR needs a subscription)
# ══════════════════════════════════════════════════════════════

EQUITY_TICKERS = [
    # US Tech
    "NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA",
    # US Finance / Other
    "GS", "JPM", "SPY",
    # European
    "EWG", "SAP",
]

# ══════════════════════════════════════════════════════════════
#  COMMODITIES  (yfinance futures; IBKR CFD needs subscription)
# ══════════════════════════════════════════════════════════════

COMMODITY_TICKERS = [
    "GC=F",   # Gold
    "SI=F",   # Silver
    "NG=F",   # Natural Gas
    "HG=F",   # Copper
    "CL=F",   # Crude Oil
    "ZW=F",   # Wheat
]

# ── Asset class map ───────────────────────────────────────────
ASSET_CLASS_MAP = {}
for t in FOREX_PAIRS:       ASSET_CLASS_MAP[t] = "forex"
for t in EQUITY_TICKERS:    ASSET_CLASS_MAP[t] = "equity"
for t in COMMODITY_TICKERS: ASSET_CLASS_MAP[t] = "commodity"
for t in CRYPTO_TICKERS:    ASSET_CLASS_MAP[t] = "crypto"

CRYPTO_SHORT = {"BTC":"crypto","ETH":"crypto","SOL":"crypto",
                "BNB":"crypto","ADA":"crypto","XRP":"crypto",
                "AVAX":"crypto","MATIC":"crypto"}
ASSET_CLASS_MAP.update(CRYPTO_SHORT)

ALL_INSTRUMENTS = FOREX_PAIRS + EQUITY_TICKERS + COMMODITY_TICKERS + CRYPTO_TICKERS

# ── Risk per asset class ──────────────────────────────────────
ASSET_RISK_PCT = {
    "forex":     0.005,   # €250 max risk/trade
    "equity":    0.015,   # €750
    "commodity": 0.010,   # €500
    "crypto":    0.002,   # €100
}

ASSET_ATR_MULTIPLIER = {
    "forex": 2.5, "equity": 2.0, "commodity": 2.0, "crypto": 5.0,
}

DATA_PERIODS = {
    "forex":"1Y","equity":"1Y","commodity":"1Y","crypto":"1Y",
}

# ── Data ──────────────────────────────────────────────────────
GRANULARITY     = "1h"
LOOKBACK_PERIOD = "1Y"
DAILY_PERIOD    = "2Y"
MIN_BARS        = 50
DATA_CACHE_DIR  = "data/cache"

# ── Features ──────────────────────────────────────────────────
FEATURE_HORIZON  = 6
SIGNAL_THRESHOLD = 0.0002
EMA_FAST, EMA_SLOW              = 9, 21
MACD_FAST, MACD_SLOW, MACD_SIG  = 12, 26, 9
RSI_PERIOD  = 14
ATR_PERIOD  = 14
BB_PERIOD, BB_STD = 20, 2.0
STOCH_K, STOCH_D  = 14, 3

# ── Regime ────────────────────────────────────────────────────
REGIME_LOOKBACK     = 50
ADX_TREND_THRESHOLD = 25
RVOL_HIGH_THRESHOLD = 1.5

# ── ML Models ─────────────────────────────────────────────────
CV_SPLITS  = 5
TRAIN_DAYS = 45
TEST_DAYS  = 10
MODEL_DIR  = "models/saved"

XGB_PARAMS = {"n_estimators":400,"max_depth":4,"learning_rate":0.04,
    "subsample":0.8,"colsample_bytree":0.8,"use_label_encoder":False,
    "eval_metric":"mlogloss","random_state":42,"n_jobs":-1}
LGBM_PARAMS = {"n_estimators":500,"max_depth":5,"learning_rate":0.03,
    "num_leaves":31,"subsample":0.8,"random_state":42,"verbose":-1,"n_jobs":-1}
RF_PARAMS = {"n_estimators":200,"max_depth":6,"random_state":42,"n_jobs":-1}
MLPC_PARAMS = {"hidden_layer_sizes":(128,64,32),"activation":"relu",
    "max_iter":500,"early_stopping":True,"validation_fraction":0.1,
    "random_state":42,"verbose":False}

MIN_CONFIDENCE          = 0.58
MIN_BARS_BETWEEN_TRADES = 0

# Per-instrument confidence overrides (fallback: MIN_CONFIDENCE)
# Thresholds selected by confidence sweep (70/30 split, criteria: max Sharpe
# with trades >= 30 and PF >= 1.0).  USDCHF and NZDUSD excluded — no valid
# threshold found in sweep (all Sharpe < 0 or trades < 30).
INSTRUMENT_MIN_CONFIDENCE = {
    "EURUSD=X": 0.57,   # Sharpe 2.730 | 172 trades | PF 1.641
    "GBPUSD=X": 0.55,   # WF sweep: best thresh=0.55, WF Sharpe 0.978
    "USDJPY=X": 0.60,   # Sharpe 1.060 |  77 trades | PF 1.316
    "AUDUSD=X": 0.55,   # Sharpe 1.653 | 222 trades | PF 1.261
    "USDCAD=X": 0.57,   # Sharpe 3.355 |  50 trades | PF 4.186
    "GBPJPY=X": 0.57,   # lowered from 0.60 to allow 57.9%+ conf signals through
    "EURJPY=X": 0.62,   # WF sweep: best thresh=0.62, WF Sharpe 1.452
    "EURGBP=X": 0.60,   # Sharpe 1.341 |  33 trades | PF 1.864
}

# Instruments permanently excluded from swing trading (no valid WF threshold)
SWING_EXCLUDED_INSTRUMENTS: set = {"NZDUSD=X", "USDCHF=X"}

# ── Risk ──────────────────────────────────────────────────────
MAX_RISK_PCT          = 0.01
MAX_DAILY_LOSS_PCT    = 0.05
MAX_WEEKLY_LOSS_PCT   = 0.15
MAX_DRAWDOWN_PCT      = 0.20
MAX_OPEN_POSITIONS    = 3
MAX_CORRELATED_TRADES = 2
MAX_SINGLE_EXPOSURE   = 0.15
MAX_HOLD_BARS         = 24
SL_ATR_MULT    = 1.5
TP_ATR_MULT    = 1.5
TRAILING_STOP  = True
TRAIL_ATR_MULT = 1.0
KELLY_FRACTION = 0.5

# ── Execution ─────────────────────────────────────────────────
BROKER         = "paper"   # "paper" | "ibkr"
SLIPPAGE_BPS   = 2
COMMISSION_BPS = 1

# ── Portfolio ─────────────────────────────────────────────────
STRATEGY_WEIGHTS = {"momentum":0.35,"mean_reversion":0.30,
                    "breakout":0.20,"regime_adaptive":0.15}
REBALANCE_HOURS = 24
CORRELATION_CAP = 0.70

# ── Logging ───────────────────────────────────────────────────
LOG_LEVEL      = "INFO"
LOG_FILE       = "logs/system.log"
TRADE_LOG_FILE = "logs/trades.log"