# Trading Firm OS

A professional multi-asset ML trading system for Forex, Equities, Commodities and Crypto.

## Architecture

trading_firm/
├── config/settings.py          # all configuration
├── data/pipeline.py            # yfinance data fetching
├── signals/features.py         # 52 features (42 base + 10 swing)
├── signals/ensemble.py         # XGB+LGBM+RF+MLP stacked ensemble
├── signals/regime.py           # HTF regime detection
├── risk/engine.py              # position sizing, drawdown halting
├── execution/broker.py         # paper broker, trade logging
├── portfolio/manager.py        # signal aggregation
├── backtest/engine.py          # walk-forward backtesting
├── dashboard/cli.py            # terminal dashboard
├── utils/logger.py             # logging
├── utils/scheduler.py          # weekly retraining scheduler
├── main.py                     # swing system (1h bars)
└── intraday_forex.py           # intraday system (5m bars)
## Setup

```bash
git clone <repo-url>
cd trading_firm
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Intraday system (BTC/ETH/SOL — 5min bars)
python intraday_forex.py --mode train
python intraday_forex.py --mode backtest
python intraday_forex.py --mode paper --poll 300

# Swing system (GC=F/XRP-USD/BNB-USD — 1h bars)
python main.py --mode train
python main.py --mode backtest
python main.py --mode paper --poll 3600

# Scheduler (weekly retraining)
python utils/scheduler.py intraday
python utils/scheduler.py swing
