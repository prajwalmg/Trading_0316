# TradingFirm OS — Professional Algorithmic Trading System

A full institutional-grade trading system for Forex, Equities, and
Commodities. Scales from €10 to €10M+ without changing core logic.

---

## Architecture

```
trading_firm/
├── config/settings.py          # All parameters — edit this first
├── main.py                     # Entry point (4 modes)
│
├── data/pipeline.py            # Multi-source data fetcher + cache
│
├── signals/
│   ├── features.py             # 40+ features (trend/momentum/vol/candle/volume/session)
│   ├── regime.py               # Market regime detection (4 regimes)
│   └── ensemble.py             # Stacked ensemble (XGB + LGBM + RF → LogReg)
│
├── risk/engine.py              # 7-layer risk management + Kelly sizing
│
├── execution/broker.py         # Paper / Alpaca / OANDA broker layer
│
├── portfolio/manager.py        # 4-strategy orchestrator + conflict resolution
│
├── backtest/engine.py          # Vectorised backtest + full metrics
│
└── dashboard/cli.py            # Real-time terminal dashboard
```

---

## Quick Start

### 1. Install Python 3.11 via pyenv
```bash
brew install pyenv
pyenv install 3.11.9
pyenv local 3.11.9
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure
Edit `config/settings.py`:
- Set `INITIAL_CAPITAL` (works from €10)
- Add broker credentials if using live/paper API
- Adjust `FOREX_PAIRS`, `EQUITY_TICKERS`, `COMMODITY_TICKERS`

### 3. Train models
```bash
python main.py --mode train
```

### 4. Backtest
```bash
python main.py --mode backtest
```

### 5. Paper trade (no real money)
```bash
python main.py --mode paper
```

---

## Strategies

| Strategy | Logic | Best Regime |
|---|---|---|
| Momentum | Follows ML signal when ADX strong + MACD confirms | Trending |
| Mean Reversion | Fades RSI + Bollinger extremes | Ranging |
| Breakout | Enters on BB squeeze breakouts with volume | Any |
| Regime Adaptive | Scales ML signal by regime confidence | All |

---

## Risk Rules

1. **1% max risk per trade** (scales with account size automatically)
2. **Quarter-Kelly sizing** (mathematically optimal, conservative)
3. **3% daily loss → halt** trading until next day
4. **6% weekly loss → halt** trading until next week
5. **10% drawdown → full halt** + human review required
6. **Max 5 open positions** at any time
7. **Max 2 correlated positions** per asset class
8. **Volatility spike filter** (ATR > 2.5× mean → skip)
9. **ATR-based trailing stops** (adjusts to volatility automatically)

---

## Supported Brokers

| Broker | Asset Classes | Setup |
|---|---|---|
| Paper (built-in) | All | No API key needed |
| Alpaca | Stocks + Crypto | Free API key at alpaca.markets |
| OANDA | Forex | Free practice account at oanda.com |

---

## Important Warnings

- Always paper trade for at least 1–3 months before going live
- Past performance does not guarantee future results
- Forex, equities, and commodity trading involves significant risk
- This system is for educational and research purposes
- Retrain models monthly as market regimes shift
