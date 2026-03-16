"""
================================================================
  execution/broker.py
  Unified execution layer — supports:
    "paper"   : in-memory paper trading (no API needed)
    "alpaca"  : Alpaca Markets API (stocks + crypto)
    "oanda"   : OANDA v20 REST API (forex)

  All brokers implement the same interface so strategies
  never need to know which broker is active.
================================================================
"""

import logging
import importlib
import time
from datetime import datetime, timezone
from typing import Optional
import json

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.settings import (
    BROKER, ENV,
    ALPACA_API_KEY, ALPACA_SECRET, ALPACA_BASE_URL,
    OANDA_API_KEY, OANDA_ACCOUNT_ID, OANDA_ENV,
    SLIPPAGE_BPS, COMMISSION_BPS, INITIAL_CAPITAL,
)

logger = logging.getLogger("trading_firm.execution")


def calibration_report(self) -> dict:
    """
    Check if model confidence matches actual win rate.
    Run weekly to tune confidence thresholds.
    
    If 60-70% confidence bucket shows only 45% win rate
    → raise threshold to 65%.
    If 60-70% shows 70% win rate → lower threshold to 55%.
    """
    import glob

    buckets = {
        "55-60%": {"wins": 0, "total": 0},
        "60-65%": {"wins": 0, "total": 0},
        "65-70%": {"wins": 0, "total": 0},
        "70-75%": {"wins": 0, "total": 0},
        "75%+":   {"wins": 0, "total": 0},
    }

    for path in glob.glob("logs/feedback/*_outcomes.jsonl"):
        with open(path) as f:
            for line in f:
                try:
                    r    = json.loads(line)
                    conf = r.get("confidence", 0)
                    won  = r.get("won", False)

                    if   conf < 0.60: bucket = "55-60%"
                    elif conf < 0.65: bucket = "60-65%"
                    elif conf < 0.70: bucket = "65-70%"
                    elif conf < 0.75: bucket = "70-75%"
                    else:             bucket = "75%+"

                    buckets[bucket]["total"] += 1
                    if won:
                        buckets[bucket]["wins"] += 1
                except Exception:
                    continue

    report = {}
    for bucket, data in buckets.items():
        if data["total"] >= 5:
            wr = data["wins"] / data["total"]
            report[bucket] = {
                "win_rate": round(wr, 3),
                "trades":   data["total"],
                "verdict":  "✅ good" if wr >= 0.55 else "⚠️ weak" if wr >= 0.45 else "❌ bad",
            }
    return report


# ── Base interface ────────────────────────────────────────────

class BrokerBase:
    """Abstract broker interface — all brokers implement these methods."""

    def get_account(self) -> dict:
        raise NotImplementedError

    def get_latest_price(self, instrument: str) -> dict:
        raise NotImplementedError

    def market_order(self, instrument: str, units: int,
                     stop_loss: float = None, take_profit: float = None) -> dict:
        raise NotImplementedError

    def close_position(self, instrument: str) -> dict:
        raise NotImplementedError

    def get_open_positions(self) -> list:
        raise NotImplementedError


# ── Paper broker ──────────────────────────────────────────────

class PaperBroker(BrokerBase):
    """
    In-memory paper trading broker.
    Simulates fills with configurable slippage and commission.
    Works with any instrument — no API key required.
    """

    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        self.capital    = initial_capital
        self.positions  = {}    # {instrument: {units, entry, direction, sl, tp}}
        self.trades     = []
        self._prices    = {}    # {instrument: last_price}

        logger.info(
            f"PaperBroker initialised | capital={initial_capital:.2f}"
        )

    def _apply_costs(self, price: float, direction: int) -> float:
        """Apply slippage and commission to fill price."""
        total_bps = SLIPPAGE_BPS + COMMISSION_BPS
        cost      = price * (total_bps / 10_000)
        return price + (direction * cost)

    def partial_close(
        self,
        instrument:     str,
        close_fraction: float = 0.5,
        reason:         str   = "partial_tp",
    ) -> dict:
        if instrument not in self.positions:
            return {"status": "no_position"}

        pos         = self.positions[instrument]
        total_units = abs(pos["units"])

        # Use stored direction if available, else derive from units sign
        direction   = pos.get("direction", 1 if pos["units"] > 0 else -1)

        # Fractional-safe close units — no int() truncation
        close_units = total_units * close_fraction
        close_units = max(close_units, total_units * 0.01)  # minimum 1% of position

        current_price = self._prices.get(instrument, pos["entry"])
        pnl           = (current_price - pos["entry"]) * direction * close_units

        # Reduce position — keep as float for crypto
        remaining = (total_units - close_units) * direction
        pos["units"] = remaining

        trade = {
            "instrument": instrument,
            "direction":  "LONG" if direction == 1 else "SHORT",
            "entry":      pos["entry"],
            "exit":       current_price,
            "units":      round(close_units, 6),
            "pnl":        round(pnl, 2),
            "reason":     reason,
            "type":       "partial",
        }
        self.trades.append(trade)

        if abs(pos["units"]) < total_units * 0.01:
            del self.positions[instrument]

        logger.info(
            f"Partial close: {instrument} | "
            f"{close_units:.6f} units @ {current_price:.5f} | "
            f"P&L: {pnl:+.2f} | Remaining: {abs(pos.get('units', 0)):.6f} units"
        )
        return {
            "status":          "partial_closed",
            "pnl":             round(pnl, 2),
            "remaining_units": abs(pos.get("units", 0)),
        }


    def update_stop_loss(self, instrument: str, new_sl: float) -> bool:
        """Update the stop loss for an open position (for trailing stops)."""
        if instrument not in self.positions:
            return False
        self.positions[instrument]["sl"] = new_sl
        return True


    def get_last_price(self, instrument: str) -> float:
        """Return the last known price for an instrument."""
        return self._prices.get(instrument, 0.0)


    def get_bars_open(self, instrument: str, current_bar: int) -> int:
        """Return how many bars a position has been open."""
        if instrument not in self.positions:
            return 0
        open_bar = self.positions[instrument].get("open_bar", current_bar)
        return current_bar - open_bar

    def update_price(self, instrument: str, price: float):
        """Feed latest price — called by live trading loop."""
        self._prices[instrument] = price
        self._check_sl_tp(instrument, price)

    def _check_sl_tp(self, instrument: str, price: float):
        """Check if any SL/TP has been hit."""
        if instrument not in self.positions:
            return

        pos = self.positions[instrument]
        dir_ = pos["direction"]
        sl   = pos.get("sl")
        tp   = pos.get("tp")

        hit = None
        if sl and dir_ ==  1 and price <= sl:  hit = ("sl", sl)
        if sl and dir_ == -1 and price >= sl:  hit = ("sl", sl)
        if tp and dir_ ==  1 and price >= tp:  hit = ("tp", tp)
        if tp and dir_ == -1 and price <= tp:  hit = ("tp", tp)

        if hit:
            self._close_position(instrument, hit[1], reason=hit[0])

    def _log_trade_outcome(
        self,
        instrument: str,
        pos:        dict,
        exit_price: float,
        pnl:        float,
        reason:     str,
    ):
        """
        Save every trade outcome with its features for feedback loop.
        Used for confidence calibration and future retraining.
        """
        record = {
            "instrument":  instrument,
            "direction":   pos.get("direction", 0),
            "entry":       round(pos.get("entry", 0), 6),
            "exit":        round(exit_price, 6),
            "pnl":         round(pnl, 4),
            "won":         pnl > 0,
            "reason":      reason,
            "confidence":  round(pos.get("confidence", 0), 4),
            "regime":      pos.get("regime", "unknown"),
            "atr":         round(pos.get("atr", 0), 6),
            "htf_trend":   pos.get("htf_trend", 0),
            "timestamp":   datetime.now(timezone.utc).isoformat(),
        }
        os.makedirs("logs/feedback", exist_ok=True)
        log_path = f"logs/feedback/{instrument}_outcomes.jsonl"
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")


    def _close_position(self, instrument: str, exit_price: float, reason: str = "manual"):
        if instrument not in self.positions:
            return

        pos    = self.positions.pop(instrument)
        entry  = pos["entry"]
        units  = pos["units"]
        dir_   = pos["direction"]
        pnl    = dir_ * (exit_price - entry) * units

        self._log_trade_outcome(instrument, pos, exit_price, pnl, reason)

        self.capital += pnl
        self.trades.append({
            "instrument": instrument,
            "direction":  "Long" if dir_ == 1 else "Short",
            "entry":      entry,
            "exit":       exit_price,
            "units":      units,
            "pnl":        round(pnl, 4),
            "reason":     reason,
            "time":       datetime.now(timezone.utc),
        })
        logger.info(
            f"Paper close: {instrument} @ {exit_price:.5f} | "
            f"P&L={pnl:+.4f} | reason={reason}"
        )

    def get_account(self) -> dict:
        unrealised = sum(
            pos["direction"] * (self._prices.get(inst, pos["entry"]) - pos["entry"]) * pos["units"]
            for inst, pos in self.positions.items()
        )
        return {
            "balance":        round(self.capital, 2),
            "nav":            round(self.capital + unrealised, 2),
            "unrealised_pnl": round(unrealised, 2),
            "open_trades":    len(self.positions),
        }

    def get_latest_price(self, instrument: str) -> dict:
        price = self._prices.get(instrument, 0.0)
        return {"mid": price, "bid": price, "ask": price, "instrument": instrument}

    def market_order(
        self,
        instrument:  str,
        units:       int,
        stop_loss:   float = None,
        take_profit: float = None,
    ) -> dict:
        direction   = 1 if units > 0 else -1
        raw_price   = self._prices.get(instrument, 0.0)
        fill_price  = self._apply_costs(raw_price, direction)

        if fill_price <= 0:
            logger.warning(f"No price available for {instrument} — order rejected")
            return {"status": "rejected", "reason": "no_price"}

        # Close any opposing position first
        if instrument in self.positions:
            existing_dir = self.positions[instrument]["direction"]
            if existing_dir != direction:
                self._close_position(instrument, fill_price, reason="reversal")

        self.positions[instrument] = {
            "instrument": instrument,
            "units":      abs(units),
            "direction":  direction,
            "entry":      fill_price,
            "sl":         stop_loss,
            "tp":         take_profit,
            "time":       datetime.now(timezone.utc),
        }

        logger.info(
            f"Paper order: {instrument} | "
            f"{'BUY' if direction==1 else 'SELL'} {abs(units)} | "
            f"fill={fill_price:.5f} | sl={stop_loss} | tp={take_profit}"
        )
        return {"status": "filled", "fill_price": fill_price, "units": units}

    def close_position(self, instrument: str) -> dict:
        if instrument not in self.positions:
            return {"status": "no_position"}
        price = self._prices.get(instrument, self.positions[instrument]["entry"])
        self._close_position(instrument, price, reason="manual")
        return {"status": "closed"}

    def get_open_positions(self) -> list:
        return [
            {**pos, "instrument": inst,
             "unrealised_pnl": pos["direction"] * (
                 self._prices.get(inst, pos["entry"]) - pos["entry"]
             ) * pos["units"]}
            for inst, pos in self.positions.items()
        ]

    @property
    def trade_log(self) -> pd.DataFrame:
        return pd.DataFrame(self.trades) if self.trades else pd.DataFrame()


# ── Alpaca broker ─────────────────────────────────────────────

class AlpacaBroker(BrokerBase):
    """
    Alpaca Markets REST API.
    Supports US equities + crypto.
    Set ALPACA_BASE_URL to paper or live in settings.py.
    """

    def __init__(self):
        try:
            tradeapi = importlib.import_module("alpaca_trade_api")
        except ImportError as exc:
            raise ImportError(
                "alpaca-trade-api not installed. "
                "Run: pip install alpaca-trade-api"
            ) from exc

        self.api = tradeapi.REST(
            ALPACA_API_KEY, ALPACA_SECRET, ALPACA_BASE_URL,
            api_version="v2"
        )
        logger.info(f"AlpacaBroker connected | url={ALPACA_BASE_URL}")

    def get_account(self) -> dict:
        acc = self.api.get_account()
        return {
            "balance":        float(acc.cash),
            "nav":            float(acc.portfolio_value),
            "unrealised_pnl": float(acc.unrealized_pl),
            "open_trades":    len(self.api.list_positions()),
        }

    def get_latest_price(self, instrument: str) -> dict:
        trade = self.api.get_latest_trade(instrument)
        price = float(trade.price)
        return {"mid": price, "bid": price, "ask": price, "instrument": instrument}

    def market_order(
        self,
        instrument:  str,
        units:       int,
        stop_loss:   float = None,
        take_profit: float = None,
    ) -> dict:
        side  = "buy" if units > 0 else "sell"
        order = self.api.submit_order(
            symbol=instrument,
            qty=abs(units),
            side=side,
            type="market",
            time_in_force="day",
            order_class="bracket" if stop_loss and take_profit else "simple",
            stop_loss={"stop_price": str(stop_loss)} if stop_loss else None,
            take_profit={"limit_price": str(take_profit)} if take_profit else None,
        )
        return {"status": "submitted", "order_id": order.id}

    def close_position(self, instrument: str) -> dict:
        self.api.close_position(instrument)
        return {"status": "closed"}

    def get_open_positions(self) -> list:
        return [
            {
                "instrument":     p.symbol,
                "units":          int(p.qty),
                "direction":      1 if float(p.qty) > 0 else -1,
                "entry":          float(p.avg_entry_price),
                "unrealised_pnl": float(p.unrealized_pl),
            }
            for p in self.api.list_positions()
        ]


# ── OANDA broker ──────────────────────────────────────────────

class OandaBroker(BrokerBase):
    """OANDA v20 REST API — forex pairs only."""

    BASE_URLS = {
        "practice": "https://api-fxpractice.oanda.com",
        "live":     "https://api-fxtrade.oanda.com",
    }

    def __init__(self):
        import requests
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization":          f"Bearer {OANDA_API_KEY}",
            "Content-Type":           "application/json",
            "Accept-Datetime-Format": "RFC3339",
        })
        self._base = self.BASE_URLS[OANDA_ENV]
        self._acct = OANDA_ACCOUNT_ID
        logger.info(f"OandaBroker connected | env={OANDA_ENV}")

    def _get(self, endpoint: str, params: dict = None) -> dict:
        r = self._session.get(f"{self._base}{endpoint}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    def _post(self, endpoint: str, payload: dict) -> dict:
        r = self._session.post(f"{self._base}{endpoint}", json=payload, timeout=10)
        r.raise_for_status()
        return r.json()

    def get_account(self) -> dict:
        data = self._get(f"/v3/accounts/{self._acct}/summary")["account"]
        return {
            "balance":        float(data["balance"]),
            "nav":            float(data["NAV"]),
            "unrealised_pnl": float(data["unrealizedPL"]),
            "open_trades":    int(data["openTradeCount"]),
        }

    def get_latest_price(self, instrument: str) -> dict:
        data = self._get(
            f"/v3/accounts/{self._acct}/pricing",
            params={"instruments": instrument}
        )["prices"][0]
        bid = float(data["bids"][0]["price"])
        ask = float(data["asks"][0]["price"])
        return {"bid": bid, "ask": ask, "mid": (bid+ask)/2, "instrument": instrument}

    def market_order(self, instrument: str, units: int,
                     stop_loss: float = None, take_profit: float = None) -> dict:
        order = {"order": {
            "type": "MARKET", "instrument": instrument,
            "units": str(units), "timeInForce": "FOK",
        }}
        if stop_loss:
            order["order"]["stopLossOnFill"] = {"price": f"{stop_loss:.5f}", "timeInForce": "GTC"}
        if take_profit:
            order["order"]["takeProfitOnFill"] = {"price": f"{take_profit:.5f}", "timeInForce": "GTC"}
        return self._post(f"/v3/accounts/{self._acct}/orders", order)

    def close_position(self, instrument: str) -> dict:
        r = self._session.put(
            f"{self._base}/v3/accounts/{self._acct}/positions/{instrument}/close",
            json={"longUnits": "ALL", "shortUnits": "ALL"}, timeout=10
        )
        return r.json()

    def get_open_positions(self) -> list:
        data = self._get(f"/v3/accounts/{self._acct}/openTrades")
        return data.get("trades", [])


# ── Factory ───────────────────────────────────────────────────

def get_broker(broker_type: str = BROKER) -> BrokerBase:
    """
    Return the correct broker instance from config.
    broker_type: "paper" | "alpaca" | "oanda"
    """
    if broker_type == "paper":
        return PaperBroker()
    elif broker_type == "alpaca":
        return AlpacaBroker()
    elif broker_type == "oanda":
        return OandaBroker()
    else:
        raise ValueError(f"Unknown broker: {broker_type}. Choose: paper, alpaca, oanda")
