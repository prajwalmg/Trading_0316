"""
================================================================
  execution/broker.py  — IBKR EDITION
  Supports:
    "paper" : in-memory paper trading (no TWS needed)
    "ibkr"  : Interactive Brokers TWS/Gateway API (Mac native)

  Prerequisites:
    1. pip install ibapi
    2. Download IB Gateway from interactivebrokers.com
    3. Open IB Gateway → login with paper account credentials
    4. Enable API: Configure → Settings → API → Enable Socket Clients
       Port: 7497 (paper) or 7496 (live)

  Your system uses "paper" mode for training/backtesting.
  Switch to "ibkr" only when ready to place real orders.
================================================================
"""

import logging
import threading
import time
import json
import os
import queue
from datetime  import datetime, timezone
from typing    import Optional

import numpy  as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.settings import (
    BROKER, SLIPPAGE_BPS, COMMISSION_BPS, INITIAL_CAPITAL,
    IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID,
)

logger = logging.getLogger("trading_firm.execution")

try:
    from data.trade_db import log_trade as _db_log_trade
except ImportError:
    _db_log_trade = None


# ── Calibration report (unchanged) ───────────────────────────

def calibration_report() -> dict:
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
    def get_account(self)                                              -> dict: raise NotImplementedError
    def get_latest_price(self, instrument: str)                        -> dict: raise NotImplementedError
    def market_order(self, instrument: str, units: int,
                     stop_loss: float = None, take_profit: float = None) -> dict: raise NotImplementedError
    def close_position(self, instrument: str)                          -> dict: raise NotImplementedError
    def get_open_positions(self)                                       -> list: raise NotImplementedError


# ── Paper broker (unchanged from original) ───────────────────

class PaperBroker(BrokerBase):
    """In-memory paper trading — identical to original."""

    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        self.capital         = initial_capital
        self.initial_capital = initial_capital
        self.positions       = {}
        self.trades          = []
        self._prices         = {}
        logger.info(f"PaperBroker initialised | capital={initial_capital:.2f}")

    def _apply_costs(self, price: float, direction: int) -> float:
        cost = price * ((SLIPPAGE_BPS + COMMISSION_BPS) / 10_000)
        return price + (direction * cost)

    def partial_close(self, instrument: str, close_fraction: float = 0.5,
                      reason: str = "partial_tp") -> dict:
        if instrument not in self.positions:
            return {"status": "no_position"}
        pos         = self.positions[instrument]
        total_units = abs(pos["units"])
        direction   = pos.get("direction", 1 if pos["units"] > 0 else -1)
        close_units = max(total_units * close_fraction, total_units * 0.01)
        current_price = self._prices.get(instrument, pos["entry"])
        pnl = (current_price - pos["entry"]) * direction * close_units
        pos["units"] = (total_units - close_units) * direction
        self.trades.append({"instrument": instrument, "pnl": round(pnl, 2),
                             "reason": reason, "type": "partial"})
        if abs(pos["units"]) < total_units * 0.01:
            del self.positions[instrument]
        return {"status": "partial_closed", "pnl": round(pnl, 2),
                "remaining_units": abs(pos.get("units", 0))}

    def update_stop_loss(self, instrument: str, new_sl: float) -> bool:
        if instrument not in self.positions:
            return False
        self.positions[instrument]["sl"] = new_sl
        return True

    def get_last_price(self, instrument: str) -> float:
        return self._prices.get(instrument, 0.0)

    def get_bars_open(self, instrument: str, current_bar: int) -> int:
        if instrument not in self.positions:
            return 0
        return current_bar - self.positions[instrument].get("open_bar", current_bar)

    def update_price(self, instrument: str, price: float):
        self._prices[instrument] = price
        self._check_sl_tp(instrument, price)

    def _check_sl_tp(self, instrument: str, price: float):
        if instrument not in self.positions:
            return
        pos  = self.positions[instrument]
        dir_ = pos["direction"]
        sl   = pos.get("sl")
        tp   = pos.get("tp")
        hit  = None
        if sl and dir_ ==  1 and price <= sl: hit = ("sl", sl)
        if sl and dir_ == -1 and price >= sl: hit = ("sl", sl)
        if tp and dir_ ==  1 and price >= tp: hit = ("tp", tp)
        if tp and dir_ == -1 and price <= tp: hit = ("tp", tp)
        if hit:
            self._close_position(instrument, hit[1], reason=hit[0])

    def _log_trade_outcome(self, instrument, pos, exit_price, pnl, reason):
        record = {
            "instrument": instrument,
            "direction":  pos.get("direction", 0),
            "entry":      round(pos.get("entry", 0), 6),
            "exit":       round(exit_price, 6),
            "pnl":        round(pnl, 4),
            "won":        pnl > 0,
            "reason":     reason,
            "confidence": round(pos.get("confidence", 0), 4),
            "regime":     pos.get("regime", "unknown"),
            "atr":        round(pos.get("atr") or 0, 6),
            "htf_trend":  pos.get("htf_trend", 0),
            "timestamp":  datetime.now(timezone.utc).isoformat(),
        }
        os.makedirs("logs/feedback", exist_ok=True)
        with open(f"logs/feedback/{instrument}_outcomes.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")

    def _log_to_questdb(self, instrument: str, pos: dict, exit_price: float, pnl: float, reason: str):
        """Write closed trade to QuestDB (non-blocking — never raises)."""
        try:
            from questdb.ingress import Sender, TimestampNanos
            conf = 'http::addr=localhost:9000;'
            entry   = pos.get('entry', 0.0)
            units   = pos.get('units', 0.0)
            nav     = self.capital
            pnl_pct = pnl / max(abs(entry * units), 1e-9)
            hold_h  = (datetime.now(timezone.utc) - pos.get('time', datetime.now(timezone.utc))).total_seconds() / 3600
            with Sender.from_conf(conf) as s:
                s.row(
                    'trades',
                    symbols={
                        'instrument': instrument,
                        'system':     pos.get('system', 'unknown'),
                        'reason':     reason,
                    },
                    columns={
                        'direction':   int(pos.get('direction', 0)),
                        'entry_price': float(entry),
                        'exit_price':  float(exit_price),
                        'sl_price':    float(pos.get('sl') or 0),
                        'tp_price':    float(pos.get('tp') or 0),
                        'units':       float(units),
                        'pnl':         float(pnl),
                        'pnl_pct':     float(pnl_pct),
                        'confidence':  float(pos.get('confidence') or 0),
                        'hold_hours':  float(hold_h),
                        'nav':         float(nav),
                    },
                    at=TimestampNanos.now(),
                )
                s.flush()
        except Exception as e:
            logger.debug(f'QuestDB trade log failed for {instrument}: {e}')

    def _close_position(self, instrument: str, exit_price: float, reason: str = "manual"):
        if instrument not in self.positions:
            return
        pos   = self.positions.pop(instrument)
        entry = pos["entry"]
        units = pos["units"]
        dir_  = pos["direction"]
        pnl   = dir_ * (exit_price - entry) * units
        self._log_trade_outcome(instrument, pos, exit_price, pnl, reason)
        self._log_to_questdb(instrument, pos, exit_price, pnl, reason)
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
        # Also log to SQLite
        if _db_log_trade is not None:
            try:
                _db_log_trade(
                    instrument=instrument,
                    direction="Long" if dir_ == 1 else "Short",
                    entry=entry,
                    exit_price=exit_price,
                    units=units,
                    pnl=round(pnl, 4),
                    reason=reason,
                    entry_time=pos.get("time", ""),
                    exit_time=datetime.now(timezone.utc),
                    confidence=pos.get("confidence"),
                    regime=pos.get("regime"),
                    system=pos.get("system", "swing"),
                )
            except Exception:
                pass  # Don't break on DB error
        try:
            from notifications.telegram import trade_closed as _tg_close
            hold_hours = (
                datetime.now(timezone.utc) - pos.get("time", datetime.now(timezone.utc))
            ).total_seconds() / 3600
            closed_pnls = [t.get("pnl", 0) for t in self.trades]
            win_streak = 0
            for _p in reversed(closed_pnls):
                if _p > 0:
                    win_streak += 1
                else:
                    break
            nav = self.capital  # capital already updated above
            total_pnl = nav - self.initial_capital
            _tg_close(
                ticker      = instrument,
                direction   = dir_,
                entry_price = entry,
                exit_price  = exit_price,
                sl_price    = pos.get("sl") or exit_price,
                tp_price    = pos.get("tp") or exit_price,
                lot_size    = abs(units),
                pnl         = pnl,
                pnl_pct     = pnl / max(self.initial_capital, 1) * 100,
                reason      = reason,
                system      = pos.get("system", "swing"),
                asset_class = pos.get("asset_class", "forex"),
                hold_hours  = hold_hours,
                confidence  = pos.get("confidence"),
                nav         = nav,
                total_pnl   = total_pnl,
                win_streak  = win_streak,
                timeframe   = pos.get("timeframe", "1h"),
            )
        except Exception:
            pass
        logger.info(f"Paper close: {instrument} @ {exit_price:.5f} | P&L={pnl:+.4f}")

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
        system:      str   = 'swing',
        asset_class: str   = 'forex',
        timeframe:   str   = '1h',
        confidence:  float = 0.0,
        regime:      str   = 'unknown',
        atr:         float = None,
        risk_pct:    float = None,
        risk_amount: float = None,
    ) -> dict:
        direction  = 1 if units > 0 else -1
        raw_price  = self._prices.get(instrument, 0.0)
        fill_price = self._apply_costs(raw_price, direction)
        if fill_price <= 0:
            return {"status": "rejected", "reason": "no_price"}
        if instrument in self.positions:
            if self.positions[instrument]["direction"] != direction:
                self._close_position(instrument, fill_price, reason="reversal")
        self.positions[instrument] = {
            "instrument":  instrument,
            "units":       abs(units),
            "direction":   direction,
            "entry":       fill_price,
            "sl":          stop_loss,
            "tp":          take_profit,
            "time":        datetime.now(timezone.utc),
            "system":      system,
            "asset_class": asset_class,
            "timeframe":   timeframe,
            "confidence":  confidence,
            "regime":      regime,
            "atr":         atr,
            "risk_pct":    risk_pct,
            "risk_amount": risk_amount,
        }
        logger.info(
            f"Paper order: {instrument} | "
            f"{'BUY' if direction==1 else 'SELL'} {abs(units)} | "
            f"fill={fill_price:.5f} | sl={stop_loss} | tp={take_profit}"
        )
        try:
            from notifications.telegram import trade_opened as _tg_open
            nav = self.capital + sum(
                p["direction"] * (self._prices.get(i, p["entry"]) - p["entry"]) * p["units"]
                for i, p in self.positions.items()
            )
            _tg_open(
                ticker      = instrument,
                direction   = direction,
                entry_price = fill_price,
                sl_price    = stop_loss or fill_price,
                tp_price    = take_profit or fill_price,
                lot_size    = abs(units),
                confidence  = confidence,
                regime      = regime,
                system      = system,
                asset_class = asset_class,
                atr         = atr,
                risk_pct    = risk_pct,
                risk_amount = risk_amount,
                nav         = nav,
                timeframe   = timeframe,
            )
        except Exception:
            pass
        return {"status": "filled", "fill_price": fill_price, "units": units}

    def close_position(self, instrument: str, reason=None) -> dict:
        if instrument not in self.positions:
            return {"status": "no_position"}
        price = self._prices.get(instrument, self.positions[instrument]["entry"])
        self._close_position(instrument, price, reason="manual")
        return {"status": "closed"}

    def get_open_positions(self) -> list:
        return [
            {**pos, "instrument": inst,
             "unrealised_pnl": pos["direction"] * (
                 self._prices.get(inst, pos["entry"]) - pos["entry"]) * pos["units"]}
            for inst, pos in self.positions.items()
        ]

    def get_open_by_class(self) -> dict:
        """Return count of open positions by asset class."""
        from models.registry import ModelRegistry
        registry = ModelRegistry()
        info_map = registry.get_active_instruments()
        counts = {}
        for inst in self.positions:
            ac = info_map.get(inst, {}).get("asset_class", "unknown")
            counts[ac] = counts.get(ac, 0) + 1
        return counts

    @property
    def trade_log(self) -> pd.DataFrame:
        return pd.DataFrame(self.trades) if self.trades else pd.DataFrame()


# ── IBKR Broker ───────────────────────────────────────────────

class IBKRBroker(BrokerBase):
    """
    Interactive Brokers TWS/Gateway Python API broker.

    Connects via TCP socket to IB Gateway running on localhost.
    Works natively on macOS — no Windows needed.

    Setup:
      1. Download IB Gateway: interactivebrokers.com/en/trading/ibgateway.php
      2. Login with your IBKR paper account
      3. Configure → API → Settings:
         - Enable ActiveX and Socket Clients ✅
         - Socket port: 7497 (paper) / 7496 (live)
         - Allow connections from localhost ✅
      4. pip install ibapi

    Instrument format (internal → IBKR):
      "EURUSD=X"  → symbol="EUR",  secType="CASH",    currency="USD", exchange="IDEALPRO"
      "GC=F"      → symbol="GC",   secType="FUT",     currency="USD", exchange="COMEX"
      "AAPL"      → symbol="AAPL", secType="STK",     currency="USD", exchange="SMART"
    """

    # Internal ticker → IBKR contract parameters
    CONTRACT_MAP = {
        # Forex pairs — free data, no subscription needed
        "EURUSD=X":  {"symbol": "EUR",  "secType": "CASH", "currency": "USD", "exchange": "IDEALPRO"},
        "GBPUSD=X":  {"symbol": "GBP",  "secType": "CASH", "currency": "USD", "exchange": "IDEALPRO"},
        "EURGBP=X":  {"symbol": "EUR",  "secType": "CASH", "currency": "GBP", "exchange": "IDEALPRO"},
        "EURJPY=X":  {"symbol": "EUR",  "secType": "CASH", "currency": "JPY", "exchange": "IDEALPRO"},
        "USDCHF=X":  {"symbol": "USD",  "secType": "CASH", "currency": "CHF", "exchange": "IDEALPRO"},
        "EURCHF=X":  {"symbol": "EUR",  "secType": "CASH", "currency": "CHF", "exchange": "IDEALPRO"},
        "AUDUSD=X":  {"symbol": "AUD",  "secType": "CASH", "currency": "USD", "exchange": "IDEALPRO"},
        "NZDUSD=X":  {"symbol": "NZD",  "secType": "CASH", "currency": "USD", "exchange": "IDEALPRO"},
        # Commodities — require market data subscription
        "GC=F":      {"symbol": "XAUUSD", "secType": "CFD", "currency": "USD", "exchange": "SMART"},
        "SI=F":      {"symbol": "XAGUSD", "secType": "CFD", "currency": "USD", "exchange": "SMART"},
        # Equities
        "AAPL":      {"symbol": "AAPL", "secType": "STK", "currency": "USD", "exchange": "SMART"},
        "NVDA":      {"symbol": "NVDA", "secType": "STK", "currency": "USD", "exchange": "SMART"},
        "GS":        {"symbol": "GS",   "secType": "STK", "currency": "USD", "exchange": "SMART"},
        "TSLA":      {"symbol": "TSLA", "secType": "STK", "currency": "USD", "exchange": "SMART"},
        "JPM":       {"symbol": "JPM",  "secType": "STK", "currency": "USD", "exchange": "SMART"},
    }

    def __init__(self, host: str = None, port: int = None, client_id: int = None):
        try:
            from ibapi.client  import EClient
            from ibapi.wrapper import EWrapper
            from ibapi.contract import Contract
            from ibapi.order    import Order
            self._Contract = Contract
            self._Order    = Order
        except ImportError:
            raise ImportError(
                "ibapi not installed.\n"
                "Run: pip install ibapi\n"
                "Then open IB Gateway and enable socket connections."
            )

        self._host      = host      or IBKR_HOST
        self._port      = port      or IBKR_PORT
        self._client_id = client_id or IBKR_CLIENT_ID
        self._next_req_id   = 1
        self._next_order_id = None
        self._lock      = threading.Lock()

        # Queues for async responses
        self._price_queue   = {}   # req_id → queue
        self._account_data  = {}
        self._positions_data = []
        self._order_status  = {}

        # SL order tracking for modify-in-place
        self._sl_order_ids  = {}   # instrument → sl order_id
        self._open_orders   = {}   # order_id   → {action, orderType, totalQuantity, auxPrice, parentId}

        # Build the app (EWrapper + EClient combined)
        self._app = self._build_app()
        self._connect()

    def _next_id(self) -> int:
        with self._lock:
            rid = self._next_req_id
            self._next_req_id += 1
        return rid

    def _build_app(self):
        """Build a combined EWrapper + EClient class dynamically."""
        from ibapi.client  import EClient
        from ibapi.wrapper import EWrapper

        broker_ref = self  # capture self for inner class

        class IBApp(EWrapper, EClient):
            def __init__(self):
                EClient.__init__(self, self)

            # ── Connection callbacks ──────────────────────────
            def nextValidId(self, orderId: int):
                broker_ref._next_order_id = orderId
                logger.info(f"IBKR connected | nextValidId={orderId}")

            def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
                # Suppress harmless info messages
                if errorCode in (2104, 2106, 2158, 2119):
                    return
                logger.warning(f"IBKR error [{errorCode}] req={reqId}: {errorString}")

            # ── Account callbacks ─────────────────────────────
            def updateAccountValue(self, key, val, currency, accountName):
                broker_ref._account_data[key] = val

            def accountDownloadEnd(self, accountName):
                broker_ref._account_data["_ready"] = True

            # ── Price callbacks ───────────────────────────────
            def tickPrice(self, reqId, tickType, price, attrib):
                if reqId in broker_ref._price_queue:
                    broker_ref._price_queue[reqId].put((tickType, price))

            # ── Position callbacks ────────────────────────────
            def position(self, account, contract, position, avgCost):
                broker_ref._positions_data.append({
                    "symbol":    contract.symbol,
                    "secType":   contract.secType,
                    "currency":  contract.currency,
                    "position":  position,
                    "avgCost":   avgCost,
                })

            def positionEnd(self):
                broker_ref._account_data["_positions_ready"] = True

            # ── Order callbacks ───────────────────────────────
            def orderStatus(self, orderId, status, filled, remaining,
                            avgFillPrice, permId, parentId, lastFillPrice,
                            clientId, whyHeld, mktCapPrice):
                broker_ref._order_status[orderId] = {
                    "status":        status,
                    "filled":        filled,
                    "avgFillPrice":  avgFillPrice,
                }

            def openOrder(self, orderId, contract, order, orderState):
                # Cache live order details so update_stop_loss can re-submit
                broker_ref._open_orders[orderId] = {
                    "action":        order.action,
                    "orderType":     order.orderType,
                    "totalQuantity": order.totalQuantity,
                    "auxPrice":      order.auxPrice,
                    "lmtPrice":      order.lmtPrice,
                    "parentId":      order.parentId,
                }

        return IBApp()

    def _connect(self):
        self._app.connect(self._host, self._port, self._client_id)
        thread = threading.Thread(target=self._app.run, daemon=True)
        thread.start()
        # Wait for nextValidId to confirm connection
        timeout = 10
        for _ in range(timeout * 10):
            if self._next_order_id is not None:
                break
            time.sleep(0.1)
        else:
            raise RuntimeError(
                f"Could not connect to IB Gateway at {self._host}:{self._port}\n"
                "Make sure IB Gateway is open, logged in, and API is enabled."
            )
        logger.info(
            f"IBKRBroker connected | "
            f"{self._host}:{self._port} | "
            f"clientId={self._client_id}"
        )

    def _make_contract(self, instrument: str):
        """Build an IBKR Contract object from internal ticker."""
        params = self.CONTRACT_MAP.get(instrument)
        if params is None:
            raise ValueError(
                f"No IBKR contract mapping for '{instrument}'.\n"
                f"Add it to IBKRBroker.CONTRACT_MAP in broker.py"
            )
        c          = self._Contract()
        c.symbol   = params["symbol"]
        c.secType  = params["secType"]
        c.currency = params["currency"]
        c.exchange = params["exchange"]
        return c

    def get_account(self) -> dict:
        self._account_data.clear()
        self._app.reqAccountUpdates(True, "")
        timeout = 5
        for _ in range(timeout * 10):
            if self._account_data.get("_ready"):
                break
            time.sleep(0.1)
        self._app.reqAccountUpdates(False, "")

        def _f(key, default=0.0):
            try:
                return float(self._account_data.get(key, default))
            except Exception:
                return default

        return {
            "balance":        _f("CashBalance"),
            "nav":            _f("NetLiquidation"),
            "unrealised_pnl": _f("UnrealizedPnL"),
            "open_trades":    int(_f("OpenPositions", 0)),
        }

    def get_latest_price(self, instrument: str) -> dict:
        req_id = self._next_id()
        q      = queue.Queue()
        self._price_queue[req_id] = q

        contract = self._make_contract(instrument)
        self._app.reqMktData(req_id, contract, "", False, False, [])

        bid = ask = mid = 0.0
        try:
            deadline = time.time() + 5
            while time.time() < deadline:
                try:
                    tick_type, price = q.get(timeout=0.5)
                    if tick_type == 1:  bid = price   # BID
                    if tick_type == 2:  ask = price   # ASK
                    if bid > 0 and ask > 0:
                        break
                except queue.Empty:
                    continue
        finally:
            self._app.cancelMktData(req_id)
            del self._price_queue[req_id]

        mid = (bid + ask) / 2 if bid and ask else max(bid, ask)
        return {"bid": bid, "ask": ask, "mid": mid, "instrument": instrument}

    def get_last_price(self, instrument: str) -> float:
        return self.get_latest_price(instrument).get("mid", 0.0)

    def market_order(
        self,
        instrument:  str,
        units:       float,
        stop_loss:   float = None,
        take_profit: float = None,
    ) -> dict:
        """
        Place a market order via IB Gateway.
        units > 0 = BUY, units < 0 = SELL
        For forex: units = number of base currency units (e.g. 10000 = 1 mini lot)
        """
        from ibapi.order import Order

        if self._next_order_id is None:
            return {"status": "rejected", "reason": "not connected"}

        direction  = 1 if units > 0 else -1
        action     = "BUY" if direction == 1 else "SELL"
        qty        = abs(units)

        contract   = self._make_contract(instrument)

        # Main market order
        order          = Order()
        order.action   = action
        order.orderType = "MKT"
        order.totalQuantity = qty
        order.transmit  = True if (stop_loss is None and take_profit is None) else False

        order_id = self._next_order_id
        self._next_order_id += 1
        self._app.placeOrder(order_id, contract, order)

        result = {"status": "submitted", "order_id": order_id}

        # Attach bracket orders if SL/TP provided
        if stop_loss:
            sl_order          = Order()
            sl_order.action   = "SELL" if direction == 1 else "BUY"
            sl_order.orderType = "STP"
            sl_order.auxPrice  = round(stop_loss, 5)
            sl_order.totalQuantity = qty
            sl_order.parentId  = order_id
            sl_order.transmit  = take_profit is None
            sl_id = self._next_order_id
            self._next_order_id += 1
            self._app.placeOrder(sl_id, contract, sl_order)
            result["sl_order_id"] = sl_id
            self._sl_order_ids[instrument] = sl_id   # track for later modification

        if take_profit:
            tp_order          = Order()
            tp_order.action   = "SELL" if direction == 1 else "BUY"
            tp_order.orderType = "LMT"
            tp_order.lmtPrice  = round(take_profit, 5)
            tp_order.totalQuantity = qty
            tp_order.parentId  = order_id
            tp_order.transmit  = True   # transmit all together
            tp_id = self._next_order_id
            self._next_order_id += 1
            self._app.placeOrder(tp_id, contract, tp_order)
            result["tp_order_id"] = tp_id

        # Also transmit the parent if it was held back
        if stop_loss or take_profit:
            order.transmit = False  # already set, just make sure

        logger.info(
            f"IBKR order: {instrument} | {action} {qty} | "
            f"sl={stop_loss} | tp={take_profit} | orderId={order_id}"
        )
        return result

    def close_position(self, instrument: str) -> dict:
        """Close all open positions for an instrument."""
        self._positions_data.clear()
        self._account_data.pop("_positions_ready", None)
        self._app.reqPositions()

        for _ in range(50):
            if self._account_data.get("_positions_ready"):
                break
            time.sleep(0.1)

        contract_params = self.CONTRACT_MAP.get(instrument, {})
        symbol = contract_params.get("symbol", instrument)

        for pos in self._positions_data:
            if pos["symbol"] == symbol and pos["position"] != 0:
                qty    = abs(pos["position"])
                action = "SELL" if pos["position"] > 0 else "BUY"

                contract = self._make_contract(instrument)
                order          = self._Order()
                order.action   = action
                order.orderType = "MKT"
                order.totalQuantity = qty
                order.transmit  = True

                order_id = self._next_order_id
                self._next_order_id += 1
                self._app.placeOrder(order_id, contract, order)
                logger.info(f"IBKR close: {instrument} {action} {qty}")
                # Clean up SL order tracking so a future trade starts fresh
                self._sl_order_ids.pop(instrument, None)
                return {"status": "closing", "order_id": order_id}

        return {"status": "no_position"}

    def update_stop_loss(self, instrument: str, new_sl: float) -> bool:
        """
        Modify the live SL child order for `instrument` by re-submitting
        the same orderId with the updated auxPrice.  IBKR treats a
        placeOrder() call with an existing orderId as a modification.

        Flow:
          1. Look up the tracked SL order ID for this instrument.
          2. If the order details are not yet in _open_orders, call
             reqOpenOrders() to trigger openOrder() callbacks that
             populate the cache, then wait up to 1 s.
          3. Re-submit the STP order with the new auxPrice.
        """
        from ibapi.order import Order

        sl_id = self._sl_order_ids.get(instrument)
        if sl_id is None:
            logger.warning(f"{instrument}: no tracked SL order — cannot modify")
            return False

        # Ensure order details are cached (openOrder callback populates _open_orders)
        if sl_id not in self._open_orders:
            self._app.reqOpenOrders()
            for _ in range(20):          # wait up to 1 s in 50 ms steps
                if sl_id in self._open_orders:
                    break
                time.sleep(0.05)

        cached = self._open_orders.get(sl_id, {})

        mod_order = Order()
        mod_order.action        = cached.get("action", "SELL")
        mod_order.orderType     = "STP"
        mod_order.auxPrice      = round(new_sl, 5)
        mod_order.totalQuantity = cached.get("totalQuantity", 0)
        mod_order.parentId      = cached.get("parentId", 0)
        mod_order.transmit      = True

        contract = self._make_contract(instrument)
        self._app.placeOrder(sl_id, contract, mod_order)

        # Update local cache immediately so next modify sees the new price
        if sl_id in self._open_orders:
            self._open_orders[sl_id]["auxPrice"] = mod_order.auxPrice

        logger.info(f"{instrument}: SL order {sl_id} modified → {new_sl:.5f}")
        return True

    def partial_close(self, instrument: str, close_fraction: float = 0.5,
                      reason: str = "partial_tp") -> dict:
        """Close a fraction of the open position."""
        self._positions_data.clear()
        self._account_data.pop("_positions_ready", None)
        self._app.reqPositions()
        for _ in range(50):
            if self._account_data.get("_positions_ready"):
                break
            time.sleep(0.1)

        contract_params = self.CONTRACT_MAP.get(instrument, {})
        symbol = contract_params.get("symbol", instrument)

        for pos in self._positions_data:
            if pos["symbol"] == symbol and pos["position"] != 0:
                total  = abs(pos["position"])
                qty    = max(1, int(total * close_fraction))
                action = "SELL" if pos["position"] > 0 else "BUY"

                contract        = self._make_contract(instrument)
                order           = self._Order()
                order.action    = action
                order.orderType = "MKT"
                order.totalQuantity = qty
                order.transmit  = True

                order_id = self._next_order_id
                self._next_order_id += 1
                self._app.placeOrder(order_id, contract, order)
                logger.info(f"IBKR partial close: {instrument} {action} {qty}")
                return {"status": "partial_closed", "pnl": 0.0}  # PnL from trade history

        return {"status": "no_position"}

    def get_open_positions(self) -> list:
        self._positions_data.clear()
        self._account_data.pop("_positions_ready", None)
        self._app.reqPositions()
        for _ in range(50):
            if self._account_data.get("_positions_ready"):
                break
            time.sleep(0.1)

        result = []
        for pos in self._positions_data:
            if pos["position"] == 0:
                continue
            # Reverse map symbol → internal ticker
            reverse = {v["symbol"]: k for k, v in self.CONTRACT_MAP.items()}
            ticker  = reverse.get(pos["symbol"], pos["symbol"])
            result.append({
                "instrument":     ticker,
                "units":          abs(pos["position"]),
                "direction":      1 if pos["position"] > 0 else -1,
                "entry":          pos["avgCost"],
                "unrealised_pnl": 0.0,  # requires reqPnLSingle
                "sl":             None,
                "tp":             None,
            })
        return result

    def get_bars_open(self, instrument: str, current_bar: int) -> int:
        """Approximate — counts bars since position was added."""
        return 0  # IBKR requires reqOpenOrders for precise tracking

    def update_price(self, instrument: str, price: float):
        """No-op — prices fetched live from IBKR on demand."""
        pass

    @property
    def trade_log(self) -> pd.DataFrame:
        """Fetch execution reports from IBKR."""
        return pd.DataFrame()  # implement reqExecutions for full history

    def disconnect(self):
        self._app.disconnect()
        logger.info("IBKRBroker disconnected")


# ── Factory ───────────────────────────────────────────────────

def get_broker(broker_type: str = BROKER) -> BrokerBase:
    """
    Return correct broker instance.
    broker_type: "paper" | "ibkr"
    """
    if broker_type == "paper":
        return PaperBroker()
    elif broker_type == "ibkr":
        return IBKRBroker()
    else:
        raise ValueError(f"Unknown broker: '{broker_type}'. Choose: paper | ibkr")