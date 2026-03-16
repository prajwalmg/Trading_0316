"""
================================================================
  risk/engine.py
  Professional risk management engine — scales from €10 to €10M+.

  Seven interlocking rules:
    1. Fixed-fractional sizing (1% risk per trade)
    2. ATR-based dynamic SL/TP
    3. Kelly criterion cap (quarter-Kelly)
    4. Daily / weekly loss circuit breakers
    5. Correlation guard (max correlated trades)
    6. Maximum drawdown halt
    7. Volatility regime scaling
================================================================
"""

import logging
from datetime import datetime, timezone, date
from typing import Optional
from collections import defaultdict

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.settings import (
    MAX_RISK_PCT, MAX_DAILY_LOSS_PCT, MAX_WEEKLY_LOSS_PCT,
    MAX_DRAWDOWN_PCT, MAX_OPEN_POSITIONS, MAX_CORRELATED_TRADES,
    MAX_SINGLE_EXPOSURE, SL_ATR_MULT, TP_ATR_MULT,
    TRAILING_STOP, TRAIL_ATR_MULT, KELLY_FRACTION,
    MIN_CONFIDENCE, INITIAL_CAPITAL,
    FOREX_PAIRS, EQUITY_TICKERS, COMMODITY_TICKERS, CRYPTO_TICKERS
)
from signals.regime import regime_position_scale

logger = logging.getLogger("trading_firm.risk")

# ── Asset class lookup ────────────────────────────────────────
ASSET_CLASS_MAP = {}
for t in FOREX_PAIRS:       ASSET_CLASS_MAP[t] = "forex"
for t in EQUITY_TICKERS:    ASSET_CLASS_MAP[t] = "equity"
for t in COMMODITY_TICKERS: ASSET_CLASS_MAP[t] = "commodity"
for t in CRYPTO_TICKERS:    ASSET_CLASS_MAP[t] = "crypto"


class RiskEngine:
    """
    Stateful risk engine — must be instantiated once and shared
    across all strategies. Tracks:
      - Account NAV and peak NAV (for drawdown)
      - Daily and weekly P&L
      - Open positions and their asset classes
      - Win/loss statistics for Kelly sizing
    """

    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.nav             = initial_capital
        self.peak_nav        = initial_capital
        self.cash            = initial_capital

        # P&L tracking
        self.daily_pnl       = 0.0
        self.weekly_pnl      = 0.0
        self.total_pnl       = 0.0
        self._last_reset_day  = None
        self._last_reset_week = None

        # Open positions: {instrument: {units, entry, sl, tp, direction, strategy}}
        self.open_positions: dict = {}

        # Trade history for Kelly calculation
        self._trade_history: list = []

        # Circuit breaker state
        self.daily_halted   = False
        self.weekly_halted  = False
        self.drawdown_halted = False

        logger.info(
            f"RiskEngine initialised | "
            f"capital={initial_capital:.2f} | "
            f"max_risk={MAX_RISK_PCT:.1%}/trade | "
            f"daily_limit={MAX_DAILY_LOSS_PCT:.1%}"
        )
        
    def update_trailing_stop(self, position: dict, current_price: float) -> float:
        from config.settings import ASSET_CLASS_MAP, ASSET_ATR_MULTIPLIER
        direction   = position.get("direction", 1)
        current_sl  = position.get("sl", 0)
        atr         = position.get("atr", current_price * 0.001)
        instrument  = position.get("instrument", "")

        asset_class = (
            ASSET_CLASS_MAP.get(instrument) or
            ASSET_CLASS_MAP.get(f"{instrument}-USD") or
            "equity"
        )
        trail_mult = ASSET_ATR_MULTIPLIER.get(asset_class, 1.5)

        if direction == 1:
            new_sl = current_price - (trail_mult * atr)
            return max(new_sl, current_sl)
        else:
            new_sl = current_price + (trail_mult * atr)
            return min(new_sl, current_sl)



    def portfolio_momentum_score(self, signals: dict) -> float:
        """
        Returns score -1.0 to +1.0 indicating overall market direction.
        +1.0 = all instruments bullish, -1.0 = all bearish.
        Use this to scale position sizes portfolio-wide.
        """
        long_count  = sum(1 for s in signals.values() if s.get("signal") ==  1)
        short_count = sum(1 for s in signals.values() if s.get("signal") == -1)
        total       = max(len(signals), 1)
        return (long_count - short_count) / total


    # ── Daily / weekly reset ──────────────────────────────────

    def _check_resets(self):
        now  = datetime.now(timezone.utc)
        today = now.date()
        week  = now.isocalendar()[1]

        if self._last_reset_day != today:
            self.daily_pnl      = 0.0
            self.daily_halted   = False
            self._last_reset_day = today
            logger.debug("Daily P&L reset")

        if self._last_reset_week != week:
            self.weekly_pnl      = 0.0
            self.weekly_halted   = False
            self._last_reset_week = week
            logger.debug("Weekly P&L reset")

    # ── NAV update ────────────────────────────────────────────

    def update_nav(self, nav: float):
        self.nav = nav
        if nav > self.peak_nav:
            self.peak_nav = nav
        current_dd = (self.peak_nav - nav) / self.peak_nav
        if current_dd >= MAX_DRAWDOWN_PCT and not self.drawdown_halted:
            self.drawdown_halted = True
            logger.critical(
                f"⛔ DRAWDOWN HALT: {current_dd:.2%} drawdown "
                f"(limit={MAX_DRAWDOWN_PCT:.2%}). ALL trading suspended."
            )

    # ── Kelly criterion ───────────────────────────────────────

    def kelly_fraction(self) -> float:
        """
        Compute quarter-Kelly from recent trade history.
        Falls back to fixed MAX_RISK_PCT if < 30 trades recorded.
        """
        if len(self._trade_history) < 30:
            return MAX_RISK_PCT

        pnls    = [t["pnl"] for t in self._trade_history[-100:]]
        winners = [p for p in pnls if p > 0]
        losers  = [p for p in pnls if p <= 0]

        if not winners or not losers:
            return MAX_RISK_PCT

        win_rate = len(winners) / len(pnls)
        avg_win  = np.mean(winners)
        avg_loss = abs(np.mean(losers))

        # Kelly formula: f = W/L - (1-W)/W  (simplified)
        b        = avg_win / (avg_loss + 1e-9)
        kelly    = (b * win_rate - (1 - win_rate)) / (b + 1e-9)
        kelly    = max(0.001, min(kelly, 0.10))   # Cap at 10%

        # Quarter-Kelly for safety
        return kelly * KELLY_FRACTION

    # ── Position sizing ───────────────────────────────────────

    def position_size(
        self,
        instrument: str,
        entry:      float,
        atr:        float,
        confidence: float,
        regime:     str = "ranging",
    ) -> dict:
        """
        Calculate position size using:
        - Base risk from MAX_RISK_PCT
        - Regime multiplier (trend = bigger, ranging = smaller)
        - Confidence multiplier (higher confidence = bigger)
        - Kelly fraction
        - Portfolio momentum boost
        """
        from config.settings import MAX_RISK_PCT, KELLY_FRACTION

        from config.settings import ASSET_RISK_PCT, ASSET_CLASS_MAP, ASSET_ATR_MULTIPLIER
        asset_class = ASSET_CLASS_MAP.get(instrument) or \
              ASSET_CLASS_MAP.get(f"{instrument}-USD") or \
              ASSET_CLASS_MAP.get(f"{instrument}=X") or \
              "equity"
        risk_pct    = ASSET_RISK_PCT.get(asset_class, MAX_RISK_PCT)
        base_risk   = self.nav * risk_pct

        # ── Progressive drawdown size reduction ──────────────────
        dd = (self.peak_nav - self.nav) / self.peak_nav if self.peak_nav > 0 else 0
        if dd >= 0.08:
            return {"units": 0, "risk_amount": 0, "regime_mult": 0,
                    "conf_mult": 0, "stop_distance": 0}
        elif dd >= 0.06: dd_factor = 0.25
        elif dd >= 0.04: dd_factor = 0.50
        elif dd >= 0.02: dd_factor = 0.75
        else:            dd_factor = 1.00

        base_risk *= dd_factor

        # ── Regime multiplier ─────────────────────────────────────
        regime_mult = {
            "trending_up":     1.5,
            "trending_down":   1.5,
            "ranging":         0.7,
            "high_volatility": 0.5,
        }.get(regime, 1.0)

        # ── Confidence multiplier (scales 0.5x → 1.5x) ───────────
        conf_mult = 0.5 + float(confidence)   # 0.5 at 0%, 1.5 at 100%
        conf_mult = max(0.5, min(conf_mult, 1.5))

        # ── Kelly fraction ────────────────────────────────────────
        risk = base_risk * regime_mult * conf_mult * KELLY_FRACTION

        # ── Hard cap at 3x base risk ──────────────────────────────
        risk = min(risk, base_risk * 3.0)
        risk = max(risk, 1.0)   # minimum $1 risk

        # ── Units from ATR-based stop distance ───────────────────
        atr_mult  = ASSET_ATR_MULTIPLIER.get(asset_class, 1.5)
        stop_dist = max(atr * atr_mult, entry * 0.001)   # at least 0.1% of price
        units     = risk / stop_dist
        
        # ── Asset-aware position sizing ───────────────────────────
        MAX_NOTIONAL_PCT = {
            "forex":     0.15,
            "equity":    0.25,
            "commodity": 0.15,
            "crypto":    0.10,
        }
        max_notional = self.nav * MAX_NOTIONAL_PCT.get(asset_class, 0.20)
        max_units    = max_notional / entry

        units = min(units, max_units)

        # Crypto with high price — allow fractional units
        if asset_class == "crypto" and entry > 100:
            units = round(max(0.0001, units), 4)
        else:
            units = max(1, int(units))      

        return {
            "units":         units,
            "risk_amount":   round(risk, 2),
            "regime_mult":   regime_mult,
            "conf_mult":     round(conf_mult, 3),
            "stop_distance": round(stop_dist, 5),
        }
    
    def correlation_adjusted_size(
        self,
        instrument:     str,
        base_units:     int,
        open_positions: list,
    ) -> int:
        """
        Reduce position size if correlated instruments already open.
        Each same-class position already open reduces size by 20%.
        """
        from config.settings import ASSET_CLASS_MAP
        asset_class = ASSET_CLASS_MAP.get(instrument) or \
              ASSET_CLASS_MAP.get(f"{instrument}-USD") or \
              ASSET_CLASS_MAP.get(f"{instrument}=X") or \
              "equity"
        same_class_count = sum(
            1 for p in open_positions
            if ASSET_CLASS_MAP.get(p.get("instrument", ""), "") == asset_class
        )
        reduction = 0.8 ** same_class_count
        adjusted  = max(1, int(base_units * reduction))
        if adjusted != base_units:
            logger.debug(
                f"{instrument}: size reduced {base_units}→{adjusted} "
                f"({same_class_count} correlated positions open)"
            )
        return adjusted


    def sl_tp(
        self,
        entry:     float,
        direction: int,       # 1=long, -1=short
        atr:       float,
        instrument: str = "",
    ) -> tuple:
        """
        Compute ATR-based stop loss and take profit prices.
        Returns (sl_price, tp_price).
        """
        from config.settings import ASSET_CLASS_MAP

        asset_class = (
            ASSET_CLASS_MAP.get(instrument) or
            ASSET_CLASS_MAP.get(f"{instrument}-USD") or
            ASSET_CLASS_MAP.get(f"{instrument}=X") or
            "equity"
        )

        SL_MULT = {
            "forex":     1.5,
            "equity":    2.0,
            "commodity": 2.0,
            "crypto":    3.0,
        }
        TP_MULT = {
            "forex":     2.0,
            "equity":    3.0,
            "commodity": 3.0,
            "crypto":    4.0,
        }

        sl_dist = atr * SL_MULT.get(asset_class, SL_ATR_MULT)
        tp_dist = atr * TP_MULT.get(asset_class, TP_ATR_MULT)


        if direction == 1:
            sl = round(entry - sl_dist, 5)
            tp = round(entry + tp_dist, 5)
        else:
            sl = round(entry + sl_dist, 5)
            tp = round(entry - tp_dist, 5)

        return sl, tp

    def trailing_stop(
        self,
        entry:          float,
        current_price:  float,
        current_sl:     float,
        direction:      int,
        atr:            float,
    ) -> float:
        """
        Update trailing stop loss. Moves SL in favour of trade
        once price moves TRAIL_ATR_MULT × ATR in our direction.
        Never moves against the trade.
        """
        trail_dist = atr * TRAIL_ATR_MULT

        if direction == 1:   # Long: trail stop upward
            new_sl = current_price - trail_dist
            return max(current_sl, new_sl)
        else:                # Short: trail stop downward
            new_sl = current_price + trail_dist
            return min(current_sl, new_sl)

    # ── Pre-trade filters ─────────────────────────────────────

    def can_trade(
        self,
        instrument:  str,
        signal:      int,
        confidence:  float,
        atr:         float,
        atr_mean:    float,
        hour_utc:    int,
    ) -> tuple:
        """
        Master pre-trade filter. Returns (allowed: bool, reason: str).

        Checks (in order):
          1. Drawdown circuit breaker
          2. Daily loss circuit breaker
          3. Weekly loss circuit breaker
          4. Flat signal guard
          5. Minimum confidence
          6. Max open positions
          7. Max correlated trades
          8. Volatility spike (news filter)
          9. Trading hours filter
        """
        self._check_resets()

        # 1. Drawdown halt
        if self.drawdown_halted:
            return False, f"Drawdown halt active ({MAX_DRAWDOWN_PCT:.0%} limit hit)"

        # 2. Daily loss
        if self.daily_halted:
            return False, "Daily loss limit hit — halted until tomorrow"
        if self.daily_pnl < 0:
            daily_loss_pct = abs(self.daily_pnl) / self.nav
            if daily_loss_pct >= MAX_DAILY_LOSS_PCT:
                self.daily_halted = True
                logger.warning(f"Daily loss limit hit: {daily_loss_pct:.2%}")
                return False, f"Daily loss {daily_loss_pct:.2%} ≥ {MAX_DAILY_LOSS_PCT:.2%}"

        # 3. Weekly loss
        if self.weekly_halted:
            return False, "Weekly loss limit hit — halted until next week"
        if self.weekly_pnl < 0:
            weekly_loss_pct = abs(self.weekly_pnl) / self.nav
            if weekly_loss_pct >= MAX_WEEKLY_LOSS_PCT:
                self.weekly_halted = True
                logger.warning(f"Weekly loss limit hit: {weekly_loss_pct:.2%}")
                return False, f"Weekly loss {weekly_loss_pct:.2%} ≥ {MAX_WEEKLY_LOSS_PCT:.2%}"

        # 4. Flat signal
        if signal == 0:
            return False, "Flat signal"

        # 5. Confidence
        if confidence < MIN_CONFIDENCE:
            return False, f"Confidence {confidence:.2%} < {MIN_CONFIDENCE:.2%}"

        # 6. Max open positions
        if len(self.open_positions) >= MAX_OPEN_POSITIONS:
            return False, f"Max open positions ({MAX_OPEN_POSITIONS}) reached"

        # 7. Correlation guard
        asset_class = ASSET_CLASS_MAP.get(instrument, "other")
        same_class  = sum(
            1 for pos in self.open_positions.values()
            if ASSET_CLASS_MAP.get(pos.get("instrument", ""), "") == asset_class
        )
        if same_class >= MAX_CORRELATED_TRADES:
            return False, (
                f"Max correlated trades ({MAX_CORRELATED_TRADES}) "
                f"in {asset_class} reached"
            )

        # 8. Volatility spike (avoid news events)
        if atr_mean > 0 and atr / atr_mean > 2.5:
            return False, f"Volatility spike: ATR={atr:.5f}, mean={atr_mean:.5f}"

        # 9. Trading hours (7am–9pm UTC for Forex; all hours for crypto/equity)
        asset_class = ASSET_CLASS_MAP.get(instrument, "equity")
        if asset_class == "forex" and hour_utc not in range(7, 21):
            return False, f"Outside forex trading hours (UTC {hour_utc})"
        if asset_class == "crypto":
            pass

        return True, "OK"

    # ── Trade lifecycle ───────────────────────────────────────

    def open_trade(self, trade: dict):
        inst = trade["instrument"]
        self.open_positions[inst] = trade
        logger.info(
            f"Position opened: {inst} | "
            f"{'LONG' if trade['direction']==1 else 'SHORT'} "
            f"{trade.get('units', '?')} | "
            f"entry={trade.get('entry', '?')} | "
            f"sl={trade.get('sl', '?')} | tp={trade.get('tp', '?')}"
        )

    def close_trade(self, instrument: str, exit_price: float, reason: str = ""):
        self._check_resets()

        if instrument not in self.open_positions:
            return

        pos     = self.open_positions.pop(instrument)
        entry   = pos.get("entry", exit_price)
        units   = pos.get("units", 0)
        dir_    = pos.get("direction", 1)
        pnl     = dir_ * (exit_price - entry) * units

        self.daily_pnl  += pnl
        self.weekly_pnl += pnl
        self.total_pnl  += pnl
        self.nav        += pnl
        self.update_nav(self.nav)

        self._trade_history.append({
            "instrument": instrument,
            "direction":  dir_,
            "entry":      entry,
            "exit":       exit_price,
            "pnl":        pnl,
            "reason":     reason,
            "time":       datetime.now(timezone.utc),
        })

        logger.info(
            f"Position closed: {instrument} | "
            f"exit={exit_price} | P&L={pnl:+.2f} | reason={reason}"
        )

    # ── Risk report ───────────────────────────────────────────

    def report(self) -> dict:
        self._check_resets()
        dd = (self.peak_nav - self.nav) / self.peak_nav if self.peak_nav > 0 else 0
        return {
            "nav":              round(self.nav, 2),
            "peak_nav":         round(self.peak_nav, 2),
            "drawdown":         f"{dd:.2%}",
            "daily_pnl":        round(self.daily_pnl, 2),
            "daily_pnl_pct":    f"{self.daily_pnl / self.nav:.2%}",
            "weekly_pnl":       round(self.weekly_pnl, 2),
            "total_pnl":        round(self.total_pnl, 2),
            "open_positions":   len(self.open_positions),
            "daily_halted":     self.daily_halted,
            "weekly_halted":    self.weekly_halted,
            "drawdown_halted":  self.drawdown_halted,
            "daily_limit_used": f"{abs(self.daily_pnl) / (self.nav * MAX_DAILY_LOSS_PCT):.1%}",
            "total_trades":     len(self._trade_history),
        }

    @property
    def trade_history(self) -> pd.DataFrame:
        if not self._trade_history:
            return pd.DataFrame()
        return pd.DataFrame(self._trade_history)
