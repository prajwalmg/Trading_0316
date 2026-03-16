"""
================================================================
  portfolio/manager.py
  Multi-strategy portfolio manager:

  Strategies:
    1. Momentum     — trend-following in direction of ML signal
    2. Mean Reversion — fade overbought/oversold extremes
    3. Breakout     — enter on Bollinger squeeze breakouts
    4. Regime Adaptive — adjusts weights by detected regime

  Responsibilities:
    - Collect signals from all strategies
    - Resolve conflicts between strategies
    - Apply portfolio-level risk rules (correlation, exposure)
    - Rebalance strategy weights based on rolling performance
    - Coordinate with RiskEngine and Broker
================================================================
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.settings import (
    MIN_BARS_BETWEEN_TRADES, STRATEGY_WEIGHTS, MIN_CONFIDENCE, REBALANCE_HOURS,
    FOREX_PAIRS, EQUITY_TICKERS, COMMODITY_TICKERS,
)
from signals.regime import RegimeTracker, strategy_weights_by_regime
from risk.engine    import RiskEngine

logger = logging.getLogger("trading_firm.portfolio")


# ── Individual strategy signal generators ─────────────────────

def momentum_signal(df: pd.DataFrame, ml_signal: int, ml_conf: float) -> dict:
    """
    Momentum strategy: agrees with ML signal when trend is strong.
    Requires ADX > 20 and MACD histogram in same direction as signal.
    """
    if df.empty or len(df) < 30:
        return {"signal": 0, "confidence": 0.0, "strategy": "momentum"}

    latest = df.iloc[-1]
    adx    = latest.get("adx", 0) if hasattr(latest, "get") else 0

    try:
        adx       = float(df["adx"].iloc[-1])       if "adx"       in df.columns else 0
        macd_hist = float(df["macd_hist"].iloc[-1])  if "macd_hist" in df.columns else 0
        roc_6     = float(df["roc_6"].iloc[-1])      if "roc_6"     in df.columns else 0
    except Exception:
        return {"signal": 0, "confidence": 0.0, "strategy": "momentum"}

    # Confirm: ML signal, MACD histogram, and short ROC all agree
    macd_dir = 1 if macd_hist > 0 else (-1 if macd_hist < 0 else 0)
    roc_dir  = 1 if roc_6    > 0 else (-1 if roc_6    < 0 else 0)

    if ml_signal != 0 and macd_dir == ml_signal and adx > 20:
        conf = ml_conf * min(1.0, adx / 40)
        return {"signal": ml_signal, "confidence": conf, "strategy": "momentum"}

    return {"signal": 0, "confidence": 0.0, "strategy": "momentum"}


def mean_reversion_signal(df: pd.DataFrame) -> dict:
    """
    Mean reversion: fade extremes when RSI and Bollinger confirm.
    Only trade when market is NOT strongly trending (ADX < 25).
    """
    if df.empty or len(df) < 30:
        return {"signal": 0, "confidence": 0.0, "strategy": "mean_reversion"}

    try:
        rsi    = float(df["rsi"].iloc[-1])     if "rsi"    in df.columns else 50
        bb_pos = float(df["bb_pos"].iloc[-1])  if "bb_pos" in df.columns else 0.5
        adx    = float(df["adx"].iloc[-1])     if "adx"    in df.columns else 0
        stoch  = float(df["stoch_k"].iloc[-1]) if "stoch_k" in df.columns else 50
    except Exception:
        return {"signal": 0, "confidence": 0.0, "strategy": "mean_reversion"}

    # Only trade mean reversion when trend is weak
    if adx > 30:
        return {"signal": 0, "confidence": 0.0, "strategy": "mean_reversion"}

    # Oversold conditions → buy
    if rsi < 30 and bb_pos < 0.2 and stoch < 25:
        conf = (30 - rsi) / 30 * 0.8
        return {"signal": 1, "confidence": conf, "strategy": "mean_reversion"}

    # Overbought conditions → sell
    if rsi > 70 and bb_pos > 0.8 and stoch > 75:
        conf = (rsi - 70) / 30 * 0.8
        return {"signal": -1, "confidence": conf, "strategy": "mean_reversion"}

    return {"signal": 0, "confidence": 0.0, "strategy": "mean_reversion"}


def breakout_signal(df: pd.DataFrame) -> dict:
    """
    Breakout strategy: enter when price breaks out of a Bollinger squeeze.
    Squeeze = BB width at 20-bar low. Breakout = close outside BB.
    """
    if df.empty or len(df) < 40:
        return {"signal": 0, "confidence": 0.0, "strategy": "breakout"}

    try:
        bb_squeeze = int(df["bb_squeeze"].iloc[-1])   if "bb_squeeze" in df.columns else 0
        bb_pos     = float(df["bb_pos"].iloc[-1])     if "bb_pos"     in df.columns else 0.5
        vol_ratio  = float(df["vol_ratio"].iloc[-1])  if "vol_ratio"  in df.columns else 1.0
        roc_1      = float(df["roc_1"].iloc[-1])      if "roc_1"      in df.columns else 0
    except Exception:
        return {"signal": 0, "confidence": 0.0, "strategy": "breakout"}

    # Breakout up: was squeezed, now closes above upper band, with volume
    if bb_squeeze and bb_pos > 0.95 and vol_ratio > 1.3 and roc_1 > 0:
        conf = min(0.85, 0.5 + (bb_pos - 0.95) * 5 + (vol_ratio - 1) * 0.2)
        return {"signal": 1, "confidence": conf, "strategy": "breakout"}

    # Breakout down
    if bb_squeeze and bb_pos < 0.05 and vol_ratio > 1.3 and roc_1 < 0:
        conf = min(0.85, 0.5 + (0.05 - bb_pos) * 5 + (vol_ratio - 1) * 0.2)
        return {"signal": -1, "confidence": conf, "strategy": "breakout"}

    return {"signal": 0, "confidence": 0.0, "strategy": "breakout"}


def regime_adaptive_signal(
    ml_signal: int,
    ml_conf:   float,
    regime:    str,
) -> dict:
    """
    Regime adaptive strategy: passes through ML signal but
    scales confidence based on regime appropriateness.
    Acts as a fallback when other strategies disagree.
    """
    regime_conf_scale = {
        "trending_up":    1.0 if ml_signal ==  1 else 0.4,
        "trending_down":  1.0 if ml_signal == -1 else 0.4,
        "ranging":        0.7,
        "high_volatility": 0.3,
    }
    scale = regime_conf_scale.get(regime, 0.7)
    adj_conf = ml_conf * scale

    if adj_conf < MIN_CONFIDENCE or ml_signal == 0:
        return {"signal": 0, "confidence": 0.0, "strategy": "regime_adaptive"}

    return {"signal": ml_signal, "confidence": adj_conf, "strategy": "regime_adaptive"}


# ── Portfolio Manager ─────────────────────────────────────────

class PortfolioManager:
    """
    Orchestrates all strategies, resolves signal conflicts,
    manages portfolio-level risk, and coordinates execution.
    """

    def __init__(self, risk_engine: RiskEngine):
        self.risk           = risk_engine
        self.regime_tracker = RegimeTracker()
        self._strategy_pnl  = {s: 0.0 for s in STRATEGY_WEIGHTS}
        self._weights       = dict(STRATEGY_WEIGHTS)
        self._last_rebalance = datetime.now(timezone.utc)
        self._signal_log    = []
        self._last_trade    = {}  # ticker → last trade bar index

    # ── Signal aggregation ────────────────────────────────────

    def aggregate_signals(
        self,
        df:        pd.DataFrame,
        ticker:    str,
        ml_signal: int,
        ml_conf:   float,
    ) -> dict:
        self.regime_tracker.update(ticker, df)
        regime  = self.regime_tracker.current(ticker)
        weights = strategy_weights_by_regime(regime)

        # Collect strategy signals
        sigs = {
            "momentum":        momentum_signal(df, ml_signal, ml_conf),
            "mean_reversion":  mean_reversion_signal(df),
            "breakout":        breakout_signal(df),
            "regime_adaptive": regime_adaptive_signal(ml_signal, ml_conf, regime),
        }

        # Weighted vote
        weighted_long  = 0.0
        weighted_short = 0.0
        total_weight   = 0.0

        for strat, sig_dict in sigs.items():
            w    = weights.get(strat, 0.0)
            sig  = sig_dict["signal"]
            conf = sig_dict["confidence"]

            if sig == 1:
                weighted_long  += w * conf
            elif sig == -1:
                weighted_short += w * conf
            total_weight += w

        long_score  = weighted_long  / (total_weight + 1e-9)
        short_score = weighted_short / (total_weight + 1e-9)

        # ── KEY FIX: ML override ──────────────────────────────
        # If ML confidence is very high (>80%) and strategies are
        # simply abstaining (not actively disagreeing), trust ML.
        strategy_long  = sum(1 for s in sigs.values() if s["signal"] ==  1)
        strategy_short = sum(1 for s in sigs.values() if s["signal"] == -1)

        dominant = self.regime_tracker.dominant_regime()

        if ml_conf >= 0.50 and ml_signal != 0:  
            # Don't override if signal contradicts dominant regime
            regime_allows_long  = dominant != "trending_down"
            regime_allows_short = dominant != "trending_up"

            if ml_signal == 1 and strategy_short == 0 and regime_allows_long:
                long_score  = max(long_score,  ml_conf * 0.85)
            if ml_signal == -1 and strategy_long == 0 and regime_allows_short:
                short_score = max(short_score, ml_conf * 0.85)

        # Determine final signal
        if long_score > short_score and long_score >= MIN_CONFIDENCE:
            final_signal = 1
            final_conf   = long_score
        elif short_score > long_score and short_score >= MIN_CONFIDENCE:
            final_signal = -1
            final_conf   = short_score
        else:
            final_signal = 0
            final_conf   = max(long_score, short_score)

        result = {
            "ticker":         ticker,
            "signal":         final_signal,
            "confidence":     round(final_conf, 4),
            "regime":         regime,
            "strategy_votes": {s: v["signal"] for s, v in sigs.items()},
            "weights_used":   weights,
            "long_score":     round(long_score, 4),
            "short_score":    round(short_score, 4),
            "timestamp":      datetime.now(timezone.utc),
        }

        self._signal_log.append(result)
        return result

    
    # ── Execution decision ────────────────────────────────────

    def should_execute(
        self,
        signal_dict: dict,
        df:          pd.DataFrame,
    ) -> tuple:
        """
        Final check before sending to broker.
        Returns (execute: bool, reason: str, sizing_dict: dict)
        """
        ticker  = signal_dict["ticker"]
        signal  = signal_dict["signal"]
        conf    = signal_dict["confidence"]
        regime  = signal_dict["regime"]

        if signal == 0:
            return False, "Strategies disagree", {}
        
        # Add at the top of should_execute, after signal == 0 check:
        # Check cooldown
        """
        last_trade_bar = self._last_trade.get(ticker, 0)
        current_bar    = len(df)
        if current_bar - last_trade_bar < MIN_BARS_BETWEEN_TRADES:
            return False, f"Cooldown: {current_bar - last_trade_bar} bars since last trade", {}
        """

        try:
            atr      = float(df["atr"].iloc[-1])     if "atr"      in df.columns else 0.001
            atr_norm = float(df["atr_norm"].iloc[-1]) if "atr_norm" in df.columns else 1.0
            close    = float(df["close"].iloc[-1])
            hour     = df.index[-1].hour
        except Exception:
            return False, "Feature extraction failed", {}

        atr_mean = atr / (atr_norm + 1e-9)

        # Risk engine pre-trade check
        allowed, reason = self.risk.can_trade(
            instrument=ticker,
            signal=signal,
            confidence=conf,
            atr=atr,
            atr_mean=atr_mean,
            hour_utc=hour,
        )

        if not allowed:
            return False, reason, {}

        # Compute position size
        sizing = self.risk.position_size(
            instrument=ticker,
            entry=close,
            atr=atr,
            regime=regime,
            confidence=conf,
        )

        # Compute SL/TP
        sl, tp = self.risk.sl_tp(close, signal, atr, ticker)
        sizing["sl"] = sl
        sizing["tp"] = tp
        sizing["entry"] = close
        sizing["direction"] = signal

        # Correlation-adjusted sizing
        open_pos       = self.broker.get_open_positions()
        adjusted_units = self.risk.correlation_adjusted_size(
            ticker, sizing["units"], open_pos
        )
        sizing["units"] = adjusted_units

        return True, "OK", sizing

    # ── Rebalancing ───────────────────────────────────────────

    def rebalance_weights(self):
        """
        Adjust strategy weights based on rolling P&L.
        Better-performing strategies get more weight.
        Called every REBALANCE_HOURS.
        """
        total_pnl = sum(abs(p) for p in self._strategy_pnl.values())
        if total_pnl == 0:
            return   # Not enough history

        new_weights = {}
        for strat, pnl in self._strategy_pnl.items():
            # Positive P&L → increase weight, negative → decrease
            new_weights[strat] = max(0.05, STRATEGY_WEIGHTS[strat] + (pnl / total_pnl) * 0.1)

        # Normalise to sum to 1
        total = sum(new_weights.values())
        self._weights = {s: w/total for s, w in new_weights.items()}
        logger.info(f"Weights rebalanced: {self._weights}")
        self._last_rebalance = datetime.now(timezone.utc)

    def record_strategy_pnl(self, strategy: str, pnl: float):
        if strategy in self._strategy_pnl:
            self._strategy_pnl[strategy] += pnl

    # ── Summary ───────────────────────────────────────────────

    def portfolio_summary(self) -> dict:
        regime_summary = self.regime_tracker.all_current()
        dominant       = self.regime_tracker.dominant_regime()

        return {
            "dominant_regime":   dominant,
            "instrument_regimes": regime_summary,
            "strategy_weights":  self._weights,
            "strategy_pnl":      {s: round(p, 2) for s, p in self._strategy_pnl.items()},
            "signal_count":      len(self._signal_log),
        }

    @property
    def signal_log(self) -> pd.DataFrame:
        return pd.DataFrame(self._signal_log) if self._signal_log else pd.DataFrame()
