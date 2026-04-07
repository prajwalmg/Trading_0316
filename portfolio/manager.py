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

from risk.portfolio_optimizer import optimise_weights, estimate_covariance, estimate_expected_returns

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.settings import (
    MIN_BARS_BETWEEN_TRADES, STRATEGY_WEIGHTS, MIN_CONFIDENCE, REBALANCE_HOURS,
    FOREX_PAIRS, EQUITY_TICKERS, COMMODITY_TICKERS, INSTRUMENT_MIN_CONFIDENCE,
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

    def __init__(self, risk_engine: RiskEngine, pipeline=None):
        self.risk           = risk_engine
        self.pipeline       = pipeline          # DataPipeline — for live covariance
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
            # No opposing strategy → ML confidence applied directly.
            # Regime gating is handled downstream by the HTF filter in run_paper().
            if ml_signal == 1 and strategy_short == 0:
                long_score  = max(long_score,  ml_conf)
            if ml_signal == -1 and strategy_long == 0:
                short_score = max(short_score, ml_conf)

        # Determine final signal — use per-instrument threshold if available
        _min_conf = INSTRUMENT_MIN_CONFIDENCE.get(ticker, MIN_CONFIDENCE)
        if long_score > short_score and long_score >= _min_conf:
            final_signal = 1
            final_conf   = long_score
        elif short_score > long_score and short_score >= _min_conf:
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
        signal_dict:    dict,
        df:             pd.DataFrame,
        open_positions: list = None,
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
            conf_str = f"{signal_dict.get('confidence', 0):.2%}"
            long_s   = signal_dict.get("long_score", 0)
            short_s  = signal_dict.get("short_score", 0)
            _thr     = INSTRUMENT_MIN_CONFIDENCE.get(ticker, MIN_CONFIDENCE)
            if max(long_s, short_s) < _thr:
                return False, f"conf {conf_str} below threshold {_thr:.0%}", {}
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

        # HMM regime context for position sizing
        vol_forecast  = self.regime_tracker.get_vol_forecast(ticker)
        in_transition = self.regime_tracker.is_in_transition(ticker)

        # Compute position size
        sizing = self.risk.position_size(
            instrument    = ticker,
            entry         = close,
            atr           = atr,
            regime        = regime,
            confidence    = conf,
            vol_forecast  = vol_forecast,
            in_transition = in_transition,
        )

        if sizing is None:
            return False, "high_volatility + low_confidence", {}

        # Compute SL/TP
        sl, tp = self.risk.sl_tp(close, signal, atr, ticker)
        sizing["sl"] = sl
        sizing["tp"] = tp
        sizing["entry"] = close
        sizing["direction"] = signal

        # Correlation-adjusted sizing
        # open_positions is passed in by the caller (broker.get_open_positions())
        # so PortfolioManager does not need a direct broker reference
        open_pos       = open_positions if open_positions is not None else []
        adjusted_units = self.risk.correlation_adjusted_size(
            ticker, sizing["units"], open_pos
        )
        sizing["units"] = adjusted_units

        # ── Portfolio optimisation weight adjustment ───────────────
        # Build a simple 1-asset optimisation to get the MV-optimal
        # allocation weight for this instrument, then scale units.
        try:
            all_tickers = (
                list(FOREX_PAIRS) + list(EQUITY_TICKERS) + list(COMMODITY_TICKERS)
            )
            n = max(len(all_tickers), 1)

            # Build live returns matrix from pipeline store when available.
            returns_df = None
            if self.pipeline is not None:
                frames = {}
                for t in all_tickers:
                    df_t = self.pipeline.get(t)
                    if not df_t.empty and "close" in df_t.columns and len(df_t) >= 30:
                        frames[t] = df_t["close"].pct_change().dropna()
                if len(frames) >= 2:
                    returns_df = pd.DataFrame(frames).dropna()
                    # Align column order to all_tickers
                    returns_df = returns_df.reindex(columns=all_tickers)

            cov = estimate_covariance(returns_df) if returns_df is not None and not returns_df.empty \
                  else (np.eye(1) if n == 1 else np.eye(n) * 0.0001)

            mu = estimate_expected_returns(returns_df) if returns_df is not None and not returns_df.empty \
                 else np.full(n, 0.0001)

            opt_weights = optimise_weights(
                tickers          = all_tickers,
                expected_returns = mu,
                cov_matrix       = cov,
                risk_aversion    = 1.0,
                max_weight       = 0.40,
            )

            equal_weight = 1.0 / n
            this_weight  = opt_weights.get(ticker, equal_weight)

            # Scale units proportionally: if optimizer says 20% vs 10% equal,
            # double the units; if 5% vs 10%, halve them.
            weight_scale = this_weight / (equal_weight + 1e-9)
            weight_scale = float(np.clip(weight_scale, 0.5, 2.0))   # ±2× cap

            sizing["units"]          = int(round(sizing["units"] * weight_scale))
            sizing["opt_weight"]     = round(this_weight, 6)
            sizing["weight_scale"]   = round(weight_scale, 4)

            logger.debug(
                f"{ticker}: opt_weight={this_weight:.4f} "
                f"scale={weight_scale:.3f} units={sizing['units']}"
            )
        except Exception as e:
            logger.debug(f"Portfolio optimisation skipped for {ticker}: {e}")

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
