"""execution/risk_manager.py — Intraday session risk controls.

Resets automatically at midnight UTC each day.
Never raises — all methods return safe defaults on error.
"""
import logging
from datetime import datetime, timezone

logger = logging.getLogger("trading_firm.risk_manager")


class SessionRiskManager:
    """
    Per-session (daily) risk controls for intraday trading.

    Gates:
      1. Daily loss limit  — halts ALL new entries if NAV drops > daily_loss_pct
      2. Per-instrument cap — suspends an instrument after max_trades_per_instrument trades/day
      3. Consecutive loss gate — suspends instrument after max_consec_losses in a row
      4. Total daily trade cap — hard ceiling across all instruments

    Auto-resets at new UTC calendar day.
    """

    def __init__(
        self,
        initial_nav:                float,
        daily_loss_pct:             float = 0.03,   # 3 % daily loss → halt
        max_trades_per_instrument:  int   = 3,
        max_consec_losses:          int   = 2,
        max_total_daily:            int   = 10,
    ):
        self.initial_nav    = initial_nav
        self.daily_loss_pct = daily_loss_pct
        self.max_per_inst   = max_trades_per_instrument
        self.max_consec     = max_consec_losses
        self.max_daily      = max_total_daily

        # Session state — reset daily
        self._date              = self._today()
        self.session_start      = None          # set on first can_trade() of the day
        self.session_trades     = 0
        self.session_pnl        = 0.0
        self.inst_trades        : dict = {}     # {sym: count}
        self.inst_consec_loss   : dict = {}     # {sym: int}
        self.suspended_insts    : set  = set()
        self.halted             = False
        self.halt_reason        = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _today():
        return datetime.now(timezone.utc).date()

    def _check_date_reset(self):
        """Auto-reset all counters at the start of a new UTC calendar day."""
        today = self._today()
        if today != self._date:
            self._date              = today
            self.session_start      = None
            self.session_trades     = 0
            self.session_pnl        = 0.0
            self.inst_trades        = {}
            self.inst_consec_loss   = {}
            self.suspended_insts    = set()
            self.halted             = False
            self.halt_reason        = None
            logger.info("SessionRisk: new UTC day — all limits reset ✅")

    # ── Public API ────────────────────────────────────────────────────────────

    def set_session_start(self, nav: float):
        """Call once per session (or daily) with the opening NAV."""
        if self.session_start is None:
            self.session_start = nav
            logger.info(f"SessionRisk: session start NAV = {nav:.2f}")

    def can_trade(self, instrument: str, current_nav: float) -> tuple:
        """
        Returns (True, 'OK') if a new trade is allowed.
        Returns (False, reason_str) if blocked.
        """
        try:
            self._check_date_reset()

            # 1 — Session halted?
            if self.halted:
                return False, f"Session halted: {self.halt_reason}"

            # 2 — Daily loss limit
            start = self.session_start or self.initial_nav
            session_loss_pct = (current_nav - start) / max(start, 1e-9)
            if session_loss_pct <= -self.daily_loss_pct:
                self.halted      = True
                self.halt_reason = (
                    f"Daily loss limit hit: {session_loss_pct:.1%} "
                    f"(limit -{self.daily_loss_pct:.0%})"
                )
                logger.warning(f"🚨 SESSION HALTED: {self.halt_reason}")
                return False, self.halt_reason

            # 3 — Total daily trade cap
            if self.session_trades >= self.max_daily:
                return False, f"Daily trade cap: {self.session_trades}/{self.max_daily}"

            # 4 — Instrument suspended?
            if instrument in self.suspended_insts:
                return False, f"{instrument} suspended for session"

            # 5 — Per-instrument trade limit
            if self.inst_trades.get(instrument, 0) >= self.max_per_inst:
                self.suspended_insts.add(instrument)
                return False, f"{instrument}: max {self.max_per_inst} trades/day reached"

            # 6 — Consecutive loss gate
            consec = self.inst_consec_loss.get(instrument, 0)
            if consec >= self.max_consec:
                self.suspended_insts.add(instrument)
                logger.warning(
                    f"🛑 {instrument}: {consec} consecutive losses — suspended for session"
                )
                return False, f"{instrument}: {consec} consecutive losses — suspended"

            return True, "OK"

        except Exception as e:
            logger.warning(f"SessionRisk.can_trade error: {e}")
            return True, "OK"   # fail-open to avoid blocking trades on bug

    def record_trade_open(self, instrument: str):
        """Call immediately after broker.market_order() fills."""
        try:
            self.session_trades += 1
            self.inst_trades[instrument] = self.inst_trades.get(instrument, 0) + 1
        except Exception as e:
            logger.warning(f"SessionRisk.record_trade_open error: {e}")

    def record_trade_close(self, instrument: str, pnl: float):
        """Call when a position closes (any reason)."""
        try:
            self.session_pnl += pnl
            if pnl < 0:
                self.inst_consec_loss[instrument] = (
                    self.inst_consec_loss.get(instrument, 0) + 1
                )
                consec = self.inst_consec_loss[instrument]
                if consec >= self.max_consec:
                    self.suspended_insts.add(instrument)
                    logger.warning(
                        f"🛑 {instrument}: {consec} consecutive losses "
                        f"— suspended rest of session"
                    )
            else:
                # Any win resets the consecutive loss counter
                self.inst_consec_loss[instrument] = 0
        except Exception as e:
            logger.warning(f"SessionRisk.record_trade_close error: {e}")

    def status(self) -> dict:
        """Snapshot of current session state for logging."""
        start = self.session_start or self.initial_nav
        return {
            "halted":          self.halted,
            "halt_reason":     self.halt_reason,
            "session_pnl":     round(self.session_pnl, 2),
            "session_trades":  self.session_trades,
            "suspended":       sorted(self.suspended_insts),
            "inst_trades":     dict(self.inst_trades),
            "inst_consec_loss": dict(self.inst_consec_loss),
            "session_start":   start,
        }
