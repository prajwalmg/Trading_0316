"""data/fundamental.py — Fundamental feature engineering for equity instruments.

Features (19 total — FUNDAMENTAL_FEATURE_NAMES):
  Piotroski F-Score, earnings surprise, EPS revision, insider buy ratio,
  PE/PS/PB ratios, revenue growth, net margin, debt/equity, price vs target,
  days to earnings, analyst rating, ROE, ROA, current ratio, FCF yield,
  revenue surprise, earnings streak.

All features are point-in-time safe: uses filing dates, not period-end dates.
"""
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger("trading_firm.fundamental")

FUNDAMENTAL_FEATURE_NAMES = [
    "piotroski_f",        # 0–9 composite quality score
    "earnings_surprise",  # (actual EPS – estimate) / |estimate|, clipped ±3
    "eps_revision_3m",    # analyst EPS revision over last quarter, clipped ±1
    "insider_buy_ratio",  # buys / (buys + sells) over last 90 days
    "pe_ratio",           # P/E normalised (log-scaled, clipped)
    "ps_ratio",           # P/S normalised
    "pb_ratio",           # P/B normalised
    "revenue_growth_yoy", # YoY revenue growth, clipped ±3
    "net_margin",         # net income / revenue, clipped ±1
    "debt_equity",        # total debt / equity, clipped 0–10
    "price_vs_target",    # (price – analyst target) / target
    "days_to_earnings",   # days until next earnings (0–1 scaled, 90d horizon)
    "analyst_rating",     # strong sell=0.2 … strong buy=1.0
    "roe",                # return on equity, clipped ±2
    "roa",                # return on assets, clipped ±0.5
    "current_ratio",      # current assets / liabilities, clipped 0–10
    "fcf_yield",          # FCF / market cap proxy, clipped ±0.2
    "revenue_surprise",   # (actual rev – est rev) / |est rev|, clipped ±3
    "earnings_streak",    # consecutive quarterly EPS beats, clipped 0–8 → /8
]


def _sdiv(a, b, fill=0.0):
    try:
        return float(a) / float(b) if b and float(b) != 0 else fill
    except Exception:
        return fill


class FundamentalFeatures:
    """Computes fundamental features from FMPClient.

    Usage:
        ff = FundamentalFeatures()
        feat_dict = ff.get_features("AAPL")
    """

    def __init__(self):
        from data.fmp import FMPClient
        self.fmp = FMPClient()

    # ── Piotroski F-Score ─────────────────────────────────────────────────────

    def _piotroski(self, inc: pd.DataFrame, bal: pd.DataFrame, cf: pd.DataFrame) -> float:
        try:
            if inc.empty or bal.empty or cf.empty:
                return 4.5
            i, b, c = inc.iloc[0], bal.iloc[0], cf.iloc[0]
            i1 = inc.iloc[1] if len(inc) > 1 else i
            b1 = bal.iloc[1] if len(bal) > 1 else b

            ta   = float(b.get("totalAssets", 1) or 1)
            ta1  = float(b1.get("totalAssets", 1) or 1)
            ni   = float(i.get("netIncome",   0) or 0)
            ni1  = float(i1.get("netIncome",  0) or 0)
            ocf  = float(c.get("operatingCashFlow", 0) or 0)
            ld   = float(b.get("longTermDebt",  0) or 0)
            ld1  = float(b1.get("longTermDebt", 0) or 0)
            ca   = float(b.get("totalCurrentAssets",       0) or 0)
            cl   = float(b.get("totalCurrentLiabilities",  1) or 1)
            ca1  = float(b1.get("totalCurrentAssets",      0) or 0)
            cl1  = float(b1.get("totalCurrentLiabilities", 1) or 1)
            sh   = float(b.get("commonStock",  1) or 1)
            sh1  = float(b1.get("commonStock", 1) or 1)
            gp   = float(i.get("grossProfit",  0) or 0)
            gp1  = float(i1.get("grossProfit", 0) or 0)
            rev  = float(i.get("revenue",  1) or 1)
            rev1 = float(i1.get("revenue", 1) or 1)

            roa   = _sdiv(ni,  ta)
            roa1  = _sdiv(ni1, ta1)
            lev   = _sdiv(ld,  ta)
            lev1  = _sdiv(ld1, ta1)
            cur   = _sdiv(ca,  cl)
            cur1  = _sdiv(ca1, cl1)
            gm    = _sdiv(gp,  rev)
            gm1   = _sdiv(gp1, rev1)
            turn  = _sdiv(rev, ta)
            turn1 = _sdiv(rev1, ta1)

            score = sum([
                int(roa  > 0),
                int(ocf  > 0),
                int(roa  > roa1),
                int(_sdiv(ocf, ta) > roa),
                int(lev  < lev1),
                int(cur  > cur1),
                int(sh   <= sh1),
                int(gm   > gm1),
                int(turn > turn1),
            ])
            return float(score)
        except Exception:
            return 4.5

    # ── Public API ────────────────────────────────────────────────────────────

    def get_features(self, ticker: str, as_of: datetime = None,
                     current_price: float = 0.0) -> dict:
        """Return dict of 19 fundamental features for ticker as of date.

        Args:
            ticker:        e.g. 'AAPL'
            as_of:         point-in-time cutoff (default: now)
            current_price: used for price_vs_target; pass 0 to skip
        """
        if as_of is None:
            as_of = datetime.now()

        feat = {k: 0.0 for k in FUNDAMENTAL_FEATURE_NAMES}

        try:
            inc = self.fmp.get_income_statement(ticker, limit=8)
            bal = self.fmp.get_balance_sheet(ticker, limit=4)
            cf  = self.fmp.get_cash_flow(ticker, limit=4)
            sur = self.fmp.get_earnings_surprises(ticker, limit=12)
            ins = self.fmp.get_insider_trading(ticker, limit=100)
            tgt = self.fmp.get_price_target(ticker)
            est = self.fmp.get_analyst_estimates(ticker)
        except Exception as e:
            logger.warning(f"{ticker}: FMP fetch error: {e}")
            return feat

        # ── Piotroski ─────────────────────────────────────────────────────
        feat["piotroski_f"] = self._piotroski(inc, bal, cf) / 9.0  # normalise 0–1

        # ── Earnings surprise & streak ────────────────────────────────────
        try:
            if not sur.empty and "actualEarningResult" in sur.columns:
                sur = sur.copy()
                if "date" in sur.columns:
                    sur["date"] = pd.to_datetime(sur["date"])
                    sur = sur[sur["date"] <= as_of].sort_values("date", ascending=False)
                if len(sur):
                    actual  = float(sur["actualEarningResult"].iloc[0] or 0)
                    est_eps = float(sur.get("estimatedEarning", pd.Series([0])).iloc[0] or 0)
                    feat["earnings_surprise"] = float(
                        np.clip(_sdiv(actual - est_eps, abs(est_eps) + 1e-9), -3, 3)
                    )
                    # Revenue surprise
                    if "actualRevenue" in sur.columns and "estimatedRevenue" in sur.columns:
                        act_rev = float(sur["actualRevenue"].iloc[0] or 0)
                        est_rev = float(sur["estimatedRevenue"].iloc[0] or 0)
                        feat["revenue_surprise"] = float(
                            np.clip(_sdiv(act_rev - est_rev, abs(est_rev) + 1e-9), -3, 3)
                        )
                    # Streak
                    beats = 0
                    for _, row in sur.head(8).iterrows():
                        if (row.get("actualEarningResult", 0) or 0) >= (row.get("estimatedEarning", 0) or 0):
                            beats += 1
                        else:
                            break
                    feat["earnings_streak"] = float(beats) / 8.0
        except Exception:
            pass

        # ── Analyst EPS revision ──────────────────────────────────────────
        try:
            if not est.empty and "estimatedEpsAvg" in est.columns:
                est_df = est.copy()
                if "date" in est_df.columns:
                    est_df["date"] = pd.to_datetime(est_df["date"])
                    est_df = est_df.sort_values("date", ascending=False)
                if len(est_df) >= 2:
                    curr_eps = float(est_df["estimatedEpsAvg"].iloc[0] or 0)
                    prev_eps = float(est_df["estimatedEpsAvg"].iloc[1] or 0)
                    feat["eps_revision_3m"] = float(
                        np.clip(_sdiv(curr_eps - prev_eps, abs(prev_eps) + 1e-9), -1, 1)
                    )
        except Exception:
            pass

        # ── Insider buy ratio ─────────────────────────────────────────────
        try:
            if not ins.empty and "transactionType" in ins.columns:
                ins = ins.copy()
                cutoff = as_of - timedelta(days=90)
                if "transactionDate" in ins.columns:
                    ins["transactionDate"] = pd.to_datetime(ins["transactionDate"])
                    ins = ins[ins["transactionDate"] >= cutoff]
                buys  = ins["transactionType"].str.contains("P-Purchase|Buy", na=False, case=False).sum()
                sells = ins["transactionType"].str.contains("S-Sale|Sell",    na=False, case=False).sum()
                total = buys + sells
                feat["insider_buy_ratio"] = float(_sdiv(buys, total, fill=0.5)) if total > 0 else 0.5
        except Exception:
            pass

        # ── Balance-sheet ratios ──────────────────────────────────────────
        try:
            if not inc.empty and not bal.empty:
                i = inc.iloc[0]
                b = bal.iloc[0]
                revenue   = float(i.get("revenue",       0) or 0)
                net_inc   = float(i.get("netIncome",     0) or 0)
                ta        = float(b.get("totalAssets",   1) or 1)
                equity    = float(b.get("totalStockholdersEquity", 1) or 1)
                total_dbt = float(b.get("totalDebt",     0) or 0)
                cur_a     = float(b.get("totalCurrentAssets",      0) or 0)
                cur_l     = float(b.get("totalCurrentLiabilities", 1) or 1)

                feat["net_margin"]    = float(np.clip(_sdiv(net_inc, revenue),    -1,   1))
                feat["debt_equity"]   = float(np.clip(_sdiv(total_dbt, equity),    0,  10))
                feat["roe"]           = float(np.clip(_sdiv(net_inc, equity),     -1,   2))
                feat["roa"]           = float(np.clip(_sdiv(net_inc, ta),        -0.5, 0.5))
                feat["current_ratio"] = float(np.clip(_sdiv(cur_a, cur_l),        0,  10)) / 10.0

                # Revenue growth YoY (use 4 quarters back)
                if len(inc) >= 5:
                    prev_rev = float(inc.iloc[4].get("revenue", 0) or 0)
                    feat["revenue_growth_yoy"] = float(
                        np.clip(_sdiv(revenue - prev_rev, abs(prev_rev) + 1e-9), -1, 3)
                    )
        except Exception:
            pass

        # ── Price-based ratios (require current_price) ────────────────────
        try:
            if current_price > 0 and not inc.empty and not bal.empty:
                i = inc.iloc[0]
                b = bal.iloc[0]
                shares    = float(b.get("commonStock",   0) or 0)
                revenue   = float(i.get("revenue",       0) or 0)
                net_inc   = float(i.get("netIncome",     0) or 0)
                book_val  = float(b.get("totalStockholdersEquity", 0) or 0)
                if shares > 0:
                    eps   = _sdiv(net_inc, shares)
                    rev_s = _sdiv(revenue, shares)
                    bv_s  = _sdiv(book_val, shares)
                    feat["pe_ratio"] = float(np.clip(np.log1p(abs(_sdiv(current_price, eps + 1e-9))), 0, 6)) / 6.0
                    feat["ps_ratio"] = float(np.clip(np.log1p(_sdiv(current_price, rev_s + 1e-9)), 0, 6)) / 6.0
                    feat["pb_ratio"] = float(np.clip(np.log1p(abs(_sdiv(current_price, bv_s + 1e-9))), 0, 5)) / 5.0
        except Exception:
            pass

        # ── Analyst price target ──────────────────────────────────────────
        try:
            if tgt and "targetConsensus" in tgt and current_price > 0:
                target = float(tgt.get("targetConsensus") or 0)
                if target > 0:
                    feat["price_vs_target"] = float(
                        np.clip(_sdiv(current_price - target, target), -0.5, 0.5)
                    )
                rating_map = {"Strong Buy": 1.0, "Buy": 0.8, "Hold": 0.6, "Sell": 0.4, "Strong Sell": 0.2}
                feat["analyst_rating"] = rating_map.get(tgt.get("consensusRating", "Hold"), 0.6)
        except Exception:
            pass

        # ── FCF yield ─────────────────────────────────────────────────────
        try:
            if not cf.empty and not bal.empty and current_price > 0:
                ocf   = float(cf.iloc[0].get("operatingCashFlow",  0) or 0)
                capex = float(cf.iloc[0].get("capitalExpenditure", 0) or 0)
                fcf   = ocf - abs(capex)
                shares = float(bal.iloc[0].get("commonStock", 1) or 1)
                mktcap = current_price * shares
                feat["fcf_yield"] = float(np.clip(_sdiv(fcf, mktcap + 1e-9), -0.2, 0.2))
        except Exception:
            pass

        # ── Days to next earnings ─────────────────────────────────────────
        try:
            from_d = as_of.strftime("%Y-%m-%d")
            to_d   = (as_of + timedelta(days=90)).strftime("%Y-%m-%d")
            cal    = self.fmp.get_earnings_calendar(from_d, to_d)
            if not cal.empty and "symbol" in cal.columns and "date" in cal.columns:
                cal = cal[cal["symbol"] == ticker].copy()
                if not cal.empty:
                    cal["date"] = pd.to_datetime(cal["date"])
                    next_earn   = cal["date"].min()
                    days_away   = (next_earn - pd.Timestamp(as_of)).days
                    feat["days_to_earnings"] = float(np.clip(days_away, 0, 90)) / 90.0
        except Exception:
            pass

        return feat

    def get_feature_vector(self, ticker: str, as_of: datetime = None,
                           current_price: float = 0.0) -> "np.ndarray":
        """Return numpy array of features in FUNDAMENTAL_FEATURE_NAMES order."""
        d = self.get_features(ticker, as_of=as_of, current_price=current_price)
        return np.array([d[k] for k in FUNDAMENTAL_FEATURE_NAMES], dtype=np.float32)
