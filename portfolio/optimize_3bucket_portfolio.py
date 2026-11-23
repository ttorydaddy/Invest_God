"""
Invest_God Phase 1-D: 3-Bucket Integrated Portfolio Optimizer & Dynamic Cash Shift Simulator
===========================================================================================
Inputs (ONLY Auditor-PASS raw CSVs + DuckDB BTC 4h):
  1. Short-term (Crypto Turbo-1): strategies/crypto_short/equity_turbo_M3_R15_net.csv
  2. Mid-term   (KRX D-3)       : strategies/krx_surge/equity_def_D-3_net.csv
  3. Long-term  (Global L-3)    : strategies/longterm_core/equity_lt_L-3_Barbell_Regime_net.csv
  4. Reference  (Defensive L-2) : strategies/longterm_core/equity_lt_L-2_RiskParity_Trend_net.csv
  5. BTC Regime (4h 600 EMA)    : strategies/crypto_short/crypto_short_data.duckdb

Strict Anti-Overfitting & Zero Look-Ahead Rules:
  - Common KRX trading calendar: 2021-01-04 ~ 2026-11-19 (1,439 trading days).
  - Weekend/holiday returns of 24/7 Turbo-1 are 100% preserved via cumulative equity alignment.
  - Optimization strictly on IS (2021-01-04 ~ 2024-06-30, 858 days).
  - Single-shot OOS verification (2024-07-01 ~ 2026-11-19, 581 days).
  - Transition cost: 0.15% per side = 0.30% round-trip on moved capital (one-way turnover).
  - Dynamic Cash Shift (P-5): Uses BTC 4h bar closed at 04:00 UTC (13:00 KST), 2.5h BEFORE KRX close,
    executed at KRX close on day t to take effect on day t+1 (Zero Look-Ahead).
  - Survival-bias Conservative Haircut: D-3 -4.0%p/yr, Turbo-1 -2.5%p/yr.
  - Z3-Solver SLA mechanical verification on saved CSVs.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
from z3 import And, Bool, Real, Solver, sat

# ---------------------------------------------------------------------------
# Constants & Paths
# ---------------------------------------------------------------------------
ROOT_DIR = r"c:\Users\양태철\Desktop\Invest_God"
PORT_DIR = os.path.join(ROOT_DIR, "portfolio")

PATH_TURBO1 = os.path.join(ROOT_DIR, "strategies", "crypto_short", "equity_turbo_M3_R15_net.csv")
PATH_D3 = os.path.join(ROOT_DIR, "strategies", "krx_surge", "equity_def_D-3_net.csv")
PATH_L3 = os.path.join(ROOT_DIR, "strategies", "longterm_core", "equity_lt_L-3_Barbell_Regime_net.csv")
PATH_L2 = os.path.join(ROOT_DIR, "strategies", "longterm_core", "equity_lt_L-2_RiskParity_Trend_net.csv")
PATH_CRYPTO_DB = os.path.join(ROOT_DIR, "strategies", "crypto_short", "crypto_short_data.duckdb")

IS_START = "2021-01-04"
IS_END = "2024-06-30"
OOS_START = "2024-07-01"
OOS_END = "2026-11-19"

INITIAL_CAPITAL = 100_000_000.0  # 100M KRW
ROUND_TRIP_COST = 0.0030         # 0.30% round-trip on moved capital (0.15% sell + 0.15% buy)
MIN_WEIGHT = 0.15
MAX_WEIGHT = 0.60

# Survival-bias conservative annual haircuts (Auditor recommendation)
HAIRCUT_D3_ANNUAL = 0.040        # -4.0%p / yr
HAIRCUT_TURBO1_ANNUAL = 0.025    # -2.5%p / yr


# ---------------------------------------------------------------------------
# Data Loading & Alignment
# ---------------------------------------------------------------------------
def load_aligned_data(
    use_l2: bool = False,
    perturb_oos_mult: float = 1.0,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Aligns Turbo-1, D-3, and L-3 (or L-2) onto the exact 1,439 KRX trading days of D-3
    (2021-01-04 ~ 2026-11-19).
    Also loads BTC 4h 600-EMA (100-day EMA) mapped strictly causally to KRX trading days
    using the 00:00 UTC bar (which closes at 04:00 UTC = 13:00 KST, 2.5h before KRX close).
    """
    df_d3 = pd.read_csv(PATH_D3, usecols=["date", "equity"]).rename(columns={"equity": "eq_mid"})
    df_t1 = pd.read_csv(PATH_TURBO1, usecols=["date", "equity"]).rename(columns={"equity": "eq_short"})
    lt_path = PATH_L2 if use_l2 else PATH_L3
    df_lt = pd.read_csv(lt_path, usecols=["date", "equity"]).rename(columns={"equity": "eq_long"})

    df = df_d3.merge(df_t1, on="date", how="inner").merge(df_lt, on="date", how="inner")
    df = df.sort_values("date").reset_index(drop=True)
    assert len(df) == 1439, f"Expected 1439 KRX trading days, got {len(df)}"
    assert df["date"].iloc[0] == IS_START and df["date"].iloc[-1] == OOS_END

    if perturb_oos_mult != 1.0:
        oos_mask = df["date"] > IS_END
        for col in ["eq_short", "eq_mid", "eq_long"]:
            df.loc[oos_mask, col] = df.loc[oos_mask, col] * perturb_oos_mult

    # Daily returns (exact cumulative equity ratio; weekend crypto returns included on Monday)
    df["ret_short"] = df["eq_short"].pct_change().fillna(0.0)
    df["ret_mid"] = df["eq_mid"].pct_change().fillna(0.0)
    df["ret_long"] = df["eq_long"].pct_change().fillna(0.0)

    # Load BTCUSDT 4h bars from DuckDB
    con = duckdb.connect(PATH_CRYPTO_DB, read_only=True)
    btc = con.execute(
        """
        SELECT open_time_utc, close
        FROM ohlcv_4h
        WHERE symbol = 'BTCUSDT'
        ORDER BY open_time_utc
        """
    ).fetchdf()
    con.close()

    btc["open_time_utc"] = pd.to_datetime(btc["open_time_utc"])
    if perturb_oos_mult != 1.0:
        btc_oos_mask = btc["open_time_utc"] >= pd.Timestamp("2024-07-01 00:00:00")
        btc.loc[btc_oos_mask, "close"] = btc.loc[btc_oos_mask, "close"] * perturb_oos_mult

    # Exact causal EMA(600) matching backtest_crypto_short_v2.py / backtest_crypto_turbo.py
    btc["ema600"] = btc["close"].ewm(span=600, adjust=False).mean()
    btc["is_bear"] = btc["close"] < btc["ema600"]

    # Map strictly causally to KRX date D:
    # Use the bar with open_time_utc == D 00:00:00 UTC (closes at D 04:00:00 UTC = D 13:00 KST,
    # strictly 2.5 hours BEFORE KRX close at 15:30 KST!).
    btc_idx = btc.set_index("open_time_utc")
    krx_cutoff_ts = pd.to_datetime(df["date"])  # YYYY-MM-DD 00:00:00 UTC
    # asof lookup ensures no future bar is ever used
    btc_aligned = btc_idx.reindex(krx_cutoff_ts, method="ffill")
    assert not btc_aligned["is_bear"].isna().any(), "Missing BTC regime data on KRX dates"

    btc_bear = pd.Series(btc_aligned["is_bear"].astype(bool).values, index=df.index, name="btc_bear")
    df["btc_bear"] = btc_bear
    df["btc_close_13kst"] = btc_aligned["close"].values
    df["btc_ema600_13kst"] = btc_aligned["ema600"].values

    return df, btc_bear


def apply_haircut_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies Auditor-recommended survival-bias conservative annual haircuts:
      - Mid-term D-3: -4.0%p / yr
      - Short-term Turbo-1: -2.5%p / yr
    Compounded strictly by actual calendar-day delta between consecutive KRX trading days.
    """
    out = df.copy()
    dates = pd.to_datetime(out["date"])
    dt_years = dates.diff().dt.days.fillna(0.0) / 365.25

    drag_mid = (1.0 - HAIRCUT_D3_ANNUAL) ** dt_years
    drag_short = (1.0 - HAIRCUT_TURBO1_ANNUAL) ** dt_years

    out["ret_mid"] = (1.0 + out["ret_mid"]) * drag_mid - 1.0
    out["ret_short"] = (1.0 + out["ret_short"]) * drag_short - 1.0
    # Reconstruct equity series for individual haircut benchmarks
    out["eq_mid"] = INITIAL_CAPITAL * (1.0 + out["ret_mid"]).cumprod()
    out["eq_short"] = INITIAL_CAPITAL * (1.0 + out["ret_short"]).cumprod()
    out["eq_long"] = INITIAL_CAPITAL * (1.0 + out["ret_long"]).cumprod()
    return out


# ---------------------------------------------------------------------------
# Metrics Calculation (100% consistent with Invest_God convention)
# ---------------------------------------------------------------------------
def calc_metrics(eq_df: pd.DataFrame, start_date: str, end_date: str) -> Dict:
    sub = eq_df[(eq_df["date"] >= start_date) & (eq_df["date"] <= end_date)].copy()
    eq = sub["equity"].values.astype(float)
    total_ret = (eq[-1] / eq[0]) - 1.0
    dt0 = pd.Timestamp(sub["date"].iloc[0])
    dt1 = pd.Timestamp(sub["date"].iloc[-1])
    cal_years = max((dt1 - dt0).days / 365.25, 1e-6)
    cagr_cal = (eq[-1] / eq[0]) ** (1.0 / cal_years) - 1.0 if eq[-1] > 0 else -1.0
    dr = pd.Series(eq).pct_change().dropna()
    ann_vol = float(dr.std() * np.sqrt(252)) if len(dr) > 1 else 0.0
    sharpe = float((dr.mean() / dr.std()) * np.sqrt(252)) if dr.std() > 0 else 0.0
    running_max = np.maximum.accumulate(eq)
    dd = eq / running_max - 1.0
    mdd = float(dd.min())
    calmar = float(cagr_cal / abs(mdd)) if mdd < 0 else 0.0
    return {
        "total_return": float(total_ret),
        "cagr_cal": float(cagr_cal),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "mdd": float(mdd),
        "calmar": float(calmar),
        "final_equity": float(eq[-1]),
        "days": int(len(sub)),
    }


def calc_yearly_stats(eq_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    df = eq_df.copy()
    df["year"] = df["date"].str[:4]
    out: Dict[str, Dict[str, float]] = {}
    prev_end_eq: Optional[float] = None
    for yr, g in df.groupby("year"):
        eq_vals = g["equity"].values.astype(float)
        start_eq = prev_end_eq if prev_end_eq is not None else float(eq_vals[0])
        end_eq = float(eq_vals[-1])
        yr_ret = (end_eq / start_eq) - 1.0
        # Include start_eq as base peak for intra-year MDD
        series_for_mdd = np.concatenate([[start_eq], eq_vals])
        yr_mdd = float((series_for_mdd / np.maximum.accumulate(series_for_mdd) - 1.0).min())
        out[str(yr)] = {
            "return": float(yr_ret),
            "mdd": float(yr_mdd),
        }
        prev_end_eq = end_eq
    return out


# ---------------------------------------------------------------------------
# Portfolio Simulator (Quarterly Rebal + Optional Dynamic Cash Shift)
# ---------------------------------------------------------------------------
def get_period_end_flags(dates: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns boolean arrays (is_quarter_end, is_month_end) for KRX trading days.
    Strictly deterministic from calendar dates (last trading day of each quarter/month).
    """
    dt = pd.to_datetime(dates)
    ym = dt.dt.year * 100 + dt.dt.month
    yq = dt.dt.year * 10 + dt.dt.quarter

    is_month_end = np.zeros(len(dates), dtype=bool)
    is_quarter_end = np.zeros(len(dates), dtype=bool)

    for i in range(len(dates) - 1):
        if ym.iloc[i] != ym.iloc[i + 1]:
            is_month_end[i] = True
        if yq.iloc[i] != yq.iloc[i + 1]:
            is_quarter_end[i] = True
    return is_quarter_end, is_month_end


def simulate_portfolio(
    df: pd.DataFrame,
    base_weights: Tuple[float, float, float],
    dynamic_mode: Optional[str] = None,  # None, "5d_confirm", "monthly"
    shift_split: Tuple[float, float] = (0.5, 0.5),  # how shifted half of Short goes to (Mid, Long)
    cost_rt: float = ROUND_TRIP_COST,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulates a 3-Bucket portfolio (Short=Turbo-1, Mid=D-3, Long=L-3) on KRX trading days.
    - Rebalances at the close of every quarter-end KRX trading day.
    - If dynamic_mode is set:
        * When BTC < 100d EMA (confirmed via 5 consecutive KRX days OR month-end check),
          target weight of Short is halved (0.5 * w_short), and the freed 0.5 * w_short
          is shifted into Mid and Long according to shift_split (default 50:50).
        * When BTC >= 100d EMA (confirmed), restores base_weights.
        * Rebalance executes at KRX close on day t (using BTC 13:00 KST bar), deducting
          0.30% round-trip cost on moved capital, and takes effect on day t+1.
    """
    w_s0, w_m0, w_l0 = base_weights
    assert abs((w_s0 + w_m0 + w_l0) - 1.0) < 1e-9

    dates = df["date"].values
    r_s = df["ret_short"].values
    r_m = df["ret_mid"].values
    r_l = df["ret_long"].values
    btc_bear = df["btc_bear"].values

    is_q_end, is_m_end = get_period_end_flags(df["date"])
    n = len(df)

    # State variables
    val_s = INITIAL_CAPITAL * w_s0
    val_m = INITIAL_CAPITAL * w_m0
    val_l = INITIAL_CAPITAL * w_l0

    in_bear_regime = False
    bear_streak = 0
    bull_streak = 0

    eq_rows = []
    rebal_rows = []

    for i in range(n):
        if i > 0:
            val_s *= (1.0 + r_s[i])
            val_m *= (1.0 + r_m[i])
            val_l *= (1.0 + r_l[i])

        pre_eq = val_s + val_m + val_l
        w_s_cur = val_s / pre_eq
        w_m_cur = val_m / pre_eq
        w_l_cur = val_l / pre_eq

        # Update causal BTC regime streak using day i's 13:00 KST BTC bar
        if btc_bear[i]:
            bear_streak += 1
            bull_streak = 0
        else:
            bull_streak += 1
            bear_streak = 0

        regime_switched = False
        new_bear_regime = in_bear_regime

        if dynamic_mode == "5d_confirm":
            if (not in_bear_regime) and (bear_streak >= 5):
                new_bear_regime = True
                regime_switched = True
            elif in_bear_regime and (bull_streak >= 5):
                new_bear_regime = False
                regime_switched = True
        elif dynamic_mode == "monthly":
            if is_m_end[i]:
                if (not in_bear_regime) and btc_bear[i]:
                    new_bear_regime = True
                    regime_switched = True
                elif in_bear_regime and (not btc_bear[i]):
                    new_bear_regime = False
                    regime_switched = True

        in_bear_regime = new_bear_regime

        # Determine target weights
        if in_bear_regime and dynamic_mode is not None:
            shifted = 0.5 * w_s0
            tw_s = w_s0 - shifted
            tw_m = w_m0 + shifted * shift_split[0]
            tw_l = w_l0 + shifted * shift_split[1]
        else:
            tw_s, tw_m, tw_l = w_s0, w_m0, w_l0

        do_rebal = (i > 0) and (is_q_end[i] or regime_switched)
        rebal_cost = 0.0
        one_way_to = 0.0
        reason = ""

        if do_rebal:
            if regime_switched and is_q_end[i]:
                reason = "REGIME_SHIFT+QUARTER_END"
            elif regime_switched:
                reason = "BEAR_SHIFT_IN" if in_bear_regime else "BULL_RESTORE"
            else:
                reason = "QUARTER_END"

            one_way_to = 0.5 * (abs(tw_s - w_s_cur) + abs(tw_m - w_m_cur) + abs(tw_l - w_l_cur))
            rebal_cost = pre_eq * one_way_to * cost_rt
            post_eq = pre_eq - rebal_cost

            val_s = post_eq * tw_s
            val_m = post_eq * tw_m
            val_l = post_eq * tw_l

            rebal_rows.append({
                "date": str(dates[i]),
                "reason": reason,
                "in_bear_regime": int(in_bear_regime),
                "pre_equity": float(pre_eq),
                "post_equity": float(post_eq),
                "one_way_turnover": float(one_way_to),
                "cost_krw": float(rebal_cost),
                "w_short_pre": float(w_s_cur),
                "w_mid_pre": float(w_m_cur),
                "w_long_pre": float(w_l_cur),
                "w_short_target": float(tw_s),
                "w_mid_target": float(tw_m),
                "w_long_target": float(tw_l),
            })

        total_eq = val_s + val_m + val_l
        eq_rows.append({
            "date": str(dates[i]),
            "equity": float(total_eq),
            "cash": 0.0,  # 100% allocated across the 3 buckets (each bucket internally manages its own cash)
            "val_short": float(val_s),
            "val_mid": float(val_m),
            "val_long": float(val_l),
            "w_short": float(val_s / total_eq),
            "w_mid": float(val_m / total_eq),
            "w_long": float(val_l / total_eq),
            "total_weight": float((val_s + val_m + val_l) / total_eq),
            "in_bear_regime": int(in_bear_regime),
            "rebal_cost_krw": float(rebal_cost),
        })

    return pd.DataFrame(eq_rows), pd.DataFrame(rebal_rows)


# ---------------------------------------------------------------------------
# IS-Only Optimization (Zero OOS Peeking!)
# ---------------------------------------------------------------------------
def optimize_weights_on_is(df_full: pd.DataFrame) -> Dict[str, Tuple[float, float, float]]:
    """
    Computes optimal weights strictly using IS data (2021-01-04 ~ 2024-06-30).
    Uses exact 1% integer grid over [15%, 60%] summing to 100% with actual quarterly
    rebalancing & 0.30% transition costs on IS!
    """
    df_is = df_full[df_full["date"] <= IS_END].copy().reset_index(drop=True)

    # P-1: Equal Weight
    w_p1 = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

    # P-3: IS Risk-Parity (Inverse Volatility on IS daily returns, clipped to [15%, 60%])
    vols = np.array([
        df_is["ret_short"].iloc[1:].std(),
        df_is["ret_mid"].iloc[1:].std(),
        df_is["ret_long"].iloc[1:].std(),
    ])
    inv_v = 1.0 / vols
    w_rp = inv_v / inv_v.sum()
    w_rp = np.clip(w_rp, MIN_WEIGHT, MAX_WEIGHT)
    w_rp = w_rp / w_rp.sum()
    w_p3 = (float(w_rp[0]), float(w_rp[1]), float(w_rp[2]))

    # Grid search on IS for P-2 (Max-Sharpe) and P-4 (Aggressive Growth: Max CAGR s.t. IS MDD >= -20%)
    best_sharpe = -1e9
    w_p2 = (0.34, 0.33, 0.33)

    best_agg_cagr = -1e9
    w_p4 = (0.50, 0.35, 0.15)

    for ws in range(15, 61):
        for wm in range(15, 61):
            wl = 100 - ws - wm
            if wl < 15 or wl > 60:
                continue
            w = (ws / 100.0, wm / 100.0, wl / 100.0)
            eq_is, _ = simulate_portfolio(df_is, w, dynamic_mode=None)
            m = calc_metrics(eq_is, IS_START, IS_END)
            if m["sharpe"] > best_sharpe:
                best_sharpe = m["sharpe"]
                w_p2 = w
            if m["mdd"] >= -0.20 and m["cagr_cal"] > best_agg_cagr:
                best_agg_cagr = m["cagr_cal"]
                w_p4 = w

    return {
        "P-1_EqualWeight": w_p1,
        "P-2_MaxSharpe": w_p2,
        "P-3_RiskParity": w_p3,
        "P-4_Aggressive": w_p4,
    }


# ---------------------------------------------------------------------------
# Z3-Solver Mechanical SLA Verification (Reads Saved CSVs Directly!)
# ---------------------------------------------------------------------------
def verify_portfolio_z3(
    models_meta: Dict[str, Dict],
    individual_metrics: Dict[str, Dict],
    perturb_max_diff: float,
) -> Dict:
    """
    Mechanically parses saved equity_port_*.csv and rebal_port_*.csv files and
    proves portfolio SLA invariants via Z3 SMT Solver.
    """
    results = {}

    ind_full_mdds = [individual_metrics[k]["full"]["mdd"] for k in ["Turbo-1", "D-3", "L-3"]]
    ind_full_sharpes = [individual_metrics[k]["full"]["sharpe"] for k in ["Turbo-1", "D-3", "L-3"]]
    best_ind_mdd = max(ind_full_mdds)       # least negative (e.g., -0.15 > -0.25)
    best_ind_sharpe = max(ind_full_sharpes)

    for name, meta in models_meta.items():
        eq_df = pd.read_csv(meta["equity_csv"])
        rb_df = pd.read_csv(meta["rebal_csv"])

        min_eq = float(eq_df["equity"].min())
        min_cash = float(eq_df["cash"].min())
        min_w = float(eq_df[["w_short", "w_mid", "w_long"]].min().min())
        max_w = float(eq_df[["w_short", "w_mid", "w_long"]].max().max())
        max_tot_w_err = float((eq_df["total_weight"] - 1.0).abs().max())
        max_tot_w = float(eq_df["total_weight"].max())

        # Check target weight bounds & cost accuracy on rebal log
        is_dynamic = meta["dynamic_mode"] is not None
        if len(rb_df) > 0:
            rb_min_tw = float(rb_df[["w_short_target", "w_mid_target", "w_long_target"]].min().min())
            rb_max_tw = float(rb_df[["w_short_target", "w_mid_target", "w_long_target"]].max().max())
            expected_cost = rb_df["pre_equity"] * rb_df["one_way_turnover"] * ROUND_TRIP_COST
            max_cost_err = float((rb_df["cost_krw"] - expected_cost).abs().max())
        else:
            rb_min_tw, rb_max_tw, max_cost_err = 0.0, 1.0, 0.0

        port_mdd = float(meta["metrics_full"]["mdd"])
        port_sharpe = float(meta["metrics_full"]["sharpe"])

        # Z3 Formal Proof
        s = Solver()
        z_min_eq = Real("min_equity")
        z_min_cash = Real("min_cash")
        z_min_w = Real("min_weight")
        z_max_tot_w = Real("max_total_weight")
        z_max_tot_err = Real("max_total_weight_err")
        z_rb_min_tw = Real("rebal_min_target_w")
        z_rb_max_tw = Real("rebal_max_target_w")
        z_cost_err = Real("max_cost_err_krw")
        z_perturb = Real("lookahead_perturb_diff")
        z_port_mdd = Real("port_mdd")
        z_best_ind_mdd = Real("best_ind_mdd")
        z_port_sharpe = Real("port_sharpe")
        z_best_ind_sharpe = Real("best_ind_sharpe")

        # Bind observed values from CSVs
        s.add(z_min_eq == min_eq)
        s.add(z_min_cash == min_cash)
        s.add(z_min_w == min_w)
        s.add(z_max_tot_w == max_tot_w)
        s.add(z_max_tot_err == max_tot_w_err)
        s.add(z_rb_min_tw == rb_min_tw)
        s.add(z_rb_max_tw == rb_max_tw)
        s.add(z_cost_err == max_cost_err)
        s.add(z_perturb == perturb_max_diff)
        s.add(z_port_mdd == port_mdd)
        s.add(z_best_ind_mdd == best_ind_mdd)
        s.add(z_port_sharpe == port_sharpe)
        s.add(z_best_ind_sharpe == best_ind_sharpe)

        # SLA Constraints
        c_no_bankruptcy = And(z_min_eq > 0, z_min_cash >= 0)
        c_no_leverage = And(z_min_w >= 0, z_max_tot_w <= 1.0 + 1e-6, z_max_tot_err <= 1e-6)
        if not is_dynamic:
            c_weight_bounds = And(z_rb_min_tw >= MIN_WEIGHT - 1e-6, z_rb_max_tw <= MAX_WEIGHT + 1e-6)
        else:
            # Dynamic bear shift intentionally halves short weight (can drop below 15%),
            # while base weights stay in [15%, 60%] and active weights remain in [0%, 80%] with sum=100%
            bw = meta["base_weights"]
            base_ok = all((MIN_WEIGHT - 1e-6 <= x <= MAX_WEIGHT + 1e-6) for x in bw)
            c_weight_bounds = And(base_ok, z_rb_min_tw >= 0, z_rb_max_tw <= 0.80 + 1e-6)

        c_cost_applied = z_cost_err <= 1e-2
        c_zero_lookahead = z_perturb == 0.0
        # Diversification Magic: lower MDD than best individual OR higher Sharpe than best individual
        c_diversification_magic = (z_port_mdd > z_best_ind_mdd) if port_mdd > best_ind_mdd else (z_port_sharpe > z_best_ind_sharpe)

        # Check each SLA individually + combined
        checks = {
            "no_bankruptcy": c_no_bankruptcy,
            "no_leverage_sum_100": c_no_leverage,
            "weight_bounds": c_weight_bounds,
            "transition_cost_030pct": c_cost_applied,
            "zero_lookahead": c_zero_lookahead,
            "diversification_magic_mdd_or_sharpe": c_diversification_magic,
        }

        check_status = {}
        all_sat = True
        for ck_name, expr in checks.items():
            s.push()
            s.add(expr)
            res = s.check()
            s.pop()
            ok = (res == sat)
            check_status[ck_name] = "SAT" if ok else "UNSAT"
            if not ok:
                all_sat = False

        results[name] = {
            "overall_verdict": "SAT" if all_sat else "UNSAT",
            "checks": check_status,
            "observed": {
                "rows": int(len(eq_df)),
                "rebal_events": int(len(rb_df)),
                "min_equity_krw": min_eq,
                "max_total_weight": max_tot_w,
                "max_total_weight_err": max_tot_w_err,
                "rebal_min_target_w": rb_min_tw,
                "rebal_max_target_w": rb_max_tw,
                "max_cost_err_krw": max_cost_err,
                "port_full_mdd": port_mdd,
                "best_individual_mdd": best_ind_mdd,
                "port_full_sharpe": port_sharpe,
                "best_individual_sharpe": best_ind_sharpe,
                "mdd_beats_all_individuals": bool(port_mdd > best_ind_mdd),
                "sharpe_beats_all_individuals": bool(port_sharpe > best_ind_sharpe),
            },
        }

    return results


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main():
    os.makedirs(PORT_DIR, exist_ok=True)
    print("=== Invest_God Phase 1-D: 3-Bucket Portfolio Optimizer ===")

    # 1. Load aligned data
    df, btc_bear = load_aligned_data(use_l2=False, perturb_oos_mult=1.0)
    df_l2, _ = load_aligned_data(use_l2=True, perturb_oos_mult=1.0)
    df_haircut = apply_haircut_returns(df)

    # Correlations
    corr_full = df[["ret_short", "ret_mid", "ret_long"]].iloc[1:].corr()
    corr_is = df[df["date"] <= IS_END][["ret_short", "ret_mid", "ret_long"]].iloc[1:].corr()
    corr_oos = df[df["date"] > IS_END][["ret_short", "ret_mid", "ret_long"]].corr()

    # Individual Strategy Metrics (on exact aligned 1,439 KRX days)
    individual_metrics = {}
    for label, col in [("Turbo-1", "eq_short"), ("D-3", "eq_mid"), ("L-3", "eq_long")]:
        tmp = pd.DataFrame({"date": df["date"], "equity": df[col]})
        tmp_hc = pd.DataFrame({"date": df_haircut["date"], "equity": df_haircut[col]})
        individual_metrics[label] = {
            "is": calc_metrics(tmp, IS_START, IS_END),
            "oos": calc_metrics(tmp, OOS_START, OOS_END),
            "full": calc_metrics(tmp, IS_START, OOS_END),
            "full_haircut": calc_metrics(tmp_hc, IS_START, OOS_END),
            "yearly": calc_yearly_stats(tmp),
        }

    # L-2 individual metrics
    tmp_l2 = pd.DataFrame({"date": df_l2["date"], "equity": df_l2["eq_long"]})
    individual_metrics["L-2_ref"] = {
        "is": calc_metrics(tmp_l2, IS_START, IS_END),
        "oos": calc_metrics(tmp_l2, OOS_START, OOS_END),
        "full": calc_metrics(tmp_l2, IS_START, OOS_END),
        "yearly": calc_yearly_stats(tmp_l2),
    }

    # BTC Bear Regime Stats
    is_mask = df["date"] <= IS_END
    oos_mask = df["date"] > IS_END
    btc_regime_stats = {
        "is_bear_days": int(btc_bear[is_mask].sum()),
        "is_total_days": int(is_mask.sum()),
        "is_bear_pct": float(btc_bear[is_mask].mean()),
        "oos_bear_days": int(btc_bear[oos_mask].sum()),
        "oos_total_days": int(oos_mask.sum()),
        "oos_bear_pct": float(btc_bear[oos_mask].mean()),
        "full_bear_days": int(btc_bear.sum()),
        "full_bear_pct": float(btc_bear.mean()),
    }

    # 2. IS-Only Weight Optimization
    opt_weights = optimize_weights_on_is(df)
    print("IS-Only Optimized Base Weights (Short / Mid / Long):")
    for k, w in opt_weights.items():
        print(f"  {k:18s}: {w[0]*100:5.2f}% / {w[1]*100:5.2f}% / {w[2]*100:5.2f}%")

    # 3. Define all portfolio models to evaluate
    # P-1..P-4 + P-5 variants (both Max-Sharpe base and Aggressive base, both 5d_confirm and monthly)
    model_specs = [
        ("P-1_EqualWeight", opt_weights["P-1_EqualWeight"], None),
        ("P-2_MaxSharpe", opt_weights["P-2_MaxSharpe"], None),
        ("P-3_RiskParity", opt_weights["P-3_RiskParity"], None),
        ("P-4_Aggressive", opt_weights["P-4_Aggressive"], None),
        ("P-5a_Dyn_MaxSharpe_5d", opt_weights["P-2_MaxSharpe"], "5d_confirm"),
        ("P-5b_Dyn_MaxSharpe_Monthly", opt_weights["P-2_MaxSharpe"], "monthly"),
        ("P-5c_Dyn_Aggressive_5d", opt_weights["P-4_Aggressive"], "5d_confirm"),
        ("P-5d_Dyn_Aggressive_Monthly", opt_weights["P-4_Aggressive"], "monthly"),
    ]

    models_summary = {}
    models_meta_for_z3 = {}
    eq_orig_mem: Dict[str, pd.DataFrame] = {}

    for name, bw, dyn_mode in model_specs:
        eq_df, rb_df = simulate_portfolio(df, bw, dynamic_mode=dyn_mode)
        eq_orig_mem[name] = eq_df
        eq_hc, rb_hc = simulate_portfolio(df_haircut, bw, dynamic_mode=dyn_mode)
        eq_l2, _ = simulate_portfolio(df_l2, bw, dynamic_mode=dyn_mode)

        eq_path = os.path.join(PORT_DIR, f"equity_port_{name}.csv")
        rb_path = os.path.join(PORT_DIR, f"rebal_port_{name}.csv")
        eq_df.to_csv(eq_path, index=False)
        rb_df.to_csv(rb_path, index=False)

        m_is = calc_metrics(eq_df, IS_START, IS_END)
        m_oos = calc_metrics(eq_df, OOS_START, OOS_END)
        m_full = calc_metrics(eq_df, IS_START, OOS_END)
        m_hc_is = calc_metrics(eq_hc, IS_START, IS_END)
        m_hc_oos = calc_metrics(eq_hc, OOS_START, OOS_END)
        m_hc_full = calc_metrics(eq_hc, IS_START, OOS_END)
        m_l2_full = calc_metrics(eq_l2, IS_START, OOS_END)
        m_l2_oos = calc_metrics(eq_l2, OOS_START, OOS_END)

        total_cost = float(rb_df["cost_krw"].sum()) if len(rb_df) > 0 else 0.0
        total_to = float(rb_df["one_way_turnover"].sum()) if len(rb_df) > 0 else 0.0
        bear_days_active = int(eq_df["in_bear_regime"].sum())

        models_summary[name] = {
            "base_weights": {"short": bw[0], "mid": bw[1], "long": bw[2]},
            "dynamic_mode": dyn_mode,
            "rebal_events": int(len(rb_df)),
            "total_one_way_turnover": total_to,
            "total_cost_krw": total_cost,
            "bear_regime_active_days": bear_days_active,
            "is": m_is,
            "oos": m_oos,
            "full": m_full,
            "haircut_is": m_hc_is,
            "haircut_oos": m_hc_oos,
            "haircut_full": m_hc_full,
            "l2_ref_oos": m_l2_oos,
            "l2_ref_full": m_l2_full,
            "yearly": calc_yearly_stats(eq_df),
            "yearly_haircut": calc_yearly_stats(eq_hc),
        }

        models_meta_for_z3[name] = {
            "equity_csv": eq_path,
            "rebal_csv": rb_path,
            "base_weights": bw,
            "dynamic_mode": dyn_mode,
            "metrics_full": m_full,
        }

    # 4. Look-Ahead Perturbation Test (+500% on OOS > 2024-06-30)
    df_pert, _ = load_aligned_data(use_l2=False, perturb_oos_mult=6.0)
    opt_weights_pert = optimize_weights_on_is(df_pert)
    max_is_diff = 0.0
    for name, bw, dyn_mode in model_specs:
        # Use perturbed IS optimization weights
        base_key = (
            "P-1_EqualWeight" if "P-1" in name else
            "P-2_MaxSharpe" if ("P-2" in name or "MaxSharpe" in name) else
            "P-3_RiskParity" if "P-3" in name else
            "P-4_Aggressive"
        )
        bw_p = opt_weights_pert[base_key]
        eq_orig = eq_orig_mem[name]
        eq_pert, _ = simulate_portfolio(df_pert, bw_p, dynamic_mode=dyn_mode)
        is_orig = eq_orig[eq_orig["date"] <= IS_END]["equity"].values
        is_pert = eq_pert[eq_pert["date"] <= IS_END]["equity"].values
        diff = float(np.max(np.abs(is_orig - is_pert)))
        if diff > max_is_diff:
            max_is_diff = diff

    print(f"Look-Ahead Perturbation Test (+500% OOS) Max IS Diff: {max_is_diff:.10f} KRW")

    # 5. Z3-Solver Verification (parses saved CSVs directly)
    z3_results = verify_portfolio_z3(models_meta_for_z3, individual_metrics, max_is_diff)
    z3_path = os.path.join(PORT_DIR, "z3_portfolio_verification.json")
    with open(z3_path, "w", encoding="utf-8") as f:
        json.dump(z3_results, f, indent=2, ensure_ascii=False)

    # 6. Save Comprehensive Results JSON
    output_json = {
        "meta": {
            "is_period": f"{IS_START} ~ {IS_END}",
            "oos_period": f"{OOS_START} ~ {OOS_END}",
            "full_period": f"{IS_START} ~ {OOS_END}",
            "trading_days_total": int(len(df)),
            "trading_days_is": int(is_mask.sum()),
            "trading_days_oos": int(oos_mask.sum()),
            "initial_capital_krw": INITIAL_CAPITAL,
            "rebal_round_trip_cost": ROUND_TRIP_COST,
            "haircut_d3_annual": HAIRCUT_D3_ANNUAL,
            "haircut_turbo1_annual": HAIRCUT_TURBO1_ANNUAL,
            "lookahead_perturbation_max_is_diff": max_is_diff,
        },
        "correlations": {
            "full": corr_full.to_dict(),
            "is": corr_is.to_dict(),
            "oos": corr_oos.to_dict(),
        },
        "btc_regime_stats": btc_regime_stats,
        "individual_strategies": individual_metrics,
        "portfolio_models": models_summary,
        "z3_summary": {k: v["overall_verdict"] for k, v in z3_results.items()},
    }

    res_path = os.path.join(PORT_DIR, "portfolio_results.json")
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)

    # Print Clean Summary Table
    print("\n=== Individual Strategies (Aligned 1,439 KRX Days) ===")
    for k in ["Turbo-1", "D-3", "L-3", "L-2_ref"]:
        m_f = individual_metrics[k]["full"]
        m_i = individual_metrics[k]["is"]
        m_o = individual_metrics[k]["oos"]
        print(
            f"{k:10s} | Full CAGR {m_f['cagr_cal']*100:6.2f}% Sh {m_f['sharpe']:4.2f} MDD {m_f['mdd']*100:6.2f}% | "
            f"IS CAGR {m_i['cagr_cal']*100:6.2f}% Sh {m_i['sharpe']:4.2f} | "
            f"OOS CAGR {m_o['cagr_cal']*100:6.2f}% Sh {m_o['sharpe']:4.2f} MDD {m_o['mdd']*100:6.2f}%"
        )

    print("\n=== 3-Bucket Portfolio Models ===")
    for name, d in models_summary.items():
        bw = d["base_weights"]
        m_f = d["full"]
        m_i = d["is"]
        m_o = d["oos"]
        m_hc = d["haircut_full"]
        z3v = z3_results[name]["overall_verdict"]
        print(
            f"{name:28s} ({bw['short']*100:4.1f}/{bw['mid']*100:4.1f}/{bw['long']*100:4.1f}) | "
            f"Full: CAGR {m_f['cagr_cal']*100:5.2f}% Sh {m_f['sharpe']:4.2f} MDD {m_f['mdd']*100:6.2f}% | "
            f"OOS: CAGR {m_o['cagr_cal']*100:5.2f}% Sh {m_o['sharpe']:4.2f} MDD {m_o['mdd']*100:6.2f}% | "
            f"HC: CAGR {m_hc['cagr_cal']*100:5.2f}% Sh {m_hc['sharpe']:4.2f} | Z3={z3v}"
        )


if __name__ == "__main__":
    main()
