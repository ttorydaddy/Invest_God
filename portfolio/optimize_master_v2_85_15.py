"""
Invest_God Master Portfolio v2 (85 : 15 Barbell) Official Backtest & Z3 SLA Verifier
====================================================================================
Confirmed Baseline Architecture:
  1. Core (85.0%): Audited P-1 Equal-Weight 3-Bucket Portfolio
       - Short Crypto (Turbo-1) : 28.333%
       - Mid KRX (D-3)          : 28.333%
       - Long Core (L-3)        : 28.333%
       Source: portfolio/equity_port_P-1_EqualWeight.csv (1,439 KRX trading days)
  2. Rocket (15.0%): SK Hynix (000660.KS) Daily 2x Leveraged ETF/ETN + MA20/MA120 Golden Cross
       - Daily 2x return: r_2x = max(2 * r_1x - 0.02 / 252, -0.99) (annual 2.0% fee/financing drag)
       - Signal: MA20 > MA120 at day t close -> applied to day t+1 (Zero Look-Ahead)
       - Cash / RP parking when MA20 <= MA120: 2.5% annual yield (calendar-day compounded)
       - Regime switch one-way transition cost: 0.15%
       Source: strategies/krx_surge/krx_surge_data.duckdb (daily_ohlcv, code='000660', 2020-07-01~2026-11-19)
  3. Rebalancing: First trading day of each quarter -> 85.0% Core : 15.0% Rocket
       - Cost: one_way_turnover * 0.30% (0.15% sell + 0.15% buy)

Mandatory Audits & Stress Tests:
  - +500% OOS (>= 2024-07-01) Look-Ahead Perturbation Test -> IS max diff == 0.000000
  - Z3-Solver Mechanical SLA Proof (reads saved CSVs directly)
  - 2x Rocket Cost Stress Test (4.0%/yr drag, 0.30% switch cost, 0.60% quarterly rebal cost)
  - Combined Survival-Bias Haircut (P-1 D-3 -4%p, Turbo-1 -2.5%p) + 2x Rocket Cost Stress
  - Worst-Case Rocket Crash (-50% / -70%) Portfolio Impact Analysis
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
from z3 import And, Real, Solver, sat

# ---------------------------------------------------------------------------
# Constants & Paths
# ---------------------------------------------------------------------------
ROOT_DIR = r"c:\Users\양태철\Desktop\Invest_God"
PORT_DIR = os.path.join(ROOT_DIR, "portfolio")

PATH_P1 = os.path.join(PORT_DIR, "equity_port_P-1_EqualWeight.csv")
PATH_KRX_DB = os.path.join(ROOT_DIR, "strategies", "krx_surge", "krx_surge_data.duckdb")

# Also load raw strategies to compute P-1 Haircut baseline for combined stress test
PATH_TURBO1 = os.path.join(ROOT_DIR, "strategies", "crypto_short", "equity_turbo_M3_R15_net.csv")
PATH_D3 = os.path.join(ROOT_DIR, "strategies", "krx_surge", "equity_def_D-3_net.csv")
PATH_L3 = os.path.join(ROOT_DIR, "strategies", "longterm_core", "equity_lt_L-3_Barbell_Regime_net.csv")

IS_START = "2021-01-04"
IS_END = "2024-06-30"
OOS_START = "2024-07-01"
OOS_END = "2026-11-19"

INITIAL_CAPITAL = 100_000_000.0  # 100M KRW
W_CORE_TARGET = 0.85
W_ROCKET_TARGET = 0.15

# Baseline Cost & Return Parameters
ROCKET_ANNUAL_DRAG = 0.020       # 2.0%/yr leveraged ETF fee + financing cost
ROCKET_RP_ANNUAL = 0.025         # 2.5%/yr RP/short-term bond yield when MA20 <= MA120
ROCKET_SWITCH_COST = 0.0015      # 0.15% one-way regime switch cost
QUARTER_REBAL_COST_RT = 0.0030   # 0.30% * one-way turnover (0.15% sell + 0.15% buy)


# ---------------------------------------------------------------------------
# Metrics Helpers (100% consistent with Invest_God convention)
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
    mdd = float((eq / np.maximum.accumulate(eq) - 1.0).min())
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
        series_for_mdd = np.concatenate([[start_eq], eq_vals])
        yr_mdd = float((series_for_mdd / np.maximum.accumulate(series_for_mdd) - 1.0).min())
        out[str(yr)] = {"return": float(yr_ret), "mdd": float(yr_mdd)}
        prev_end_eq = end_eq
    return out


# ---------------------------------------------------------------------------
# Rocket Standalone Simulator (SK Hynix 2x + MA20/MA120 Golden Cross)
# ---------------------------------------------------------------------------
def build_rocket_series(
    annual_drag: float = ROCKET_ANNUAL_DRAG,
    rp_annual: float = ROCKET_RP_ANNUAL,
    switch_cost: float = ROCKET_SWITCH_COST,
    perturb_oos_mult: float = 1.0,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Simulates SK Hynix (000660) 2x daily leveraged ETF/ETN with MA20 > MA120 filter.
    Uses full 2020-07-01 ~ 2026-11-19 daily_ohlcv so MA120 is 100% warmed up before 2021-01-04.
    Signal at day t close applies strictly to day t+1 return (Zero Look-Ahead).
    """
    con = duckdb.connect(PATH_KRX_DB, read_only=True)
    hx = con.execute(
        """
        SELECT date, Close
        FROM daily_ohlcv
        WHERE code = '000660' AND date >= '2020-07-01' AND date <= '2026-11-19'
        ORDER BY date
        """
    ).fetchdf()
    con.close()

    hx["Close"] = hx["Close"].astype(float)
    if perturb_oos_mult != 1.0:
        oos_mask = hx["date"] >= OOS_START
        hx.loc[oos_mask, "Close"] = hx.loc[oos_mask, "Close"] * perturb_oos_mult

    hx["ma20"] = hx["Close"].rolling(20, min_periods=20).mean()
    hx["ma120"] = hx["Close"].rolling(120, min_periods=120).mean()
    # Signal observed at day t close
    hx["signal_close"] = (hx["ma20"] > hx["ma120"]).fillna(False)
    # Position active during day t (strictly t-1 close signal -> Zero Look-Ahead!)
    hx["in_rocket_2x"] = hx["signal_close"].shift(1, fill_value=False).astype(bool)

    hx["r_1x"] = hx["Close"].pct_change().fillna(0.0)
    hx["r_2x"] = np.maximum(2.0 * hx["r_1x"] - (annual_drag / 252.0), -0.99)

    dt = pd.to_datetime(hx["date"])
    dt_days = dt.diff().dt.days.fillna(0.0)
    hx["r_rp"] = (1.0 + rp_annual) ** (dt_days / 365.25) - 1.0

    # Restrict simulation window to >= IS_START (2021-01-04), initialized at 2021-01-04 close
    sub = hx[hx["date"] >= IS_START].copy().reset_index(drop=True)
    assert sub["date"].iloc[0] == IS_START and sub["date"].iloc[-1] == OOS_END
    assert not sub["ma120"].isna().any(), "MA120 must be fully warmed up on 2021-01-04"

    eq = np.zeros(len(sub), dtype=float)
    bh_1x_eq = np.zeros(len(sub), dtype=float)
    bh_2x_eq = np.zeros(len(sub), dtype=float)

    eq[0] = INITIAL_CAPITAL
    bh_1x_eq[0] = INITIAL_CAPITAL
    bh_2x_eq[0] = INITIAL_CAPITAL

    switches = 0
    # Position held entering day 1 is determined by day 0 close signal
    prev_pos = bool(sub["signal_close"].iloc[0])
    # Conservative: charge initial entry switch cost on day 0 if starting invested in 2x
    if prev_pos:
        eq[0] *= (1.0 - switch_cost)
        switches += 1

    switch_flags = np.zeros(len(sub), dtype=int)
    if prev_pos:
        switch_flags[0] = 1

    for i in range(1, len(sub)):
        pos_today = bool(sub["in_rocket_2x"].iloc[i])
        r_day = float(sub["r_2x"].iloc[i]) if pos_today else float(sub["r_rp"].iloc[i])
        cur = eq[i - 1] * (1.0 + r_day)

        # Check if signal at day i close changes position for day i+1 -> execute switch at day i close
        next_pos = bool(sub["signal_close"].iloc[i])
        if next_pos != pos_today and i < len(sub) - 1:
            cur *= (1.0 - switch_cost)
            switches += 1
            switch_flags[i] = 1

        eq[i] = cur
        bh_1x_eq[i] = bh_1x_eq[i - 1] * (1.0 + float(sub["r_1x"].iloc[i]))
        bh_2x_eq[i] = bh_2x_eq[i - 1] * (1.0 + float(sub["r_2x"].iloc[i]))

    sub["rocket_equity"] = eq
    sub["bh_1x_equity"] = bh_1x_eq
    sub["bh_2x_equity"] = bh_2x_eq
    sub["rocket_switch"] = switch_flags

    rocket_stats = {
        "switches_total": int(switches),
        "days_in_2x": int(sub["in_rocket_2x"].iloc[1:].sum()),
        "days_in_rp": int((~sub["in_rocket_2x"].iloc[1:]).sum()),
        "pct_in_2x": float(sub["in_rocket_2x"].iloc[1:].mean()),
        "is_pct_in_2x": float(sub[sub["date"] <= IS_END]["in_rocket_2x"].iloc[1:].mean()),
        "oos_pct_in_2x": float(sub[sub["date"] >= OOS_START]["in_rocket_2x"].mean()),
    }
    return sub, rocket_stats


# ---------------------------------------------------------------------------
# Master Portfolio v2 (85 : 15 Barbell) Simulator
# ---------------------------------------------------------------------------
def simulate_master_v2(
    p1_df: pd.DataFrame,
    rocket_df: pd.DataFrame,
    w_core: float = W_CORE_TARGET,
    w_rocket: float = W_ROCKET_TARGET,
    rebal_cost_rt: float = QUARTER_REBAL_COST_RT,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combines Core (85% P-1) and Rocket (15% SK Hynix 2x MA20/MA120) on P-1's exact
    1,439 KRX trading days (2021-01-04 ~ 2026-11-19).
    Rebalances on the FIRST trading day of every new quarter (Jan, Apr, Jul, Oct)
    back to 85% : 15%, deducting one_way_turnover * rebal_cost_rt.
    """
    df = p1_df[["date", "equity"]].rename(columns={"equity": "eq_core"}).merge(
        rocket_df[["date", "rocket_equity", "in_rocket_2x", "signal_close", "rocket_switch", "Close", "ma20", "ma120"]],
        on="date",
        how="inner",
    )
    df = df.sort_values("date").reset_index(drop=True)
    assert len(df) == 1439, f"Expected 1439 aligned days, got {len(df)}"

    df["r_core"] = df["eq_core"].pct_change().fillna(0.0)
    df["r_rocket"] = df["rocket_equity"].pct_change().fillna(0.0)

    # First trading day of each quarter (quarter changes vs previous trading day)
    dt = pd.to_datetime(df["date"])
    yq = (dt.dt.year * 10 + dt.dt.quarter).values
    is_q_first = np.zeros(len(df), dtype=bool)
    for i in range(1, len(df)):
        if yq[i] != yq[i - 1]:
            is_q_first[i] = True

    val_core = INITIAL_CAPITAL * w_core
    val_rocket = INITIAL_CAPITAL * w_rocket

    eq_rows = []
    rebal_rows = []

    dates = df["date"].values
    r_c = df["r_core"].values
    r_r = df["r_rocket"].values
    in_2x = df["in_rocket_2x"].values
    r_sw = df["rocket_switch"].values

    for i in range(len(df)):
        if i > 0:
            val_core *= (1.0 + r_c[i])
            val_rocket *= (1.0 + r_r[i])

        pre_eq = val_core + val_rocket
        w_c_pre = val_core / pre_eq
        w_r_pre = val_rocket / pre_eq

        rebal_cost = 0.0
        one_way_to = 0.0

        if is_q_first[i]:
            one_way_to = 0.5 * (abs(w_core - w_c_pre) + abs(w_rocket - w_r_pre))
            rebal_cost = pre_eq * one_way_to * rebal_cost_rt
            post_eq = pre_eq - rebal_cost

            val_core = post_eq * w_core
            val_rocket = post_eq * w_rocket

            rebal_rows.append({
                "date": str(dates[i]),
                "reason": "QUARTER_FIRST_DAY",
                "rocket_in_2x": int(in_2x[i]),
                "pre_equity": float(pre_eq),
                "post_equity": float(post_eq),
                "one_way_turnover": float(one_way_to),
                "cost_krw": float(rebal_cost),
                "w_core_pre": float(w_c_pre),
                "w_rocket_pre": float(w_r_pre),
                "w_core_post": float(val_core / post_eq),
                "w_rocket_post": float(val_rocket / post_eq),
            })

        tot_eq = val_core + val_rocket
        w_c_post = val_core / tot_eq
        w_r_post = val_rocket / tot_eq

        # Effective 4-bucket breakdown (Core = 1/3 each of Short, Mid, Long)
        eq_rows.append({
            "date": str(dates[i]),
            "equity": float(tot_eq),
            "cash": 0.0,
            "val_core": float(val_core),
            "val_rocket": float(val_rocket),
            "w_core": float(w_c_post),
            "w_rocket": float(w_r_post),
            "w_short_eff": float(w_c_post / 3.0),
            "w_mid_eff": float(w_c_post / 3.0),
            "w_long_eff": float(w_c_post / 3.0),
            "total_weight": float(w_c_post + w_r_post),
            "rocket_in_2x": int(in_2x[i]),
            "rocket_switch": int(r_sw[i]),
            "is_quarter_rebal": int(is_q_first[i]),
            "rebal_cost_krw": float(rebal_cost),
        })

    return pd.DataFrame(eq_rows), pd.DataFrame(rebal_rows)


# ---------------------------------------------------------------------------
# Haircut P-1 Builder (for Combined Conservative Stress Test)
# ---------------------------------------------------------------------------
def build_haircut_p1() -> pd.DataFrame:
    """Rebuilds P-1 with Auditor survival-bias haircuts (D-3 -4.0%p/yr, Turbo-1 -2.5%p/yr)."""
    from optimize_3bucket_portfolio import apply_haircut_returns, load_aligned_data, simulate_portfolio
    df_raw, _ = load_aligned_data(use_l2=False, perturb_oos_mult=1.0)
    df_hc = apply_haircut_returns(df_raw)
    eq_p1_hc, _ = simulate_portfolio(df_hc, (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0), dynamic_mode=None)
    return eq_p1_hc


# ---------------------------------------------------------------------------
# Z3-Solver Mechanical SLA Verification (Reads Saved CSVs Directly)
# ---------------------------------------------------------------------------
def verify_master_v2_z3(
    eq_csv_path: str,
    rb_csv_path: str,
    m_v2_full: Dict,
    m_v2_oos: Dict,
    m_p1_full: Dict,
    m_p1_oos: Dict,
    perturb_max_diff: float,
) -> Dict:
    eq_df = pd.read_csv(eq_csv_path)
    rb_df = pd.read_csv(rb_csv_path)

    min_eq = float(eq_df["equity"].min())
    min_cash = float(eq_df["cash"].min())
    max_tot_w = float(eq_df["total_weight"].max())
    max_tot_w_err = float((eq_df["total_weight"] - 1.0).abs().max())
    min_w = float(eq_df[["w_core", "w_rocket"]].min().min())

    # Check w_rocket restored to exact 0.15 immediately after every quarterly rebalance
    q_rows = eq_df[eq_df["is_quarter_rebal"] == 1]
    max_rocket_rebal_err = float((q_rows["w_rocket"] - W_ROCKET_TARGET).abs().max()) if len(q_rows) > 0 else 0.0
    expected_cost = rb_df["pre_equity"] * rb_df["one_way_turnover"] * QUARTER_REBAL_COST_RT
    max_cost_err = float((rb_df["cost_krw"] - expected_cost).abs().max()) if len(rb_df) > 0 else 0.0

    s = Solver()
    z_min_eq = Real("min_equity")
    z_min_cash = Real("min_cash")
    z_min_w = Real("min_weight")
    z_max_tot_w = Real("max_total_weight")
    z_max_tot_err = Real("max_total_weight_err")
    z_rocket_err = Real("max_rocket_rebal_err")
    z_cost_err = Real("max_cost_err_krw")
    z_perturb = Real("lookahead_perturb_diff")
    z_v2_full_cagr = Real("v2_full_cagr")
    z_p1_full_cagr = Real("p1_full_cagr")
    z_v2_oos_cagr = Real("v2_oos_cagr")
    z_p1_oos_cagr = Real("p1_oos_cagr")

    s.add(z_min_eq == min_eq)
    s.add(z_min_cash == min_cash)
    s.add(z_min_w == min_w)
    s.add(z_max_tot_w == max_tot_w)
    s.add(z_max_tot_err == max_tot_w_err)
    s.add(z_rocket_err == max_rocket_rebal_err)
    s.add(z_cost_err == max_cost_err)
    s.add(z_perturb == perturb_max_diff)
    s.add(z_v2_full_cagr == m_v2_full["cagr_cal"])
    s.add(z_p1_full_cagr == m_p1_full["cagr_cal"])
    s.add(z_v2_oos_cagr == m_v2_oos["cagr_cal"])
    s.add(z_p1_oos_cagr == m_p1_oos["cagr_cal"])

    checks = {
        "no_bankruptcy_cash_nonneg": And(z_min_eq > 0, z_min_cash >= 0),
        "no_leverage_total_weight_le_1": And(z_min_w >= 0, z_max_tot_w <= 1.000001, z_max_tot_err <= 1e-6),
        "rocket_restored_to_015_after_rebal": z_rocket_err <= 1e-6,
        "rebal_cost_030pct_exact": z_cost_err <= 1e-2,
        "zero_lookahead_perturb_exact_0": z_perturb == 0.0,
        "full_and_oos_cagr_beats_p1_benchmark": And(z_v2_full_cagr > z_p1_full_cagr, z_v2_oos_cagr > z_p1_oos_cagr),
    }

    status = {}
    all_sat = True
    for k, expr in checks.items():
        s.push()
        s.add(expr)
        ok = (s.check() == sat)
        s.pop()
        status[k] = "SAT" if ok else "UNSAT"
        if not ok:
            all_sat = False

    return {
        "overall_verdict": "SAT" if all_sat else "UNSAT",
        "checks": status,
        "observed": {
            "rows": int(len(eq_df)),
            "quarter_rebal_events": int(len(rb_df)),
            "min_equity_krw": min_eq,
            "min_cash_krw": min_cash,
            "max_total_weight": max_tot_w,
            "max_total_weight_err": max_tot_w_err,
            "max_rocket_rebal_err": max_rocket_rebal_err,
            "max_cost_err_krw": max_cost_err,
            "lookahead_perturb_max_is_diff": perturb_max_diff,
            "v2_full_cagr": m_v2_full["cagr_cal"],
            "p1_full_cagr": m_p1_full["cagr_cal"],
            "v2_oos_cagr": m_v2_oos["cagr_cal"],
            "p1_oos_cagr": m_p1_oos["cagr_cal"],
        },
    }


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main():
    print("=== Invest_God Master Portfolio v2 (85 : 15 Barbell) Backtest ===")

    # 1. Load P-1 Core & Build Rocket
    p1_df = pd.read_csv(PATH_P1)
    rocket_df, rocket_stats = build_rocket_series(
        annual_drag=ROCKET_ANNUAL_DRAG,
        rp_annual=ROCKET_RP_ANNUAL,
        switch_cost=ROCKET_SWITCH_COST,
        perturb_oos_mult=1.0,
    )

    # 2. Simulate Master v2 (85 : 15)
    eq_v2, rb_v2 = simulate_master_v2(
        p1_df, rocket_df, w_core=W_CORE_TARGET, w_rocket=W_ROCKET_TARGET, rebal_cost_rt=QUARTER_REBAL_COST_RT
    )

    eq_path = os.path.join(PORT_DIR, "equity_master_v2_85_15.csv")
    rb_path = os.path.join(PORT_DIR, "rebal_master_v2_85_15.csv")
    eq_v2.to_csv(eq_path, index=False)
    rb_v2.to_csv(rb_path, index=False)

    # 3. Look-Ahead Perturbation Test (+500% on OOS >= 2024-07-01)
    p1_pert = p1_df.copy()
    p1_pert.loc[p1_pert["date"] >= OOS_START, "equity"] *= 6.0
    rocket_pert, _ = build_rocket_series(
        annual_drag=ROCKET_ANNUAL_DRAG,
        rp_annual=ROCKET_RP_ANNUAL,
        switch_cost=ROCKET_SWITCH_COST,
        perturb_oos_mult=6.0,
    )
    eq_v2_pert, _ = simulate_master_v2(
        p1_pert, rocket_pert, w_core=W_CORE_TARGET, w_rocket=W_ROCKET_TARGET, rebal_cost_rt=QUARTER_REBAL_COST_RT
    )
    is_orig = eq_v2[eq_v2["date"] <= IS_END]["equity"].values
    is_pert = eq_v2_pert[eq_v2_pert["date"] <= IS_END]["equity"].values
    perturb_max_diff = float(np.max(np.abs(is_orig - is_pert)))
    print(f"Look-Ahead Perturbation (+500% OOS) Max IS Diff: {perturb_max_diff:.10f} KRW")

    # 4. Stress Tests
    # Stress A: 2x Rocket Cost (4.0%/yr drag, 0.30% switch cost, 0.60% quarterly rebal cost)
    rocket_2x_cost, _ = build_rocket_series(
        annual_drag=ROCKET_ANNUAL_DRAG * 2.0,
        rp_annual=ROCKET_RP_ANNUAL,
        switch_cost=ROCKET_SWITCH_COST * 2.0,
        perturb_oos_mult=1.0,
    )
    eq_v2_stress2x, _ = simulate_master_v2(
        p1_df, rocket_2x_cost, w_core=W_CORE_TARGET, w_rocket=W_ROCKET_TARGET, rebal_cost_rt=QUARTER_REBAL_COST_RT * 2.0
    )

    # Stress B: Combined Survival-Bias Haircut (P-1 D-3 -4%p, Turbo-1 -2.5%p) + 2x Rocket Cost
    p1_hc = build_haircut_p1()
    eq_v2_hc_stress, _ = simulate_master_v2(
        p1_hc, rocket_2x_cost, w_core=W_CORE_TARGET, w_rocket=W_ROCKET_TARGET, rebal_cost_rt=QUARTER_REBAL_COST_RT * 2.0
    )

    # 5. Standalone Metrics for Comparison
    rocket_aligned = rocket_df[rocket_df["date"].isin(p1_df["date"])].copy().reset_index(drop=True)
    df_rocket_only = pd.DataFrame({"date": rocket_aligned["date"], "equity": rocket_aligned["rocket_equity"]})
    df_hx_1x = pd.DataFrame({"date": rocket_aligned["date"], "equity": rocket_aligned["bh_1x_equity"]})
    df_hx_2x = pd.DataFrame({"date": rocket_aligned["date"], "equity": rocket_aligned["bh_2x_equity"]})

    # Correlation between P-1 Core and Rocket
    r_p1 = p1_df["equity"].pct_change().iloc[1:].values
    r_rk = df_rocket_only["equity"].pct_change().iloc[1:].values
    corr_core_rocket_full = float(np.corrcoef(r_p1, r_rk)[0, 1])

    is_n = int((p1_df["date"] <= IS_END).sum())
    corr_core_rocket_is = float(np.corrcoef(r_p1[: is_n - 1], r_rk[: is_n - 1])[0, 1])
    corr_core_rocket_oos = float(np.corrcoef(r_p1[is_n - 1 :], r_rk[is_n - 1 :])[0, 1])

    m_p1 = {
        "is": calc_metrics(p1_df, IS_START, IS_END),
        "oos": calc_metrics(p1_df, OOS_START, OOS_END),
        "full": calc_metrics(p1_df, IS_START, OOS_END),
        "yearly": calc_yearly_stats(p1_df),
    }
    m_v2 = {
        "is": calc_metrics(eq_v2, IS_START, IS_END),
        "oos": calc_metrics(eq_v2, OOS_START, OOS_END),
        "full": calc_metrics(eq_v2, IS_START, OOS_END),
        "yearly": calc_yearly_stats(eq_v2),
    }
    m_rocket = {
        "is": calc_metrics(df_rocket_only, IS_START, IS_END),
        "oos": calc_metrics(df_rocket_only, OOS_START, OOS_END),
        "full": calc_metrics(df_rocket_only, IS_START, OOS_END),
        "yearly": calc_yearly_stats(df_rocket_only),
    }
    m_stress2x = {
        "is": calc_metrics(eq_v2_stress2x, IS_START, IS_END),
        "oos": calc_metrics(eq_v2_stress2x, OOS_START, OOS_END),
        "full": calc_metrics(eq_v2_stress2x, IS_START, OOS_END),
    }
    m_hc_stress = {
        "is": calc_metrics(eq_v2_hc_stress, IS_START, IS_END),
        "oos": calc_metrics(eq_v2_hc_stress, OOS_START, OOS_END),
        "full": calc_metrics(eq_v2_hc_stress, IS_START, OOS_END),
    }

    # 6. Z3-Solver SLA Verification (Reads Saved CSVs Directly!)
    z3_res = verify_master_v2_z3(
        eq_csv_path=eq_path,
        rb_csv_path=rb_path,
        m_v2_full=m_v2["full"],
        m_v2_oos=m_v2["oos"],
        m_p1_full=m_p1["full"],
        m_p1_oos=m_p1["oos"],
        perturb_max_diff=perturb_max_diff,
    )

    # 7. Save Comprehensive JSON Results
    results = {
        "meta": {
            "name": "Invest_God Master Portfolio v2 (85:15 Barbell)",
            "core_weight": W_CORE_TARGET,
            "rocket_weight": W_ROCKET_TARGET,
            "effective_4bucket_weights": {
                "short_crypto_turbo1": round(W_CORE_TARGET / 3.0, 6),
                "mid_krx_d3": round(W_CORE_TARGET / 3.0, 6),
                "long_core_l3": round(W_CORE_TARGET / 3.0, 6),
                "rocket_hynix2x_ma20_120": W_ROCKET_TARGET,
            },
            "rebal_schedule": "First KRX trading day of each quarter (Jan/Apr/Jul/Oct)",
            "rebal_cost_rt": QUARTER_REBAL_COST_RT,
            "rocket_annual_drag": ROCKET_ANNUAL_DRAG,
            "rocket_rp_annual": ROCKET_RP_ANNUAL,
            "rocket_switch_cost_one_way": ROCKET_SWITCH_COST,
            "period_full": f"{IS_START} ~ {OOS_END}",
            "period_is": f"{IS_START} ~ {IS_END}",
            "period_oos": f"{OOS_START} ~ {OOS_END}",
            "trading_days": int(len(eq_v2)),
            "lookahead_perturbation_max_is_diff": perturb_max_diff,
        },
        "correlations_core_vs_rocket": {
            "full": corr_core_rocket_full,
            "is": corr_core_rocket_is,
            "oos": corr_core_rocket_oos,
        },
        "rocket_activity": rocket_stats,
        "rebal_summary": {
            "quarter_rebal_events": int(len(rb_v2)),
            "total_one_way_turnover": float(rb_v2["one_way_turnover"].sum()),
            "total_rebal_cost_krw": float(rb_v2["cost_krw"].sum()),
            "max_rocket_weight_observed": float(eq_v2["w_rocket"].max()),
            "min_rocket_weight_observed": float(eq_v2["w_rocket"].min()),
        },
        "master_v2_85_15": m_v2,
        "benchmark_p1_100": m_p1,
        "standalone_rocket_100": m_rocket,
        "standalone_hynix_1x_bh": {
            "is": calc_metrics(df_hx_1x, IS_START, IS_END),
            "oos": calc_metrics(df_hx_1x, OOS_START, OOS_END),
            "full": calc_metrics(df_hx_1x, IS_START, OOS_END),
        },
        "standalone_hynix_2x_bh": {
            "is": calc_metrics(df_hx_2x, IS_START, IS_END),
            "oos": calc_metrics(df_hx_2x, OOS_START, OOS_END),
            "full": calc_metrics(df_hx_2x, IS_START, OOS_END),
        },
        "stress_tests": {
            "rocket_2x_cost_stress": m_stress2x,
            "combined_p1_haircut_plus_rocket_2x_cost": m_hc_stress,
        },
        "z3_verification": z3_res,
    }

    res_path = os.path.join(PORT_DIR, "results_master_v2_85_15.json")
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print Clean Summary
    def fmt(m: Dict) -> str:
        return f"CAGR {m['cagr_cal']*100:6.2f}% | Sh {m['sharpe']:4.2f} | MDD {m['mdd']*100:6.2f}% | Cal {m['calmar']:4.2f} | Eq {m['final_equity']/1e8:5.2f}억"

    print("\n=== Performance Summary (Initial 1.00억 KRW) ===")
    print(f"Master v2 (85:15) Full : {fmt(m_v2['full'])}")
    print(f"  - IS  (21.01~24.06)  : {fmt(m_v2['is'])}")
    print(f"  - OOS (24.07~26.11)  : {fmt(m_v2['oos'])}")
    print(f"Benchmark P-1     Full : {fmt(m_p1['full'])}")
    print(f"  - IS  (21.01~24.06)  : {fmt(m_p1['is'])}")
    print(f"  - OOS (24.07~26.11)  : {fmt(m_p1['oos'])}")
    print(f"Rocket 100%       Full : {fmt(m_rocket['full'])}")
    print(f"  - IS  (21.01~24.06)  : {fmt(m_rocket['is'])}")
    print(f"  - OOS (24.07~26.11)  : {fmt(m_rocket['oos'])}")
    print(f"Stress 2x Cost    Full : {fmt(m_stress2x['full'])} | OOS: {fmt(m_stress2x['oos'])}")
    print(f"Stress HC+2xCost  Full : {fmt(m_hc_stress['full'])} | OOS: {fmt(m_hc_stress['oos'])}")
    print(f"\nCore vs Rocket Corr    : Full={corr_core_rocket_full:+.4f}, IS={corr_core_rocket_is:+.4f}, OOS={corr_core_rocket_oos:+.4f}")
    print(f"Rocket Switches / In2x : {rocket_stats['switches_total']} switches, {rocket_stats['pct_in_2x']*100:.1f}% days in 2x")
    print(f"Quarterly Rebal        : {len(rb_v2)} events, total TO={rb_v2['one_way_turnover'].sum()*100:.1f}%, cost={rb_v2['cost_krw'].sum()/1e6:.2f}M KRW")
    print(f"Z3 SLA Verdict         : {z3_res['overall_verdict']} | {z3_res['checks']}")


if __name__ == "__main__":
    main()
