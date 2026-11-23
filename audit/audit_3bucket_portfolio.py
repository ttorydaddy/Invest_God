"""
Invest_God Final Integrated Red Team Audit Script for Phase 1-D (3-Bucket Portfolio)
- Author: quant-auditor (Independent Red Team Auditor)
- Target: portfolio/* (Read-Only! Never modifies portfolio-allocator files)
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from z3 import And, RealVal, Solver, sat

PROJECT_ROOT = Path(r"c:\Users\양태철\Desktop\Invest_God")
PORT_DIR = PROJECT_ROOT / "portfolio"
RESULTS_JSON = PORT_DIR / "portfolio_results.json"

PATH_TURBO1 = PROJECT_ROOT / "strategies" / "crypto_short" / "equity_turbo_M3_R15_net.csv"
PATH_D3 = PROJECT_ROOT / "strategies" / "krx_surge" / "equity_def_D-3_net.csv"
PATH_L3 = PROJECT_ROOT / "strategies" / "longterm_core" / "equity_lt_L-3_Barbell_Regime_net.csv"

sys.path.insert(0, str(PORT_DIR))
import optimize_3bucket_portfolio as opt

IS_START = "2021-01-04"
IS_END = "2024-06-30"
OOS_START = "2024-07-01"
OOS_END = "2026-11-19"


def section(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def indep_metrics(eq_df: pd.DataFrame, s_dt: str, e_dt: str):
    sub = eq_df[(eq_df["date"] >= s_dt) & (eq_df["date"] <= e_dt)].copy()
    eq = sub["equity"].values.astype(float)
    tot_ret = (eq[-1] / eq[0]) - 1.0
    cal_yrs = (pd.Timestamp(sub["date"].iloc[-1]) - pd.Timestamp(sub["date"].iloc[0])).days / 365.25
    cagr = (eq[-1] / eq[0]) ** (1.0 / cal_yrs) - 1.0
    dr = pd.Series(eq).pct_change().dropna()
    sharpe = float((dr.mean() / dr.std()) * np.sqrt(252)) if dr.std() > 0 else 0.0
    mdd = float((eq / np.maximum.accumulate(eq) - 1.0).min())
    return {"total_return": tot_ret, "cagr_cal": cagr, "sharpe": sharpe, "mdd": mdd, "final_eq": eq[-1]}


def main():
    section("1. 원시 증거물 완비 & 원본 CSV 정합성 + 주말 코인 수익률 보존 검증")
    models = [
        "P-1_EqualWeight", "P-2_MaxSharpe", "P-3_RiskParity", "P-4_Aggressive",
        "P-5a_Dyn_MaxSharpe_5d", "P-5b_Dyn_MaxSharpe_Monthly",
        "P-5c_Dyn_Aggressive_5d", "P-5d_Dyn_Aggressive_Monthly",
    ]
    req = ["optimize_3bucket_portfolio.py", "portfolio_results.json", "z3_portfolio_verification.json", "REPORT_3BUCKET_PORTFOLIO.md"]
    for m in models:
        req += [f"equity_port_{m}.csv", f"rebal_port_{m}.csv"]
    missing = [f for f in req if not (PORT_DIR / f).exists() or (PORT_DIR / f).stat().st_size == 0]
    print(f"필수 파일 {len(req)}개 중 누락/빈 파일: {len(missing)}개 {missing}")

    # 주말/24시간 코인 수익률 보존 수학적 증명
    df_t1_raw = pd.read_csv(PATH_TURBO1)
    t1_start_eq = float(df_t1_raw[df_t1_raw["date"] == IS_START]["equity"].iloc[0])
    t1_end_eq = float(df_t1_raw[df_t1_raw["date"] == OOS_END]["equity"].iloc[0])
    raw_cum_ratio = t1_end_eq / t1_start_eq

    df_aligned, _ = opt.load_aligned_data(use_l2=False, perturb_oos_mult=1.0)
    aligned_cum_ratio = float((1.0 + df_aligned["ret_short"]).prod())
    weekend_err = abs(raw_cum_ratio - aligned_cum_ratio)
    print(f"  - Turbo-1 원본 누적배수({IS_START}~{OOS_END})={raw_cum_ratio:.8f} vs KRX정렬 복리배수={aligned_cum_ratio:.8f} -> 주말 수익률 손실 오차={weekend_err:.12f}")

    section("2. P-1 / P-3 / P-4 / P-5 독립 재계산 & 비용(0.30%) & 생존편향 할인 검산")
    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        res_json = json.load(f)

    max_err_all = 0.0
    max_cost_err_all = 0.0
    for m in models:
        eq = pd.read_csv(PORT_DIR / f"equity_port_{m}.csv")
        rb = pd.read_csv(PORT_DIR / f"rebal_port_{m}.csv")
        mf = indep_metrics(eq, IS_START, OOS_END)
        mi = indep_metrics(eq, IS_START, IS_END)
        mo = indep_metrics(eq, OOS_START, OOS_END)

        # 리밸런싱 비용 0.30% 공식 검산
        exp_cost = rb["pre_equity"] * rb["one_way_turnover"] * 0.0030
        c_err = float((rb["cost_krw"] - exp_cost).abs().max())
        max_cost_err_all = max(max_cost_err_all, c_err)

        for seg, calc in [("full", mf), ("is", mi), ("oos", mo)]:
            rep = res_json["portfolio_models"][m][seg]
            err = max(abs(calc["cagr_cal"] - rep["cagr_cal"]) * 100, abs(calc["mdd"] - rep["mdd"]) * 100, abs(calc["sharpe"] - rep["sharpe"]))
            max_err_all = max(max_err_all, err)

        if m in ("P-1_EqualWeight", "P-3_RiskParity", "P-4_Aggressive", "P-5c_Dyn_Aggressive_5d"):
            print(
                f"  [{m:24s}] Full: CAGR={mf['cagr_cal']*100:+6.2f}% Shp={mf['sharpe']:.2f} MDD={mf['mdd']*100:6.2f}% ({mf['final_eq']/1e8:.2f}억) | "
                f"OOS: CAGR={mo['cagr_cal']*100:+6.2f}% Shp={mo['sharpe']:.2f} MDD={mo['mdd']*100:6.2f}% | CostErr={c_err:.2e}"
            )
    print(f"전 8개 모델 x 3구간 JSON 대비 최대 오차: {max_err_all:.10f}%p (허용치 0.01%p), 리밸런싱 비용 최대 오차: {max_cost_err_all:.2e} KRW")

    # P-1 연도별 수익률 및 생존편향 할인 독립 검산
    eq_p1 = pd.read_csv(PORT_DIR / "equity_port_P-1_EqualWeight.csv")
    eq_p1["year"] = eq_p1["date"].str[:4]
    prev_e = None
    yr_s = []
    for yr, g in eq_p1.groupby("year"):
        se = prev_e if prev_e is not None else float(g["equity"].iloc[0])
        ee = float(g["equity"].iloc[-1])
        yr_s.append(f"{yr}: {(ee/se - 1)*100:+.1f}%")
        prev_e = ee
    print("  [P-1 연도별 독립 재계산]:", ", ".join(yr_s))

    df_hc = opt.apply_haircut_returns(df_aligned)
    opt_w = opt.optimize_weights_on_is(df_aligned)
    for m_key in ["P-1_EqualWeight", "P-3_RiskParity", "P-4_Aggressive"]:
        eq_hc, _ = opt.simulate_portfolio(df_hc, opt_w[m_key], dynamic_mode=None)
        mhf = indep_metrics(eq_hc, IS_START, OOS_END)
        mho = indep_metrics(eq_hc, OOS_START, OOS_END)
        rep_hc = res_json["portfolio_models"][m_key]["haircut_full"]
        hc_err = abs(mhf["cagr_cal"] - rep_hc["cagr_cal"]) * 100
        print(f"  [생존편향 할인 {m_key:16s}] Full CAGR={mhf['cagr_cal']*100:+6.2f}% (Shp {mhf['sharpe']:.2f}) | OOS CAGR={mho['cagr_cal']*100:+6.2f}% (MDD {mho['mdd']*100:.2f}%) | diff={hc_err:.8f}%p")

    section("3. IS 단독 최적화 & 극단적 미래 교란(+500% / -80%) 독립 검증")
    for lbl, mult in [("OOS +500% (x6.0)", 6.0), ("OOS -80% (x0.2)", 0.2)]:
        df_p, _ = opt.load_aligned_data(use_l2=False, perturb_oos_mult=mult)
        opt_w_p = opt.optimize_weights_on_is(df_p)
        w_diff = max(max(abs(a - b) for a, b in zip(opt_w[k], opt_w_p[k])) for k in opt_w)
        max_eq_diff = 0.0
        for m_key in ["P-1_EqualWeight", "P-2_MaxSharpe", "P-3_RiskParity", "P-4_Aggressive"]:
            eq_orig = pd.read_csv(PORT_DIR / f"equity_port_{m_key}.csv")
            eq_p, _ = opt.simulate_portfolio(df_p, opt_w_p[m_key], dynamic_mode=None)
            d = float(np.max(np.abs(eq_orig[eq_orig["date"] <= IS_END]["equity"].values - eq_p[eq_p["date"] <= IS_END]["equity"].values)))
            max_eq_diff = max(max_eq_diff, d)
        # P-5c 동적 모델도 교란 테스트
        eq_p5c_orig = pd.read_csv(PORT_DIR / "equity_port_P-5c_Dyn_Aggressive_5d.csv")
        eq_p5c_p, _ = opt.simulate_portfolio(df_p, opt_w_p["P-4_Aggressive"], dynamic_mode="5d_confirm")
        d5 = float(np.max(np.abs(eq_p5c_orig[eq_p5c_orig["date"] <= IS_END]["equity"].values - eq_p5c_p[eq_p5c_p["date"] <= IS_END]["equity"].values)))
        print(f"  [교란 {lbl:16s}] IS 최적비중 오차={w_diff:.10f} | 고정 P-1~P-4 IS오차={max_eq_diff:.10f} | 동적 P-5c IS오차={d5:.10f}")

    section("4. 감사관 독립 Z3-Solver 종합 SLA 판정 (JSON 배제, 엄격한 AND 분산마법)")
    # 개별 3개 전략 지표를 CSV에서 직접 계산
    ind_sharpes = []
    ind_mdds = []
    for col in ["eq_short", "eq_mid", "eq_long"]:
        m_ind = indep_metrics(pd.DataFrame({"date": df_aligned["date"], "equity": df_aligned[col]}), IS_START, OOS_END)
        ind_sharpes.append(m_ind["sharpe"])
        ind_mdds.append(m_ind["mdd"])
    best_ind_sharpe = max(ind_sharpes)
    best_ind_mdd = max(ind_mdds)
    print(f"  개별 3전략 최고 샤프={best_ind_sharpe:.4f}, 최저 MDD={best_ind_mdd*100:.2f}%")

    for m_key in ["P-1_EqualWeight", "P-2_MaxSharpe", "P-3_RiskParity", "P-4_Aggressive"]:
        eq = pd.read_csv(PORT_DIR / f"equity_port_{m_key}.csv")
        mf = indep_metrics(eq, IS_START, OOS_END)
        mo = indep_metrics(eq, OOS_START, OOS_END)
        max_w_err = float((eq["w_short"] + eq["w_mid"] + eq["w_long"] - 1.0).abs().max())
        min_w = float(eq[["w_short", "w_mid", "w_long"]].min().min())
        min_eq = float(eq["equity"].min())
        acc_err = float((eq["equity"] - (eq["val_short"] + eq["val_mid"] + eq["val_long"])).abs().max())

        s = Solver()
        s.add(RealVal(f"{min_eq:.6f}") > RealVal("0"))
        s.add(RealVal(f"{min_w:.10f}") >= RealVal("0"))
        s.add(RealVal(f"{max_w_err:.10f}") <= RealVal("0.000001"))
        s.add(RealVal(f"{acc_err:.6f}") <= RealVal("0.01"))
        # 엄격한 AND 조건: 개별 3개 전략 모두보다 샤프 높음 AND MDD 낮음!
        s.add(RealVal(f"{mf['sharpe']:.10f}") > RealVal(f"{best_ind_sharpe:.10f}"))
        s.add(RealVal(f"{mf['mdd']:.10f}") > RealVal(f"{best_ind_mdd:.10f}"))
        s.add(RealVal(f"{mo['cagr_cal']:.10f}") > RealVal("0"))
        s.add(RealVal(f"{mo['mdd']:.10f}") > RealVal("-0.15"))

        print(f"  [{m_key:18s}] Shp={mf['sharpe']:.2f}(>{best_ind_sharpe:.2f}), MDD={mf['mdd']*100:.2f}%(>{best_ind_mdd*100:.2f}%), OOS_MDD={mo['mdd']*100:.2f}%, w_err={max_w_err:.2e} -> 엄격 AND 독립 Z3: {'SAT' if s.check()==sat else 'UNSAT'}")


if __name__ == "__main__":
    main()
