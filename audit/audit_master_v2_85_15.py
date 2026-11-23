"""
Invest_God Independent Red Team Audit Script for Master Portfolio v2 (85 : 15 Barbell)
- Author: quant-auditor (Independent Red Team Auditor)
- Target: portfolio/optimize_master_v2_85_15.py & 4 evidence files (Read-Only! Never modifies portfolio/*)
"""

import json
import sys
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from z3 import And, RealVal, Solver, sat

PROJECT_ROOT = Path(r"c:\Users\양태철\Desktop\Invest_God")
PORT_DIR = PROJECT_ROOT / "portfolio"
KRX_DB = PROJECT_ROOT / "strategies" / "krx_surge" / "krx_surge_data.duckdb"

sys.path.insert(0, str(PORT_DIR))
import optimize_master_v2_85_15 as mv2

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
    calmar = cagr / abs(mdd) if mdd < 0 else 0.0
    return {"total_return": tot_ret, "cagr_cal": cagr, "sharpe": sharpe, "mdd": mdd, "calmar": calmar, "final_eq": eq[-1]}


def main():
    section("1. 산출물 5종 완비 점검 & 누락/불일치 거래일 조사")
    req = [
        "optimize_master_v2_85_15.py",
        "equity_master_v2_85_15.csv",
        "rebal_master_v2_85_15.csv",
        "results_master_v2_85_15.json",
        "FINAL_MASTER_PORTFOLIO_V2.md",
    ]
    missing = [f for f in req if not (PORT_DIR / f).exists() or (PORT_DIR / f).stat().st_size == 0]
    print(f"필수 파일 5종 중 누락/빈 파일: {len(missing)}개 {missing}")

    eq_saved = pd.read_csv(PORT_DIR / "equity_master_v2_85_15.csv")
    rb_saved = pd.read_csv(PORT_DIR / "rebal_master_v2_85_15.csv")
    p1_df = pd.read_csv(PORT_DIR / "equity_port_P-1_EqualWeight.csv")
    print(f"  equity rows={len(eq_saved)} (NaN={int(eq_saved.isna().sum().sum())}), rebal rows={len(rb_saved)} (NaN={int(rb_saved.isna().sum().sum())})")

    # SK하이닉스 거래일 vs P-1 거래일 차이 조사
    con = duckdb.connect(str(KRX_DB), read_only=True)
    hx_dates = set(con.execute("SELECT date FROM daily_ohlcv WHERE code='000660' AND date>='2021-01-04' AND date<='2026-11-19'").df()["date"])
    con.close()
    p1_dates = set(p1_df["date"])
    extra_hx = sorted(hx_dates - p1_dates)
    extra_p1 = sorted(p1_dates - hx_dates)
    print(f"  SK하이닉스 거래일({len(hx_dates)}) vs P-1 거래일({len(p1_dates)}): hx에만 있음={extra_hx}, p1에만 있음={extra_p1}")

    section("2. 원천 코드 독립 재실행 & CSV/JSON/리포트 수치 교차 대조")
    with open(PORT_DIR / "results_master_v2_85_15.json", "r", encoding="utf-8") as f:
        res_json = json.load(f)

    rocket_df, rocket_stats = mv2.build_rocket_series(0.020, 0.025, 0.0015, 1.0)
    eq_run, rb_run = mv2.simulate_master_v2(p1_df, rocket_df, 0.85, 0.15, 0.0030)

    repro_eq_err = float(np.max(np.abs(eq_run["equity"].values - eq_saved["equity"].values)))
    repro_rb_err = float(np.max(np.abs(rb_run["cost_krw"].values - rb_saved["cost_krw"].values)))
    print(f"  [재현성] 재실행 vs 저장 CSV: equity 최대 오차={repro_eq_err:.10f}, rebal 비용 오차={repro_rb_err:.10f}")

    # Master v2 독립 재계산 vs JSON
    for seg, s_dt, e_dt in [("full", IS_START, OOS_END), ("is", IS_START, IS_END), ("oos", OOS_START, OOS_END)]:
        calc = indep_metrics(eq_saved, s_dt, e_dt)
        rep = res_json["master_v2_85_15"][seg]
        err = max(abs(calc["cagr_cal"] - rep["cagr_cal"]) * 100, abs(calc["mdd"] - rep["mdd"]) * 100, abs(calc["sharpe"] - rep["sharpe"]))
        print(
            f"  [Master v2 {seg:4s}] CAGR={calc['cagr_cal']*100:+6.2f}% | Shp={calc['sharpe']:.2f} | "
            f"MDD={calc['mdd']*100:6.2f}% | Calmar={calc['calmar']:.2f} | Eq={calc['final_eq']/1e8:.2f}억 | diff_vs_json={err:.10f}%p"
        )

    # 연도별 수익률 & MDD 독립 검산
    eq_saved["year"] = eq_saved["date"].str[:4]
    prev_e = None
    yr_list = []
    for yr, g in eq_saved.groupby("year"):
        ev = g["equity"].values.astype(float)
        se = prev_e if prev_e is not None else float(ev[0])
        ee = float(ev[-1])
        s_mdd = np.concatenate([[se], ev])
        ymdd = (s_mdd / np.maximum.accumulate(s_mdd) - 1.0).min() * 100
        yr_list.append(f"{yr}: {(ee/se-1)*100:+.1f}% ({ymdd:.1f}%)")
        prev_e = ee
    print("  [Master v2 연도별]:", ", ".join(yr_list))

    # 스트레스 A/B 독립 재계산
    rk_2x, _ = mv2.build_rocket_series(0.040, 0.025, 0.0030, 1.0)
    eq_sa, _ = mv2.simulate_master_v2(p1_df, rk_2x, 0.85, 0.15, 0.0060)
    p1_hc = mv2.build_haircut_p1()
    eq_sb, _ = mv2.simulate_master_v2(p1_hc, rk_2x, 0.85, 0.15, 0.0060)
    msa = indep_metrics(eq_sa, IS_START, OOS_END)
    msb = indep_metrics(eq_sb, IS_START, OOS_END)
    print(f"  [스트레스 A (비용2배)] Full CAGR={msa['cagr_cal']*100:+.2f}%, Shp={msa['sharpe']:.2f}, MDD={msa['mdd']*100:.2f}%, Eq={msa['final_eq']/1e8:.2f}억")
    print(f"  [스트레스 B (HC+비용2배)] Full CAGR={msb['cagr_cal']*100:+.2f}%, Shp={msb['sharpe']:.2f}, MDD={msb['mdd']*100:.2f}%, Eq={msb['final_eq']/1e8:.2f}억")

    # 참고 벤치마크(하이닉스 1배/2배 단순보유) IS/OOS JSON vs 리포트 불일치 확인
    print("\n  [문서 대조 적발 체크: 하이닉스 1x/2x 단순보유 IS/OOS JSON 실측치]")
    for k in ["standalone_hynix_1x_bh", "standalone_hynix_2x_bh"]:
        ji, jo = res_json[k]["is"], res_json[k]["oos"]
        print(f"    {k}: IS CAGR={ji['cagr_cal']*100:+.2f}% (MDD {ji['mdd']*100:.1f}%) | OOS CAGR={jo['cagr_cal']*100:+.2f}% (MDD {jo['mdd']*100:.1f}%)")

    section("3. 미래참조(Look-Ahead) 독립 교란 테스트 (+500% 및 -80%)")
    base_is = eq_run[eq_run["date"] <= IS_END]["equity"].values
    for lbl, mult in [("OOS +500% (x6.0)", 6.0), ("OOS -80% (x0.2)", 0.2)]:
        p1_p = p1_df.copy()
        p1_p.loc[p1_p["date"] >= OOS_START, "equity"] *= mult
        rk_p, _ = mv2.build_rocket_series(0.020, 0.025, 0.0015, mult)
        eq_p, _ = mv2.simulate_master_v2(p1_p, rk_p, 0.85, 0.15, 0.0030)
        diff = float(np.max(np.abs(eq_p[eq_p["date"] <= IS_END]["equity"].values - base_is)))
        print(f"  [교란 {lbl:16s}] IS 일별 잔고 최대 오차 = {diff:.12f} KRW")

    section("4. Z3-Solver 독립 검증 (JSON 배제, CSV 직접 파싱)")
    min_eq = float(eq_saved["equity"].min())
    min_cash = float(eq_saved["cash"].min())
    max_tw = float(eq_saved["total_weight"].max())
    max_tw_err = float((eq_saved["w_core"] + eq_saved["w_rocket"] - 1.0).abs().max())
    acc_err = float((eq_saved["equity"] - (eq_saved["val_core"] + eq_saved["val_rocket"])).abs().max())

    q_rows = eq_saved[eq_saved["is_quarter_rebal"] == 1]
    max_rk_rebal_err = float((q_rows["w_rocket"] - 0.15).abs().max())
    exp_cost = rb_saved["pre_equity"] * rb_saved["one_way_turnover"] * 0.0030
    max_cost_err = float((rb_saved["cost_krw"] - exp_cost).abs().max())

    m_v2_f = indep_metrics(eq_saved, IS_START, OOS_END)
    m_v2_o = indep_metrics(eq_saved, OOS_START, OOS_END)
    m_p1_f = indep_metrics(p1_df, IS_START, OOS_END)
    m_p1_o = indep_metrics(p1_df, OOS_START, OOS_END)

    s = Solver()
    s.add(RealVal(f"{min_eq:.6f}") > RealVal("0"))
    s.add(RealVal(f"{min_cash:.6f}") >= RealVal("0"))
    s.add(RealVal(f"{max_tw:.10f}") <= RealVal("1.000001"))
    s.add(RealVal(f"{max_tw_err:.10f}") <= RealVal("0.000001"))
    s.add(RealVal(f"{acc_err:.6f}") <= RealVal("0.01"))
    s.add(RealVal(f"{max_rk_rebal_err:.10f}") <= RealVal("0.000001"))
    s.add(RealVal(f"{max_cost_err:.10f}") <= RealVal("0.01"))
    # P-1 대비 Full & OOS CAGR, Sharpe 동시 우월성
    s.add(RealVal(f"{m_v2_f['cagr_cal']:.10f}") > RealVal(f"{m_p1_f['cagr_cal']:.10f}"))
    s.add(RealVal(f"{m_v2_o['cagr_cal']:.10f}") > RealVal(f"{m_p1_o['cagr_cal']:.10f}"))
    s.add(RealVal(f"{m_v2_f['sharpe']:.10f}") > RealVal(f"{m_p1_f['sharpe']:.10f}"))
    s.add(RealVal(f"{m_v2_o['sharpe']:.10f}") > RealVal(f"{m_p1_o['sharpe']:.10f}"))

    print(
        f"  CSV 직접 추출: min_eq={min_eq:,.0f}, min_cash={min_cash:.2f}, max_tw={max_tw:.12f}, "
        f"acc_err={acc_err:.2e}, rk_rebal_err={max_rk_rebal_err:.2e}, cost_err={max_cost_err:.2e}"
    )
    print(f"  감사관 독립 Z3-Solver 종합 SLA(무결성 + 복원 + 비용 + P-1 대비 CAGR&Sharpe 우월) 판정: {'SAT' if s.check()==sat else 'UNSAT'}")


if __name__ == "__main__":
    main()
