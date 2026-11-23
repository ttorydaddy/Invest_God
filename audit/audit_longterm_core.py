"""
Invest_God Independent Red Team Audit Script for Phase 1-C (Long-Term Core: L-1, L-2, L-3)
- Author: quant-auditor (Independent Red Team Auditor)
- Target: strategies/longterm_core/* (Read-Only! Never modifies developer files)
"""

import json
import sys
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from z3 import And, RealVal, Solver, sat

PROJECT_ROOT = Path(r"c:\Users\양태철\Desktop\Invest_God")
LT_DIR = PROJECT_ROOT / "strategies" / "longterm_core"
DB_PATH = LT_DIR / "longterm_core_data.duckdb"
RESULTS_JSON = LT_DIR / "longterm_results.json"

D3_EQ = PROJECT_ROOT / "strategies" / "krx_surge" / "equity_def_D-3_net.csv"
TB_EQ = PROJECT_ROOT / "strategies" / "crypto_short" / "equity_turbo_M3_R15_net.csv"

sys.path.insert(0, str(LT_DIR))
import backtest_longterm_core as bt

STRATS = ["L-1_DualMomentum", "L-2_RiskParity_Trend", "L-3_Barbell_Regime", "BM_KOSPI", "BM_6040"]
BT_START = "2021-01-04"
IS_END = "2024-06-28"
OOS_START = "2024-07-01"
DATA_END = "2026-11-19"


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
    return {
        "eq0": float(eq[0]),
        "final_eq": float(eq[-1]),
        "total_return": float(tot_ret),
        "cagr_cal": float(cagr),
        "sharpe": sharpe,
        "mdd": mdd,
        "calmar": calmar,
    }


def main():
    section("1. 원시 증거물 4종 완비 & DuckDB 원천 데이터(6 KRX ETF + BTC-KRW) 감사")
    req = ["backtest_longterm_core.py", "longterm_core_data.duckdb", "longterm_results.json", "z3_longterm_verification.json", "REPORT_LONGTERM_CORE.md"]
    for st in STRATS:
        req.append(f"equity_lt_{st}_net.csv")
        req.append(f"trades_lt_{st}_net.csv")
    missing = [f for f in req if not (LT_DIR / f).exists() or (LT_DIR / f).stat().st_size == 0]
    print(f"필수 파일 {len(req)}개 중 누락/빈 파일: {len(missing)}개 {missing}")

    con = duckdb.connect(str(DB_PATH), read_only=True)
    summary = con.execute("""
        SELECT asset, ticker, name, COUNT(*) AS rows,
               MIN(date) AS min_dt, MAX(date) AS max_dt,
               MIN(Close) AS min_c, MAX(Close) AS max_c,
               SUM(CASE WHEN Open <= 0 OR Close <= 0 OR Open IS NULL OR Close IS NULL THEN 1 ELSE 0 END) AS bad_px
        FROM asset_ohlcv
        GROUP BY 1, 2, 3
        ORDER BY 1
    """).df()
    dup = con.execute("SELECT COUNT(*) FROM (SELECT asset, date, COUNT(*) c FROM asset_ohlcv GROUP BY 1,2 HAVING c > 1)").fetchone()[0]
    print(f"DuckDB asset_ohlcv 중복 행: {dup}건")
    print(summary.to_string(index=False))

    # BTC-KRW 연도별 가격 Sanity Check (환율 왜곡 여부)
    btc_yr = con.execute("""
        SELECT SUBSTR(date, 1, 4) AS yr, COUNT(*) AS days,
               ROUND(MIN(Close)/1e6, 1) AS min_m_krw,
               ROUND(MAX(Close)/1e6, 1) AS max_m_krw
        FROM asset_ohlcv WHERE asset='BTC'
        GROUP BY 1 ORDER BY 1
    """).df()
    print("\n[BTC-KRW 연도별 가격 범위 (백만 원 단위 — 환율 Sanity Check)]")
    print(btc_yr.to_string(index=False))
    con.close()

    section("2. L-1 / L-2 / L-3 / BM 독립 재계산 vs 리포트 & JSON 교차 대조 (0.01%p 이내)")
    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        res_json = json.load(f)

    max_err_all = 0.0
    for st in STRATS:
        eq = pd.read_csv(LT_DIR / f"equity_lt_{st}_net.csv")
        tr = pd.read_csv(LT_DIR / f"trades_lt_{st}_net.csv")
        na_eq = int(eq.isna().sum().sum())
        mf = indep_metrics(eq, BT_START, DATA_END)
        mi = indep_metrics(eq, BT_START, IS_END)
        mo = indep_metrics(eq, OOS_START, DATA_END)

        for sname, calc in [("full", mf), ("is", mi), ("oos", mo)]:
            rep = res_json["strategies"][st]["net"][sname]
            err = max(
                abs(calc["cagr_cal"] - rep["cagr_cal"]) * 100,
                abs(calc["total_return"] - rep["total_return"]) * 100,
                abs(calc["mdd"] - rep["mdd"]) * 100,
            )
            max_err_all = max(max_err_all, err)

        print(
            f"  [{st:20s}] eq0={mf['eq0']/1e8:.4f}억 -> end={mf['final_eq']/1e8:.3f}억 | "
            f"CalCAGR={mf['cagr_cal']*100:+6.2f}% | Tot={mf['total_return']*100:+6.1f}% | "
            f"Shp={mf['sharpe']:.2f} | MDD={mf['mdd']*100:6.2f}% | "
            f"IS={mi['cagr_cal']*100:+6.2f}% | OOS={mo['cagr_cal']*100:+6.2f}% (Shp {mo['sharpe']:.2f}, MDD {mo['mdd']*100:6.2f}%) | NaN={na_eq}, Rebal={len(tr)}"
        )
    print(f"전 5개 전략/BM x 3구간 JSON 대비 최대 오차: {max_err_all:.10f}%p (허용치 0.01%p)")

    # L-3 연도별 수익률 독립 검산
    eq_l3 = pd.read_csv(LT_DIR / "equity_lt_L-3_Barbell_Regime_net.csv")
    eq_l3["year"] = eq_l3["date"].str[:4]
    prev_e = None
    yr_s = []
    for yr, g in eq_l3.groupby("year"):
        s_e = prev_e if prev_e is not None else float(g["equity"].iloc[0])
        e_e = float(g["equity"].iloc[-1])
        yr_s.append(f"{yr}: {(e_e/s_e - 1)*100:+.1f}%")
        prev_e = e_e
    print("  [L-3 연도별 독립 재계산]:", ", ".join(yr_s))

    section("3. 독립 재실행 재현성 & 극단적 미래 교란 테스트 (+500% / -80%)")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    all_df = con.execute("SELECT * FROM asset_ohlcv ORDER BY asset, date").df()
    con.close()
    open_p, close_p = bt.build_aligned_panels(all_df)

    for st in ["L-1_DualMomentum", "L-2_RiskParity_Trend", "L-3_Barbell_Regime"]:
        eq_saved = pd.read_csv(LT_DIR / f"equity_lt_{st}_net.csv")
        eq_run, _, _ = bt.run_longterm_strategy(st, open_p, close_p, apply_costs=True)
        repro = float(np.max(np.abs(eq_run["equity"].values - eq_saved["equity"].values)))

        base_is = eq_run[eq_run["date"] <= IS_END]["equity"].values
        diffs = {}
        for lbl, mult in [("+500%", 6.0), ("-80%", 0.2)]:
            po, pc = open_p.copy(), close_p.copy()
            m = pc.index >= pd.Timestamp(OOS_START)
            po.loc[m] = po.loc[m] * mult
            pc.loc[m] = pc.loc[m] * mult
            peq, _, _ = bt.run_longterm_strategy(st, po, pc, apply_costs=True)
            diffs[lbl] = float(np.max(np.abs(peq[peq["date"] <= IS_END]["equity"].values - base_is)))
        print(f"  [{st:20s}] 재실행 오차={repro:.10f} | 교란(+500%) IS오차={diffs['+500%']:.10f} | 교란(-80%) IS오차={diffs['-80%']:.10f}")

    section("4. 3-Bucket 일별 수익률 상관계수 독립 교차 계산")
    d3 = pd.read_csv(D3_EQ)[["date", "equity"]].rename(columns={"equity": "D3"})
    tb = pd.read_csv(TB_EQ)[["date", "equity"]].rename(columns={"equity": "Turbo1"})
    d3["date"] = pd.to_datetime(d3["date"])
    tb["date"] = pd.to_datetime(tb["date"])
    base_m = d3.set_index("date").join(tb.set_index("date"), how="inner")

    for st in ["L-1_DualMomentum", "L-2_RiskParity_Trend", "L-3_Barbell_Regime"]:
        lt = pd.read_csv(LT_DIR / f"equity_lt_{st}_net.csv")[["date", "equity"]].rename(columns={"equity": st})
        lt["date"] = pd.to_datetime(lt["date"])
        m = base_m.join(lt.set_index("date"), how="inner")
        c = m.pct_change().dropna().corr()
        print(f"  [{st:20s}] vs D-3={c.loc[st, 'D3']:+.3f} | vs Turbo-1={c.loc[st, 'Turbo1']:+.3f} | D-3 vs Turbo-1={c.loc['D3', 'Turbo1']:+.3f} (n={len(m.pct_change().dropna())})")

    section("5. 감사관 독립 Z3-Solver 종합 SLA 판정 (JSON 배제, CSV 직접 계산)")
    eq_k = pd.read_csv(LT_DIR / "equity_lt_BM_KOSPI_net.csv")
    eq_64 = pd.read_csv(LT_DIR / "equity_lt_BM_6040_net.csv")
    mk = indep_metrics(eq_k, BT_START, DATA_END)
    m64 = indep_metrics(eq_64, BT_START, DATA_END)

    for st in ["L-1_DualMomentum", "L-2_RiskParity_Trend", "L-3_Barbell_Regime"]:
        eq = pd.read_csv(LT_DIR / f"equity_lt_{st}_net.csv")
        mf = indep_metrics(eq, BT_START, DATA_END)
        mo = indep_metrics(eq, OOS_START, DATA_END)
        max_w = float(eq["invested_weight"].max())
        min_c = float(eq["cash"].min())
        min_e = float(eq["equity"].min())
        acc_err = float(np.max(np.abs(eq["equity"] - (eq["cash"] + eq["asset_val"]))))

        # Gross 재실행으로 Net < Gross 직접 확인
        eq_gross, _, _ = bt.run_longterm_strategy(st, open_p, close_p, apply_costs=False)
        gf = indep_metrics(eq_gross, BT_START, DATA_END)

        s = Solver()
        s.add(RealVal(f"{max_w:.10f}") <= RealVal("1.0001"))
        s.add(RealVal(f"{min_c:.10f}") >= RealVal("0"))
        s.add(RealVal(f"{min_e:.10f}") > RealVal("0"))
        s.add(RealVal(f"{acc_err:.10f}") <= RealVal("0.01"))
        s.add(RealVal(f"{mf['total_return']:.10f}") < RealVal(f"{gf['total_return']:.10f}"))
        s.add(RealVal(f"{mo['cagr_cal']:.10f}") > RealVal("0"))
        s.add(RealVal(f"{mo['sharpe']:.10f}") > RealVal("0"))
        s.add(RealVal(f"{mf['sharpe']:.10f}") > RealVal(f"{mk['sharpe']:.10f}"))
        s.add(RealVal(f"{mf['sharpe']:.10f}") > RealVal(f"{m64['sharpe']:.10f}"))
        s.add(RealVal(f"{mf['mdd']:.10f}") > RealVal(f"{mk['mdd']:.10f}"))

        print(f"  [{st:20s}] max_w={max_w:.6f}, min_c={min_c:.2f}, acc_err={acc_err:.8f}, Net<Gross={mf['total_return']<gf['total_return']} -> 독립 Z3: {'SAT' if s.check()==sat else 'UNSAT'}")


if __name__ == "__main__":
    main()
