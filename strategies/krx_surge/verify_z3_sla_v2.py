"""
Invest_God Phase 1-A-2: Z3-Solver 기반 KRX 전략 A Ablation Study SLA 무결성 검증기
- 실제 백테스트 결과 JSON 및 일별 계좌 로그 CSV(1,439거래일 전수) 직접 파싱
- 구조적 SLA(1~5) + OOS 수익성(6) + KOSPI 대비 OOS 우월성(7) + KOSPI 대비 Full&OOS 3관왕 우월성(8)
"""

import json
from pathlib import Path
from typing import Dict

import pandas as pd
from z3 import And, Real, RealVal, Solver, sat

BASE_DIR = Path(__file__).resolve().parent
RESULTS_V2_PATH = BASE_DIR / "backtest_results_v2.json"
SLA_V2_OUTPUT_PATH = BASE_DIR / "z3_sla_verification_v2.json"


def check_clause(name: str, expr) -> Dict:
    s = Solver()
    s.add(expr)
    res = s.check()
    return {
        "clause": name,
        "status": "SAT" if res == sat else "UNSAT",
        "passed": res == sat,
    }


def verify_variant(name: str, vdata: Dict, kospi: Dict, la_diff: float) -> Dict:
    csv_path = BASE_DIR / f"equity_v2_{name}_net.csv"
    df = pd.read_csv(csv_path)

    csv_max_weight = float(df["invested_weight"].max())
    csv_min_cash = float(df["cash"].min())
    csv_min_equity = float(df["equity"].min())
    csv_max_positions = int(df["num_positions"].max())

    audit = vdata["net"]["audit"]
    nf = vdata["net"]["full"]
    no = vdata["net"]["oos"]
    gf = vdata["gross"]["full"]

    kf = kospi["full"]
    ko = kospi["oos"]

    # Z3 심볼릭 변수
    max_w = Real("max_total_weight")
    min_cash = Real("min_cash")
    min_eq = Real("min_equity")
    max_single_w = Real("max_single_entry_weight")
    max_pos = Real("max_positions_held")
    rt_cost = Real("round_trip_cost")
    net_ret = Real("net_full_return")
    gross_ret = Real("gross_full_return")
    sl_pct = Real("stop_loss_pct")
    la_err = Real("lookahead_max_diff")

    full_cagr = Real("full_net_cagr")
    full_sharpe = Real("full_net_sharpe")
    full_mdd = Real("full_net_mdd")
    oos_cagr = Real("oos_net_cagr")
    oos_sharpe = Real("oos_net_sharpe")
    oos_mdd = Real("oos_net_mdd")

    k_full_cagr = Real("kospi_full_cagr")
    k_full_sharpe = Real("kospi_full_sharpe")
    k_full_mdd = Real("kospi_full_mdd")
    k_oos_cagr = Real("kospi_oos_cagr")
    k_oos_sharpe = Real("kospi_oos_sharpe")
    k_oos_mdd = Real("kospi_oos_mdd")

    bindings = And(
        max_w == RealVal(f"{max(csv_max_weight, audit['max_observed_total_weight']):.10f}"),
        min_cash == RealVal(f"{min(csv_min_cash, audit['min_observed_cash']):.10f}"),
        min_eq == RealVal(f"{min(csv_min_equity, audit['min_observed_equity']):.10f}"),
        max_single_w == RealVal(f"{audit['max_observed_single_entry_weight']:.10f}"),
        max_pos == RealVal(str(csv_max_positions)),
        rt_cost == RealVal(f"{audit['round_trip_cost']:.10f}"),
        net_ret == RealVal(f"{nf['total_return']:.10f}"),
        gross_ret == RealVal(f"{gf['total_return']:.10f}"),
        sl_pct == RealVal(f"{audit['stop_loss_pct']:.10f}"),
        la_err == RealVal(f"{la_diff:.10f}"),
        full_cagr == RealVal(f"{nf['cagr']:.10f}"),
        full_sharpe == RealVal(f"{nf['sharpe']:.10f}"),
        full_mdd == RealVal(f"{nf['mdd']:.10f}"),
        oos_cagr == RealVal(f"{no['cagr']:.10f}"),
        oos_sharpe == RealVal(f"{no['sharpe']:.10f}"),
        oos_mdd == RealVal(f"{no['mdd']:.10f}"),
        k_full_cagr == RealVal(f"{kf['cagr']:.10f}"),
        k_full_sharpe == RealVal(f"{kf['sharpe']:.10f}"),
        k_full_mdd == RealVal(f"{kf['mdd']:.10f}"),
        k_oos_cagr == RealVal(f"{ko['cagr']:.10f}"),
        k_oos_sharpe == RealVal(f"{ko['sharpe']:.10f}"),
        k_oos_mdd == RealVal(f"{ko['mdd']:.10f}"),
    )

    clauses = [
        check_clause("SLA-1_NoLeverage_Weight_LE_100pct", And(bindings, max_w <= RealVal("1.0001"), max_pos <= RealVal("5"))),
        check_clause("SLA-2_NoBankruptcy_Cash_GE_0_Equity_GT_0", And(bindings, min_cash >= RealVal("0"), min_eq > RealVal("0"))),
        check_clause("SLA-3_SingleStockCap_LE_20pct", And(bindings, max_single_w <= RealVal("0.2001"))),
        check_clause("SLA-4_ConservativeCost_GE_60bps_Deducted", And(bindings, rt_cost >= RealVal("0.006"), net_ret < gross_ret)),
        check_clause("SLA-5_HardStop_And_ZeroLookAhead", And(bindings, sl_pct >= RealVal("0.05"), sl_pct <= RealVal("0.10"), la_err <= RealVal("0.000000001"))),
        check_clause("SLA-6_OOS_NetProfit_Positive", And(bindings, oos_cagr > RealVal("0"), oos_sharpe > RealVal("0"))),
        check_clause("SLA-7_OOS_Beats_KOSPI_Sharpe_And_MDD", And(bindings, oos_sharpe > k_oos_sharpe, oos_mdd > k_oos_mdd)),
        check_clause(
            "SLA-8_Full_And_OOS_Dominates_KOSPI_CAGR_Sharpe_MDD",
            And(
                bindings,
                full_cagr > k_full_cagr,
                full_sharpe > k_full_sharpe,
                full_mdd > k_full_mdd,
                oos_cagr > k_oos_cagr,
                oos_sharpe > k_oos_sharpe,
                oos_mdd > k_oos_mdd,
            ),
        ),
    ]

    return {
        "variant": name,
        "label": vdata["label"],
        "trading_days_checked": len(df),
        "clauses": clauses,
        "structural_1_to_5": "SAT" if all(c["passed"] for c in clauses[:5]) else "UNSAT",
        "oos_positive_6": clauses[5]["status"],
        "beats_kospi_oos_7": clauses[6]["status"],
        "dominates_kospi_all_8": clauses[7]["status"],
    }


def main():
    with open(RESULTS_V2_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)

    kospi = results["benchmarks"]["KOSPI"]
    summary = {"variants": {}}
    print("=== Invest_God Phase 1-A-2 Z3-Solver SLA Verification ===\n")

    for name, vdata in results["variants"].items():
        la_diff = results["lookahead_perturbation_max_diff"][name]
        v = verify_variant(name, vdata, kospi, la_diff)
        summary["variants"][name] = v

        print(f"[{name:4s}] ({v['label']})")
        print(
            f"  SLA 1~5(구조/무결성): {v['structural_1_to_5']} | "
            f"SLA-6(OOS>0): {v['oos_positive_6']} | "
            f"SLA-7(OOS>KOSPI Shp&MDD): {v['beats_kospi_oos_7']} | "
            f"SLA-8(Full&OOS>KOSPI 3관왕): {v['dominates_kospi_all_8']}"
        )

    with open(SLA_V2_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved V2 Z3 verification report to {SLA_V2_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
