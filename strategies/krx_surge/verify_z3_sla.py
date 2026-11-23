"""
Invest_God Phase 1-A: Z3-Solver 기반 KRX 급등주 전략 SLA 무결성 검증기
- 3중 무결성 검증 철칙 제2번 명제: 실제 백테스트 산출물(JSON + 일별 CSV)을 직접 파싱해
  대표님 투자 헌장 SLA를 Microsoft Z3 SMT Solver로 수학적 증명(SAT/UNSAT).
"""

import json
from pathlib import Path
from typing import Dict

import pandas as pd
from z3 import And, Real, RealVal, Solver, sat

BASE_DIR = Path(__file__).resolve().parent
RESULTS_JSON_PATH = BASE_DIR / "backtest_results.json"
SLA_OUTPUT_PATH = BASE_DIR / "z3_sla_verification.json"


def check_clause(name: str, expr) -> Dict:
    s = Solver()
    s.add(expr)
    res = s.check()
    return {
        "clause": name,
        "status": "SAT" if res == sat else "UNSAT",
        "passed": res == sat,
    }


def verify_strategy(name: str, strat_data: Dict, la_diff: float) -> Dict:
    csv_path = BASE_DIR / f"equity_{name}_net.csv"
    df = pd.read_csv(csv_path)

    # 일별 실제 로그에서 직접 산출한 엄격한 극값
    csv_max_weight = float(df["invested_weight"].max())
    csv_min_cash = float(df["cash"].min())
    csv_min_equity = float(df["equity"].min())
    csv_max_positions = int(df["num_positions"].max())

    audit = strat_data["net"]["audit"]
    net_full = strat_data["net"]["full"]
    gross_full = strat_data["gross"]["full"]
    net_oos = strat_data["net"]["oos"]

    # Z3 심볼릭 변수에 실측 데이터 바인딩
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
    oos_cagr = Real("oos_net_cagr")
    oos_sharpe = Real("oos_net_sharpe")
    full_cagr = Real("full_net_cagr")

    bindings = And(
        max_w == RealVal(f"{max(csv_max_weight, audit['max_observed_total_weight']):.10f}"),
        min_cash == RealVal(f"{min(csv_min_cash, audit['min_observed_cash']):.10f}"),
        min_eq == RealVal(f"{min(csv_min_equity, audit['min_observed_equity']):.10f}"),
        max_single_w == RealVal(f"{audit['max_observed_single_entry_weight']:.10f}"),
        max_pos == RealVal(str(csv_max_positions)),
        rt_cost == RealVal(f"{audit['round_trip_cost']:.10f}"),
        net_ret == RealVal(f"{net_full['total_return']:.10f}"),
        gross_ret == RealVal(f"{gross_full['total_return']:.10f}"),
        sl_pct == RealVal(f"{audit['stop_loss_pct']:.10f}"),
        la_err == RealVal(f"{la_diff:.10f}"),
        oos_cagr == RealVal(f"{net_oos['cagr']:.10f}"),
        oos_sharpe == RealVal(f"{net_oos['sharpe']:.10f}"),
        full_cagr == RealVal(f"{net_full['cagr']:.10f}"),
    )

    clauses = [
        check_clause("SLA-1_NoLeverage_TotalWeight_LE_100pct", And(bindings, max_w <= RealVal("1.0001"), max_pos <= RealVal("5"))),
        check_clause("SLA-2_NoBankruptcy_Cash_GE_0_Equity_GT_0", And(bindings, min_cash >= RealVal("0"), min_eq > RealVal("0"))),
        check_clause("SLA-3_SingleStockCap_LE_20pct", And(bindings, max_single_w <= RealVal("0.2001"))),
        check_clause("SLA-4_ConservativeCost_GE_60bps_And_Deducted", And(bindings, rt_cost >= RealVal("0.006"), net_ret < gross_ret)),
        check_clause("SLA-5_HardStop_Enforced_And_ZeroLookAhead", And(bindings, sl_pct >= RealVal("0.05"), sl_pct <= RealVal("0.10"), la_err <= RealVal("0.000000001"))),
        check_clause("SLA-6_OOS_NetProfit_Positive", And(bindings, oos_cagr > RealVal("0"), oos_sharpe > RealVal("0"))),
        check_clause("SLA-7_ProductionCandidate_FullCAGR_GT_0_And_OOSSharpe_GE_0.5", And(bindings, full_cagr > RealVal("0"), oos_sharpe >= RealVal("0.5"))),
    ]

    structural_pass = all(c["passed"] for c in clauses[:5])
    oos_pass = clauses[5]["passed"]
    prod_pass = clauses[6]["passed"]

    return {
        "strategy": name,
        "observed_from_csv": {
            "trading_days_checked": len(df),
            "max_invested_weight": csv_max_weight,
            "max_positions": csv_max_positions,
            "min_cash": csv_min_cash,
            "min_equity": csv_min_equity,
            "max_single_entry_weight": audit["max_observed_single_entry_weight"],
            "round_trip_cost": audit["round_trip_cost"],
            "stop_loss_pct": audit["stop_loss_pct"],
            "lookahead_max_diff": la_diff,
            "full_net_cagr": net_full["cagr"],
            "oos_net_cagr": net_oos["cagr"],
            "oos_net_sharpe": net_oos["sharpe"],
        },
        "clauses": clauses,
        "structural_sla_1_to_5": "SAT" if structural_pass else "UNSAT",
        "oos_profit_sla_6": "SAT" if oos_pass else "UNSAT",
        "production_candidate_sla_7": "SAT" if prod_pass else "UNSAT",
    }


def main():
    with open(RESULTS_JSON_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)

    summary = {"strategies": {}}
    print("=== Invest_God Z3-Solver SLA Integrity Verification ===\n")

    for name, strat_data in results["strategies"].items():
        la_diff = results["lookahead_perturbation_max_diff"][name]
        v = verify_strategy(name, strat_data, la_diff)
        summary["strategies"][name] = v

        print(f"[{name}]")
        for c in v["clauses"]:
            print(f"  - {c['clause']}: {c['status']}")
        print(
            f"  => Structural(1~5): {v['structural_sla_1_to_5']} | "
            f"OOS Profit(6): {v['oos_profit_sla_6']} | "
            f"Production Candidate(7): {v['production_candidate_sla_7']}\n"
        )

    with open(SLA_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved Z3 verification report to {SLA_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
