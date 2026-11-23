"""
Invest_God Phase 1-B: Z3-Solver 기반 가상자산 단기 전략 3종 SLA 무결성 검증기
- 실제 백테스트 결과 JSON 및 일별 계좌 로그 CSV(2,151일 전수), 매매 체결 내역 CSV 직접 파싱
- 구조적 무결성 SLA(1~6) + OOS 순수익(7) + OOS BTC 대비 우월성(8) + Full BTC 대비 우월성(9) 기계적 SAT/UNSAT 판정
"""

import json
from pathlib import Path
from typing import Dict

import pandas as pd
from z3 import And, Real, RealVal, Solver, sat

BASE_DIR = Path(__file__).resolve().parent
RESULTS_PATH = BASE_DIR / "backtest_crypto_results.json"
OUTPUT_PATH = BASE_DIR / "z3_crypto_verification.json"


def check_clause(name: str, expr) -> Dict:
    s = Solver()
    s.add(expr)
    res = s.check()
    return {
        "clause": name,
        "status": "SAT" if res == sat else "UNSAT",
        "passed": res == sat,
    }


def verify_strategy(code: str, sdata: Dict, btc: Dict) -> Dict:
    eq_csv = BASE_DIR / f"equity_crypto_{code}_net.csv"
    tr_csv = BASE_DIR / f"trades_crypto_{code}_net.csv"
    df_eq = pd.read_csv(eq_csv)
    df_tr = pd.read_csv(tr_csv)

    csv_max_weight = float(df_eq["invested_weight"].max())
    csv_min_cash = float(df_eq["cash"].min())
    csv_min_equity = float(df_eq["equity"].min())
    csv_max_positions = int(df_eq["num_positions"].max())

    audit = sdata["net"]["audit"]
    nf = sdata["net"]["full"]
    no = sdata["net"]["oos"]
    gf = sdata["gross"]["full"]
    la_diff = float(sdata["lookahead_is_max_abs_diff"])

    bf = btc["full"]
    bo = btc["oos"]

    # Z3 Symbolic Variables
    max_w = Real("max_total_weight")
    min_cash = Real("min_cash")
    min_eq = Real("min_equity")
    max_single_entry_w = Real("max_single_entry_weight")
    max_pos = Real("max_positions_held")
    sl_pct = Real("stop_loss_pct")
    rt_cost = Real("round_trip_cost")
    net_ret = Real("net_full_return")
    gross_ret = Real("gross_full_return")
    la_err = Real("lookahead_is_max_abs_diff")

    full_cagr = Real("full_net_cagr")
    full_sharpe = Real("full_net_sharpe")
    full_mdd = Real("full_net_mdd")
    oos_cagr = Real("oos_net_cagr")
    oos_sharpe = Real("oos_net_sharpe")
    oos_mdd = Real("oos_net_mdd")

    b_full_sharpe = Real("btc_full_sharpe")
    b_full_mdd = Real("btc_full_mdd")
    b_oos_sharpe = Real("btc_oos_sharpe")
    b_oos_mdd = Real("btc_oos_mdd")

    bindings = And(
        max_w == RealVal(f"{max(csv_max_weight, audit['max_observed_total_weight']):.10f}"),
        min_cash == RealVal(f"{min(csv_min_cash, audit['min_observed_cash']):.10f}"),
        min_eq == RealVal(f"{min(csv_min_equity, audit['min_observed_equity']):.10f}"),
        max_single_entry_w == RealVal(f"{audit['max_observed_single_entry_weight']:.10f}"),
        max_pos == RealVal(str(max(csv_max_positions, audit["max_observed_positions"]))),
        sl_pct == RealVal(f"{audit['stop_loss_pct']:.10f}"),
        rt_cost == RealVal(f"{audit['round_trip_cost']:.10f}"),
        net_ret == RealVal(f"{nf['total_return']:.10f}"),
        gross_ret == RealVal(f"{gf['total_return']:.10f}"),
        la_err == RealVal(f"{la_diff:.10f}"),
        full_cagr == RealVal(f"{nf['cagr']:.10f}"),
        full_sharpe == RealVal(f"{nf['sharpe']:.10f}"),
        full_mdd == RealVal(f"{nf['mdd']:.10f}"),
        oos_cagr == RealVal(f"{no['cagr']:.10f}"),
        oos_sharpe == RealVal(f"{no['sharpe']:.10f}"),
        oos_mdd == RealVal(f"{no['mdd']:.10f}"),
        b_full_sharpe == RealVal(f"{bf['sharpe']:.10f}"),
        b_full_mdd == RealVal(f"{bf['mdd']:.10f}"),
        b_oos_sharpe == RealVal(f"{bo['sharpe']:.10f}"),
        b_oos_mdd == RealVal(f"{bo['mdd']:.10f}"),
    )

    clauses = [
        check_clause(
            "SLA-1_NoLeverage_Weight_LE_100pct",
            And(bindings, max_w <= RealVal("1.0001"), max_pos <= RealVal("4")),
        ),
        check_clause(
            "SLA-2_NoBankruptcy_Cash_GE_0_Equity_GT_0",
            And(bindings, min_cash >= RealVal("0"), min_eq > RealVal("0")),
        ),
        check_clause(
            "SLA-3_SingleCoinCap_LE_25pct",
            And(bindings, max_single_entry_w <= RealVal("0.2501")),
        ),
        check_clause(
            "SLA-4_TightHardStop_2pct_to_3pct",
            And(bindings, sl_pct >= RealVal("0.02"), sl_pct <= RealVal("0.03")),
        ),
        check_clause(
            "SLA-5_ConservativeCost_GE_30bps_Deducted",
            And(bindings, rt_cost >= RealVal("0.003"), net_ret < gross_ret),
        ),
        check_clause(
            "SLA-6_ZeroLookAhead_Perturbation_0",
            And(bindings, la_err <= RealVal("0.000000001")),
        ),
        check_clause(
            "SLA-7_OOS_NetProfit_Positive",
            And(bindings, oos_cagr > RealVal("0"), oos_sharpe > RealVal("0")),
        ),
        check_clause(
            "SLA-8_OOS_Beats_BTC_Sharpe_And_MDD",
            And(bindings, oos_sharpe > b_oos_sharpe, oos_mdd > b_oos_mdd),
        ),
        check_clause(
            "SLA-9_Full_NetProfit_And_Beats_BTC_Sharpe_MDD",
            And(bindings, full_cagr > RealVal("0"), full_sharpe > b_full_sharpe, full_mdd > b_full_mdd),
        ),
    ]

    structural_pass = all(c["passed"] for c in clauses[:6])
    oos_pass = all(c["passed"] for c in clauses[6:8])
    all_pass = all(c["passed"] for c in clauses)

    return {
        "strategy_code": code,
        "daily_rows_verified": int(len(df_eq)),
        "trade_rows_verified": int(len(df_tr)),
        "structural_sla_1_to_6": "SAT" if structural_pass else "UNSAT",
        "oos_sla_7_to_8": "SAT" if oos_pass else "UNSAT",
        "overall_all_9_sla": "SAT" if all_pass else "UNSAT",
        "clauses": clauses,
    }


def main() -> None:
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        res = json.load(f)

    btc = res["benchmark_btc_bh"]
    output = {
        "verifier": "Z3 SMT Solver (Phase 1-B Crypto Short-Term SLA)",
        "strategies": {},
    }

    for code, sdata in res["strategies"].items():
        v = verify_strategy(code, sdata, btc)
        output["strategies"][code] = v
        print(
            f"[{code:18s}] Structural(1-6)={v['structural_sla_1_to_6']:5s} | "
            f"OOS(7-8)={v['oos_sla_7_to_8']:5s} | All(1-9)={v['overall_all_9_sla']:5s}"
        )
        for c in v["clauses"]:
            print(f"    - {c['clause']:48s}: {c['status']}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[DONE] Saved Z3 verification to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
