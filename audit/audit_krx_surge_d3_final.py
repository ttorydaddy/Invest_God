"""
Invest_God Final Independent Audit Script for Phase 1-A-4 (KRX Mid-Term Bucket Final Candidate: D-3)
- Author: quant-auditor (Independent Red Team Auditor)
- Target: strategies/krx_surge/backtest_krx_surge_defense.py & D-3 evidence files (Read-Only)
"""

import json
import sys
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from z3 import And, RealVal, Solver, sat

PROJECT_ROOT = Path(r"c:\Users\양태철\Desktop\Invest_God")
STRAT_DIR = PROJECT_ROOT / "strategies" / "krx_surge"
DB_PATH = STRAT_DIR / "krx_surge_data.duckdb"
RESULTS_JSON = STRAT_DIR / "defense_test_results.json"
Z3_JSON = STRAT_DIR / "z3_sla_defense_test.json"

sys.path.insert(0, str(STRAT_DIR))
import backtest_krx_surge_defense as bt

BT_START = "2021-01-04"
IS_END = "2024-06-30"
OOS_START = "2024-07-01"
DATA_END = "2026-11-20"


def section(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def independent_metrics(eq_df: pd.DataFrame, tr_df: pd.DataFrame, start_date: str, end_date: str):
    sub = eq_df[(eq_df["date"] >= start_date) & (eq_df["date"] <= end_date)].copy()
    eq = sub["equity"].values.astype(float)
    total_ret = (eq[-1] / eq[0]) - 1.0
    dt0 = pd.Timestamp(sub["date"].iloc[0])
    dt1 = pd.Timestamp(sub["date"].iloc[-1])
    cal_years = (dt1 - dt0).days / 365.25
    cagr_cal = (eq[-1] / eq[0]) ** (1.0 / cal_years) - 1.0
    cagr_252 = (eq[-1] / eq[0]) ** (1.0 / (len(sub) / 252.0)) - 1.0
    dr = pd.Series(eq).pct_change().dropna()
    sharpe = float((dr.mean() / dr.std()) * np.sqrt(252)) if dr.std() > 0 else 0.0
    mdd = float((eq / np.maximum.accumulate(eq) - 1.0).min())
    calmar = cagr_cal / abs(mdd) if mdd < 0 else 0.0

    sub_tr = tr_df[(tr_df["entry_date"] >= start_date) & (tr_df["entry_date"] <= end_date)]
    n_tr = len(sub_tr)
    rets = sub_tr["net_ret"].values.astype(float)
    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    wr = len(wins) / n_tr if n_tr > 0 else 0.0
    payoff = (np.mean(wins) / abs(np.mean(losses))) if len(wins) > 0 and len(losses) > 0 else 0.0
    pf = (np.sum(wins) / abs(np.sum(losses))) if len(losses) > 0 and abs(np.sum(losses)) > 0 else 0.0

    return {
        "total_return": total_ret,
        "cagr_cal": cagr_cal,
        "cagr_252": cagr_252,
        "sharpe": sharpe,
        "mdd": mdd,
        "calmar": calmar,
        "num_trades": n_tr,
        "win_rate": wr,
        "payoff_ratio": float(payoff),
        "profit_factor": float(pf),
        "avg_exposure": float(sub["invested_weight"].mean()),
    }


def main():
    section("1. 원시 증거물 4종 완비 점검 (D-3 및 전체 6개 모델)")
    cases = ["BASE", "D-1", "D-2", "D-2H", "D-3", "D-4"]
    required = [
        "backtest_krx_surge_defense.py",
        "defense_test_results.json",
        "z3_sla_defense_test.json",
        "REPORT_KRX_SURGE_DEFENSE.md",
    ] + [f"equity_def_{c}_net.csv" for c in cases] + [f"trades_def_{c}_net.csv" for c in cases]
    missing = [f for f in required if not (STRAT_DIR / f).exists() or (STRAT_DIR / f).stat().st_size == 0]
    print(f"필수 파일 {len(required)}개 중 누락/빈 파일: {len(missing)}개 {missing}")

    eq_d3 = pd.read_csv(STRAT_DIR / "equity_def_D-3_net.csv")
    tr_d3 = pd.read_csv(STRAT_DIR / "trades_def_D-3_net.csv", dtype={"code": str})
    tr_d3["code"] = tr_d3["code"].str.zfill(6)
    print(f"D-3 equity rows={len(eq_d3)} (NaN={int(eq_d3.isna().sum().sum())}), trades rows={len(tr_d3)} (NaN={int(tr_d3.isna().sum().sum())})")

    section("2. D-3 독립 재계산 vs 리포트 & JSON 교차 대조 (0.01%p 이내)")
    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        res_json = json.load(f)

    m_full = independent_metrics(eq_d3, tr_d3, BT_START, DATA_END)
    m_is = independent_metrics(eq_d3, tr_d3, BT_START, IS_END)
    m_oos = independent_metrics(eq_d3, tr_d3, OOS_START, DATA_END)

    for seg_name, calc in [("full", m_full), ("is", m_is), ("oos", m_oos)]:
        rep = res_json["cases"]["D-3"]["net"][seg_name]
        err_cagr = abs(calc["cagr_cal"] - rep["cagr_cal"]) * 100
        err_mdd = abs(calc["mdd"] - rep["mdd"]) * 100
        err_shp = abs(calc["sharpe"] - rep["sharpe"])
        err_ret = abs(calc["total_return"] - rep["total_return"]) * 100
        print(
            f"  [D-3 {seg_name:4s}] CalCAGR={calc['cagr_cal']*100:+.2f}% | Ret={calc['total_return']*100:+.2f}% | "
            f"Shp={calc['sharpe']:.2f} | MDD={calc['mdd']*100:.2f}% | Calmar={calc['calmar']:.2f} | "
            f"Trd={calc['num_trades']} | Exp={calc['avg_exposure']*100:.1f}% | "
            f"max_diff={max(err_cagr, err_mdd, err_ret):.10f}%p (shp_diff={err_shp:.10f})"
        )

    # 리포트 표 수치 대조 (달력 CAGR +22.19%, Full MDD -30.81%, OOS 달력 CAGR +39.23%, OOS MDD -15.00%, 샤프 1.00)
    report_checks = [
        ("Full CalCAGR", m_full["cagr_cal"] * 100, 22.19),
        ("Full MDD", m_full["mdd"] * 100, -30.81),
        ("Full Sharpe", m_full["sharpe"], 1.00),
        ("OOS CalCAGR", m_oos["cagr_cal"] * 100, 39.23),
        ("OOS Sharpe", m_oos["sharpe"], 1.43),
        ("OOS MDD", m_oos["mdd"] * 100, -15.00),
        ("IS CalCAGR", m_is["cagr_cal"] * 100, 11.71),
        ("IS MDD", m_is["mdd"] * 100, -30.81),
    ]
    print("\n  [리포트 기재 수치 vs 독립 재계산 오차]")
    for k, v_calc, v_rep in report_checks:
        print(f"    - {k:14s}: 계산={v_calc:+.4f}, 리포트={v_rep:+.2f}, 오차={abs(v_calc - v_rep):.4f}")

    section("3. D-3 낙폭 연동 감속(HWM -15%->10%, -8%->20%) 로직 독립 실증 감사")
    hwm = 100_000_000.0
    th = False
    recalc_th = []
    for eq_val in eq_d3["equity"].values:
        if eq_val > hwm:
            hwm = eq_val
        dd = eq_val / hwm - 1.0
        if not th and dd <= -0.15:
            th = True
        elif th and dd >= -0.08:
            th = False
        recalc_th.append(int(th))
    th_mismatch = int(np.sum(np.array(recalc_th) != eq_d3["throttled"].values))
    print(f"  - HWM 히스테리시스 상태 독립 재계산 불일치 일수: {th_mismatch}일 / {len(eq_d3)}일")
    print(f"  - 감속(throttled=1) 활성화 총 일수: {int(eq_d3['throttled'].sum())}일 ({eq_d3['throttled'].mean()*100:.1f}%)")

    # 전일(t-1) throttled 상태에 따른 당일(t) 진입 거래 비중 전수 검증
    eq_d3["prev_throttled"] = eq_d3["throttled"].shift(1).fillna(0).astype(int)
    prev_th_map = eq_d3.set_index("date")["prev_throttled"].to_dict()
    tr_d3["prev_throttled"] = tr_d3["entry_date"].map(prev_th_map)
    th_trades = tr_d3[tr_d3["prev_throttled"] == 1]
    norm_trades = tr_d3[tr_d3["prev_throttled"] == 0]
    th_cap_violations = int((th_trades["weight_at_entry"] > 0.1001).sum())
    norm_cap_violations = int((norm_trades["weight_at_entry"] > 0.2001).sum())
    print(f"  - 감속 구간 진입 거래: {len(th_trades)}건, 최대 진입비중={th_trades['weight_at_entry'].max()*100:.2f}%, 10% 한도 위반={th_cap_violations}건")
    print(f"  - 정상 구간 진입 거래: {len(norm_trades)}건, 최대 진입비중={norm_trades['weight_at_entry'].max()*100:.2f}%, 20% 한도 위반={norm_cap_violations}건")

    section("4. D-3 체결 363건 전수 DuckDB OHLCV 대조 & 비용 0.73% 검산")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    ohlcv_idx = con.execute("SELECT date, code, Open, High, Low, Close FROM daily_ohlcv").df().set_index(["date", "code"])
    con.close()
    buy_c = 0.00015 + 0.0025
    sell_c = 0.00015 + 0.0020 + 0.0025
    bad_ep = bad_xp = bad_cost = 0
    for _, r in tr_d3.iterrows():
        c, ed, xd = r["code"], r["entry_date"], r["exit_date"]
        ep, xp, sh = float(r["entry_price"]), float(r["exit_price"]), int(r["shares"])
        if abs(ep - float(ohlcv_idx.loc[(ed, c), "Open"])) > 1e-4:
            bad_ep += 1
        xrow = ohlcv_idx.loc[(xd, c)]
        if r["exit_reason"] in ("below_ma20", "max_hold_days"):
            if abs(xp - float(xrow["Open"])) > 1e-4:
                bad_xp += 1
        else:
            if not (float(xrow["Low"]) - 1e-4 <= xp <= float(xrow["High"]) + 1e-4):
                bad_xp += 1
        proc = sh * xp * (1.0 - sell_c)
        cb = sh * ep * (1.0 + buy_c)
        if abs((proc / cb - 1.0) - float(r["net_ret"])) > 1e-6 or abs((proc - cb) - float(r["pnl"])) > 1.0:
            bad_cost += 1
    print(f"  - 총 체결 {len(tr_d3)}건: 진입가 오류={bad_ep}건, 청산가 범위 오류={bad_xp}건, 0.73% 비용 오류={bad_cost}건")

    section("5. 독립 재실행 재현성 & 극단적 미래참조 교란 테스트 (D-3)")
    ohlcv, meta, bench, flow = bt.load_duckdb_data()
    data_by_code = bt.compute_indicators(ohlcv, meta, flow, bt.compute_benchmark_regimes(bench))
    all_dates = pd.to_datetime(bench[bench["code"] == "KS11"]["date"].unique())
    trading_dates = sorted([d for d in all_dates if pd.Timestamp(BT_START) <= d <= pd.Timestamp(DATA_END)])

    cfg_d3 = bt.DefenseConfig("D-3", "Audit D-3", False, "none", True, True)
    eq_rerun, tr_rerun, _ = bt.run_defense_case(cfg_d3, data_by_code, trading_dates, True)
    repro_diff = float(np.max(np.abs(eq_rerun["equity"].values - eq_d3["equity"].values)))
    print(f"  - [재현성] D-3 재실행 vs 저장 CSV 최대 절대 오차: {repro_diff:.12f} (거래수 {len(tr_rerun)} vs {len(tr_d3)})")

    base_is = eq_rerun[eq_rerun["date"] <= IS_END]["equity"].values
    for label, p_mult, f_mult, v_mult in [
        ("OOS 주가+500% & 수급-500% & 거래량+900%", 6.0, -5.0, 10.0),
        ("OOS 주가-80% 폭락 & 수급+500% & 거래량-90%", 0.2, 6.0, 0.1),
    ]:
        p_ohlcv = ohlcv.copy()
        m1 = p_ohlcv["date"] >= OOS_START
        for col in ["Open", "High", "Low", "Close"]:
            p_ohlcv[col] = p_ohlcv[col].astype(float)
            p_ohlcv.loc[m1, col] = p_ohlcv.loc[m1, col] * p_mult
        p_ohlcv["Volume"] = p_ohlcv["Volume"].astype(float)
        p_ohlcv.loc[m1, "Volume"] = p_ohlcv.loc[m1, "Volume"] * v_mult

        p_bench = bench.copy()
        m2 = p_bench["date"] >= OOS_START
        for col in ["Open", "High", "Low", "Close"]:
            p_bench[col] = p_bench[col].astype(float)
            p_bench.loc[m2, col] = p_bench.loc[m2, col] * p_mult

        p_flow = flow.copy()
        m3 = p_flow["date"] >= OOS_START
        for col in ["frgn_net_qty", "inst_net_qty", "frgn_net_val", "inst_net_val", "raw_close", "raw_volume"]:
            p_flow[col] = p_flow[col].astype(float)
            if col in ("raw_close",):
                p_flow.loc[m3, col] = p_flow.loc[m3, col] * p_mult
            elif col in ("raw_volume",):
                p_flow.loc[m3, col] = p_flow.loc[m3, col] * v_mult
            else:
                p_flow.loc[m3, col] = p_flow.loc[m3, col] * f_mult

        p_data = bt.compute_indicators(p_ohlcv, meta, p_flow, bt.compute_benchmark_regimes(p_bench))
        p_eq, _, _ = bt.run_defense_case(cfg_d3, p_data, trading_dates, True)
        diff = float(np.max(np.abs(p_eq[p_eq["date"] <= IS_END]["equity"].values - base_is)))
        print(f"  - [교란 테스트] {label} -> IS 최대 오차 = {diff:.12f}")

    section("6. 감사관 독립 Z3-Solver SLA 판정 (JSON 배제, CSV/DuckDB 직접 계산)")
    ks = bench[bench["code"] == "KS11"].sort_values("date")
    def k_met(s_dt, e_dt):
        sub = ks[(ks["date"] >= s_dt) & (ks["date"] <= e_dt)]
        c = sub["Close"].values.astype(float)
        yrs = (pd.Timestamp(sub["date"].iloc[-1]) - pd.Timestamp(sub["date"].iloc[0])).days / 365.25
        cagr = (c[-1] / c[0]) ** (1.0 / yrs) - 1.0
        dr = pd.Series(c).pct_change().dropna()
        shp = float((dr.mean() / dr.std()) * np.sqrt(252))
        mdd = float((c / np.maximum.accumulate(c) - 1.0).min())
        return cagr, shp, mdd

    kf_c, kf_s, kf_m = k_met(BT_START, DATA_END)
    ko_c, ko_s, ko_m = k_met(OOS_START, DATA_END)

    max_w = float(eq_d3["invested_weight"].max())
    min_cash = float(eq_d3["cash"].min())
    min_eq = float(eq_d3["equity"].min())
    max_pos = int(eq_d3["num_positions"].max())
    max_sw = float(tr_d3["weight_at_entry"].max())
    acc_err = float(np.max(np.abs(eq_d3["equity"] - (eq_d3["cash"] + eq_d3["stock_val"]))))

    s = Solver()
    s.add(RealVal(f"{max_w:.10f}") <= RealVal("1.0001"))
    s.add(RealVal(str(max_pos)) <= RealVal("5"))
    s.add(RealVal(f"{min_cash:.10f}") >= RealVal("0"))
    s.add(RealVal(f"{min_eq:.10f}") > RealVal("0"))
    s.add(RealVal(f"{max_sw:.10f}") <= RealVal("0.2001"))
    s.add(RealVal(str(th_cap_violations)) == RealVal("0"))
    s.add(RealVal(f"{acc_err:.10f}") <= RealVal("0.01"))
    s.add(RealVal(f"{m_oos['cagr_cal']:.10f}") > RealVal("0"))
    s.add(RealVal(f"{m_oos['sharpe']:.10f}") > RealVal("0"))
    s.add(RealVal(f"{m_full['cagr_cal']:.10f}") > RealVal(f"{kf_c:.10f}"))
    s.add(RealVal(f"{m_full['sharpe']:.10f}") > RealVal(f"{kf_s:.10f}"))
    s.add(RealVal(f"{m_full['mdd']:.10f}") > RealVal(f"{kf_m:.10f}"))
    s.add(RealVal(f"{m_oos['cagr_cal']:.10f}") > RealVal(f"{ko_c:.10f}"))
    s.add(RealVal(f"{m_oos['sharpe']:.10f}") > RealVal(f"{ko_s:.10f}"))
    s.add(RealVal(f"{m_oos['mdd']:.10f}") > RealVal(f"{ko_m:.10f}"))

    print(f"  - CSV 직접 추출: max_w={max_w:.6f}, min_cash={min_cash:,.2f}, min_eq={min_eq:,.2f}, max_pos={max_pos}, max_sw={max_sw:.6f}, acc_err={acc_err:.10f}")
    print(f"  - 감사관 독립 Z3-Solver 종합 SLA(1~7 + 회계항등식 + D-3 감속준수) 판정: {'SAT' if s.check() == sat else 'UNSAT'}")

    # 연도별 D-3 성과 요약
    eq_d3["year"] = eq_d3["date"].str[:4]
    print("\n  [D-3 연도별 성과 분해]")
    for yr, g in eq_d3.groupby("year"):
        e = g["equity"].values
        ret = (e[-1] / e[0] - 1.0) * 100
        mdd = (e / np.maximum.accumulate(e) - 1.0).min() * 100
        th_d = int(g["throttled"].sum())
        print(f"    {yr}: Return={ret:+7.2f}%, MDD={mdd:7.2f}%, ThrottledDays={th_d:3d}/{len(g)}")


if __name__ == "__main__":
    main()
