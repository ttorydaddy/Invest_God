"""
Invest_God Final Independent Audit Script for Phase 1-B-3 (Crypto Short-Term Turbo-1: M3_R15)
- Author: quant-auditor (Independent Red Team Auditor)
- Target: strategies/crypto_short/backtest_crypto_turbo.py & turbo evidence files (Read-Only)
"""

import json
import math
import sys
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from z3 import And, RealVal, Solver, sat

PROJECT_ROOT = Path(r"c:\Users\양태철\Desktop\Invest_God")
STRAT_DIR = PROJECT_ROOT / "strategies" / "crypto_short"
DB_PATH = STRAT_DIR / "crypto_short_data.duckdb"
RESULTS_JSON = STRAT_DIR / "turbo_results.json"

sys.path.insert(0, str(STRAT_DIR))
import backtest_crypto_turbo as bt
from backtest_crypto_short_v2 import load_data_4h


def section(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def indep_metrics(eq_df: pd.DataFrame, tr_df: pd.DataFrame, col: str = "equity"):
    eq = eq_df[col].to_numpy(dtype=np.float64)
    dts = pd.to_datetime(eq_df["date"])
    tot_ret = float(eq[-1] / eq[0] - 1.0)
    n_days = max(1, (dts.iloc[-1] - dts.iloc[0]).days)
    cagr = float((eq[-1] / eq[0]) ** (365.0 / n_days) - 1.0)
    dr = pd.Series(eq).pct_change().dropna().to_numpy()
    sharpe = float((np.mean(dr) / np.std(dr, ddof=1)) * math.sqrt(365.0)) if np.std(dr, ddof=1) > 1e-12 else 0.0
    mdd = float(np.min(eq / np.maximum.accumulate(eq) - 1.0))
    calmar = float(cagr / abs(mdd)) if abs(mdd) > 1e-12 else 0.0
    out = {"total_return": tot_ret, "cagr": cagr, "sharpe": sharpe, "mdd": mdd, "calmar": calmar}
    if tr_df is not None and len(tr_df) > 0:
        out["num_trades"] = len(tr_df)
        out["max_loss_acct"] = float(tr_df["account_impact_pct"].min())
        out["max_entry_w"] = float(tr_df["entry_weight"].max())
    return out


def main():
    section("1. 로직 무변경 수학적 증명 (M3_R07 vs Phase 1-B-2 M-3 동일성)")
    eq_v2 = pd.read_csv(STRAT_DIR / "equity_crypto_v2_M3_Turtle_Regime_DDThrottle_net.csv")
    eq_t0 = pd.read_csv(STRAT_DIR / "equity_turbo_M3_R07_net.csv")
    tr_v2 = pd.read_csv(STRAT_DIR / "trades_crypto_v2_M3_Turtle_Regime_DDThrottle_net.csv")
    tr_t0 = pd.read_csv(STRAT_DIR / "trades_turbo_M3_R07_net.csv")
    eq_diff = float(np.max(np.abs(eq_v2["equity"].values - eq_t0["equity"].values)))
    tr_pnl_diff = float(np.max(np.abs(tr_v2["net_pnl"].values - tr_t0["net_pnl"].values))) if len(tr_v2) == len(tr_t0) else -1.0
    print(f"  - Base M3_R07 vs 기존 감사통과 M-3 일별 Equity 최대 오차: {eq_diff:.12f}")
    print(f"  - Base M3_R07 vs 기존 감사통과 M-3 체결 건수({len(tr_t0)} vs {len(tr_v2)}) & PnL 최대 오차: {tr_pnl_diff:.12f}")

    section("2. Turbo-1 (M3_R15) 독립 재계산 vs 리포트 & JSON 교차 대조")
    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        res_json = json.load(f)

    eq_r15 = pd.read_csv(STRAT_DIR / "equity_turbo_M3_R15_net.csv")
    tr_r15 = pd.read_csv(STRAT_DIR / "trades_turbo_M3_R15_net.csv")
    print(f"  - M3_R15 equity rows={len(eq_r15)} (NaN={int(eq_r15.isna().sum().sum())}), trades rows={len(tr_r15)} (NaN={int(tr_r15.isna().sum().sum())})")

    dts = pd.to_datetime(eq_r15["date"])
    mf = indep_metrics(eq_r15, tr_r15)
    mi = indep_metrics(eq_r15[dts <= pd.Timestamp("2024-06-30")].reset_index(drop=True), tr_r15[tr_r15["period"] == "IS"])
    mo = indep_metrics(eq_r15[dts >= pd.Timestamp("2024-07-01")].reset_index(drop=True), tr_r15[tr_r15["period"] == "OOS"])

    for sname, calc in [("full", mf), ("is", mi), ("oos", mo)]:
        rep = res_json["cases"]["M3_R15"]["net"][sname]
        err = max(
            abs(calc["total_return"] - rep["total_return"]) * 100,
            abs(calc["cagr"] - rep["cagr"]) * 100,
            abs(calc["mdd"] - rep["mdd"]) * 100,
        )
        print(
            f"  [M3_R15 {sname:4s}] Ret={calc['total_return']*100:+7.2f}% | CAGR={calc['cagr']*100:+6.2f}% | "
            f"Shp={calc['sharpe']:+5.2f} | MDD={calc['mdd']*100:+6.2f}% | Calmar={calc['calmar']:4.2f} | "
            f"Trd={calc['num_trades']} | MaxAcctLoss={calc['max_loss_acct']*100:+.2f}% | diff={err:.8f}%p"
        )

    # 연도별 수익률
    eq_r15["year"] = eq_r15["date"].str[:4]
    prev_end = float(eq_r15["equity"].iloc[0])
    yr_list = []
    for yr, g in eq_r15.groupby("year"):
        ev = float(g["equity"].iloc[-1])
        yr_list.append(f"{yr}: {(ev/prev_end - 1)*100:+.1f}%")
        prev_end = ev
    print("  [M3_R15 연도별 독립 재계산]:", ", ".join(yr_list))

    section("3. M3_R15 체결 497건 전수 DuckDB 4h OHLCV 대조 & 비중/비용 검산")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    ohlcv = con.execute("SELECT symbol, CAST(open_time_utc AS VARCHAR) AS ts, open, high, low, close FROM ohlcv_4h").df()
    con.close()
    ohlcv["ts"] = pd.to_datetime(ohlcv["ts"]).astype(str)
    idx = ohlcv.set_index(["symbol", "ts"])

    bad_ep = bad_xp = bad_cost = cap_viol = 0
    side_cost = 0.0018
    for _, r in tr_r15.iterrows():
        sym, et, xt = r["symbol"], str(r["entry_time_utc"]), str(r["exit_time_utc"])
        rep_p, rxp_p = float(r["raw_entry_price"]), float(r["raw_exit_price"])
        erow, xrow = idx.loc[(sym, et)], idx.loc[(sym, xt)]
        if abs(rep_p - float(erow["open"])) > 1e-4:
            bad_ep += 1
        if r["exit_reason"] == "CHANNEL_EXIT":
            if abs(rxp_p - float(xrow["open"])) > 1e-4:
                bad_xp += 1
        else:
            if not (float(xrow["low"]) - 1e-4 <= rxp_p <= float(xrow["high"]) + 1e-4):
                bad_xp += 1
        exp_ret = (rxp_p * (1.0 - side_cost)) / (rep_p * (1.0 + side_cost)) - 1.0
        if abs(exp_ret - float(r["net_return"])) > 1e-4:
            bad_cost += 1
        if float(r["entry_weight"]) > 0.335:
            cap_viol += 1

    print(f"  - 총 체결 {len(tr_r15)}건: 진입가 오류={bad_ep}건, 청산가 범위 오류={bad_xp}건, 0.36% 비용 오류={bad_cost}건, 단일 33.5% 초과={cap_viol}건")
    print(f"  - 최대 단일 진입 비중={tr_r15['entry_weight'].max()*100:.2f}%, 최악 1회 계좌손실={tr_r15['account_impact_pct'].min()*100:.3f}%")

    section("4. 독립 재실행 재현성 & 극단적 미래 교란(+500% / -80%) & 비용 2배(0.72%) 스트레스")
    tl_4h, data_4h = load_data_4h(perturb_oos_mult=1.0)
    cfg_r15 = [c for c in bt.TURBO_CASES if c.code == "M3_R15"][0]

    eq_run, tr_run, _ = bt.run_turbo_case(cfg_r15, tl_4h, data_4h, side_cost=0.0018)
    repro_diff = float(np.max(np.abs(eq_run["equity"].values - eq_r15["equity"].values)))
    print(f"  - [재현성] M3_R15 재실행 vs 저장 CSV 최대 절대 오차: {repro_diff:.10f} (거래수 {len(tr_run)} vs {len(tr_r15)})")

    is_mask = pd.to_datetime(eq_run["date"]) <= pd.Timestamp("2024-06-30")
    base_is = eq_run.loc[is_mask, "equity"].to_numpy()
    for label, mult in [("OOS 가격 +500% (x6.0)", 6.0), ("OOS 가격 -80% 폭락 (x0.2)", 0.2)]:
        _, p_data = load_data_4h(perturb_oos_mult=mult)
        p_eq, _, _ = bt.run_turbo_case(cfg_r15, tl_4h, p_data, side_cost=0.0018)
        diff = float(np.max(np.abs(p_eq.loc[is_mask, "equity"].to_numpy() - base_is)))
        print(f"  - [교란 테스트] {label:26s} -> M3_R15 IS 오차 = {diff:.10f}")

    print("\n  [Red Team 비용/슬리피지 극한 스트레스 테스트 (M3_R15)]")
    for label, sc in [("기본 왕복 0.36% (편도 0.18%)", 0.0018), ("보수 왕복 0.52% (편도 0.26%)", 0.0026), ("극한 왕복 0.72% (편도 0.36%)", 0.0036)]:
        s_eq, s_tr, _ = bt.run_turbo_case(cfg_r15, tl_4h, data_4h, side_cost=sc)
        smf = indep_metrics(s_eq, s_tr)
        sdts = pd.to_datetime(s_eq["date"])
        smo = indep_metrics(s_eq[sdts >= pd.Timestamp("2024-07-01")].reset_index(drop=True), s_tr[s_tr["period"] == "OOS"])
        print(f"    {label}: Full Ret={smf['total_return']*100:+7.2f}%, CAGR={smf['cagr']*100:+6.2f}%, Shp={smf['sharpe']:.2f}, MDD={smf['mdd']*100:.2f}% | OOS Ret={smo['total_return']*100:+6.2f}%")

    section("5. 감사관 독립 Z3-Solver 종합 SLA 판정 (JSON 배제, CSV/DB 직접 계산)")
    bf = indep_metrics(eq_r15, None, "btc_bh_equity")
    bo = indep_metrics(eq_r15[dts >= pd.Timestamp("2024-07-01")].reset_index(drop=True), None, "btc_bh_equity")

    max_w = float(eq_r15["invested_weight"].max())
    min_cash = float(eq_r15["cash"].min())
    min_eq = float(eq_r15["equity"].min())
    max_pos = int(eq_r15["num_positions"].max())
    max_sw = float(tr_r15["entry_weight"].max())
    acc_err = float(np.max(np.abs(eq_r15["equity"] - (eq_r15["cash"] + eq_r15["invested_value"]))))

    s = Solver()
    s.add(RealVal(f"{max_w:.10f}") <= RealVal("1.0001"))
    s.add(RealVal(str(max_pos)) <= RealVal("4"))
    s.add(RealVal(f"{min_cash:.10f}") >= RealVal("0"))
    s.add(RealVal(f"{min_eq:.10f}") > RealVal("0"))
    s.add(RealVal(f"{max_sw:.10f}") <= RealVal("0.335"))
    s.add(RealVal(f"{acc_err:.10f}") <= RealVal("0.01"))
    s.add(RealVal(f"{mo['cagr']:.10f}") > RealVal("0"))
    s.add(RealVal(f"{mo['sharpe']:.10f}") > RealVal("0"))
    s.add(RealVal(f"{mf['cagr']:.10f}") > RealVal(f"{bf['cagr']:.10f}"))
    s.add(RealVal(f"{mf['sharpe']:.10f}") > RealVal(f"{bf['sharpe']:.10f}"))
    s.add(RealVal(f"{mf['mdd']:.10f}") > RealVal(f"{bf['mdd']:.10f}"))
    s.add(RealVal(f"{mo['cagr']:.10f}") > RealVal(f"{bo['cagr']:.10f}"))
    s.add(RealVal(f"{mo['sharpe']:.10f}") > RealVal(f"{bo['sharpe']:.10f}"))
    s.add(RealVal(f"{mo['mdd']:.10f}") > RealVal(f"{bo['mdd']:.10f}"))

    print(f"  - CSV 직접 추출(M3_R15): max_w={max_w:.6f}, min_cash={min_cash:,.2f}, min_eq={min_eq:,.2f}, max_pos={max_pos}, max_sw={max_sw:.5f}, acc_err={acc_err:.6f}")
    print(f"  - 감사관 독립 Z3-Solver 종합 SLA(1~8 + 회계항등식 + 33.5% 한도) 판정: {'SAT' if s.check() == sat else 'UNSAT'}")


if __name__ == "__main__":
    main()
