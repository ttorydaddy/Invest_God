"""
Invest_God Red Team Independent Audit Script for Phase 1-B-2 (Crypto Short-Term V2: M-1 ~ M-4)
- Author: quant-auditor (Independent Red Team Auditor)
- Target: strategies/crypto_short/* (Read-Only! Never modifies developer files)
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
RESULTS_JSON = STRAT_DIR / "backtest_crypto_v2_results.json"
Z3_JSON = STRAT_DIR / "z3_crypto_v2_verification.json"

sys.path.insert(0, str(STRAT_DIR))
import backtest_crypto_short_v2 as bt

MODEL_CODES = [
    "M1_Major3_Tight3pct",
    "M2_Turtle_2ATR_Risk07",
    "M3_Turtle_Regime_DDThrottle",
    "M4_Daily1D_Turtle_LowTurn",
]


def section(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def check_evidence():
    section("1. 원시 증거물 4종 완비 점검")
    req = [
        "crypto_short_data.duckdb",
        "backtest_crypto_short_v2.py",
        "verify_z3_crypto_v2.py",
        "REPORT_CRYPTO_SHORT_V2.md",
        "backtest_crypto_v2_results.json",
        "z3_crypto_v2_verification.json",
    ]
    for code in MODEL_CODES:
        req.append(f"equity_crypto_v2_{code}_net.csv")
        req.append(f"trades_crypto_v2_{code}_net.csv")
    missing = [f for f in req if not (STRAT_DIR / f).exists() or (STRAT_DIR / f).stat().st_size == 0]
    print(f"필수 파일 {len(req)}개 중 누락/빈 파일: {len(missing)}개 {missing}")
    for code in MODEL_CODES:
        eq = pd.read_csv(STRAT_DIR / f"equity_crypto_v2_{code}_net.csv")
        tr = pd.read_csv(STRAT_DIR / f"trades_crypto_v2_{code}_net.csv")
        print(f"  [{code:28s}] equity rows={len(eq)} (NaN={int(eq.isna().sum().sum())}), trades rows={len(tr)} (NaN={int(tr.isna().sum().sum())})")


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
        wins = tr_df[tr_df["net_return"] > 0]
        losses = tr_df[tr_df["net_return"] <= 0]
        out["num_trades"] = len(tr_df)
        out["win_rate"] = len(wins) / len(tr_df)
        out["pf"] = float(wins["net_pnl"].sum() / abs(losses["net_pnl"].sum())) if abs(losses["net_pnl"].sum()) > 1e-9 else 0.0
        out["avg_loss_acct"] = float(losses["account_impact_pct"].mean()) if len(losses) > 0 else 0.0
        out["max_loss_acct"] = float(tr_df["account_impact_pct"].min())
    return out


def audit_metrics_and_db():
    section("2. 독립 재계산 vs 리포트/JSON 교차 대조 (M-1 ~ M-4, 0.01%p 이내)")
    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        res_json = json.load(f)

    max_err_all = 0.0
    for code in MODEL_CODES:
        eq = pd.read_csv(STRAT_DIR / f"equity_crypto_v2_{code}_net.csv")
        tr = pd.read_csv(STRAT_DIR / f"trades_crypto_v2_{code}_net.csv")
        dts = pd.to_datetime(eq["date"])
        segs = [
            ("full", eq, tr),
            ("is", eq[dts <= pd.Timestamp("2024-06-30")].reset_index(drop=True), tr[tr["period"] == "IS"]),
            ("oos", eq[dts >= pd.Timestamp("2024-07-01")].reset_index(drop=True), tr[tr["period"] == "OOS"]),
        ]
        for sname, s_eq, s_tr in segs:
            calc = indep_metrics(s_eq, s_tr)
            rep = res_json["models"][code]["net"][sname]
            err = max(
                abs(calc["total_return"] - rep["total_return"]) * 100,
                abs(calc["cagr"] - rep["cagr"]) * 100,
                abs(calc["mdd"] - rep["mdd"]) * 100,
            )
            max_err_all = max(max_err_all, err)
            if code in ("M2_Turtle_2ATR_Risk07", "M3_Turtle_Regime_DDThrottle"):
                print(
                    f"  [{code[:4]} {sname:4s}] Ret={calc['total_return']*100:+7.2f}% | CAGR={calc['cagr']*100:+6.2f}% | "
                    f"Shp={calc['sharpe']:+5.2f} | MDD={calc['mdd']*100:+6.2f}% | Trd={calc['num_trades']:3d} | "
                    f"MaxAcctLoss={calc['max_loss_acct']*100:+.2f}% | diff={err:.8f}%p"
                )
    print(f"전 4개 모델 x 3구간 JSON 대비 최대 오차: {max_err_all:.10f}%p (허용치 0.01%p)")

    # M-3 연도별 수익률 독립 검산
    eq_m3 = pd.read_csv(STRAT_DIR / "equity_crypto_v2_M3_Turtle_Regime_DDThrottle_net.csv")
    eq_m3["year"] = eq_m3["date"].str[:4]
    prev_end = float(eq_m3["equity"].iloc[0])
    yr_str = []
    for yr, g in eq_m3.groupby("year"):
        end_v = float(g["equity"].iloc[-1])
        yr_str.append(f"{yr}: {(end_v/prev_end - 1)*100:+.1f}%")
        prev_end = end_v
    print("  [M-3 연도별 독립 재계산]:", ", ".join(yr_str))

    section("3. M-3 체결 502건 전수 DuckDB 4h OHLCV 대조 & 비용 0.36% 검산")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    dup = con.execute("SELECT COUNT(*) FROM (SELECT symbol, open_time_utc, COUNT(*) c FROM ohlcv_4h GROUP BY 1,2 HAVING c > 1)").fetchone()[0]
    tot_rows = con.execute("SELECT COUNT(*) FROM ohlcv_4h").fetchone()[0]
    print(f"  DuckDB ohlcv_4h: {tot_rows:,} rows, duplicates={dup}")
    ohlcv = con.execute("SELECT symbol, CAST(open_time_utc AS VARCHAR) AS ts, open, high, low, close FROM ohlcv_4h").df()
    con.close()
    ohlcv["ts"] = pd.to_datetime(ohlcv["ts"]).astype(str)
    idx = ohlcv.set_index(["symbol", "ts"])

    tr_m3 = pd.read_csv(STRAT_DIR / "trades_crypto_v2_M3_Turtle_Regime_DDThrottle_net.csv")
    bad_ep = bad_xp = bad_cost = gap_stops = same_bar_stop = 0
    side_cost = 0.0018

    for _, r in tr_m3.iterrows():
        sym, et, xt = r["symbol"], str(r["entry_time_utc"]), str(r["exit_time_utc"])
        rep_p, rxp_p = float(r["raw_entry_price"]), float(r["raw_exit_price"])
        eep, exp = float(r["exec_entry_price"]), float(r["exec_exit_price"])

        erow = idx.loc[(sym, et)]
        xrow = idx.loc[(sym, xt)]

        if abs(rep_p - float(erow["open"])) > 1e-4:
            bad_ep += 1

        if r["exit_reason"] == "CHANNEL_EXIT":
            if abs(rxp_p - float(xrow["open"])) > 1e-4:
                bad_xp += 1
        else:
            if not (float(xrow["low"]) - 1e-4 <= rxp_p <= float(xrow["high"]) + 1e-4):
                bad_xp += 1
            if abs(rxp_p - float(xrow["open"])) < 1e-6 and float(xrow["open"]) < float(xrow["high"]):
                gap_stops += 1

        if et == xt:
            same_bar_stop += 1

        # 왕복 비용 0.36% 공식 검산
        exp_eep = rep_p * (1.0 + side_cost)
        exp_exp = rxp_p * (1.0 - side_cost)
        exp_ret = (exp_exp / exp_eep) - 1.0
        if abs(exp_ret - float(r["net_return"])) > 1e-4:
            bad_cost += 1

    print(f"  총 체결 {len(tr_m3)}건:")
    print(f"    - 진입가 != 4h 시가(open) 오류: {bad_ep}건")
    print(f"    - 청산가 4h [low, high]/open 이탈 오류: {bad_xp}건")
    print(f"    - 왕복 0.36% 비용 수익률 불일치: {bad_cost}건")
    print(f"    - 진입 당봉(same-bar) 손절: {same_bar_stop}건, 청산 사유: {tr_m3['exit_reason'].value_counts().to_dict()}")

    # 종목별 PnL 쏠림도
    tot_pnl = tr_m3["net_pnl"].sum()
    sym_pnl = tr_m3.groupby("symbol")["net_pnl"].agg(["count", "sum"]).sort_values("sum", ascending=False)
    sym_pnl["share_pct"] = sym_pnl["sum"] / tot_pnl * 100
    print(f"\n  [M-3 종목별 실현 PnL 분해 (총 PnL={tot_pnl:,.2f} USDT)]")
    print(sym_pnl.to_string())


def audit_repro_lookahead_and_stress():
    section("4. 독립 재실행 재현성 & 극단적 미래교란 테스트(+500% / -80%) & 슬리피지 스트레스")
    tl_4h, data_4h = bt.load_data_4h(perturb_oos_mult=1.0)
    cfg_m3 = [m for m in bt.MODELS if m.code == "M3_Turtle_Regime_DDThrottle"][0]
    cfg_m2 = [m for m in bt.MODELS if m.code == "M2_Turtle_2ATR_Risk07"][0]

    eq_m3_saved = pd.read_csv(STRAT_DIR / "equity_crypto_v2_M3_Turtle_Regime_DDThrottle_net.csv")
    eq_m3_run, tr_m3_run, _ = bt.run_model(cfg_m3, tl_4h, data_4h, side_cost=0.0018)
    repro_diff = float(np.max(np.abs(eq_m3_run["equity"].values - eq_m3_saved["equity"].values)))
    print(f"  [재현성] M-3 재실행 vs 저장 CSV 최대 절대 오차: {repro_diff:.10f} (거래수 {len(tr_m3_run)} vs 502)")

    # 극단적 교란 테스트: +500%(x6.0) 및 -80%(x0.2)
    is_mask = pd.to_datetime(eq_m3_run["date"]) <= pd.Timestamp("2024-06-30")
    base_is_m3 = eq_m3_run.loc[is_mask, "equity"].to_numpy()
    eq_m2_run, _, _ = bt.run_model(cfg_m2, tl_4h, data_4h, side_cost=0.0018)
    base_is_m2 = eq_m2_run.loc[is_mask, "equity"].to_numpy()

    for label, mult in [("OOS 가격 +500% (x6.0)", 6.0), ("OOS 가격 -80% 폭락 (x0.2)", 0.2)]:
        _, p_data = bt.load_data_4h(perturb_oos_mult=mult)
        p_m3, _, _ = bt.run_model(cfg_m3, tl_4h, p_data, side_cost=0.0018)
        p_m2, _, _ = bt.run_model(cfg_m2, tl_4h, p_data, side_cost=0.0018)
        d3 = float(np.max(np.abs(p_m3.loc[is_mask, "equity"].to_numpy() - base_is_m3)))
        d2 = float(np.max(np.abs(p_m2.loc[is_mask, "equity"].to_numpy() - base_is_m2)))
        print(f"  [교란 테스트] {label:28s} -> M-3 IS 오차={d3:.10f}, M-2 IS 오차={d2:.10f}")

    # 슬리피지 2배(왕복 0.52%) / 3배(왕복 0.68%) 스트레스 테스트 (실전 급변동 감안)
    print("\n  [Red Team 비용/슬리피지 스트레스 테스트 (M-3)]")
    for label, sc in [("기본 왕복 0.36% (편도 0.18%)", 0.0018), ("보수 왕복 0.52% (편도 0.26%)", 0.0026), ("극한 왕복 0.72% (편도 0.36%)", 0.0036)]:
        s_eq, s_tr, _ = bt.run_model(cfg_m3, tl_4h, data_4h, side_cost=sc)
        m_f = indep_metrics(s_eq, s_tr)
        dts = pd.to_datetime(s_eq["date"])
        m_o = indep_metrics(s_eq[dts >= pd.Timestamp("2024-07-01")].reset_index(drop=True), s_tr[s_tr["period"] == "OOS"])
        print(f"    {label}: Full Ret={m_f['total_return']*100:+6.2f}%, CAGR={m_f['cagr']*100:+5.2f}%, Shp={m_f['sharpe']:.2f}, MDD={m_f['mdd']*100:.2f}% | OOS Ret={m_o['total_return']*100:+5.2f}%")


def audit_independent_z3():
    section("5. 감사관 독립 Z3-Solver 종합 판정 (JSON 배제, CSV/DuckDB 직접 계산)")
    eq_m3 = pd.read_csv(STRAT_DIR / "equity_crypto_v2_M3_Turtle_Regime_DDThrottle_net.csv")
    tr_m3 = pd.read_csv(STRAT_DIR / "trades_crypto_v2_M3_Turtle_Regime_DDThrottle_net.csv")
    dts = pd.to_datetime(eq_m3["date"])

    mf = indep_metrics(eq_m3, tr_m3, "equity")
    mo = indep_metrics(eq_m3[dts >= pd.Timestamp("2024-07-01")].reset_index(drop=True), tr_m3[tr_m3["period"] == "OOS"], "equity")
    bf = indep_metrics(eq_m3, None, "btc_bh_equity")
    bo = indep_metrics(eq_m3[dts >= pd.Timestamp("2024-07-01")].reset_index(drop=True), None, "btc_bh_equity")

    max_w = float(eq_m3["invested_weight"].max())
    min_cash = float(eq_m3["cash"].min())
    min_eq = float(eq_m3["equity"].min())
    max_pos = int(eq_m3["num_positions"].max())
    max_entry_w = float(tr_m3["entry_weight"].max())
    worst_loss = float(tr_m3["account_impact_pct"].min())
    acc_err = float(np.max(np.abs(eq_m3["equity"] - (eq_m3["cash"] + eq_m3["invested_value"]))))

    print(f"  CSV 직접 추출(M-3): max_w={max_w:.6f}, min_cash={min_cash:,.2f}, min_eq={min_eq:,.2f}, max_pos={max_pos}, max_entry_w={max_entry_w:.5f}, worst_loss={worst_loss*100:.3f}%, acc_err={acc_err:.6f}")

    s = Solver()
    s.add(RealVal(f"{max_w:.10f}") <= RealVal("1.0001"))
    s.add(RealVal(str(max_pos)) <= RealVal("4"))
    s.add(RealVal(f"{min_cash:.10f}") >= RealVal("0"))
    s.add(RealVal(f"{min_eq:.10f}") > RealVal("0"))
    s.add(RealVal(f"{max_entry_w:.10f}") <= RealVal("0.2501"))
    s.add(RealVal(f"{worst_loss:.10f}") >= RealVal("-0.015"))
    s.add(RealVal(f"{acc_err:.10f}") <= RealVal("0.01"))
    s.add(RealVal(f"{mo['cagr']:.10f}") > RealVal("0"))
    s.add(RealVal(f"{mo['sharpe']:.10f}") > RealVal("0"))
    s.add(RealVal(f"{mf['cagr']:.10f}") > RealVal(f"{bf['cagr']:.10f}"))
    s.add(RealVal(f"{mf['sharpe']:.10f}") > RealVal(f"{bf['sharpe']:.10f}"))
    s.add(RealVal(f"{mf['mdd']:.10f}") > RealVal(f"{bf['mdd']:.10f}"))
    s.add(RealVal(f"{mo['cagr']:.10f}") > RealVal(f"{bo['cagr']:.10f}"))
    s.add(RealVal(f"{mo['sharpe']:.10f}") > RealVal(f"{bo['sharpe']:.10f}"))
    s.add(RealVal(f"{mo['mdd']:.10f}") > RealVal(f"{bo['mdd']:.10f}"))

    print(f"  감사관 독립 Z3-Solver 종합 SLA(1~9 + 회계항등식) 판정 (M-3): {'SAT' if s.check() == sat else 'UNSAT'}")


if __name__ == "__main__":
    check_evidence()
    audit_metrics_and_db()
    audit_repro_lookahead_and_stress()
    audit_independent_z3()
