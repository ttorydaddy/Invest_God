"""
Invest_God Red Team Independent Audit Script for Phase 1-A-2 (KRX Surge V2)
- Author: quant-auditor (Independent Red Team Auditor)
- Target: strategies/krx_surge/* (Read-Only! Never modifies developer files)
"""

import json
import sys
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from z3 import And, Real, RealVal, Solver, sat

PROJECT_ROOT = Path(r"c:\Users\양태철\Desktop\Invest_God")
STRAT_DIR = PROJECT_ROOT / "strategies" / "krx_surge"
DB_PATH = STRAT_DIR / "krx_surge_data.duckdb"
RESULTS_JSON = STRAT_DIR / "backtest_results_v2.json"
Z3_JSON = STRAT_DIR / "z3_sla_verification_v2.json"

sys.path.insert(0, str(STRAT_DIR))
import backtest_krx_surge_v2 as bt

VARIANTS = ["A-0", "A-1", "A-2", "A-3", "A-R", "A-FR", "A-LR"]
BT_START = "2021-01-04"
IS_END = "2024-06-30"
OOS_START = "2024-07-01"
DATA_END = "2026-11-20"


def section(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def check_evidence_files():
    section("1. 원시 증거물 4종 완비 점검")
    required = [
        "backtest_krx_surge_v2.py",
        "verify_z3_sla_v2.py",
        "REPORT_KRX_SURGE_V2.md",
        "backtest_results_v2.json",
        "z3_sla_verification_v2.json",
        "krx_surge_data.duckdb",
    ]
    for v in VARIANTS:
        required.append(f"equity_v2_{v}_net.csv")
        required.append(f"trades_v2_{v}_net.csv")

    missing = []
    for fname in required:
        p = STRAT_DIR / fname
        if not p.exists() or p.stat().st_size == 0:
            missing.append(fname)
    print(f"총 필수 파일 {len(required)}개 중 누락/빈 파일: {len(missing)}개 {missing}")
    for v in VARIANTS:
        eq = pd.read_csv(STRAT_DIR / f"equity_v2_{v}_net.csv")
        tr = pd.read_csv(STRAT_DIR / f"trades_v2_{v}_net.csv")
        na_eq = int(eq.isna().sum().sum())
        na_tr = int(tr.isna().sum().sum())
        print(f"  [{v:4s}] equity rows={len(eq)} (NaN={na_eq}), trades rows={len(tr)} (NaN={na_tr})")
    return len(missing) == 0


def audit_duckdb():
    section("2. DuckDB 원천 데이터 & investor_flow 단위/무결성 감사")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
    print(f"DuckDB Tables: {tables}")

    for t in tables:
        cnt = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  Table '{t}': {cnt:,} rows")

    # 중복 체크
    dup_ohlcv = con.execute(
        "SELECT COUNT(*) FROM (SELECT date, code, COUNT(*) c FROM daily_ohlcv GROUP BY date, code HAVING c > 1)"
    ).fetchone()[0]
    dup_flow = con.execute(
        "SELECT COUNT(*) FROM (SELECT date, code, COUNT(*) c FROM investor_flow GROUP BY date, code HAVING c > 1)"
    ).fetchone()[0]
    dup_bench = con.execute(
        "SELECT COUNT(*) FROM (SELECT date, code, COUNT(*) c FROM benchmarks GROUP BY date, code HAVING c > 1)"
    ).fetchone()[0]
    print(f"중복 행 검사 -> daily_ohlcv: {dup_ohlcv}, investor_flow: {dup_flow}, benchmarks: {dup_bench}")

    # 날짜 범위
    for t in ["daily_ohlcv", "investor_flow", "benchmarks"]:
        mn, mx = con.execute(f"SELECT MIN(date), MAX(date) FROM {t}").fetchone()
        print(f"  {t} date range: {mn} ~ {mx}")

    # investor_flow 샘플 및 단위 교차 대조 (daily_ohlcv vs investor_flow)
    sample = con.execute("""
        SELECT o.date, o.code, m.name,
               o.Close AS adj_close, o.Volume AS adj_vol,
               f.raw_close, f.raw_volume,
               f.frgn_net_qty, f.inst_net_qty, f.indiv_net_qty,
               (f.frgn_net_qty + f.inst_net_qty + f.indiv_net_qty) AS sum_qty,
               f.frgn_net_val, f.inst_net_val,
               (f.frgn_net_qty + f.inst_net_qty) / NULLIF(f.raw_volume, 0) AS sf_ratio
        FROM daily_ohlcv o
        JOIN investor_flow f ON o.date = f.date AND o.code = f.code
        LEFT JOIN universe_meta m ON o.code = m.code
        WHERE o.code IN ('005930', '000660', '247540', '035720')
          AND o.date IN ('2021-01-04', '2023-06-15', '2025-07-01', '2026-11-19')
        ORDER BY o.code, o.date
    """).df()
    print("\n[샘플 대조: daily_ohlcv vs investor_flow]")
    print(sample.to_string(index=False))

    # 수급 비율 이상치(|smart_flow_ratio| > 1.0) 점검
    anomalies = con.execute("""
        SELECT COUNT(*) AS total_rows,
               SUM(CASE WHEN raw_volume <= 0 THEN 1 ELSE 0 END) AS zero_vol,
               SUM(CASE WHEN ABS((frgn_net_qty + inst_net_qty) / NULLIF(raw_volume, 0)) > 1.0 THEN 1 ELSE 0 END) AS gt_100pct,
               MAX((frgn_net_qty + inst_net_qty) / NULLIF(raw_volume, 0)) AS max_sf,
               MIN((frgn_net_qty + inst_net_qty) / NULLIF(raw_volume, 0)) AS min_sf
        FROM investor_flow
    """).df()
    print("\n[investor_flow 수급 비율 이상치 통계]")
    print(anomalies.to_string(index=False))

    # 수정주가 vs raw_close 괴리 (액면분할 종목 등)
    split_diff = con.execute("""
        SELECT o.code, m.name, COUNT(*) AS days,
               MIN(o.Close / NULLIF(f.raw_close, 0)) AS min_ratio,
               MAX(o.Close / NULLIF(f.raw_close, 0)) AS max_ratio
        FROM daily_ohlcv o
        JOIN investor_flow f ON o.date = f.date AND o.code = f.code
        LEFT JOIN universe_meta m ON o.code = m.code
        GROUP BY o.code, m.name
        HAVING ABS(MIN(o.Close / NULLIF(f.raw_close, 0)) - 1.0) > 0.3
            OR ABS(MAX(o.Close / NULLIF(f.raw_close, 0)) - 1.0) > 0.3
        ORDER BY min_ratio
        LIMIT 10
    """).df()
    print("\n[수정주가(ohlcv.Close) vs 비수정주가(flow.raw_close) 괴리 상위 종목 (액면분할/증자 등)]")
    print(split_diff.to_string(index=False))

    # 종목별 결측 거래일 확인 (벤치마크 거래일 대비)
    ks_days = con.execute(f"SELECT COUNT(DISTINCT date) FROM benchmarks WHERE code='KS11' AND date >= '{BT_START}' AND date <= '{DATA_END}'").fetchone()[0]
    partial_tickers = con.execute(f"""
        SELECT o.code, m.name, COUNT(DISTINCT o.date) AS days, MIN(o.date) AS first_dt, MAX(o.date) AS last_dt
        FROM daily_ohlcv o
        LEFT JOIN universe_meta m ON o.code = m.code
        WHERE o.date >= '{BT_START}' AND o.date <= '{DATA_END}'
        GROUP BY o.code, m.name
        HAVING COUNT(DISTINCT o.date) < {ks_days}
        ORDER BY days ASC
        LIMIT 15
    """).df()
    print(f"\n[KOSPI 거래일({ks_days}일) 미만 상장/결측 종목 (상위 15개)]")
    print(partial_tickers.to_string(index=False))
    con.close()


def independent_calc_metrics(eq_df: pd.DataFrame, tr_df: pd.DataFrame, start_date: str, end_date: str):
    sub = eq_df[(eq_df["date"] >= start_date) & (eq_df["date"] <= end_date)].copy()
    eq = sub["equity"].values.astype(float)
    total_ret = (eq[-1] / eq[0]) - 1.0
    n_days = len(sub)
    years_252 = n_days / 252.0
    cagr_252 = (eq[-1] / eq[0]) ** (1.0 / years_252) - 1.0

    # 실제 달력 일수 기준 CAGR도 산출
    dt0 = pd.Timestamp(sub["date"].iloc[0])
    dt1 = pd.Timestamp(sub["date"].iloc[-1])
    years_cal = (dt1 - dt0).days / 365.25
    cagr_cal = (eq[-1] / eq[0]) ** (1.0 / years_cal) - 1.0 if years_cal > 0 else 0.0

    dr = pd.Series(eq).pct_change().dropna()
    sharpe = float((dr.mean() / dr.std()) * np.sqrt(252)) if dr.std() > 0 else 0.0
    mdd = float((eq / np.maximum.accumulate(eq) - 1.0).min())

    sub_tr = tr_df[(tr_df["entry_date"] >= start_date) & (tr_df["entry_date"] <= end_date)]
    n_tr = len(sub_tr)
    if n_tr > 0:
        rets = sub_tr["net_ret"].values.astype(float)
        wins = rets[rets > 0]
        losses = rets[rets <= 0]
        win_rate = len(wins) / n_tr
        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
        avg_loss = float(abs(np.mean(losses))) if len(losses) > 0 else 0.0
        payoff = avg_win / avg_loss if avg_loss > 0 else 0.0
        pf = float(np.sum(wins) / abs(np.sum(losses))) if len(losses) > 0 and abs(np.sum(losses)) > 0 else 0.0
        avg_hold = float(sub_tr["holding_days"].mean())
    else:
        win_rate = payoff = pf = avg_hold = 0.0

    return {
        "n_days": n_days,
        "years_252": years_252,
        "years_cal": years_cal,
        "total_return": total_ret,
        "cagr_252": cagr_252,
        "cagr_cal": cagr_cal,
        "sharpe": sharpe,
        "mdd": mdd,
        "num_trades": n_tr,
        "win_rate": win_rate,
        "payoff_ratio": payoff,
        "profit_factor": pf,
        "avg_holding_days": avg_hold,
        "avg_exposure": float(sub["invested_weight"].mean()),
    }


def audit_metrics_and_trades():
    section("3. 독립 재계산 vs 리포트/JSON 교차 대조 (0.01%p 이내 일치 검증)")
    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        res_json = json.load(f)

    max_err_pct_pt = 0.0
    for v in VARIANTS:
        eq_df = pd.read_csv(STRAT_DIR / f"equity_v2_{v}_net.csv")
        tr_df = pd.read_csv(STRAT_DIR / f"trades_v2_{v}_net.csv")
        for seg_name, s_dt, e_dt in [("full", BT_START, DATA_END), ("is", BT_START, IS_END), ("oos", OOS_START, DATA_END)]:
            calc = independent_calc_metrics(eq_df, tr_df, s_dt, e_dt)
            rep = res_json["variants"][v]["net"][seg_name]

            diffs = {
                "cagr": abs(calc["cagr_252"] - rep["cagr"]) * 100,
                "total_ret": abs(calc["total_return"] - rep["total_return"]) * 100,
                "sharpe": abs(calc["sharpe"] - rep["sharpe"]),
                "mdd": abs(calc["mdd"] - rep["mdd"]) * 100,
                "win_rate": abs(calc["win_rate"] - rep["win_rate"]) * 100,
                "num_trades": abs(calc["num_trades"] - rep["num_trades"]),
            }
            seg_max = max(diffs["cagr"], diffs["total_ret"], diffs["mdd"], diffs["win_rate"])
            max_err_pct_pt = max(max_err_pct_pt, seg_max)
            if v == "A-FR":
                print(
                    f"  [A-FR {seg_name:4s}] CAGR(252d)={calc['cagr_252']*100:+.2f}% (Cal365={calc['cagr_cal']*100:+.2f}%) | "
                    f"Ret={calc['total_return']*100:+.2f}% | Shp={calc['sharpe']:.2f} | MDD={calc['mdd']*100:.2f}% | "
                    f"Trd={calc['num_trades']} | WR={calc['win_rate']*100:.1f}% | Payoff={calc['payoff_ratio']:.2f} | "
                    f"PF={calc['profit_factor']:.2f} | max_diff_vs_json={seg_max:.8f}%p"
                )
    print(f"전 7개 모델 x 3구간(Full/IS/OOS) JSON 대비 최대 오차: {max_err_pct_pt:.10f}%p (허용치 0.01%p)")

    # A-FR 매매 체결 내역 363건 전수 DuckDB OHLCV 대조 및 비용 공식 검산
    section("3-B. A-FR 전체 체결 내역(trades_v2_A-FR_net.csv) 363건 전수 DuckDB 대조")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    ohlcv = con.execute("SELECT date, code, Open, High, Low, Close FROM daily_ohlcv").df()
    con.close()
    ohlcv_idx = ohlcv.set_index(["date", "code"])

    tr = pd.read_csv(STRAT_DIR / "trades_v2_A-FR_net.csv", dtype={"code": str})
    tr["code"] = tr["code"].str.zfill(6)

    buy_c = 0.00015 + 0.0025
    sell_c = 0.00015 + 0.0020 + 0.0025

    bad_entry_px = 0
    bad_exit_range = 0
    bad_cost_formula = 0
    same_day_stop = 0
    for idx, row in tr.iterrows():
        c = row["code"]
        ed = row["entry_date"]
        xd = row["exit_date"]
        ep = float(row["entry_price"])
        xp = float(row["exit_price"])
        sh = int(row["shares"])

        # 1) 진입가가 당일 시가(Open)와 일치하는지
        o_entry = float(ohlcv_idx.loc[(ed, c), "Open"])
        if abs(ep - o_entry) > 1e-4:
            bad_entry_px += 1

        # 2) 청산가가 청산일 [Low, High] 범위 내인지 (시가 청산은 Open과 일치)
        xrow = ohlcv_idx.loc[(xd, c)]
        if row["exit_reason"] in ("below_ma20", "max_hold_days"):
            if abs(xp - float(xrow["Open"])) > 1e-4:
                bad_exit_range += 1
        else:
            if not (float(xrow["Low"]) - 1e-4 <= xp <= float(xrow["High"]) + 1e-4):
                bad_exit_range += 1

        if ed == xd:
            same_day_stop += 1

        # 3) 왕복 비용 0.73% 반영 검산
        expected_proceeds = sh * xp * (1.0 - sell_c)
        expected_cost = sh * ep * (1.0 + buy_c)
        expected_net_ret = (expected_proceeds / expected_cost) - 1.0
        expected_pnl = expected_proceeds - expected_cost
        if abs(expected_net_ret - float(row["net_ret"])) > 1e-6 or abs(expected_pnl - float(row["pnl"])) > 1.0:
            bad_cost_formula += 1

    print(f"총 체결 건수: {len(tr)}건")
    print(f"  - 진입가 != 시가(Open) 오류: {bad_entry_px}건")
    print(f"  - 청산가 범위 이탈([Low, High] / Open) 오류: {bad_exit_range}건")
    print(f"  - 거래비용(0.73%) PnL/수익률 계산 불일치: {bad_cost_formula}건")
    print(f"  - 당일 시가 진입 후 당일 장중 손절 발생: {same_day_stop}건")
    print("  - 청산 사유 분포:", tr["exit_reason"].value_counts().to_dict())

    # 마지막 날 미청산 포지션 확인
    eq_afr = pd.read_csv(STRAT_DIR / "equity_v2_A-FR_net.csv")
    last_row = eq_afr.iloc[-1]
    print(f"  - 최종일({last_row['date']}) 미청산 포지션 수: {int(last_row['num_positions'])}개 (보유비중 {last_row['invested_weight']*100:.2f}%)")


def audit_reproducibility_and_lookahead():
    section("4. 독립 재실행 재현성 & 극단적 미래참조(+500%/-500%/+1000%) 교란 테스트")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    ohlcv = con.execute("SELECT * FROM daily_ohlcv ORDER BY code, date").df()
    meta = con.execute("SELECT * FROM universe_meta").df()
    bench = con.execute("SELECT * FROM benchmarks ORDER BY code, date").df()
    flow = con.execute("SELECT * FROM investor_flow ORDER BY code, date").df()
    con.close()

    br = bt.compute_benchmark_regime(bench)
    data_by_code = bt.compute_enriched_indicators(ohlcv, meta, flow, br)
    all_dates = pd.to_datetime(bench[bench["code"] == "KS11"]["date"].unique())
    trading_dates = sorted([d for d in all_dates if pd.Timestamp(BT_START) <= d <= pd.Timestamp(DATA_END)])

    cfg_afr = bt.VariantConfig("A-FR", "Audit A-FR", True, False, True, True)
    eq_rerun, tr_rerun, audit_rerun = bt.run_variant(cfg_afr, data_by_code, trading_dates, apply_costs=True)

    # 기존 저장된 CSV와 100% 비트단위 재현성 대조
    eq_saved = pd.read_csv(STRAT_DIR / "equity_v2_A-FR_net.csv")
    repro_diff = float(np.max(np.abs(eq_rerun["equity"].values - eq_saved["equity"].values)))
    print(f"[재현성] A-FR 재실행 vs 저장된 CSV 최대 절대 오차: {repro_diff:.12f} (거래수 {len(tr_rerun)} vs {len(pd.read_csv(STRAT_DIR / 'trades_v2_A-FR_net.csv'))})")

    # 극단적 교란 테스트: OOS 가격 +500%(x6), 거래량 x10, 벤치마크 x6, 수급 -500%(x-5) + 반대 방향(-80% 폭락)
    base_is = eq_rerun[eq_rerun["date"] <= IS_END]["equity"].values

    for label, p_mult, f_mult, v_mult in [
        ("OOS 주가+500% & 수급-500% & 거래량+900%", 6.0, -5.0, 10.0),
        ("OOS 주가-80% 폭락 & 수급+500% & 거래량-90%", 0.2, 6.0, 0.1),
    ]:
        p_ohlcv = ohlcv.copy()
        m1 = p_ohlcv["date"] >= OOS_START
        for col in ["Open", "High", "Low", "Close"]:
            p_ohlcv.loc[m1, col] = p_ohlcv.loc[m1, col] * p_mult
        p_ohlcv.loc[m1, "Volume"] = p_ohlcv.loc[m1, "Volume"] * v_mult

        p_bench = bench.copy()
        m2 = p_bench["date"] >= OOS_START
        for col in ["Open", "High", "Low", "Close"]:
            p_bench.loc[m2, col] = p_bench.loc[m2, col] * p_mult

        p_flow = flow.copy()
        m3 = p_flow["date"] >= OOS_START
        for col in ["frgn_net_qty", "inst_net_qty", "frgn_net_val", "inst_net_val"]:
            p_flow.loc[m3, col] = p_flow.loc[m3, col] * f_mult

        p_br = bt.compute_benchmark_regime(p_bench)
        p_data = bt.compute_enriched_indicators(p_ohlcv, meta, p_flow, p_br)
        p_eq, _, _ = bt.run_variant(cfg_afr, p_data, trading_dates, apply_costs=True)
        p_is = p_eq[p_eq["date"] <= IS_END]["equity"].values
        max_diff = float(np.max(np.abs(p_is - base_is)))
        print(f"[교란 테스트] {label} -> IS 최대 오차 = {max_diff:.12f}")


def audit_independent_z3():
    section("5. 감사관 독립 Z3-Solver SLA 검증 (JSON 미신뢰, CSV/DuckDB 직접 계산)")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    bench = con.execute("SELECT * FROM benchmarks WHERE code='KS11' ORDER BY date").df()
    con.close()

    def b_met(s_dt, e_dt):
        sub = bench[(bench["date"] >= s_dt) & (bench["date"] <= e_dt)]
        c = sub["Close"].values.astype(float)
        yrs = len(sub) / 252.0
        cagr = (c[-1] / c[0]) ** (1.0 / yrs) - 1.0
        dr = pd.Series(c).pct_change().dropna()
        shp = float((dr.mean() / dr.std()) * np.sqrt(252))
        mdd = float((c / np.maximum.accumulate(c) - 1.0).min())
        return cagr, shp, mdd

    kf_cagr, kf_shp, kf_mdd = b_met(BT_START, DATA_END)
    ko_cagr, ko_shp, ko_mdd = b_met(OOS_START, DATA_END)

    eq_df = pd.read_csv(STRAT_DIR / "equity_v2_A-FR_net.csv")
    tr_df = pd.read_csv(STRAT_DIR / "trades_v2_A-FR_net.csv")
    nf = independent_calc_metrics(eq_df, tr_df, BT_START, DATA_END)
    no = independent_calc_metrics(eq_df, tr_df, OOS_START, DATA_END)

    max_w = float(eq_df["invested_weight"].max())
    min_cash = float(eq_df["cash"].min())
    min_eq = float(eq_df["equity"].min())
    max_pos = int(eq_df["num_positions"].max())
    max_single_w = float(tr_df["weight_at_entry"].max())
    # 모든 일별 계좌에서 equity == cash + stock_val 일치 여부
    accounting_err = float(np.max(np.abs(eq_df["equity"] - (eq_df["cash"] + eq_df["stock_val"]))))

    print(f"A-FR CSV 직접 추출: max_w={max_w:.6f}, min_cash={min_cash:,.2f}, min_eq={min_eq:,.2f}, max_pos={max_pos}, max_single_w={max_single_w:.6f}, accounting_err={accounting_err:.10f}")

    s = Solver()
    s.add(RealVal(f"{max_w:.10f}") <= RealVal("1.0001"))
    s.add(RealVal(str(max_pos)) <= RealVal("5"))
    s.add(RealVal(f"{min_cash:.10f}") >= RealVal("0"))
    s.add(RealVal(f"{min_eq:.10f}") > RealVal("0"))
    s.add(RealVal(f"{max_single_w:.10f}") <= RealVal("0.2001"))
    s.add(RealVal(f"{accounting_err:.10f}") <= RealVal("0.01"))
    s.add(RealVal(f"{no['cagr_252']:.10f}") > RealVal("0"))
    s.add(RealVal(f"{no['sharpe']:.10f}") > RealVal("0"))
    s.add(RealVal(f"{nf['cagr_252']:.10f}") > RealVal(f"{kf_cagr:.10f}"))
    s.add(RealVal(f"{nf['sharpe']:.10f}") > RealVal(f"{kf_shp:.10f}"))
    s.add(RealVal(f"{nf['mdd']:.10f}") > RealVal(f"{kf_mdd:.10f}"))
    s.add(RealVal(f"{no['cagr_252']:.10f}") > RealVal(f"{ko_cagr:.10f}"))
    s.add(RealVal(f"{no['sharpe']:.10f}") > RealVal(f"{ko_shp:.10f}"))
    s.add(RealVal(f"{no['mdd']:.10f}") > RealVal(f"{ko_mdd:.10f}"))

    res = s.check()
    print(f"감사관 독립 Z3-Solver 종합 SLA(1~8 + 회계항등식) 판정: {'SAT' if res == sat else 'UNSAT'}")


if __name__ == "__main__":
    check_evidence_files()
    audit_duckdb()
    audit_metrics_and_trades()
    audit_reproducibility_and_lookahead()
    audit_independent_z3()
