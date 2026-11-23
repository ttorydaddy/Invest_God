"""
Invest_God Phase 1-C: 3-Bucket 마지막 바구니 — 장기 코어(Long-Term Core) 전략 3종 백테스트
- 평가 기간: 2021-01-04 ~ 2026-11-19 (1,439 KRX 거래일; 워밍업 2020-01-02부터 실데이터)
  - IS (60%): 2021-01-04 ~ 2024-06-28 | OOS (40%): 2024-07-01 ~ 2026-11-19
- 실자산 (전 종목 2020-01-02 이전 상장 KRX ETF + 원화 BTC-KRW, 백필 왜곡 0%):
  - 069500 (KODEX 200), 133690 (TIGER 미국나스닥100), 143850 (TIGER 미국S&P500선물(H)),
    132030 (KODEX 골드선물(H)), 148070 (KOSEF 국고채10년), 153130 (KODEX 단기채권), BTC-KRW
- 보수적 거래비용: KRX ETF 왕복 0.15% (편도 0.075%, 거래세 0%), BTC 왕복 0.36% (편도 0.18%)
- Zero Look-Ahead: 월말 종가 확정 신호 -> 익월 첫 거래일 시가 리밸런싱 + OOS +500% 교란 검증 + Z3 SLA 검증
- 3-Bucket 상관관계: 기존 확정 중기(D-3) & 단기(Turbo-1)와 일별 수익률 상관계수 산출
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import duckdb
import FinanceDataReader as fdr
import numpy as np
import pandas as pd
import yfinance as yf
from z3 import And, Real, RealVal, Solver, sat

# ==============================================================================
# 1. 전역 설정
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "longterm_core_data.duckdb"
RESULTS_PATH = BASE_DIR / "longterm_results.json"
Z3_PATH = BASE_DIR / "z3_longterm_verification.json"

D3_EQUITY_PATH = BASE_DIR.parent / "krx_surge" / "equity_def_D-3_net.csv"
TURBO_EQUITY_PATH = BASE_DIR.parent / "crypto_short" / "equity_turbo_M3_R15_net.csv"

DATA_START = "2020-01-01"
BT_START = "2021-01-04"
IS_END = "2024-06-28"
OOS_START = "2024-07-01"
DATA_END = "2026-11-19"

INITIAL_CAPITAL = 100_000_000.0

ASSETS = {
    "KR_EQ": ("069500", "KODEX 200", 0.00075),
    "US_NDX": ("133690", "TIGER 미국나스닥100", 0.00075),
    "US_SPX": ("143850", "TIGER 미국S&P500선물(H)", 0.00075),
    "GOLD": ("132030", "KODEX 골드선물(H)", 0.00075),
    "KR_BOND": ("148070", "KOSEF 국고채10년", 0.00075),
    "CASH_BOND": ("153130", "KODEX 단기채권", 0.00075),
    "BTC": ("BTC-KRW", "Bitcoin (KRW)", 0.00180),
}


# ==============================================================================
# 2. 데이터 수집 및 DuckDB 캐싱 (실자산 100%, 백필 왜곡 0%)
# ==============================================================================
def sync_longterm_data(force_refresh: bool = False) -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH))
    tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
    if not force_refresh and "asset_ohlcv" in tables:
        cnt = con.execute("SELECT COUNT(DISTINCT asset) FROM asset_ohlcv").fetchone()[0]
        if cnt == len(ASSETS):
            df = con.execute("SELECT * FROM asset_ohlcv ORDER BY asset, date").df()
            con.close()
            print(f"[DuckDB Cache] Loaded {cnt} assets ({len(df):,} rows) from {DB_PATH.name}")
            return df

    print("[Data Sync] Downloading 6 KRX ETFs + BTC-KRW (2020-01-01 ~ 2026-11-20)...")
    frames = []
    for key, (ticker, name, _cost) in ASSETS.items():
        if key == "BTC":
            raw = yf.Ticker(ticker).history(start=DATA_START, end="2026-11-21")
            raw = raw.reset_index()
            raw = raw.rename(columns={raw.columns[0]: "date"})
            raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None).dt.strftime("%Y-%m-%d")
        else:
            raw = fdr.DataReader(ticker, DATA_START, "2026-11-20").reset_index()
            raw = raw.rename(columns={raw.columns[0]: "date"})
            raw["date"] = pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d")

        sub = raw[["date", "Open", "High", "Low", "Close", "Volume"]].copy()
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
        sub["asset"] = key
        sub["ticker"] = ticker
        sub["name"] = name
        sub = sub.dropna(subset=["Open", "Close"])
        frames.append(sub)
        print(f"  {key:9s} ({ticker:7s} {name}): {len(sub)} rows ({sub['date'].iloc[0]} ~ {sub['date'].iloc[-1]})")

    all_df = pd.concat(frames, ignore_index=True)
    con.execute("DROP TABLE IF EXISTS asset_ohlcv")
    con.execute("CREATE TABLE asset_ohlcv AS SELECT * FROM all_df")
    con.close()
    print(f"[Data Sync] Saved {len(all_df):,} rows to {DB_PATH.name}")
    return all_df


def build_aligned_panels(all_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """KRX 거래일 캘린더(KODEX 200 기준)에 맞춰 Open/Close 패널 정렬 (BTC는 해당 거래일 가격 매칭)"""
    krx_dates = sorted(all_df[all_df["asset"] == "KR_EQ"]["date"].unique())
    idx = pd.to_datetime(krx_dates)

    open_panel = pd.DataFrame(index=idx)
    close_panel = pd.DataFrame(index=idx)

    for key in ASSETS:
        sub = all_df[all_df["asset"] == key].copy()
        sub["date_dt"] = pd.to_datetime(sub["date"])
        sub = sub.drop_duplicates("date_dt").set_index("date_dt").sort_index()
        # KRX 거래일에 정렬 (BTC는 주말 포함 연중무휴라 모든 KRX 거래일에 정확히 존재, 혹시 결측 시 과거 ffill만 사용 -> Zero Look-Ahead)
        open_panel[key] = sub["Open"].reindex(idx).ffill()
        close_panel[key] = sub["Close"].reindex(idx).ffill()

    return open_panel, close_panel


# ==============================================================================
# 3. 전략별 월말 목표 비중 계산기 (Strict Causal: t일 종가까지 확정 데이터만 사용)
# ==============================================================================
def compute_target_weights(strategy: str, close_hist: pd.DataFrame) -> Dict[str, float]:
    """
    close_hist: t일(월말)까지의 종가 이력 (미래 데이터 포함 불가!)
    반환값: 자산별 목표 비중 dict (합계 <= 1.0)
    """
    if strategy == "BM_KOSPI":
        return {"KR_EQ": 1.0}

    if strategy == "BM_6040":
        return {"US_NDX": 0.30, "KR_EQ": 0.30, "KR_BOND": 0.40}

    if strategy == "L-1_DualMomentum":
        # 공격 자산 5종: 3개월(63d) + 6개월(126d) 평균 모멘텀 상위 3개 (BTC 최대 25% 캡)
        # 절대 모멘텀(단기채권 대비) 음수 슬롯은 전부 CASH_BOND 대피
        offense = ["KR_EQ", "US_NDX", "US_SPX", "GOLD", "BTC"]
        c_now = close_hist.iloc[-1]
        c_63 = close_hist.iloc[-64]
        c_126 = close_hist.iloc[-127]

        scores = {}
        for a in offense + ["CASH_BOND"]:
            r3 = (c_now[a] / c_63[a]) - 1.0
            r6 = (c_now[a] / c_126[a]) - 1.0
            scores[a] = 0.5 * r3 + 0.5 * r6

        cash_score = scores["CASH_BOND"]
        ranked = sorted(offense, key=lambda a: scores[a], reverse=True)
        top3 = ranked[:3]

        # 기본 슬롯 비중: BTC 포함 시 BTC 25%, 나머지 2개 37.5%씩 / 미포함 시 33.33%씩
        if "BTC" in top3:
            slot_w = {a: (0.25 if a == "BTC" else 0.375) for a in top3}
        else:
            slot_w = {a: 1.0 / 3.0 for a in top3}

        weights = {}
        cash_alloc = 0.0
        for a, w in slot_w.items():
            if scores[a] > cash_score:
                weights[a] = w
            else:
                cash_alloc += w
        if cash_alloc > 0:
            weights["CASH_BOND"] = cash_alloc
        return weights

    if strategy == "L-2_RiskParity_Trend":
        # 5대 자산 최근 60일 변동성 역가중 + 120일 이평선 하회 자산은 CASH_BOND로 자동 전환
        universe = ["KR_EQ", "US_NDX", "GOLD", "KR_BOND", "BTC"]
        rets = close_hist[universe].pct_change().iloc[-60:]
        vols = rets.std()
        inv_vol = 1.0 / vols.replace(0, np.nan)
        base_w = inv_vol / inv_vol.sum()

        ma120 = close_hist[universe].iloc[-120:].mean()
        c_now = close_hist[universe].iloc[-1]

        weights = {}
        cash_alloc = 0.0
        for a in universe:
            w = float(base_w[a])
            if c_now[a] > ma120[a]:
                weights[a] = w
            else:
                cash_alloc += w
        if cash_alloc > 0:
            weights["CASH_BOND"] = cash_alloc
        return weights

    if strategy == "L-3_Barbell_Regime":
        # 공격형 바벨: 나스닥100 40%, 코스피200 30%, BTC 30%
        # 각 자산이 120일선 위면 보유, 아래면 해당 비중을 금 50% + 단기채 50%로 방어 전환
        core_targets = {"US_NDX": 0.40, "KR_EQ": 0.30, "BTC": 0.30}
        ma120 = close_hist[list(core_targets.keys())].iloc[-120:].mean()
        c_now = close_hist[list(core_targets.keys())].iloc[-1]

        weights = {"GOLD": 0.0, "CASH_BOND": 0.0}
        for a, w in core_targets.items():
            if c_now[a] > ma120[a]:
                weights[a] = w
            else:
                weights["GOLD"] += w * 0.5
                weights["CASH_BOND"] += w * 0.5
        return {k: v for k, v in weights.items() if v > 1e-9}

    raise ValueError(f"Unknown strategy: {strategy}")


# ==============================================================================
# 4. 월간 리밸런싱 백테스트 엔진 (Zero Look-Ahead)
# ==============================================================================
@dataclass
class RebalLog:
    rebal_date: str
    signal_date: str
    turnover_oneway: float
    tx_cost_krw: float
    equity_after: float
    weights_json: str


def run_longterm_strategy(
    strategy: str,
    open_panel: pd.DataFrame,
    close_panel: pd.DataFrame,
    apply_costs: bool = True,
) -> Tuple[pd.DataFrame, List[RebalLog], Dict]:
    bt_dates = [d for d in close_panel.index if pd.Timestamp(BT_START) <= d <= pd.Timestamp(DATA_END)]

    # 매월 첫 거래일 식별 (리밸런싱 실행일) -> 직전 거래일(전월 말일) 종가로 신호 생성!
    rebal_dates = set()
    prev_ym = None
    for d in bt_dates:
        ym = (d.year, d.month)
        if ym != prev_ym:
            rebal_dates.add(d)
            prev_ym = ym

    cash = INITIAL_CAPITAL
    holdings: Dict[str, float] = {k: 0.0 for k in ASSETS}  # 보유 수량 (float 허용: BTC 소수점/ETF 정수 근사 오차 제거하되 비용 정확 차감)
    rebal_logs: List[RebalLog] = []
    daily_logs = []

    max_observed_weight = 0.0
    min_observed_cash = INITIAL_CAPITAL
    min_observed_equity = INITIAL_CAPITAL
    total_oneway_turnover = 0.0
    total_tx_cost = 0.0

    for dt in bt_dates:
        # (1) 매월 첫 거래일 시가 리밸런싱 (직전 거래일 종가까지의 히스토리만 사용!)
        if dt in rebal_dates:
            # 직전 거래일 인덱스
            pos_idx = close_panel.index.get_loc(dt)
            sig_hist = close_panel.iloc[:pos_idx]  # STRICT CAUSAL: dt 당일 데이터 절대 미포함!
            sig_date = sig_hist.index[-1]

            target_w = compute_target_weights(strategy, sig_hist)

            # 현재 시가 기준 평가자산
            open_px = open_panel.loc[dt]
            cur_vals = {a: holdings[a] * float(open_px[a]) for a in ASSETS}
            mtm_open = cash + sum(cur_vals.values())

            # 목표 금액 vs 현재 금액 차이 계산 후 매도 먼저, 매수 나중 (cash >= 0 엄격 보장)
            target_vals = {a: mtm_open * target_w.get(a, 0.0) for a in ASSETS}
            oneway_trade_val = 0.0
            step_cost = 0.0

            # 1단계: 비중 축소 자산 매도
            for a in ASSETS:
                diff = target_vals[a] - cur_vals[a]
                if diff < -1.0:
                    sell_val = -diff
                    cost_rate = ASSETS[a][2] if apply_costs else 0.0
                    c_krw = sell_val * cost_rate
                    sell_qty = sell_val / float(open_px[a])
                    holdings[a] = max(0.0, holdings[a] - sell_qty)
                    cash += (sell_val - c_krw)
                    oneway_trade_val += sell_val
                    step_cost += c_krw

            # 2단계: 비중 확대 자산 매수 (거래비용 차감 후 가용 현금 내에서 비례 배분하여 cash >= 0 수학적 보장)
            buy_needs = {}
            for a in ASSETS:
                diff = target_vals[a] - (holdings[a] * float(open_px[a]))
                if diff > 1.0:
                    buy_needs[a] = diff

            total_buy_gross = sum(val * (1.0 + (ASSETS[a][2] if apply_costs else 0.0)) for a, val in buy_needs.items())
            scale = min(1.0, cash / total_buy_gross) if total_buy_gross > 0 else 0.0

            for a, raw_val in buy_needs.items():
                buy_val = raw_val * scale
                cost_rate = ASSETS[a][2] if apply_costs else 0.0
                c_krw = buy_val * cost_rate
                total_out = buy_val + c_krw
                if total_out > cash:
                    buy_val = cash / (1.0 + cost_rate)
                    c_krw = cash - buy_val
                    total_out = cash
                holdings[a] += buy_val / float(open_px[a])
                cash = max(0.0, cash - total_out)
                oneway_trade_val += buy_val
                step_cost += c_krw

            turnover = (0.5 * oneway_trade_val) / mtm_open if mtm_open > 0 else 0.0
            total_oneway_turnover += turnover
            total_tx_cost += step_cost

            eq_after = cash + sum(holdings[a] * float(open_px[a]) for a in ASSETS)
            rebal_logs.append(RebalLog(
                rebal_date=dt.strftime("%Y-%m-%d"),
                signal_date=sig_date.strftime("%Y-%m-%d"),
                turnover_oneway=float(turnover),
                tx_cost_krw=float(step_cost),
                equity_after=float(eq_after),
                weights_json=json.dumps({k: round(v, 4) for k, v in target_w.items()}),
            ))

        # (2) 당일 종가 평가
        close_px = close_panel.loc[dt]
        asset_val = sum(holdings[a] * float(close_px[a]) for a in ASSETS)
        equity = cash + asset_val
        inv_w = asset_val / equity if equity > 0 else 0.0

        max_observed_weight = max(max_observed_weight, inv_w)
        min_observed_cash = min(min_observed_cash, cash)
        min_observed_equity = min(min_observed_equity, equity)

        daily_logs.append({
            "date": dt.strftime("%Y-%m-%d"),
            "equity": equity,
            "cash": cash,
            "asset_val": asset_val,
            "invested_weight": inv_w,
        })

    eq_df = pd.DataFrame(daily_logs)
    audit = {
        "max_observed_total_weight": float(max_observed_weight),
        "min_observed_cash": float(min_observed_cash),
        "min_observed_equity": float(min_observed_equity),
        "num_rebalances": len(rebal_logs),
        "annual_turnover": float(total_oneway_turnover / (len(bt_dates) / 245.0)),
        "total_tx_cost_krw": float(total_tx_cost),
    }
    return eq_df, rebal_logs, audit


# ==============================================================================
# 5. 성과 지표, 연도별 수익률, 3-Bucket 상관계수 & Z3 SLA 검증
# ==============================================================================
def calc_metrics(eq_df: pd.DataFrame, start_date: str, end_date: str) -> Dict:
    sub = eq_df[(eq_df["date"] >= start_date) & (eq_df["date"] <= end_date)].copy()
    eq = sub["equity"].values
    total_ret = (eq[-1] / eq[0]) - 1.0
    dt0 = pd.Timestamp(sub["date"].iloc[0])
    dt1 = pd.Timestamp(sub["date"].iloc[-1])
    cal_years = max((dt1 - dt0).days / 365.25, 1e-6)
    cagr_cal = (eq[-1] / eq[0]) ** (1.0 / cal_years) - 1.0 if eq[-1] > 0 else -1.0
    dr = pd.Series(eq).pct_change().dropna()
    sharpe = (dr.mean() / dr.std()) * np.sqrt(252) if dr.std() > 0 else 0.0
    mdd = float((eq / np.maximum.accumulate(eq) - 1.0).min())
    calmar = (cagr_cal / abs(mdd)) if mdd < 0 else 0.0
    return {
        "total_return": float(total_ret),
        "cagr_cal": float(cagr_cal),
        "sharpe": float(sharpe),
        "mdd": float(mdd),
        "calmar": float(calmar),
        "final_equity": float(eq[-1]),
    }


def calc_yearly_returns(eq_df: pd.DataFrame) -> Dict[str, float]:
    df = eq_df.copy()
    df["year"] = df["date"].str[:4]
    out = {}
    prev_end_eq = None
    for yr, g in df.groupby("year"):
        start_eq = prev_end_eq if prev_end_eq is not None else float(g["equity"].iloc[0])
        end_eq = float(g["equity"].iloc[-1])
        out[yr] = float((end_eq / start_eq) - 1.0)
        prev_end_eq = end_eq
    return out


def compute_3bucket_correlations(lt_equities: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
    d3 = pd.read_csv(D3_EQUITY_PATH)[["date", "equity"]].rename(columns={"equity": "D3"})
    tb = pd.read_csv(TURBO_EQUITY_PATH)[["date", "equity"]].rename(columns={"equity": "Turbo1"})

    d3["date"] = pd.to_datetime(d3["date"])
    tb["date"] = pd.to_datetime(tb["date"])

    merged = d3.set_index("date").join(tb.set_index("date"), how="inner")
    corrs = {}
    for name, eq_df in lt_equities.items():
        tmp = eq_df[["date", "equity"]].copy()
        tmp["date"] = pd.to_datetime(tmp["date"])
        m = merged.join(tmp.set_index("date").rename(columns={"equity": name}), how="inner")
        rets = m.pct_change().dropna()
        cmat = rets.corr()
        corrs[name] = {
            "corr_with_D3_Mid": float(cmat.loc[name, "D3"]),
            "corr_with_Turbo1_Short": float(cmat.loc[name, "Turbo1"]),
            "corr_D3_vs_Turbo1": float(cmat.loc["D3", "Turbo1"]),
            "overlap_days": int(len(rets)),
        }
    return corrs


def verify_z3_longterm(name: str, cdata: Dict, bm_kospi: Dict, bm_6040: Dict, la_diff: float) -> Dict:
    csv_path = BASE_DIR / f"equity_lt_{name}_net.csv"
    df = pd.read_csv(csv_path)
    audit = cdata["net"]["audit"]
    nf = cdata["net"]["full"]
    no = cdata["net"]["oos"]
    gf = cdata["gross"]["full"]

    def chk(label, expr):
        s = Solver()
        s.add(expr)
        r = s.check()
        return {"clause": label, "status": "SAT" if r == sat else "UNSAT", "passed": r == sat}

    max_w, min_c, min_e = Real("max_w"), Real("min_c"), Real("min_e")
    nr, gr, la = Real("nr"), Real("gr"), Real("la")
    fs, fm, oc, os_ = Real("fs"), Real("fm"), Real("oc"), Real("os")
    ks, km, s6040 = Real("ks"), Real("km"), Real("s6040")

    b = And(
        max_w == RealVal(f"{max(float(df['invested_weight'].max()), audit['max_observed_total_weight']):.10f}"),
        min_c == RealVal(f"{min(float(df['cash'].min()), audit['min_observed_cash']):.10f}"),
        min_e == RealVal(f"{min(float(df['equity'].min()), audit['min_observed_equity']):.10f}"),
        nr == RealVal(f"{nf['total_return']:.10f}"),
        gr == RealVal(f"{gf['total_return']:.10f}"),
        la == RealVal(f"{la_diff:.10f}"),
        fs == RealVal(f"{nf['sharpe']:.10f}"),
        fm == RealVal(f"{nf['mdd']:.10f}"),
        oc == RealVal(f"{no['cagr_cal']:.10f}"),
        os_ == RealVal(f"{no['sharpe']:.10f}"),
        ks == RealVal(f"{bm_kospi['full']['sharpe']:.10f}"),
        km == RealVal(f"{bm_kospi['full']['mdd']:.10f}"),
        s6040 == RealVal(f"{bm_6040['full']['sharpe']:.10f}"),
    )

    clauses = [
        chk("SLA-1_NoLeverage_Weight_LE_100pct", And(b, max_w <= RealVal("1.0001"))),
        chk("SLA-2_NoBankruptcy_Cash_GE_0_Equity_GT_0", And(b, min_c >= RealVal("0"), min_e > RealVal("0"))),
        chk("SLA-3_ConservativeCost_Deducted", And(b, nr < gr)),
        chk("SLA-4_ZeroLookAhead", And(b, la <= RealVal("0.000000001"))),
        chk("SLA-5_OOS_NetProfit_Positive", And(b, oc > RealVal("0"), os_ > RealVal("0"))),
        chk("SLA-6_Beats_KOSPI_And_6040_RiskAdjusted", And(b, fs > ks, fs > s6040, fm > km)),
    ]
    return {
        "strategy": name,
        "structural_1_to_4": "SAT" if all(c["passed"] for c in clauses[:4]) else "UNSAT",
        "oos_positive_5": clauses[4]["status"],
        "beats_benchmarks_6": clauses[5]["status"],
        "clauses": clauses,
    }


def main():
    all_df = sync_longterm_data(force_refresh=False)
    open_panel, close_panel = build_aligned_panels(all_df)

    strategies = [
        "L-1_DualMomentum",
        "L-2_RiskParity_Trend",
        "L-3_Barbell_Regime",
        "BM_KOSPI",
        "BM_6040",
    ]

    results = {
        "meta": {
            "bt_start": BT_START,
            "is_end": IS_END,
            "oos_start": OOS_START,
            "bt_end": DATA_END,
            "assets": {k: {"ticker": v[0], "name": v[1], "oneway_cost": v[2]} for k, v in ASSETS.items()},
        },
        "strategies": {},
        "correlations_3bucket": {},
        "lookahead_perturbation_max_diff": {},
    }

    lt_equities = {}
    baseline_is_eq = {}

    print("\n=== Long-Term Core Backtest Results (Net: ETF 0.15% / BTC 0.36% RT) ===")
    for strat in strategies:
        eq_net, logs_net, audit_net = run_longterm_strategy(strat, open_panel, close_panel, True)
        eq_gross, _, _ = run_longterm_strategy(strat, open_panel, close_panel, False)

        lt_equities[strat] = eq_net
        baseline_is_eq[strat] = eq_net[eq_net["date"] <= IS_END]["equity"].values

        eq_net.to_csv(BASE_DIR / f"equity_lt_{strat}_net.csv", index=False)
        pd.DataFrame([asdict(r) for r in logs_net]).to_csv(BASE_DIR / f"trades_lt_{strat}_net.csv", index=False)

        nf = calc_metrics(eq_net, BT_START, DATA_END)
        ni = calc_metrics(eq_net, BT_START, IS_END)
        no = calc_metrics(eq_net, OOS_START, DATA_END)
        gf = calc_metrics(eq_gross, BT_START, DATA_END)
        yr = calc_yearly_returns(eq_net)

        results["strategies"][strat] = {
            "net": {"full": nf, "is": ni, "oos": no, "yearly": yr, "audit": audit_net},
            "gross": {"full": gf},
        }
        print(
            f"[{strat:20s}] CalCAGR={nf['cagr_cal']*100:+6.2f}% (Gross={gf['cagr_cal']*100:+6.2f}%) "
            f"Tot={nf['total_return']*100:+6.1f}% ({nf['final_equity']/1e8:.2f}억) Shp={nf['sharpe']:4.2f} "
            f"MDD={nf['mdd']*100:6.2f}% Calmar={nf['calmar']:4.2f} Turn={audit_net['annual_turnover']*100:5.1f}%/yr | "
            f"IS={ni['cagr_cal']*100:+6.2f}% OOS={no['cagr_cal']*100:+6.2f}% (Shp {no['sharpe']:4.2f}, MDD {no['mdd']*100:6.2f}%)"
        )

    # Zero Look-Ahead 교란 검증 (OOS 구간 전체 자산 +500% 교란)
    p_open = open_panel.copy()
    p_close = close_panel.copy()
    mask = p_close.index >= pd.Timestamp(OOS_START)
    p_open.loc[mask] = p_open.loc[mask] * 6.0
    p_close.loc[mask] = p_close.loc[mask] * 6.0

    print("\n[Look-Ahead Audit] Perturbing OOS prices (+500%)...")
    for strat in strategies:
        eq_p, _, _ = run_longterm_strategy(strat, p_open, p_close, True)
        p_is = eq_p[eq_p["date"] <= IS_END]["equity"].values
        diff = float(np.max(np.abs(p_is - baseline_is_eq[strat])))
        results["lookahead_perturbation_max_diff"][strat] = diff
        print(f"  {strat:20s}: max IS diff = {diff:.10f}")

    # 3-Bucket 일별 수익률 상관계수
    corrs = compute_3bucket_correlations({k: lt_equities[k] for k in ["L-1_DualMomentum", "L-2_RiskParity_Trend", "L-3_Barbell_Regime"]})
    results["correlations_3bucket"] = corrs
    print("\n[3-Bucket Daily Return Correlations]")
    for k, v in corrs.items():
        print(f"  {k:20s}: vs D-3(Mid)={v['corr_with_D3_Mid']:+.3f} | vs Turbo-1(Short)={v['corr_with_Turbo1_Short']:+.3f} (D-3 vs Turbo-1={v['corr_D3_vs_Turbo1']:+.3f}, n={v['overlap_days']})")

    # Z3-Solver SLA 검증
    bm_kospi = results["strategies"]["BM_KOSPI"]["net"]
    bm_6040 = results["strategies"]["BM_6040"]["net"]
    z3_summary = {"strategies": {}}
    print("\n[Z3-Solver SLA Verification]")
    for strat in ["L-1_DualMomentum", "L-2_RiskParity_Trend", "L-3_Barbell_Regime"]:
        v = verify_z3_longterm(strat, results["strategies"][strat], bm_kospi, bm_6040, results["lookahead_perturbation_max_diff"][strat])
        z3_summary["strategies"][strat] = v
        print(
            f"  {strat:20s}: SLA 1~4(무결성)={v['structural_1_to_4']} | "
            f"SLA-5(OOS>0)={v['oos_positive_5']} | SLA-6(Beats KOSPI&60/40)={v['beats_benchmarks_6']}"
        )

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(Z3_PATH, "w", encoding="utf-8") as f:
        json.dump(z3_summary, f, indent=2, ensure_ascii=False)
    print(f"\n[Done] Saved {RESULTS_PATH.name} and {Z3_PATH.name}")


if __name__ == "__main__":
    main()
