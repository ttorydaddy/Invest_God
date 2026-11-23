"""
Invest_God Phase 1-A-3: KRX A-FR 손절선 민감도 비교 테스트 (-3% vs -5% vs -8% + 서킷브레이커)
- quant-auditor 지적 미세 결함 2건 반영:
  1) 거래대금 계산 시 액면분할 전 축소 방지를 위해 비수정 raw_close * raw_volume 사용 (결측 시 Close * Volume 폴백)
  2) 보유 중 거래정지/결측일 평가 시 entry_price 대신 last_known_close 추적
- 달력 연수 기준 실제 CAGR 및 252 거래일 기준 CAGR 병기
- 원시 증거물 4종(.py, equity_*.csv, trades_*.csv, .json) + Zero Look-Ahead 교란 검증 + Z3-Solver SLA 검증
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
from z3 import And, Real, RealVal, Solver, sat

# ==============================================================================
# 1. 전역 설정 (Phase 1-A / 1-A-2와 동일 기준)
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "krx_surge_data.duckdb"
RESULTS_PATH = BASE_DIR / "stop_test_results.json"
Z3_PATH = BASE_DIR / "z3_sla_stop_test.json"

BT_START = "2021-01-04"
IS_END = "2024-06-30"
OOS_START = "2024-07-01"
DATA_END = "2026-11-20"

INITIAL_CAPITAL = 100_000_000.0
MAX_POSITIONS = 5
MAX_SINGLE_WEIGHT = 0.20

BUY_COST = 0.00015 + 0.0025                 # 0.265%
SELL_COST = 0.00015 + 0.0020 + 0.0025       # 0.465%
ROUND_TRIP_COST = BUY_COST + SELL_COST      # 0.730%

MIN_PRICE = 2000.0
MIN_TRADING_VALUE = 10_000_000_000.0        # 100억 원
MAX_ENTRY_GAP = 0.15


# ==============================================================================
# 2. 데이터 로드 및 인과적 지표 계산 (결함 1 반영: raw_close * raw_volume 거래대금)
# ==============================================================================
def load_duckdb_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    ohlcv = con.execute("SELECT * FROM daily_ohlcv ORDER BY code, date").df()
    meta = con.execute("SELECT * FROM universe_meta").df()
    bench = con.execute("SELECT * FROM benchmarks ORDER BY code, date").df()
    flow = con.execute("SELECT * FROM investor_flow ORDER BY code, date").df()
    con.close()
    return ohlcv, meta, bench, flow


def compute_benchmark_regime(bench: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out = {}
    for bcode in ["KS11", "KQ11"]:
        sub = bench[bench["code"] == bcode].sort_values("date").copy()
        sub["date"] = pd.to_datetime(sub["date"])
        sub = sub.set_index("date")
        c = sub["Close"]
        sub["ma60"] = c.rolling(60, min_periods=60).mean()
        sub["ret_5d"] = c.pct_change(5)
        sub["regime_on"] = (c > sub["ma60"]) & (sub["ret_5d"] > -0.05)
        out[bcode] = sub
    return out


def compute_indicators_fixed(
    ohlcv: pd.DataFrame,
    meta: pd.DataFrame,
    flow: pd.DataFrame,
    bench_regime: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    market_map = meta.set_index("code")["market"].to_dict()
    merged = ohlcv.merge(
        flow[["date", "code", "frgn_net_qty", "inst_net_qty", "raw_close", "raw_volume", "frgn_net_val", "inst_net_val"]],
        on=["date", "code"],
        how="left",
    )

    data_by_code = {}
    for code, g in merged.groupby("code", sort=False):
        df = g.sort_values("date").copy()
        df["date_dt"] = pd.to_datetime(df["date"])
        df = df.set_index("date_dt")

        c = df["Close"]
        o = df["Open"]
        h = df["High"]
        l = df["Low"]
        v = df["Volume"]

        # [감사 결함 1 수정] 실제 거래대금은 비수정 raw_close * raw_volume 우선 사용 (결측 시 Close * Volume 폴백)
        raw_tval = df["raw_close"] * df["raw_volume"]
        adj_tval = c * v
        df["tval"] = raw_tval.where(raw_tval.notna() & (raw_tval > 0), adj_tval)

        df["ma20"] = c.rolling(20, min_periods=20).mean()
        df["ma60"] = c.rolling(60, min_periods=60).mean()
        df["ma20_slope5"] = df["ma20"] - df["ma20"].shift(5)

        df["avg_tval20_prev"] = df["tval"].rolling(20, min_periods=20).mean().shift(1)
        df["high20_prev"] = h.rolling(20, min_periods=20).max().shift(1)
        df["ret_1d"] = c.pct_change()
        df["prev_close"] = c.shift(1)
        df["is_frozen"] = (h == l) & (o == c)

        # 외인+기관 확정 수급 지표 (결측 시 NaN 유지)
        smart_qty = df["frgn_net_qty"] + df["inst_net_qty"]
        raw_vol = df["raw_volume"].replace(0, np.nan)
        df["smart_flow_ratio"] = smart_qty / raw_vol

        smart_val = df["frgn_net_val"] + df["inst_net_val"]
        raw_tval_nz = raw_tval.replace(0, np.nan)
        df["smart_flow_3d_ratio"] = (
            smart_val.rolling(3, min_periods=3).sum()
            / raw_tval_nz.rolling(3, min_periods=3).sum()
        )

        mkt = market_map.get(code, "KOSPI")
        bcode = "KS11" if mkt == "KOSPI" else "KQ11"
        bdf = bench_regime[bcode]
        df["bench_regime_on"] = bdf["regime_on"].reindex(df.index).astype("boolean").fillna(False).astype(bool)

        data_by_code[code] = df

    return data_by_code


# ==============================================================================
# 3. A-FR 백테스트 엔진 (결함 2 반영: last_known_close + 서킷브레이커 옵션)
# ==============================================================================
@dataclass
class TradeRecord:
    code: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    shares: int
    weight_at_entry: float
    gross_ret: float
    net_ret: float
    pnl: float
    exit_reason: str
    holding_days: int


@dataclass
class Position:
    code: str
    entry_date: pd.Timestamp
    entry_price: float
    last_known_close: float   # [감사 결함 2 수정] 거래정지/결측일 평가용
    shares: int
    weight_at_entry: float
    prev_highest_high: float
    holding_days: int = 0
    pending_exit_reason: Optional[str] = None


@dataclass
class StopCase:
    name: str
    label: str
    stop_loss_pct: float
    use_circuit_breaker: bool = False
    cb_lookback_days: int = 20
    cb_dd_threshold: float = -0.15


def run_afr_case(
    case: StopCase,
    data_by_code: Dict[str, pd.DataFrame],
    trading_dates: List[pd.Timestamp],
    apply_costs: bool = True,
) -> Tuple[pd.DataFrame, List[TradeRecord], Dict]:
    buy_c = BUY_COST if apply_costs else 0.0
    sell_c = SELL_COST if apply_costs else 0.0

    trailing_stop_pct = 0.15
    exit_ma_col = "ma20"
    max_hold_days = 60

    cash = INITIAL_CAPITAL
    positions: Dict[str, Position] = {}
    pending_entries: List[Tuple[str, float]] = []
    trades: List[TradeRecord] = []
    daily_logs = []
    equity_history: List[float] = []

    max_observed_total_weight = 0.0
    max_observed_single_entry_weight = 0.0
    min_observed_cash = INITIAL_CAPITAL
    min_observed_equity = INITIAL_CAPITAL
    cb_blocked_days = 0

    for dt in trading_dates:
        # (1) 시가 청산
        to_close = [code for code, pos in positions.items() if pos.pending_exit_reason is not None]
        for code in to_close:
            pos = positions[code]
            df = data_by_code[code]
            if dt not in df.index:
                continue
            row = df.loc[dt]
            if row["is_frozen"]:
                continue
            exit_px = float(row["Open"])
            proceeds = pos.shares * exit_px * (1.0 - sell_c)
            cost_basis = pos.shares * pos.entry_price * (1.0 + buy_c)
            cash += proceeds
            trades.append(TradeRecord(
                code=code,
                entry_date=pos.entry_date.strftime("%Y-%m-%d"),
                exit_date=dt.strftime("%Y-%m-%d"),
                entry_price=pos.entry_price,
                exit_price=exit_px,
                shares=pos.shares,
                weight_at_entry=pos.weight_at_entry,
                gross_ret=(exit_px / pos.entry_price) - 1.0,
                net_ret=(proceeds / cost_basis) - 1.0,
                pnl=proceeds - cost_basis,
                exit_reason=pos.pending_exit_reason,
                holding_days=pos.holding_days,
            ))
            del positions[code]

        # (2) 시가 진입 (서킷브레이커 활성화 시 전일 확정 계좌 20일 고점 대비 -15% 이하면 신규 진입 중단)
        cb_active = False
        if case.use_circuit_breaker and len(equity_history) >= 2:
            window = equity_history[-case.cb_lookback_days:]
            peak_20d = max(window)
            prev_eq = equity_history[-1]
            if peak_20d > 0 and (prev_eq / peak_20d - 1.0) <= case.cb_dd_threshold:
                cb_active = True
                cb_blocked_days += 1

        if (not cb_active) and pending_entries and len(positions) < MAX_POSITIONS:
            mtm_open = cash + sum(
                pos.shares * (float(data_by_code[c].loc[dt, "Open"]) if dt in data_by_code[c].index else pos.last_known_close)
                for c, pos in positions.items()
            )
            target_alloc = mtm_open * MAX_SINGLE_WEIGHT

            for code, _score in pending_entries:
                if len(positions) >= MAX_POSITIONS:
                    break
                if code in positions:
                    continue
                df = data_by_code[code]
                if dt not in df.index:
                    continue
                row = df.loc[dt]
                if row["is_frozen"]:
                    continue
                open_px = float(row["Open"])
                prev_close = float(row["prev_close"]) if pd.notna(row["prev_close"]) else open_px
                if prev_close > 0 and (open_px / prev_close - 1.0) > MAX_ENTRY_GAP:
                    continue

                alloc = min(target_alloc, cash)
                shares = int(alloc // (open_px * (1.0 + buy_c)))
                if shares <= 0:
                    continue
                total_cost = shares * open_px * (1.0 + buy_c)
                if total_cost > cash:
                    continue

                cash -= total_cost
                entry_w = total_cost / mtm_open if mtm_open > 0 else 0.0
                max_observed_single_entry_weight = max(max_observed_single_entry_weight, entry_w)
                positions[code] = Position(
                    code=code,
                    entry_date=dt,
                    entry_price=open_px,
                    last_known_close=open_px,
                    shares=shares,
                    weight_at_entry=entry_w,
                    prev_highest_high=open_px,
                    holding_days=0,
                )
        pending_entries = []

        # (3) 장중 하드 손절 & 트레일링 스탑 (전일 확정 최고가 기준)
        intraday_closed = []
        for code, pos in list(positions.items()):
            df = data_by_code[code]
            if dt not in df.index:
                continue
            row = df.loc[dt]
            if row["is_frozen"]:
                continue

            open_px = float(row["Open"])
            low_px = float(row["Low"])
            high_px = float(row["High"])

            hard_stop_px = pos.entry_price * (1.0 - case.stop_loss_pct)
            trail_stop_px = pos.prev_highest_high * (1.0 - trailing_stop_pct)
            effective_stop_px = max(hard_stop_px, trail_stop_px)
            reason = "hard_stop" if hard_stop_px >= trail_stop_px else "trailing_stop"

            if low_px <= effective_stop_px:
                exit_px = min(open_px, effective_stop_px)
                proceeds = pos.shares * exit_px * (1.0 - sell_c)
                cost_basis = pos.shares * pos.entry_price * (1.0 + buy_c)
                cash += proceeds
                trades.append(TradeRecord(
                    code=code,
                    entry_date=pos.entry_date.strftime("%Y-%m-%d"),
                    exit_date=dt.strftime("%Y-%m-%d"),
                    entry_price=pos.entry_price,
                    exit_price=exit_px,
                    shares=pos.shares,
                    weight_at_entry=pos.weight_at_entry,
                    gross_ret=(exit_px / pos.entry_price) - 1.0,
                    net_ret=(proceeds / cost_basis) - 1.0,
                    pnl=proceeds - cost_basis,
                    exit_reason=reason,
                    holding_days=pos.holding_days + 1,
                ))
                intraday_closed.append(code)
            else:
                pos.prev_highest_high = max(pos.prev_highest_high, high_px)
                pos.holding_days += 1

        for code in intraday_closed:
            del positions[code]

        # (4) 종가 평가 (결함 2 수정: last_known_close 추적) 및 익일 시그널
        stock_val = 0.0
        for code, pos in positions.items():
            df = data_by_code[code]
            if dt in df.index:
                row = df.loc[dt]
                close_px = float(row["Close"])
                pos.last_known_close = close_px
                stock_val += pos.shares * close_px
                ma_val = row[exit_ma_col]
                if pd.notna(ma_val) and close_px < float(ma_val):
                    pos.pending_exit_reason = f"below_{exit_ma_col}"
                elif pos.holding_days >= max_hold_days:
                    pos.pending_exit_reason = "max_hold_days"
            else:
                stock_val += pos.shares * pos.last_known_close

        equity = cash + stock_val
        equity_history.append(equity)
        invested_weight = stock_val / equity if equity > 0 else 0.0
        max_observed_total_weight = max(max_observed_total_weight, invested_weight)
        min_observed_cash = min(min_observed_cash, cash)
        min_observed_equity = min(min_observed_equity, equity)

        daily_logs.append({
            "date": dt.strftime("%Y-%m-%d"),
            "equity": equity,
            "cash": cash,
            "stock_val": stock_val,
            "invested_weight": invested_weight,
            "num_positions": len(positions),
            "cb_active": int(cb_active),
        })

        # 익일 매수 후보 스캔 (A-FR 조건 100% 동일)
        candidates = []
        for code, df in data_by_code.items():
            if code in positions or dt not in df.index:
                continue
            row = df.loc[dt]
            if row["is_frozen"]:
                continue
            c = float(row["Close"])
            o = float(row["Open"])
            tval = float(row["tval"])
            if c < MIN_PRICE or tval < MIN_TRADING_VALUE:
                continue
            if not (
                pd.notna(row["ma60"])
                and pd.notna(row["avg_tval20_prev"])
                and pd.notna(row["high20_prev"])
                and row["avg_tval20_prev"] > 0
            ):
                continue

            tval_ratio = tval / float(row["avg_tval20_prev"])
            ret1d = float(row["ret_1d"]) if pd.notna(row["ret_1d"]) else 0.0
            if not (
                c > float(row["ma20"]) > float(row["ma60"])
                and float(row["ma20_slope5"]) > 0
                and tval_ratio >= 2.5
                and c >= float(row["high20_prev"])
                and c > o
                and 0.03 <= ret1d <= 0.20
            ):
                continue

            # 시장 레짐 필터
            if not bool(row["bench_regime_on"]):
                continue

            # 외인+기관 확정 수급 필터
            sf1 = row["smart_flow_ratio"]
            sf3 = row["smart_flow_3d_ratio"]
            if pd.isna(sf1) or pd.isna(sf3) or float(sf1) < 0.02 or float(sf3) <= 0.0:
                continue

            candidates.append((code, tval_ratio))

        candidates.sort(key=lambda x: x[1], reverse=True)
        pending_entries = candidates[: MAX_POSITIONS * 2]

    equity_df = pd.DataFrame(daily_logs)
    audit = {
        "max_observed_total_weight": float(max_observed_total_weight),
        "max_observed_single_entry_weight": float(max_observed_single_entry_weight),
        "min_observed_cash": float(min_observed_cash),
        "min_observed_equity": float(min_observed_equity),
        "stop_loss_pct": float(case.stop_loss_pct),
        "trailing_stop_pct": float(trailing_stop_pct),
        "round_trip_cost": float(ROUND_TRIP_COST if apply_costs else 0.0),
        "cb_blocked_days": int(cb_blocked_days),
    }
    return equity_df, trades, audit


# ==============================================================================
# 4. 성과 지표 (달력 연수 CAGR & 252일 CAGR 병기) + Look-Ahead + Z3 SLA 검증
# ==============================================================================
def calc_metrics(equity_df: pd.DataFrame, trades: List[TradeRecord], start_date: str, end_date: str) -> Dict:
    sub = equity_df[(equity_df["date"] >= start_date) & (equity_df["date"] <= end_date)].copy()
    eq = sub["equity"].values
    total_ret = (eq[-1] / eq[0]) - 1.0
    n_days = len(sub)

    # (1) 달력 연수 기준 실제 CAGR (한국 거래일 ~244일/년 반영)
    dt0 = pd.Timestamp(sub["date"].iloc[0])
    dt1 = pd.Timestamp(sub["date"].iloc[-1])
    cal_years = max((dt1 - dt0).days / 365.25, 1e-6)
    cagr_cal = (eq[-1] / eq[0]) ** (1.0 / cal_years) - 1.0 if eq[-1] > 0 else -1.0

    # (2) 252 거래일 기준 CAGR (기존 비교용)
    years_252 = n_days / 252.0
    cagr_252 = (eq[-1] / eq[0]) ** (1.0 / years_252) - 1.0 if years_252 > 0 and eq[-1] > 0 else -1.0

    daily_ret = pd.Series(eq).pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0.0
    mdd = float((eq / np.maximum.accumulate(eq) - 1.0).min())

    sub_trades = [t for t in trades if start_date <= t.entry_date <= end_date]
    n_trades = len(sub_trades)
    if n_trades > 0:
        rets = [t.net_ret for t in sub_trades]
        wins = [r for r in rets if r > 0]
        losses = [r for r in rets if r <= 0]
        win_rate = len(wins) / n_trades
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(abs(np.mean(losses))) if losses else 0.0
        payoff_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0
        profit_factor = (sum(wins) / abs(sum(losses))) if losses and abs(sum(losses)) > 0 else 0.0
        avg_hold = float(np.mean([t.holding_days for t in sub_trades]))
        hard_stops = sum(1 for t in sub_trades if t.exit_reason == "hard_stop")
        hard_stop_rate = hard_stops / n_trades
    else:
        win_rate = avg_win = avg_loss = payoff_ratio = profit_factor = avg_hold = hard_stop_rate = 0.0
        hard_stops = 0

    return {
        "start_date": sub["date"].iloc[0],
        "end_date": sub["date"].iloc[-1],
        "trading_days": int(n_days),
        "calendar_years": float(cal_years),
        "total_return": float(total_ret),
        "cagr_cal": float(cagr_cal),
        "cagr_252": float(cagr_252),
        "sharpe": float(sharpe),
        "mdd": float(mdd),
        "num_trades": int(n_trades),
        "hard_stops": int(hard_stops),
        "hard_stop_rate": float(hard_stop_rate),
        "win_rate": float(win_rate),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "payoff_ratio": float(payoff_ratio),
        "profit_factor": float(profit_factor),
        "avg_holding_days": float(avg_hold),
        "avg_exposure": float(sub["invested_weight"].mean()),
        "final_equity": float(eq[-1]),
    }


def calc_benchmark_metrics(bench: pd.DataFrame, code: str, start_date: str, end_date: str) -> Dict:
    sub = bench[(bench["code"] == code) & (bench["date"] >= start_date) & (bench["date"] <= end_date)].copy()
    c = sub["Close"].values
    total_ret = (c[-1] / c[0]) - 1.0
    dt0 = pd.Timestamp(sub["date"].iloc[0])
    dt1 = pd.Timestamp(sub["date"].iloc[-1])
    cal_years = max((dt1 - dt0).days / 365.25, 1e-6)
    cagr_cal = (c[-1] / c[0]) ** (1.0 / cal_years) - 1.0
    cagr_252 = (c[-1] / c[0]) ** (1.0 / (len(sub) / 252.0)) - 1.0
    dr = pd.Series(c).pct_change().dropna()
    sharpe = (dr.mean() / dr.std()) * np.sqrt(252) if dr.std() > 0 else 0.0
    mdd = float((c / np.maximum.accumulate(c) - 1.0).min())
    return {
        "total_return": float(total_ret),
        "cagr_cal": float(cagr_cal),
        "cagr_252": float(cagr_252),
        "sharpe": float(sharpe),
        "mdd": float(mdd),
    }


def verify_z3_case(name: str, cdata: Dict, kospi: Dict, la_diff: float) -> Dict:
    csv_path = BASE_DIR / f"equity_stop_{name}_net.csv"
    df = pd.read_csv(csv_path)
    audit = cdata["net"]["audit"]
    nf = cdata["net"]["full"]
    no = cdata["net"]["oos"]
    gf = cdata["gross"]["full"]
    kf = kospi["full"]
    ko = kospi["oos"]

    def chk(label, expr):
        s = Solver()
        s.add(expr)
        r = s.check()
        return {"clause": label, "status": "SAT" if r == sat else "UNSAT", "passed": r == sat}

    max_w = Real("max_w")
    min_c = Real("min_c")
    min_e = Real("min_e")
    max_sw = Real("max_sw")
    max_p = Real("max_p")
    rtc = Real("rtc")
    nr = Real("nr")
    gr = Real("gr")
    sl = Real("sl")
    la = Real("la")
    fc, fs, fm = Real("fc"), Real("fs"), Real("fm")
    oc, os_, om = Real("oc"), Real("os"), Real("om")
    kfc, kfs, kfm = Real("kfc"), Real("kfs"), Real("kfm")
    koc, kos, kom = Real("koc"), Real("kos"), Real("kom")

    b = And(
        max_w == RealVal(f"{max(float(df['invested_weight'].max()), audit['max_observed_total_weight']):.10f}"),
        min_c == RealVal(f"{min(float(df['cash'].min()), audit['min_observed_cash']):.10f}"),
        min_e == RealVal(f"{min(float(df['equity'].min()), audit['min_observed_equity']):.10f}"),
        max_sw == RealVal(f"{audit['max_observed_single_entry_weight']:.10f}"),
        max_p == RealVal(str(int(df["num_positions"].max()))),
        rtc == RealVal(f"{audit['round_trip_cost']:.10f}"),
        nr == RealVal(f"{nf['total_return']:.10f}"),
        gr == RealVal(f"{gf['total_return']:.10f}"),
        sl == RealVal(f"{audit['stop_loss_pct']:.10f}"),
        la == RealVal(f"{la_diff:.10f}"),
        fc == RealVal(f"{nf['cagr_cal']:.10f}"),
        fs == RealVal(f"{nf['sharpe']:.10f}"),
        fm == RealVal(f"{nf['mdd']:.10f}"),
        oc == RealVal(f"{no['cagr_cal']:.10f}"),
        os_ == RealVal(f"{no['sharpe']:.10f}"),
        om == RealVal(f"{no['mdd']:.10f}"),
        kfc == RealVal(f"{kf['cagr_cal']:.10f}"),
        kfs == RealVal(f"{kf['sharpe']:.10f}"),
        kfm == RealVal(f"{kf['mdd']:.10f}"),
        koc == RealVal(f"{ko['cagr_cal']:.10f}"),
        kos == RealVal(f"{ko['sharpe']:.10f}"),
        kom == RealVal(f"{ko['mdd']:.10f}"),
    )

    clauses = [
        chk("SLA-1_NoLeverage", And(b, max_w <= RealVal("1.0001"), max_p <= RealVal("5"))),
        chk("SLA-2_NoBankruptcy", And(b, min_c >= RealVal("0"), min_e > RealVal("0"))),
        chk("SLA-3_SingleStockCap_LE_20pct", And(b, max_sw <= RealVal("0.2001"))),
        chk("SLA-4_ConservativeCost_Deducted", And(b, rtc >= RealVal("0.006"), nr < gr)),
        chk("SLA-5_HardStop_And_ZeroLookAhead", And(b, sl >= RealVal("0.02"), sl <= RealVal("0.10"), la <= RealVal("0.000000001"))),
        chk("SLA-6_OOS_Positive", And(b, oc > RealVal("0"), os_ > RealVal("0"))),
        chk("SLA-7_Dominates_KOSPI_Full_And_OOS", And(b, fc > kfc, fs > kfs, fm > kfm, oc > koc, os_ > kos, om > kom)),
    ]
    return {
        "case": name,
        "structural_1_to_5": "SAT" if all(c["passed"] for c in clauses[:5]) else "UNSAT",
        "oos_positive_6": clauses[5]["status"],
        "dominates_kospi_7": clauses[6]["status"],
        "clauses": clauses,
    }


def main():
    ohlcv, meta, bench, flow = load_duckdb_data()
    bench_regime = compute_benchmark_regime(bench)
    data_by_code = compute_indicators_fixed(ohlcv, meta, flow, bench_regime)

    all_dates = pd.to_datetime(bench[bench["code"] == "KS11"]["date"].unique())
    trading_dates = sorted([d for d in all_dates if pd.Timestamp(BT_START) <= d <= pd.Timestamp(DATA_END)])

    cases = [
        StopCase("SL3", "A-FR (손절 -3%)", 0.03, False),
        StopCase("SL5", "A-FR (손절 -5%)", 0.05, False),
        StopCase("SL8", "A-FR (손절 -8%, 기준)", 0.08, False),
        StopCase("SL5_CB", "A-FR (손절 -5% + 20일 고점 -15% 서킷브레이커)", 0.05, True, 20, -0.15),
        StopCase("SL8_CB", "A-FR (손절 -8% + 20일 고점 -15% 서킷브레이커)", 0.08, True, 20, -0.15),
    ]

    kospi = {
        "full": calc_benchmark_metrics(bench, "KS11", BT_START, DATA_END),
        "is": calc_benchmark_metrics(bench, "KS11", BT_START, IS_END),
        "oos": calc_benchmark_metrics(bench, "KS11", OOS_START, DATA_END),
    }
    kosdaq = {
        "full": calc_benchmark_metrics(bench, "KQ11", BT_START, DATA_END),
        "is": calc_benchmark_metrics(bench, "KQ11", BT_START, IS_END),
        "oos": calc_benchmark_metrics(bench, "KQ11", OOS_START, DATA_END),
    }

    results = {
        "meta": {
            "bt_start": trading_dates[0].strftime("%Y-%m-%d"),
            "is_end": IS_END,
            "oos_start": OOS_START,
            "bt_end": trading_dates[-1].strftime("%Y-%m-%d"),
            "trading_days": len(trading_dates),
            "round_trip_cost": ROUND_TRIP_COST,
            "auditor_fixes_applied": [
                "Fix 1: Use unadjusted raw_close * raw_volume for trading value (fallback Close * Volume)",
                "Fix 2: Track last_known_close for suspended/missing trading days",
            ],
        },
        "benchmarks": {"KOSPI": kospi, "KOSDAQ": kosdaq},
        "cases": {},
        "lookahead_perturbation_max_diff": {},
    }

    baseline_is_eq = {}
    print("=== A-FR Stop-Loss & Circuit Breaker Sensitivity Test (Net 0.73%) ===")
    for case in cases:
        eq_net, trades_net, audit_net = run_afr_case(case, data_by_code, trading_dates, True)
        eq_gross, trades_gross, _ = run_afr_case(case, data_by_code, trading_dates, False)

        baseline_is_eq[case.name] = eq_net[eq_net["date"] <= IS_END]["equity"].values
        eq_net.to_csv(BASE_DIR / f"equity_stop_{case.name}_net.csv", index=False)
        pd.DataFrame([asdict(t) for t in trades_net]).to_csv(BASE_DIR / f"trades_stop_{case.name}_net.csv", index=False)

        nf = calc_metrics(eq_net, trades_net, BT_START, DATA_END)
        ni = calc_metrics(eq_net, trades_net, BT_START, IS_END)
        no = calc_metrics(eq_net, trades_net, OOS_START, DATA_END)
        gf = calc_metrics(eq_gross, trades_gross, BT_START, DATA_END)

        # 2022년 하락장 별도 MDD 확인
        eq_2022 = eq_net[(eq_net["date"] >= "2022-01-01") & (eq_net["date"] <= "2022-12-31")]["equity"].values
        mdd_2022 = float((eq_2022 / np.maximum.accumulate(eq_2022) - 1.0).min()) if len(eq_2022) > 1 else 0.0
        ret_2022 = float((eq_2022[-1] / eq_2022[0]) - 1.0) if len(eq_2022) > 1 else 0.0

        results["cases"][case.name] = {
            "label": case.label,
            "net": {
                "full": nf,
                "is": ni,
                "oos": no,
                "year_2022": {"return": ret_2022, "mdd": mdd_2022},
                "audit": audit_net,
            },
            "gross": {"full": gf},
        }
        print(
            f"[{case.name:6s}] CalCAGR={nf['cagr_cal']*100:+6.2f}% (252d={nf['cagr_252']*100:+6.2f}%) "
            f"Tot={nf['total_return']*100:+6.1f}% Shp={nf['sharpe']:4.2f} MDD={nf['mdd']*100:6.2f}% (22MDD={mdd_2022*100:6.2f}%) "
            f"Trd={nf['num_trades']:3d} SL율={nf['hard_stop_rate']*100:4.1f}% Win={nf['win_rate']*100:4.1f}% Pay={nf['payoff_ratio']:4.2f} | "
            f"OOS Cal={no['cagr_cal']*100:+6.2f}% Shp={no['sharpe']:4.2f} MDD={no['mdd']*100:6.2f}%"
        )

    # Zero Look-Ahead 교란 검증
    p_ohlcv = ohlcv.copy()
    m1 = p_ohlcv["date"] >= OOS_START
    for col in ["Open", "High", "Low", "Close"]:
        p_ohlcv.loc[m1, col] = p_ohlcv.loc[m1, col] * 6.0
    p_bench = bench.copy()
    m2 = p_bench["date"] >= OOS_START
    for col in ["Open", "High", "Low", "Close"]:
        p_bench.loc[m2, col] = p_bench.loc[m2, col] * 6.0
    p_flow = flow.copy()
    m3 = p_flow["date"] >= OOS_START
    for col in ["frgn_net_qty", "inst_net_qty", "frgn_net_val", "inst_net_val"]:
        p_flow.loc[m3, col] = p_flow.loc[m3, col] * -5.0

    p_data = compute_indicators_fixed(p_ohlcv, meta, p_flow, compute_benchmark_regime(p_bench))
    print("\n[Look-Ahead Audit] Perturbing OOS prices (+500%) & flows (-500%)...")
    for case in cases:
        eq_p, _, _ = run_afr_case(case, p_data, trading_dates, True)
        p_is = eq_p[eq_p["date"] <= IS_END]["equity"].values
        diff = float(np.max(np.abs(p_is - baseline_is_eq[case.name])))
        results["lookahead_perturbation_max_diff"][case.name] = diff
        print(f"  {case.name:6s}: max IS diff = {diff:.10f}")

    # Z3-Solver SLA 검증
    z3_summary = {"cases": {}}
    print("\n[Z3-Solver SLA Verification]")
    for case in cases:
        v = verify_z3_case(case.name, results["cases"][case.name], kospi, results["lookahead_perturbation_max_diff"][case.name])
        z3_summary["cases"][case.name] = v
        print(
            f"  {case.name:6s}: SLA 1~5={v['structural_1_to_5']} | "
            f"OOS>0={v['oos_positive_6']} | Dominates KOSPI(Full&OOS)={v['dominates_kospi_7']}"
        )

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(Z3_PATH, "w", encoding="utf-8") as f:
        json.dump(z3_summary, f, indent=2, ensure_ascii=False)
    print(f"\n[Done] Saved {RESULTS_PATH.name} and {Z3_PATH.name}")


if __name__ == "__main__":
    main()
