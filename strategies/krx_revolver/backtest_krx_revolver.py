"""
Invest_God Phase 1-A-5: 대표님 제안 "3연발 회전탄창 + 20일선 개미털기 복귀 + 외인·기관 매집" 전략 백테스트
- 동일 데이터(krx_surge_data.duckdb 200종목 + 외인·기관 확정 수급 31만 행), 동일 왕복 비용(0.73%), 동일 IS/OOS
- 감사관 지적 2건 반영(비수정 raw_close*raw_volume 거래대금, 결측일 last_known_close 추적)
- 원시 증거물 4종 + Zero Look-Ahead 교란 검증 + Z3-Solver SLA 검증 + D-3 & KOSPI 1:1 비교
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
# 1. 전역 설정 (대표님 원형 규칙 + Phase 1-A 공통 현실 제약)
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR.parent / "krx_surge" / "krx_surge_data.duckdb"
RESULTS_PATH = BASE_DIR / "revolver_results.json"
Z3_PATH = BASE_DIR / "z3_sla_revolver.json"

BT_START = "2021-01-04"
IS_END = "2024-06-30"
OOS_START = "2024-07-01"
DATA_END = "2026-11-20"

INITIAL_CAPITAL = 100_000_000.0
MAX_POSITIONS = 3                  # 3연발 회전탄창
MAX_SINGLE_WEIGHT = 1.0 / 3.0      # 종목당 33.33%

BUY_COST = 0.00015 + 0.0025                 # 0.265%
SELL_COST = 0.00015 + 0.0020 + 0.0025       # 0.465%
ROUND_TRIP_COST = BUY_COST + SELL_COST      # 0.730%

MIN_PRICE = 2000.0
MIN_PEAK_TVAL_60D = 50_000_000_000.0        # 최근 60일 내 일 거래대금 500억 이상 이력
MAX_ENTRY_GAP = 0.15

STOP_LOSS_PCT = 0.04               # -4% 전량 손절
HALF_TP_PCT = 0.12                 # +12% 절반 익절
TRAIL_AFTER_TP_PCT = 0.05          # 절반 익절 후 고점 대비 -5% 트레일링 스탑


# ==============================================================================
# 2. 데이터 로드 및 인과적 지표 계산 (Zero Look-Ahead)
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


def compute_revolver_indicators(
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

        # [결함 1 수정 적용] 비수정 거래대금 우선 사용
        raw_tval = df["raw_close"] * df["raw_volume"]
        adj_tval = c * v
        df["tval"] = raw_tval.where(raw_tval.notna() & (raw_tval > 0), adj_tval)
        df["max_tval_60d"] = df["tval"].rolling(60, min_periods=20).max()

        df["ma20"] = c.rolling(20, min_periods=20).mean()
        df["ma60"] = c.rolling(60, min_periods=60).mean()
        df["ma60_rising"] = df["ma60"] > df["ma60"].shift(10)

        # 최근 60일 내 20일선 대비 +15% 이상 급등 이력
        ext_over_ma20 = (h / df["ma20"]) - 1.0
        df["had_surge_60d"] = ext_over_ma20.rolling(60, min_periods=20).max() >= 0.15

        # 개미털기: 직전 1~5거래일(t-5 ~ t-1) 중 저가 또는 종가가 MA20 아래로 이탈한 이력
        below_ma20 = ((l < df["ma20"]) | (c < df["ma20"])).astype(int)
        df["recent_below_ma20_5d"] = below_ma20.shift(1).rolling(5, min_periods=1).max()

        df["prev_close"] = c.shift(1)
        df["is_frozen"] = (h == l) & (o == c)

        # 외인+기관 확정 수급 (결측 시 NaN 유지)
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
# 3. 3연발 회전탄창 백테스트 엔진
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
    is_partial: bool = False


@dataclass
class Position:
    code: str
    entry_date: pd.Timestamp
    entry_price: float
    last_known_close: float
    shares: int
    weight_at_entry: float
    prev_highest_high: float
    holding_days: int = 0
    half_taken: bool = False
    pending_exit_reason: Optional[str] = None


@dataclass
class RevolverConfig:
    name: str
    label: str
    mode_5d: str = "strict"       # "strict": 5일 경과 시 -4%<ret<+3% 횡보만 청산 / "force": +12% 미도달 시 무조건 청산
    use_regime_filter: bool = False
    max_hold_days: int = 30       # 횡보 외 장기 고착 방지 상한


def run_revolver(
    cfg: RevolverConfig,
    data_by_code: Dict[str, pd.DataFrame],
    trading_dates: List[pd.Timestamp],
    apply_costs: bool = True,
) -> Tuple[pd.DataFrame, List[TradeRecord], Dict]:
    buy_c = BUY_COST if apply_costs else 0.0
    sell_c = SELL_COST if apply_costs else 0.0

    cash = INITIAL_CAPITAL
    positions: Dict[str, Position] = {}
    pending_entries: List[Tuple[str, float]] = []
    trades: List[TradeRecord] = []
    daily_logs = []

    max_observed_total_weight = 0.0
    max_observed_single_entry_weight = 0.0
    min_observed_cash = INITIAL_CAPITAL
    min_observed_equity = INITIAL_CAPITAL

    total_entries = 0
    half_tp_hits = 0
    hard_stop_hits = 0
    sideways_5d_hits = 0

    for dt in trading_dates:
        # (1) 시가 청산 (5일 횡보 회전 / 최대 보유기간)
        for code in list(positions.keys()):
            pos = positions[code]
            if pos.pending_exit_reason is None:
                continue
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
            if pos.pending_exit_reason.startswith("sideways_5d"):
                sideways_5d_hits += 1
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
                is_partial=False,
            ))
            del positions[code]

        # (2) 시가 진입 (3연발 회전탄창: 최대 3종목, 각 33.33%)
        if pending_entries and len(positions) < MAX_POSITIONS:
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
                total_entries += 1
                positions[code] = Position(
                    code=code,
                    entry_date=dt,
                    entry_price=open_px,
                    last_known_close=open_px,
                    shares=shares,
                    weight_at_entry=entry_w,
                    prev_highest_high=open_px,
                    holding_days=0,
                    half_taken=False,
                )
        pending_entries = []

        # (3) 장중 기계적 매도: -4% 손절 (최우선 보수적 체결) / +12% 절반 익절 / 익절 후 -5% 트레일링
        for code in list(positions.keys()):
            pos = positions[code]
            df = data_by_code[code]
            if dt not in df.index:
                continue
            row = df.loc[dt]
            if row["is_frozen"]:
                continue

            open_px = float(row["Open"])
            low_px = float(row["Low"])
            high_px = float(row["High"])

            if not pos.half_taken:
                hard_stop_px = pos.entry_price * (1.0 - STOP_LOSS_PCT)
                tp_px = pos.entry_price * (1.0 + HALF_TP_PCT)

                # [보수적 원칙] 같은 봉에서 손절(-4%)과 익절(+12%) 동시 도달 시 손절 우선 체결!
                if low_px <= hard_stop_px:
                    exit_px = min(open_px, hard_stop_px)
                    proceeds = pos.shares * exit_px * (1.0 - sell_c)
                    cost_basis = pos.shares * pos.entry_price * (1.0 + buy_c)
                    cash += proceeds
                    hard_stop_hits += 1
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
                        exit_reason="hard_stop_4pct",
                        holding_days=pos.holding_days + 1,
                        is_partial=False,
                    ))
                    del positions[code]
                    continue

                # +12% 절반 익절
                if high_px >= tp_px:
                    sell_shares = pos.shares // 2
                    if sell_shares > 0:
                        exit_px = max(open_px, tp_px)
                        proceeds = sell_shares * exit_px * (1.0 - sell_c)
                        cost_basis = sell_shares * pos.entry_price * (1.0 + buy_c)
                        cash += proceeds
                        pos.shares -= sell_shares
                        pos.half_taken = True
                        half_tp_hits += 1
                        trades.append(TradeRecord(
                            code=code,
                            entry_date=pos.entry_date.strftime("%Y-%m-%d"),
                            exit_date=dt.strftime("%Y-%m-%d"),
                            entry_price=pos.entry_price,
                            exit_price=exit_px,
                            shares=sell_shares,
                            weight_at_entry=pos.weight_at_entry * 0.5,
                            gross_ret=(exit_px / pos.entry_price) - 1.0,
                            net_ret=(proceeds / cost_basis) - 1.0,
                            pnl=proceeds - cost_basis,
                            exit_reason="half_tp_12pct",
                            holding_days=pos.holding_days + 1,
                            is_partial=True,
                        ))
                    else:
                        pos.half_taken = True
                    # 익절 당일 마감 후 최고가 갱신 (익일부터 -5% 트레일링 적용 -> Zero Look-Ahead)
                    pos.prev_highest_high = max(pos.prev_highest_high, high_px)
                    pos.holding_days += 1
                    continue

                # 미체결 시 최고가 및 보유일 갱신
                pos.prev_highest_high = max(pos.prev_highest_high, high_px)
                pos.holding_days += 1

            else:
                # 절반 익절 완료된 잔여 50%: 전일 확정 최고가 대비 -5% 트레일링 스탑
                trail_stop_px = pos.prev_highest_high * (1.0 - TRAIL_AFTER_TP_PCT)
                hard_stop_px = pos.entry_price * (1.0 - STOP_LOSS_PCT)
                eff_stop = max(trail_stop_px, hard_stop_px)

                if low_px <= eff_stop:
                    exit_px = min(open_px, eff_stop)
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
                        weight_at_entry=pos.weight_at_entry * 0.5,
                        gross_ret=(exit_px / pos.entry_price) - 1.0,
                        net_ret=(proceeds / cost_basis) - 1.0,
                        pnl=proceeds - cost_basis,
                        exit_reason="trailing_stop_5pct",
                        holding_days=pos.holding_days + 1,
                        is_partial=False,
                    ))
                    del positions[code]
                else:
                    pos.prev_highest_high = max(pos.prev_highest_high, high_px)
                    pos.holding_days += 1

        # (4) 종가 평가 및 5일 횡보 강제 회전 판정
        stock_val = 0.0
        for code, pos in positions.items():
            df = data_by_code[code]
            if dt in df.index:
                row = df.loc[dt]
                close_px = float(row["Close"])
                pos.last_known_close = close_px
                stock_val += pos.shares * close_px

                if not pos.half_taken and pos.holding_days >= 5:
                    cur_ret = (close_px / pos.entry_price) - 1.0
                    if cfg.mode_5d == "strict":
                        # 5일 경과 시 +12% 미도달 & 횡보(-4% < ret < +3%)면 익일 시가 회전
                        if -STOP_LOSS_PCT < cur_ret < 0.03:
                            pos.pending_exit_reason = "sideways_5d_strict"
                    elif cfg.mode_5d == "force":
                        # 5일 경과 시 +12% 미도달이면 무조건 익일 시가 회전
                        pos.pending_exit_reason = "sideways_5d_force"

                if pos.pending_exit_reason is None and pos.holding_days >= cfg.max_hold_days:
                    pos.pending_exit_reason = "max_hold_days"
            else:
                stock_val += pos.shares * pos.last_known_close

        equity = cash + stock_val
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
        })

        # (5) 익일 매수 후보 스캔 (대표님 원형 규칙 엄수)
        candidates = []
        for code, df in data_by_code.items():
            if code in positions or dt not in df.index:
                continue
            row = df.loc[dt]
            if row["is_frozen"]:
                continue

            c = float(row["Close"])
            o = float(row["Open"])
            if c < MIN_PRICE:
                continue

            # 1) 급등/주도주 이력: 최근 60일 내 거래대금 500억 이상 + (최근 60일 MA20 대비 +15% 급등 OR MA60 상승 추세)
            if pd.isna(row["max_tval_60d"]) or float(row["max_tval_60d"]) < MIN_PEAK_TVAL_60D:
                continue
            if not (bool(row["had_surge_60d"]) or bool(row["ma60_rising"])):
                continue

            # 2) 개미털기: 직전 1~5일 내 20일선 아래 이탈 후 당일 종가 20일선 위 복귀 + 양봉
            if pd.isna(row["ma20"]) or pd.isna(row["recent_below_ma20_5d"]):
                continue
            if not (float(row["recent_below_ma20_5d"]) >= 1.0 and c > float(row["ma20"]) and c > o):
                continue

            # 3) 외인+기관 몰래 매집: 당일 순매수 > 0 AND 최근 3일 누적 순매수 > 0
            sf1 = row["smart_flow_ratio"]
            sf3 = row["smart_flow_3d_ratio"]
            if pd.isna(sf1) or pd.isna(sf3) or float(sf1) <= 0.0 or float(sf3) <= 0.0:
                continue

            # 4) 보너스 레짐 필터
            if cfg.use_regime_filter and not bool(row["bench_regime_on"]):
                continue

            candidates.append((code, float(sf1)))

        # 외인·기관 매집 강도 상위 순 선정
        candidates.sort(key=lambda x: x[1], reverse=True)
        pending_entries = candidates[: MAX_POSITIONS * 2]

    equity_df = pd.DataFrame(daily_logs)
    audit = {
        "max_observed_total_weight": float(max_observed_total_weight),
        "max_observed_single_entry_weight": float(max_observed_single_entry_weight),
        "min_observed_cash": float(min_observed_cash),
        "min_observed_equity": float(min_observed_equity),
        "stop_loss_pct": float(STOP_LOSS_PCT),
        "round_trip_cost": float(ROUND_TRIP_COST if apply_costs else 0.0),
        "total_entries": int(total_entries),
        "hard_stop_rate_entry": float(hard_stop_hits / total_entries) if total_entries > 0 else 0.0,
        "half_tp_rate_entry": float(half_tp_hits / total_entries) if total_entries > 0 else 0.0,
        "sideways_5d_rate_entry": float(sideways_5d_hits / total_entries) if total_entries > 0 else 0.0,
    }
    return equity_df, trades, audit


# ==============================================================================
# 4. 성과 지표 & Z3 SLA 검증
# ==============================================================================
def calc_metrics(equity_df: pd.DataFrame, trades: List[TradeRecord], start_date: str, end_date: str) -> Dict:
    sub = equity_df[(equity_df["date"] >= start_date) & (equity_df["date"] <= end_date)].copy()
    eq = sub["equity"].values
    total_ret = (eq[-1] / eq[0]) - 1.0
    n_days = len(sub)
    dt0 = pd.Timestamp(sub["date"].iloc[0])
    dt1 = pd.Timestamp(sub["date"].iloc[-1])
    cal_years = max((dt1 - dt0).days / 365.25, 1e-6)
    cagr_cal = (eq[-1] / eq[0]) ** (1.0 / cal_years) - 1.0 if eq[-1] > 0 else -1.0
    cagr_252 = (eq[-1] / eq[0]) ** (1.0 / (n_days / 252.0)) - 1.0 if eq[-1] > 0 else -1.0

    dr = pd.Series(eq).pct_change().dropna()
    sharpe = (dr.mean() / dr.std()) * np.sqrt(252) if dr.std() > 0 else 0.0
    mdd = float((eq / np.maximum.accumulate(eq) - 1.0).min())

    sub_trades = [t for t in trades if start_date <= t.entry_date <= end_date]
    n_legs = len(sub_trades)
    # 진입 건수(완전 청산 건 기준: is_partial == False)
    full_closes = [t for t in sub_trades if not t.is_partial]
    n_entries = len(full_closes)
    if n_legs > 0:
        wins = [t.net_ret for t in sub_trades if t.net_ret > 0]
        losses = [t.net_ret for t in sub_trades if t.net_ret <= 0]
        win_rate = len(wins) / n_legs
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(abs(np.mean(losses))) if losses else 0.0
        payoff_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0
        profit_factor = (sum(wins) / abs(sum(losses))) if losses and abs(sum(losses)) > 0 else 0.0
        avg_hold = float(np.mean([t.holding_days for t in full_closes])) if full_closes else 0.0
    else:
        win_rate = avg_win = avg_loss = payoff_ratio = profit_factor = avg_hold = 0.0

    reason_counts = {}
    for t in sub_trades:
        reason_counts[t.exit_reason] = reason_counts.get(t.exit_reason, 0) + 1

    return {
        "total_return": float(total_ret),
        "cagr_cal": float(cagr_cal),
        "cagr_252": float(cagr_252),
        "sharpe": float(sharpe),
        "mdd": float(mdd),
        "num_trade_legs": int(n_legs),
        "num_closed_positions": int(n_entries),
        "win_rate_leg": float(win_rate),
        "payoff_ratio": float(payoff_ratio),
        "profit_factor": float(profit_factor),
        "avg_holding_days": float(avg_hold),
        "avg_exposure": float(sub["invested_weight"].mean()),
        "final_equity": float(eq[-1]),
        "exit_reasons": reason_counts,
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
    return {"total_return": float(total_ret), "cagr_cal": float(cagr_cal), "cagr_252": float(cagr_252), "sharpe": float(sharpe), "mdd": float(mdd)}


def verify_z3_revolver(name: str, cdata: Dict, kospi: Dict, la_diff: float) -> Dict:
    csv_path = BASE_DIR / f"equity_rev_{name}_net.csv"
    df = pd.read_csv(csv_path)
    audit = cdata["net"]["audit"]
    nf = cdata["net"]["full"]
    no = cdata["net"]["oos"]
    gf = cdata["gross"]["full"]
    kf, ko = kospi["full"], kospi["oos"]

    def chk(label, expr):
        s = Solver()
        s.add(expr)
        r = s.check()
        return {"clause": label, "status": "SAT" if r == sat else "UNSAT", "passed": r == sat}

    max_w, min_c, min_e, max_sw, max_p = Real("max_w"), Real("min_c"), Real("min_e"), Real("max_sw"), Real("max_p")
    rtc, nr, gr, sl, la = Real("rtc"), Real("nr"), Real("gr"), Real("sl"), Real("la")
    fc, fs, fm, oc, os_, om = Real("fc"), Real("fs"), Real("fm"), Real("oc"), Real("os"), Real("om")
    kfc, kfs, kfm, koc, kos, kom = Real("kfc"), Real("kfs"), Real("kfm"), Real("koc"), Real("kos"), Real("kom")

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
        chk("SLA-1_NoLeverage_Max3Pos", And(b, max_w <= RealVal("1.0001"), max_p <= RealVal("3"))),
        chk("SLA-2_NoBankruptcy", And(b, min_c >= RealVal("0"), min_e > RealVal("0"))),
        chk("SLA-3_SingleStockCap_LE_33.4pct", And(b, max_sw <= RealVal("0.334"))),
        chk("SLA-4_ConservativeCost_0.73pct_Deducted", And(b, rtc >= RealVal("0.006"), nr < gr)),
        chk("SLA-5_HardStop4pct_And_ZeroLookAhead", And(b, sl == RealVal("0.04"), la <= RealVal("0.000000001"))),
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
    data_by_code = compute_revolver_indicators(ohlcv, meta, flow, bench_regime)

    all_dates = pd.to_datetime(bench[bench["code"] == "KS11"]["date"].unique())
    trading_dates = sorted([d for d in all_dates if pd.Timestamp(BT_START) <= d <= pd.Timestamp(DATA_END)])

    configs = [
        RevolverConfig("R3_Strict5d", "원형 해석 A (5일 경과 시 -4%~+3% 횡보만 청산)", "strict", False, 30),
        RevolverConfig("R3_Force5d", "원형 해석 B (5일 경과 시 +12% 미도달 무조건 청산)", "force", False, 30),
        RevolverConfig("R3_Regime", "보너스 (원형 A + 시장 레짐 필터 MA60)", "strict", True, 30),
    ]

    kospi = {
        "full": calc_benchmark_metrics(bench, "KS11", BT_START, DATA_END),
        "is": calc_benchmark_metrics(bench, "KS11", BT_START, IS_END),
        "oos": calc_benchmark_metrics(bench, "KS11", OOS_START, DATA_END),
    }

    results = {
        "meta": {
            "bt_start": trading_dates[0].strftime("%Y-%m-%d"),
            "is_end": IS_END,
            "oos_start": OOS_START,
            "bt_end": trading_dates[-1].strftime("%Y-%m-%d"),
            "trading_days": len(trading_dates),
            "max_positions": MAX_POSITIONS,
            "max_single_weight": MAX_SINGLE_WEIGHT,
            "round_trip_cost": ROUND_TRIP_COST,
            "data_limitations": [
                "연기금 단독 세부분류 미제공 -> 외인+기관 합산 확정 순매수로 근사",
                "재무제표 무차입 흑자 데이터 미제공 -> 시총 상위 200종목 + 최근 60일 거래대금 500억 이상 이력으로 근사",
            ],
        },
        "benchmarks": {"KOSPI": kospi},
        "cases": {},
        "lookahead_perturbation_max_diff": {},
    }

    baseline_is_eq = {}
    print("=== 3-Shot Revolver Strategy Backtest (Net 0.73%) ===")
    for cfg in configs:
        eq_net, tr_net, audit_net = run_revolver(cfg, data_by_code, trading_dates, True)
        eq_gross, tr_gross, _ = run_revolver(cfg, data_by_code, trading_dates, False)

        baseline_is_eq[cfg.name] = eq_net[eq_net["date"] <= IS_END]["equity"].values
        eq_net.to_csv(BASE_DIR / f"equity_rev_{cfg.name}_net.csv", index=False)
        pd.DataFrame([asdict(t) for t in tr_net]).to_csv(BASE_DIR / f"trades_rev_{cfg.name}_net.csv", index=False)

        nf = calc_metrics(eq_net, tr_net, BT_START, DATA_END)
        ni = calc_metrics(eq_net, tr_net, BT_START, IS_END)
        no = calc_metrics(eq_net, tr_net, OOS_START, DATA_END)
        gf = calc_metrics(eq_gross, tr_gross, BT_START, DATA_END)

        results["cases"][cfg.name] = {
            "label": cfg.label,
            "net": {"full": nf, "is": ni, "oos": no, "audit": audit_net},
            "gross": {"full": gf},
        }
        print(
            f"[{cfg.name:11s}] CalCAGR={nf['cagr_cal']*100:+6.2f}% (Gross={gf['cagr_cal']*100:+6.2f}%) Tot={nf['total_return']*100:+6.1f}% "
            f"Shp={nf['sharpe']:5.2f} MDD={nf['mdd']*100:6.2f}% Entries={audit_net['total_entries']:4d} "
            f"SL4%={audit_net['hard_stop_rate_entry']*100:4.1f}% TP12%={audit_net['half_tp_rate_entry']*100:4.1f}% 5d={audit_net['sideways_5d_rate_entry']*100:4.1f}% | "
            f"IS={ni['cagr_cal']*100:+6.2f}% OOS={no['cagr_cal']*100:+6.2f}% (Shp {no['sharpe']:4.2f}, MDD {no['mdd']*100:6.2f}%)"
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

    p_data = compute_revolver_indicators(p_ohlcv, meta, p_flow, compute_benchmark_regime(p_bench))
    print("\n[Look-Ahead Audit] Perturbing OOS prices (+500%) & flows (-500%)...")
    for cfg in configs:
        eq_p, _, _ = run_revolver(cfg, p_data, trading_dates, True)
        p_is = eq_p[eq_p["date"] <= IS_END]["equity"].values
        diff = float(np.max(np.abs(p_is - baseline_is_eq[cfg.name])))
        results["lookahead_perturbation_max_diff"][cfg.name] = diff
        print(f"  {cfg.name:11s}: max IS diff = {diff:.10f}")

    # Z3-Solver SLA 검증
    z3_summary = {"cases": {}}
    print("\n[Z3-Solver SLA Verification]")
    for cfg in configs:
        v = verify_z3_revolver(cfg.name, results["cases"][cfg.name], kospi, results["lookahead_perturbation_max_diff"][cfg.name])
        z3_summary["cases"][cfg.name] = v
        print(
            f"  {cfg.name:11s}: SLA 1~5={v['structural_1_to_5']} | OOS>0={v['oos_positive_6']} | "
            f"Dominates KOSPI(Full&OOS)={v['dominates_kospi_7']}"
        )

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(Z3_PATH, "w", encoding="utf-8") as f:
        json.dump(z3_summary, f, indent=2, ensure_ascii=False)
    print(f"\n[Done] Saved {RESULTS_PATH.name} and {Z3_PATH.name}")


if __name__ == "__main__":
    main()
