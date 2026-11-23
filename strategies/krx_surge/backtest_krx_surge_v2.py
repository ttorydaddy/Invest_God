"""
Invest_God Phase 1-A-2: KRX 전략 A 고도화 4단계 Ablation Study
- 실제 KRX 종목별 일별 외국인/기관 확정 순매수(Naver Finance Trend API) 200종목 수집 및 DuckDB 병합
- 4단계 순차 Ablation (A-0 -> A-1 -> A-2 -> A-3) + 순수 조합 분해 (A-R, A-FR, A-LR)
- 동일 왕복 거래비용 0.73%, 동일 IS(21.01~24.06) / OOS(24.07~26.11), Zero Look-Ahead 교란 재검증
"""

import datetime
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
import requests

# ==============================================================================
# 1. 전역 설정 (Phase 1-A와 100% 동일 기준 유지)
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "krx_surge_data.duckdb"
RESULTS_V2_PATH = BASE_DIR / "backtest_results_v2.json"

DATA_START = "2020-07-01"
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
MIN_TRADING_VALUE = 10_000_000_000.0
MAX_ENTRY_GAP = 0.15


# ==============================================================================
# 2. 실제 KRX 외인·기관 확정 수급 수집 및 DuckDB 병합
# ==============================================================================
def _parse_int(val) -> int:
    if val is None:
        return 0
    s = str(val).replace(",", "").replace("+", "").strip()
    try:
        return int(s)
    except ValueError:
        return 0


def fetch_investor_flow_single(code: str, start_ymd: str = "20200615", end_ymd: str = "20261120") -> Optional[pd.DataFrame]:
    """Naver Finance 모바일 API에서 종목별 일별 외인·기관·개인 확정 순매수 수집"""
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    bizdate = end_ymd
    rows_map = {}

    for _step in range(32):  # 32 * 60 = 최대 ~1,920거래일 (약 7.5년)
        url = f"https://m.stock.naver.com/api/stock/{code}/trend?pageSize=60&bizdate={bizdate}"
        ok = False
        for attempt in range(3):
            try:
                r = s.get(url, timeout=10)
                if r.status_code == 200:
                    arr = r.json()
                    ok = True
                    break
            except Exception:
                time.sleep(0.3 * (attempt + 1))
        if not ok or not arr:
            break

        for item in arr:
            bd = str(item.get("bizdate", ""))
            if len(bd) != 8:
                continue
            rows_map[bd] = item

        oldest = str(arr[-1].get("bizdate", ""))
        if not oldest or oldest <= start_ymd or oldest == bizdate:
            break
        dt = datetime.datetime.strptime(oldest, "%Y%m%d") - datetime.timedelta(days=1)
        bizdate = dt.strftime("%Y%m%d")

    if not rows_map:
        return None

    records = []
    for bd, item in rows_map.items():
        if bd < start_ymd or bd > end_ymd:
            continue
        date_str = f"{bd[:4]}-{bd[4:6]}-{bd[6:8]}"
        frgn_q = _parse_int(item.get("foreignerPureBuyQuant"))
        inst_q = _parse_int(item.get("organPureBuyQuant"))
        indiv_q = _parse_int(item.get("individualPureBuyQuant"))
        raw_close = _parse_int(item.get("closePrice"))
        raw_vol = _parse_int(item.get("accumulatedTradingVolume"))
        records.append({
            "date": date_str,
            "code": code,
            "frgn_net_qty": frgn_q,
            "inst_net_qty": inst_q,
            "indiv_net_qty": indiv_q,
            "raw_close": float(raw_close),
            "raw_volume": float(raw_vol),
            "frgn_net_val": float(frgn_q * raw_close),
            "inst_net_val": float(inst_q * raw_close),
        })

    if not records:
        return None
    return pd.DataFrame(records)


def sync_investor_flow_to_duckdb(force_refresh: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """기존 DuckDB에서 OHLCV/메타/벤치마크 로드 + 외인·기관 수급 테이블 동기화"""
    con = duckdb.connect(str(DB_PATH))
    tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]

    ohlcv = con.execute("SELECT * FROM daily_ohlcv ORDER BY code, date").df()
    meta = con.execute("SELECT * FROM universe_meta").df()
    bench = con.execute("SELECT * FROM benchmarks ORDER BY code, date").df()
    codes = sorted(ohlcv["code"].unique().tolist())

    if not force_refresh and "investor_flow" in tables:
        cnt = con.execute("SELECT COUNT(DISTINCT code) FROM investor_flow").fetchone()[0]
        if cnt >= 190:
            print(f"[DuckDB Cache] Loaded investor_flow for {cnt} tickers.")
            flow = con.execute("SELECT * FROM investor_flow ORDER BY code, date").df()
            con.close()
            return ohlcv, meta, bench, flow

    print(f"[Flow Sync] Fetching real KRX foreign/institutional net flow for {len(codes)} tickers...")
    frames = []
    failed = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=14) as ex:
        futs = {ex.submit(fetch_investor_flow_single, c): c for c in codes}
        done = 0
        for fut in as_completed(futs):
            c = futs[fut]
            done += 1
            res = fut.result()
            if res is not None and not res.empty:
                frames.append(res)
            else:
                failed.append(c)
            if done % 40 == 0 or done == len(codes):
                print(f"  Flow Progress: {done}/{len(codes)} (ok={len(frames)}, fail={len(failed)}, {time.time()-t0:.1f}s)")

    flow = pd.concat(frames, ignore_index=True)
    con.execute("DROP TABLE IF EXISTS investor_flow")
    con.execute("CREATE TABLE investor_flow AS SELECT * FROM flow")
    con.close()
    print(f"[Flow Sync] Saved investor_flow: {flow['code'].nunique()} tickers, {len(flow):,} rows (failed={len(failed)}).")
    return ohlcv, meta, bench, flow


# ==============================================================================
# 3. 인과적 지표 계산 (OHLCV + 수급 + 거래대금 순위 + RS + 시장 레짐)
# ==============================================================================
def compute_benchmark_regime(bench: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """KOSPI(KS11), KOSDAQ(KQ11) 지수의 60일 이평 및 5일/20일 수익률 계산 (Strict Causal)"""
    out = {}
    for bcode in ["KS11", "KQ11"]:
        sub = bench[bench["code"] == bcode].sort_values("date").copy()
        sub["date"] = pd.to_datetime(sub["date"])
        sub = sub.set_index("date")
        c = sub["Close"]
        sub["ma60"] = c.rolling(60, min_periods=60).mean()
        sub["ret_5d"] = c.pct_change(5)
        sub["ret_20d"] = c.pct_change(20)
        # 레짐 ON: 지수 > 60일 이평 AND 최근 5일 급락(-5% 이하) 없음
        sub["regime_on"] = (c > sub["ma60"]) & (sub["ret_5d"] > -0.05)
        out[bcode] = sub
    return out


def compute_enriched_indicators(
    ohlcv: pd.DataFrame,
    meta: pd.DataFrame,
    flow: pd.DataFrame,
    bench_regime: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """
    종목별 패널 딕셔너리 생성 (Strict Causal / Zero Look-Ahead).
    - 거래대금 순위(tval_rank)는 매일 동시점 유니버스 내 cross-sectional rank로 계산 (미래 참조 0%)
    - 수급 비율은 당일 확정 수량 / 당일 확정 거래량 (수정주가 왜곡 0%)
    """
    market_map = meta.set_index("code")["market"].to_dict()

    # Cross-sectional 일별 거래대금 순위(1 = 최고) 및 직전 20일 내 Top30 진입 빈도 계산
    tmp = ohlcv[["date", "code", "Close", "Volume"]].copy()
    tmp["tval"] = tmp["Close"] * tmp["Volume"]
    tval_pivot = tmp.pivot(index="date", columns="code", values="tval").sort_index()
    rank_pivot = tval_pivot.rank(axis=1, ascending=False, method="min")
    in_top30_pivot = (rank_pivot <= 30).astype(float)
    # 직전 20거래일(t-1 ~ t-20) 중 Top 30 진입 일수 (t일 제외!)
    prev20_top30_cnt_pivot = in_top30_pivot.shift(1).rolling(20, min_periods=1).sum()

    # 수급 병합 준비
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

        df["tval"] = c * v
        df["ma10"] = c.rolling(10, min_periods=10).mean()
        df["ma20"] = c.rolling(20, min_periods=20).mean()
        df["ma60"] = c.rolling(60, min_periods=60).mean()
        df["ma20_slope5"] = df["ma20"] - df["ma20"].shift(5)

        df["avg_tval20_prev"] = df["tval"].rolling(20, min_periods=20).mean().shift(1)
        df["high20_prev"] = h.rolling(20, min_periods=20).max().shift(1)
        df["ret_1d"] = c.pct_change()
        df["ret_20d"] = c.pct_change(20)
        df["prev_close"] = c.shift(1)
        df["is_frozen"] = (h == l) & (o == c)

        # 외인+기관 확정 수급 지표 (결측 시 NaN 유지 -> 억지 보간 금지!)
        smart_qty = df["frgn_net_qty"] + df["inst_net_qty"]
        raw_vol = df["raw_volume"].replace(0, np.nan)
        df["smart_flow_ratio"] = smart_qty / raw_vol

        smart_val = df["frgn_net_val"] + df["inst_net_val"]
        raw_tval = (df["raw_close"] * df["raw_volume"]).replace(0, np.nan)
        df["smart_flow_3d_ratio"] = (
            smart_val.rolling(3, min_periods=3).sum()
            / raw_tval.rolling(3, min_periods=3).sum()
        )

        # 거래대금 Top30 순위 및 직전 20일 Top30 빈도 매핑
        if code in rank_pivot.columns:
            r_ser = rank_pivot[code]
            r_ser.index = pd.to_datetime(r_ser.index)
            c_ser = prev20_top30_cnt_pivot[code]
            c_ser.index = pd.to_datetime(c_ser.index)
            df["tval_rank"] = r_ser.reindex(df.index)
            df["prev20_top30_cnt"] = c_ser.reindex(df.index)
        else:
            df["tval_rank"] = np.nan
            df["prev20_top30_cnt"] = np.nan

        # 소속 시장 벤치마크 매핑 (RS 및 시장 레짐)
        mkt = market_map.get(code, "KOSPI")
        bcode = "KS11" if mkt == "KOSPI" else "KQ11"
        bdf = bench_regime[bcode]
        df["bench_ret_20d"] = bdf["ret_20d"].reindex(df.index)
        df["bench_regime_on"] = bdf["regime_on"].reindex(df.index).astype("boolean").fillna(False).astype(bool)
        df["market"] = mkt

        data_by_code[code] = df

    return data_by_code


# ==============================================================================
# 4. 전략 A Ablation 백테스트 엔진
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
    shares: int
    weight_at_entry: float
    prev_highest_high: float
    holding_days: int = 0
    pending_exit_reason: Optional[str] = None


@dataclass
class VariantConfig:
    name: str
    label: str
    use_flow_filter: bool = False       # 외인+기관 확정 수급 (당일 >= 2% & 3일 누적 > 0%)
    use_leader_filter: bool = False     # 이정윤 주도주 (Top30 신규진입 & RS > 지수)
    use_regime_filter: bool = False     # 시장 60일 이평 위 & 5일 급락 없음
    relaxed_exit: bool = False          # MA10->MA20, 트레일링 -12%->-15%, 최대보유 30->60일
    stop_loss_pct: float = 0.08


def run_variant(
    cfg: VariantConfig,
    data_by_code: Dict[str, pd.DataFrame],
    trading_dates: List[pd.Timestamp],
    apply_costs: bool = True,
) -> Tuple[pd.DataFrame, List[TradeRecord], Dict]:
    buy_c = BUY_COST if apply_costs else 0.0
    sell_c = SELL_COST if apply_costs else 0.0

    trailing_stop_pct = 0.15 if cfg.relaxed_exit else 0.12
    exit_ma_col = "ma20" if cfg.relaxed_exit else "ma10"
    max_hold_days = 60 if cfg.relaxed_exit else 30

    cash = INITIAL_CAPITAL
    positions: Dict[str, Position] = {}
    pending_entries: List[Tuple[str, float]] = []
    trades: List[TradeRecord] = []
    daily_logs = []

    max_observed_total_weight = 0.0
    max_observed_single_entry_weight = 0.0
    min_observed_cash = INITIAL_CAPITAL
    min_observed_equity = INITIAL_CAPITAL

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
            pnl = proceeds - cost_basis
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
                pnl=pnl,
                exit_reason=pos.pending_exit_reason,
                holding_days=pos.holding_days,
            ))
            del positions[code]

        # (2) 시가 진입
        if pending_entries and len(positions) < MAX_POSITIONS:
            mtm_open = cash + sum(
                pos.shares * (float(data_by_code[c].loc[dt, "Open"]) if dt in data_by_code[c].index else pos.entry_price)
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

            hard_stop_px = pos.entry_price * (1.0 - cfg.stop_loss_pct)
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

        # (4) 종가 평가 및 익일 청산/매수 시그널
        stock_val = 0.0
        for code, pos in positions.items():
            df = data_by_code[code]
            if dt in df.index:
                row = df.loc[dt]
                close_px = float(row["Close"])
                stock_val += pos.shares * close_px
                ma_val = row[exit_ma_col]
                if pd.notna(ma_val) and close_px < float(ma_val):
                    pos.pending_exit_reason = f"below_{exit_ma_col}"
                elif pos.holding_days >= max_hold_days:
                    pos.pending_exit_reason = "max_hold_days"
            else:
                stock_val += pos.shares * pos.entry_price

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

        # 익일 매수 후보 스캔 (t일 종가 확정 지표만 사용)
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

            # [A-0 기본 조건]
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

            # [레짐 필터] 소속 시장 지수가 60일 이평 위 & 최근 5일 급락(-5% 이하) 없음
            if cfg.use_regime_filter:
                if not bool(row["bench_regime_on"]):
                    continue

            # [외인+기관 확정 수급 필터] 당일 순매수 >= 2% & 3일 누적 순매수 > 0%
            if cfg.use_flow_filter:
                sf1 = row["smart_flow_ratio"]
                sf3 = row["smart_flow_3d_ratio"]
                if pd.isna(sf1) or pd.isna(sf3) or float(sf1) < 0.02 or float(sf3) <= 0.0:
                    continue

            # [이정윤 주도주 필터] 거래대금 Top30 신규 진입(직전 20일 Top30 빈도 <= 5) & 20일 RS > 소속 지수
            if cfg.use_leader_filter:
                trank = row["tval_rank"]
                p20cnt = row["prev20_top30_cnt"]
                r20 = row["ret_20d"]
                br20 = row["bench_ret_20d"]
                if (
                    pd.isna(trank)
                    or pd.isna(p20cnt)
                    or pd.isna(r20)
                    or pd.isna(br20)
                    or float(trank) > 30.0
                    or float(p20cnt) > 5.0
                    or float(r20) <= float(br20)
                ):
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
        "stop_loss_pct": float(cfg.stop_loss_pct),
        "trailing_stop_pct": float(trailing_stop_pct),
        "exit_ma_col": exit_ma_col,
        "max_hold_days": max_hold_days,
        "round_trip_cost": float(ROUND_TRIP_COST if apply_costs else 0.0),
    }
    return equity_df, trades, audit


# ==============================================================================
# 5. 성과 지표 산출 & Zero Look-Ahead 교란 검증
# ==============================================================================
def calc_metrics(equity_df: pd.DataFrame, trades: List[TradeRecord], start_date: str, end_date: str) -> Dict:
    sub = equity_df[(equity_df["date"] >= start_date) & (equity_df["date"] <= end_date)].copy()
    if len(sub) < 2:
        return {}
    eq = sub["equity"].values
    total_ret = (eq[-1] / eq[0]) - 1.0
    n_days = len(sub)
    years = n_days / 252.0
    cagr = (eq[-1] / eq[0]) ** (1.0 / years) - 1.0 if years > 0 and eq[-1] > 0 else -1.0
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
    else:
        win_rate = avg_win = avg_loss = payoff_ratio = profit_factor = avg_hold = 0.0

    return {
        "start_date": sub["date"].iloc[0],
        "end_date": sub["date"].iloc[-1],
        "trading_days": int(n_days),
        "total_return": float(total_ret),
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "mdd": float(mdd),
        "num_trades": int(n_trades),
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
    years = len(sub) / 252.0
    cagr = (c[-1] / c[0]) ** (1.0 / years) - 1.0
    dr = pd.Series(c).pct_change().dropna()
    sharpe = (dr.mean() / dr.std()) * np.sqrt(252) if dr.std() > 0 else 0.0
    mdd = float((c / np.maximum.accumulate(c) - 1.0).min())
    return {"total_return": float(total_ret), "cagr": float(cagr), "sharpe": float(sharpe), "mdd": float(mdd)}


def verify_zero_lookahead_v2(
    ohlcv: pd.DataFrame,
    meta: pd.DataFrame,
    bench: pd.DataFrame,
    flow: pd.DataFrame,
    trading_dates: List[pd.Timestamp],
    variants: List[VariantConfig],
    baseline_is_eq: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """OOS 구간(2024-07-01 이후) 주가 +500% 및 수급 -500% 교란 시 IS 구간 자산 곡선 불변성 검증"""
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

    p_br = compute_benchmark_regime(p_bench)
    p_data = compute_enriched_indicators(p_ohlcv, meta, p_flow, p_br)

    diffs = {}
    for cfg in variants:
        eq_df, _, _ = run_variant(cfg, p_data, trading_dates, apply_costs=True)
        p_is = eq_df[eq_df["date"] <= IS_END]["equity"].values
        diffs[cfg.name] = float(np.max(np.abs(p_is - baseline_is_eq[cfg.name])))
    return diffs


# ==============================================================================
# 6. 메인 실행
# ==============================================================================
def main():
    ohlcv, meta, bench, flow = sync_investor_flow_to_duckdb(force_refresh=False)

    # 수급 데이터 커버리지 통계
    flow_in_bt = flow[(flow["date"] >= BT_START) & (flow["date"] <= DATA_END)]
    ohlcv_in_bt = ohlcv[(ohlcv["date"] >= BT_START) & (ohlcv["date"] <= DATA_END)]
    coverage_ratio = len(flow_in_bt) / len(ohlcv_in_bt) if len(ohlcv_in_bt) > 0 else 0.0
    print(f"[Flow Coverage] Tickers: {flow['code'].nunique()}/{ohlcv['code'].nunique()}, Row coverage: {coverage_ratio*100:.2f}%")

    print("[Indicators] Computing enriched causal indicators (OHLCV + Flow + Top30 Rank + RS + Regime)...")
    bench_regime = compute_benchmark_regime(bench)
    data_by_code = compute_enriched_indicators(ohlcv, meta, flow, bench_regime)

    all_dates = pd.to_datetime(bench[bench["code"] == "KS11"]["date"].unique())
    trading_dates = sorted([d for d in all_dates if pd.Timestamp(BT_START) <= d <= pd.Timestamp(DATA_END)])

    variants = [
        # [핵심 4단계 순차 Ablation]
        VariantConfig("A-0", "Baseline (Phase 1-A 전략 A)", False, False, False, False),
        VariantConfig("A-1", "A-0 + 외인·기관 확정 수급", True, False, False, False),
        VariantConfig("A-2", "A-1 + 이정윤 주도주(Top30신규+RS)", True, True, False, False),
        VariantConfig("A-3", "A-2 + 레짐필터 & 회전율완화", True, True, True, True),
        # [핵심 추가 조합: 필터별 순수 기여도 분해]
        VariantConfig("A-R", "A-0 + 레짐필터 & 회전율완화 (수급/주도주 제외)", False, False, True, True),
        VariantConfig("A-FR", "A-0 + 외인·기관 수급 + 레짐/회전율 (주도주 제외)", True, False, True, True),
        VariantConfig("A-LR", "A-0 + 이정윤 주도주 + 레짐/회전율 (수급 제외)", False, True, True, True),
    ]

    results = {
        "meta": {
            "universe_size": int(ohlcv["code"].nunique()),
            "flow_tickers_collected": int(flow["code"].nunique()),
            "flow_row_coverage_pct": float(coverage_ratio * 100.0),
            "bt_start": trading_dates[0].strftime("%Y-%m-%d"),
            "is_end": IS_END,
            "oos_start": OOS_START,
            "bt_end": trading_dates[-1].strftime("%Y-%m-%d"),
            "trading_days": len(trading_dates),
            "round_trip_cost": ROUND_TRIP_COST,
        },
        "benchmarks": {
            "KOSPI": {
                "full": calc_benchmark_metrics(bench, "KS11", BT_START, DATA_END),
                "is": calc_benchmark_metrics(bench, "KS11", BT_START, IS_END),
                "oos": calc_benchmark_metrics(bench, "KS11", OOS_START, DATA_END),
            },
            "KOSDAQ": {
                "full": calc_benchmark_metrics(bench, "KQ11", BT_START, DATA_END),
                "is": calc_benchmark_metrics(bench, "KQ11", BT_START, IS_END),
                "oos": calc_benchmark_metrics(bench, "KQ11", OOS_START, DATA_END),
            },
        },
        "variants": {},
        "lookahead_perturbation_max_diff": {},
    }

    baseline_is_eq = {}
    print("\n=== Ablation Study Results (Net = 0.73% Round-Trip Cost) ===")
    for cfg in variants:
        eq_net, trades_net, audit_net = run_variant(cfg, data_by_code, trading_dates, apply_costs=True)
        eq_gross, trades_gross, _ = run_variant(cfg, data_by_code, trading_dates, apply_costs=False)

        baseline_is_eq[cfg.name] = eq_net[eq_net["date"] <= IS_END]["equity"].values
        eq_net.to_csv(BASE_DIR / f"equity_v2_{cfg.name}_net.csv", index=False)
        pd.DataFrame([asdict(t) for t in trades_net]).to_csv(BASE_DIR / f"trades_v2_{cfg.name}_net.csv", index=False)

        reason_counts = {}
        for t in trades_net:
            reason_counts[t.exit_reason] = reason_counts.get(t.exit_reason, 0) + 1

        nf = calc_metrics(eq_net, trades_net, BT_START, DATA_END)
        ni = calc_metrics(eq_net, trades_net, BT_START, IS_END)
        no = calc_metrics(eq_net, trades_net, OOS_START, DATA_END)
        gf = calc_metrics(eq_gross, trades_gross, BT_START, DATA_END)

        results["variants"][cfg.name] = {
            "label": cfg.label,
            "config": {
                "use_flow_filter": cfg.use_flow_filter,
                "use_leader_filter": cfg.use_leader_filter,
                "use_regime_filter": cfg.use_regime_filter,
                "relaxed_exit": cfg.relaxed_exit,
                "stop_loss_pct": cfg.stop_loss_pct,
            },
            "net": {
                "full": nf,
                "is": ni,
                "oos": no,
                "audit": audit_net,
                "exit_reasons": reason_counts,
            },
            "gross": {
                "full": gf,
                "is": calc_metrics(eq_gross, trades_gross, BT_START, IS_END),
                "oos": calc_metrics(eq_gross, trades_gross, OOS_START, DATA_END),
            },
        }
        print(
            f"[{cfg.name:4s}] Full: CAGR={nf['cagr']*100:+6.2f}% Shp={nf['sharpe']:5.2f} MDD={nf['mdd']*100:6.2f}% Trd={nf['num_trades']:3d} Hold={nf['avg_holding_days']:4.1f}d | "
            f"IS: {ni['cagr']*100:+6.2f}% (Shp {ni['sharpe']:5.2f}) | "
            f"OOS: {no['cagr']*100:+6.2f}% (Shp {no['sharpe']:5.2f}, MDD {no['mdd']*100:6.2f}%) | Gross: {gf['cagr']*100:+6.2f}%"
        )

    print("\n[Look-Ahead Audit V2] Perturbing OOS prices (+500%) & flows (-500%)...")
    la_diffs = verify_zero_lookahead_v2(ohlcv, meta, bench, flow, trading_dates, variants, baseline_is_eq)
    results["lookahead_perturbation_max_diff"] = la_diffs
    for k, v in la_diffs.items():
        print(f"  {k:4s}: max IS diff = {v:.10f}")

    with open(RESULTS_V2_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[Done] V2 results saved to {RESULTS_V2_PATH}")


if __name__ == "__main__":
    main()
