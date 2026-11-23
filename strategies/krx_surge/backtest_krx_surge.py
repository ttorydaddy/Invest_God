"""
KRX 건강한 급등주 초입 포착 퀀트 전략 3종 백테스트 엔진
- 프로젝트: Invest_God (Phase 1-A)
- 원칙: Zero Look-Ahead Bias, 보수적 거래비용(왕복 0.73%), IS/OOS 분리, DuckDB 캐싱
- 데이터 출처: KRX KIND 상장법인목록 + Naver Finance 시총 순위 + FinanceDataReader(Naver 수정주가)
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import FinanceDataReader as fdr
import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ==============================================================================
# 1. 전역 설정 (불변 전제조건 및 현실 제약)
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "krx_surge_data.duckdb"
RESULTS_JSON_PATH = BASE_DIR / "backtest_results.json"

DATA_START = "2020-07-01"   # 60/120일 이평 워밍업 포함
BT_START = "2021-01-04"     # 백테스트 시작일
IS_END = "2024-06-30"       # In-Sample 종료일 (~60%)
OOS_START = "2024-07-01"    # Out-of-Sample 시작일 (~40%)
DATA_END = "2026-11-20"     # 최근 거래일

INITIAL_CAPITAL = 100_000_000.0  # 1억 원
MAX_POSITIONS = 5                # 최대 동시 보유 5종목
MAX_SINGLE_WEIGHT = 0.20         # 단일 종목 최대 진입 비중 20%

# 보수적 거래비용 (KRX 주식 기준)
BUY_FEE = 0.00015       # 매수 수수료 0.015%
BUY_SLIPPAGE = 0.0025   # 급등주 매수 슬리피지 0.25%
SELL_FEE = 0.00015      # 매도 수수료 0.015%
SELL_TAX = 0.0020       # 증권거래세 0.20% (2021~2026 보수적 일괄)
SELL_SLIPPAGE = 0.0025  # 급등주 매도 슬리피지 0.25%

BUY_COST = BUY_FEE + BUY_SLIPPAGE                # 0.265%
SELL_COST = SELL_FEE + SELL_TAX + SELL_SLIPPAGE  # 0.465%
ROUND_TRIP_COST = BUY_COST + SELL_COST           # 0.730%

MIN_PRICE = 2000.0               # 동전주 제외
MIN_TRADING_VALUE = 10_000_000_000.0  # 일 거래대금 최소 100억 원
MAX_ENTRY_GAP = 0.15             # 익일 시가 갭상승 +15% 초과 시 추격 매수 방지 스킵


# ==============================================================================
# 2. 데이터 수집 및 DuckDB 캐싱
# ==============================================================================
def fetch_kind_common_stocks() -> pd.DataFrame:
    """KRX KIND 공식 상장법인목록 다운로드 (유가증권 + 코스닥 보통주)"""
    url = "https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    r.raise_for_status()
    df = pd.read_html(StringIO(r.text))[0]
    df["code"] = df["종목코드"].astype(str).str.zfill(6)
    df = df[df["시장구분"].isin(["유가", "코스닥"])].copy()
    # 스팩, 리츠, 우선주 등 제외
    df = df[~df["회사명"].str.contains("스팩|SPAC|리츠", case=False, na=False)]
    df["market"] = df["시장구분"].map({"유가": "KOSPI", "코스닥": "KOSDAQ"})
    return df[["code", "회사명", "market", "업종", "상장일"]].rename(
        columns={"회사명": "name", "업종": "sector", "상장일": "list_date"}
    )


def fetch_naver_market_cap_codes(market: str, pages: int = 3) -> List[str]:
    """네이버 금융 모바일 API에서 시가총액 상위 종목코드 조회"""
    codes = []
    for p in range(1, pages + 1):
        url = f"https://m.stock.naver.com/api/stocks/marketValue/{market}?page={p}&pageSize=100"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        if r.status_code != 200:
            break
        stocks = r.json().get("stocks", [])
        for s in stocks:
            codes.append(str(s["itemCode"]).zfill(6))
    return codes


def build_universe(target_per_market: int = 100) -> pd.DataFrame:
    """KOSPI 100 + KOSDAQ 100 = 200종목 유니버스 구성 (보통주 중심, 2021년 이전 상장 우대)"""
    kind_df = fetch_kind_common_stocks()
    kind_map = kind_df.set_index("code")

    selected_rows = []
    for market in ["KOSPI", "KOSDAQ"]:
        ranked_codes = fetch_naver_market_cap_codes(market, pages=3)
        # KIND 보통주 매칭 + 2021-01-01 이전 상장 종목 우선, 부족 시 이후 상장 포함
        valid_early = []
        valid_late = []
        seen = set()
        for rank_idx, code in enumerate(ranked_codes, start=1):
            if code in seen or code not in kind_map.index:
                continue
            seen.add(code)
            row = kind_map.loc[code]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            # 우선주 끝자리(5, 7, 9 등) 제외: 보통주는 대개 끝자리 0
            if not code.endswith("0"):
                continue
            info = {
                "code": code,
                "name": str(row["name"]),
                "market": market,
                "sector": str(row["sector"]),
                "list_date": str(row["list_date"]),
                "mcap_rank": rank_idx,
            }
            if str(row["list_date"]) <= "2021-01-01":
                valid_early.append(info)
            else:
                valid_late.append(info)

        picked = valid_early[:target_per_market]
        if len(picked) < target_per_market:
            picked.extend(valid_late[: target_per_market - len(picked)])
        selected_rows.extend(picked)

    return pd.DataFrame(selected_rows)


def download_single_ticker(code: str, start: str, end: str) -> Optional[pd.DataFrame]:
    try:
        df = fdr.DataReader(code, start, end)
        if df is None or df.empty or len(df) < 120:
            return None
        df = df.reset_index()
        date_col = df.columns[0]
        df = df.rename(columns={date_col: "date"})
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df["code"] = code
        cols = ["date", "code", "Open", "High", "Low", "Close", "Volume"]
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df[cols].dropna()
        df = df[(df["Close"] > 0) & (df["Open"] > 0) & (df["High"] >= df["Low"])]
        return df
    except Exception:
        return None


def download_benchmark(bcode: str, yf_sym: str, fallback_etf: str, start: str, end: str) -> Optional[pd.DataFrame]:
    try:
        df = yf.Ticker(yf_sym).history(start=start, end=end)
        if df is not None and not df.empty and len(df) >= 120:
            df = df.reset_index()
            date_col = df.columns[0]
            df = df.rename(columns={date_col: "date"})
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.strftime("%Y-%m-%d")
            df["code"] = bcode
            cols = ["date", "code", "Open", "High", "Low", "Close", "Volume"]
            for c in ["Open", "High", "Low", "Close", "Volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            return df[cols].dropna()
    except Exception:
        pass
    # Fallback: 대표 지수 ETF (Naver FDR)
    df = download_single_ticker(fallback_etf, start, end)
    if df is not None:
        df["code"] = bcode
    return df


def sync_data_to_duckdb(force_refresh: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """KRX 200종목 + 벤치마크(KS11, KQ11) 수집 및 DuckDB 캐싱"""
    con = duckdb.connect(str(DB_PATH))
    tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]

    if not force_refresh and "daily_ohlcv" in tables and "universe_meta" in tables and "benchmarks" in tables:
        cnt = con.execute("SELECT COUNT(DISTINCT code) FROM daily_ohlcv").fetchone()[0]
        if cnt >= 150:
            print(f"[DuckDB Cache] Loaded {cnt} tickers from {DB_PATH.name}")
            ohlcv = con.execute("SELECT * FROM daily_ohlcv ORDER BY code, date").df()
            meta = con.execute("SELECT * FROM universe_meta").df()
            bench = con.execute("SELECT * FROM benchmarks ORDER BY code, date").df()
            con.close()
            return ohlcv, meta, bench

    print("[Data Sync] Building 200-stock KRX universe (KOSPI 100 + KOSDAQ 100)...")
    meta = build_universe(target_per_market=100)
    print(f"[Data Sync] Universe built: {len(meta)} tickers. Downloading OHLCV ({DATA_START} ~ {DATA_END})...")

    frames = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(download_single_ticker, c, DATA_START, DATA_END): c for c in meta["code"]}
        done = 0
        for fut in as_completed(futs):
            done += 1
            res = fut.result()
            if res is not None:
                frames.append(res)
            if done % 50 == 0 or done == len(meta):
                print(f"  Progress: {done}/{len(meta)} ({time.time() - t0:.1f}s)")

    ohlcv = pd.concat(frames, ignore_index=True)
    # 벤치마크 지수 (KOSPI, KOSDAQ)
    bench_frames = []
    for bcode, yf_sym, fb_etf in [("KS11", "^KS11", "069500"), ("KQ11", "^KQ11", "229200")]:
        bdf = download_benchmark(bcode, yf_sym, fb_etf, DATA_START, DATA_END)
        if bdf is not None:
            bench_frames.append(bdf)
    bench = pd.concat(bench_frames, ignore_index=True) if bench_frames else pd.DataFrame(
        columns=["date", "code", "Open", "High", "Low", "Close", "Volume"]
    )

    # DuckDB 저장
    con.execute("DROP TABLE IF EXISTS daily_ohlcv")
    con.execute("DROP TABLE IF EXISTS universe_meta")
    con.execute("DROP TABLE IF EXISTS benchmarks")
    con.execute("CREATE TABLE daily_ohlcv AS SELECT * FROM ohlcv")
    con.execute("CREATE TABLE universe_meta AS SELECT * FROM meta")
    con.execute("CREATE TABLE benchmarks AS SELECT * FROM bench")
    con.close()
    print(f"[Data Sync] Saved {ohlcv['code'].nunique()} tickers ({len(ohlcv):,} rows) to DuckDB.")
    return ohlcv, meta, bench


# ==============================================================================
# 3. 기술적 지표 계산 (Strict Causal / Zero Look-Ahead)
# ==============================================================================
def compute_indicators(ohlcv: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    종목별 패널 딕셔너리 생성 (date index).
    모든 지표는 t일 종가까지 확정된 과거 데이터만 사용 (shift/rolling).
    """
    data_by_code = {}
    for code, g in ohlcv.groupby("code", sort=False):
        df = g.sort_values("date").copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

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
        df["avg_vol20_prev"] = v.rolling(20, min_periods=20).mean().shift(1)
        df["high20_prev"] = h.rolling(20, min_periods=20).max().shift(1)

        # 볼린저 밴드 (20일, 2σ)
        std20 = c.rolling(20, min_periods=20).std()
        df["bb_upper"] = df["ma20"] + 2.0 * std20
        df["bb_lower"] = df["ma20"] - 2.0 * std20
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["ma20"]
        df["bb_width_q25"] = df["bb_width"].rolling(60, min_periods=60).quantile(0.25)
        df["is_squeeze"] = (df["bb_width"] <= df["bb_width_q25"]).astype(int)
        # 직전 5일 내 스퀴즈 존재 여부 (t-1 ~ t-5, 신호일 t 자체의 급등으로 인한 밴드 확장과 분리)
        df["recent_squeeze"] = df["is_squeeze"].shift(1).rolling(5, min_periods=1).max()

        df["ret_1d"] = c.pct_change()
        df["body_ret"] = (c - o) / o
        df["prev_close"] = c.shift(1)
        df["prev_range"] = (h - l).shift(1)
        df["prev_tval"] = df["tval"].shift(1)
        df["prev_close_gt_ma20"] = (c.shift(1) > df["ma20"].shift(1))
        df["prev_close_gt_ma60"] = (c.shift(1) > df["ma60"].shift(1))

        # 상한가 고착/거래정지 판별 (당일 가격 변동 없음)
        df["is_frozen"] = (h == l) & (o == c)

        data_by_code[code] = df
    return data_by_code


# ==============================================================================
# 4. 포트폴리오 백테스트 엔진 (전략 A, B, C)
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
    prev_highest_high: float  # 전일까지 확정된 최고가 (Zero Look-Ahead)
    holding_days: int = 0
    pending_exit_reason: Optional[str] = None


def run_strategy_ab(
    strategy_name: str,
    data_by_code: Dict[str, pd.DataFrame],
    trading_dates: List[pd.Timestamp],
    apply_costs: bool = True,
    stop_loss_pct: float = 0.08,
    trailing_stop_pct: float = 0.12,
    max_hold_days: int = 30,
    exit_ma_col: str = "ma10",
) -> Tuple[pd.DataFrame, List[TradeRecord], Dict]:
    """
    전략 A (VCP/Minervini 거래대금 폭발+신고가 돌파) 및
    전략 B (볼린저 스퀴즈 압축 후 거래량 급등 돌파) 공통 이벤트 드리븐 엔진.
    - t일 종가 확정 신호 -> t+1일 시가 진입 (Zero Look-Ahead)
    - 보유 중 장중 손절/트레일링 스탑은 전일 확정 최고가 기준
    - MA 이탈 및 최대 보유기간 도달은 t일 종가 확정 -> t+1일 시가 청산
    """
    buy_c = BUY_COST if apply_costs else 0.0
    sell_c = SELL_COST if apply_costs else 0.0

    cash = INITIAL_CAPITAL
    positions: Dict[str, Position] = {}
    pending_entries: List[Tuple[str, float]] = []  # (code, score)
    trades: List[TradeRecord] = []
    daily_logs = []

    max_observed_total_weight = 0.0
    max_observed_single_entry_weight = 0.0
    min_observed_cash = INITIAL_CAPITAL
    min_observed_equity = INITIAL_CAPITAL

    for i, dt in enumerate(trading_dates):
        # ------------------------------------------------------------------
        # (1) 장 시작(시가) 체결: 전일 확정된 청산 주문 실행
        # ------------------------------------------------------------------
        to_close = [code for code, pos in positions.items() if pos.pending_exit_reason is not None]
        for code in to_close:
            pos = positions[code]
            df = data_by_code[code]
            if dt not in df.index:
                continue
            row = df.loc[dt]
            if row["is_frozen"]:
                continue  # 거래정지/점하한가 시 청산 불가, 다음날 재시도

            exit_px = float(row["Open"])
            proceeds = pos.shares * exit_px * (1.0 - sell_c)
            cost_basis = pos.shares * pos.entry_price * (1.0 + buy_c)
            pnl = proceeds - cost_basis
            gross_ret = (exit_px / pos.entry_price) - 1.0
            net_ret = (proceeds / cost_basis) - 1.0
            cash += proceeds

            trades.append(TradeRecord(
                code=code,
                entry_date=pos.entry_date.strftime("%Y-%m-%d"),
                exit_date=dt.strftime("%Y-%m-%d"),
                entry_price=pos.entry_price,
                exit_price=exit_px,
                shares=pos.shares,
                weight_at_entry=pos.weight_at_entry,
                gross_ret=gross_ret,
                net_ret=net_ret,
                pnl=pnl,
                exit_reason=pos.pending_exit_reason,
                holding_days=pos.holding_days,
            ))
            del positions[code]

        # ------------------------------------------------------------------
        # (2) 장 시작(시가) 체결: 전일 확정된 매수 신호 실행
        # ------------------------------------------------------------------
        if pending_entries and len(positions) < MAX_POSITIONS:
            # 현재 평가자산 계산 (시가 기준)
            mtm_open = cash
            for code, pos in positions.items():
                df = data_by_code[code]
                px = float(df.loc[dt, "Open"]) if dt in df.index else pos.entry_price
                mtm_open += pos.shares * px

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
                # 과도한 시가 갭상승(+15% 초과) 추격 매수 방지
                if prev_close > 0 and (open_px / prev_close - 1.0) > MAX_ENTRY_GAP:
                    continue

                alloc = min(target_alloc, cash)
                if alloc < open_px * (1.0 + buy_c):
                    continue
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
                    prev_highest_high=open_px,  # 진입일 장중에는 진입가 기준
                    holding_days=0,
                )
        pending_entries = []

        # ------------------------------------------------------------------
        # (3) 장중 리스크 관리: 하드 손절 & 트레일링 스탑 (전일 확정 최고가 기준)
        # ------------------------------------------------------------------
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

            hard_stop_px = pos.entry_price * (1.0 - stop_loss_pct)
            trail_stop_px = pos.prev_highest_high * (1.0 - trailing_stop_pct)
            effective_stop_px = max(hard_stop_px, trail_stop_px)
            reason = "hard_stop" if hard_stop_px >= trail_stop_px else "trailing_stop"

            if low_px <= effective_stop_px:
                # 시가가 이미 스탑가 아래로 갭하락했다면 시가로 체결 (현실적 갭 리스크 반영)
                exit_px = min(open_px, effective_stop_px)
                proceeds = pos.shares * exit_px * (1.0 - sell_c)
                cost_basis = pos.shares * pos.entry_price * (1.0 + buy_c)
                pnl = proceeds - cost_basis
                gross_ret = (exit_px / pos.entry_price) - 1.0
                net_ret = (proceeds / cost_basis) - 1.0
                cash += proceeds

                trades.append(TradeRecord(
                    code=code,
                    entry_date=pos.entry_date.strftime("%Y-%m-%d"),
                    exit_date=dt.strftime("%Y-%m-%d"),
                    entry_price=pos.entry_price,
                    exit_price=exit_px,
                    shares=pos.shares,
                    weight_at_entry=pos.weight_at_entry,
                    gross_ret=gross_ret,
                    net_ret=net_ret,
                    pnl=pnl,
                    exit_reason=reason,
                    holding_days=pos.holding_days + 1,
                ))
                intraday_closed.append(code)
            else:
                # 장 마감 후 최고가 업데이트 (익일 트레일링 스탑에 사용)
                pos.prev_highest_high = max(pos.prev_highest_high, high_px)
                pos.holding_days += 1

        for code in intraday_closed:
            del positions[code]

        # ------------------------------------------------------------------
        # (4) 장 마감(종가) 평가 및 익일 청산/매수 신호 생성
        # ------------------------------------------------------------------
        stock_val = 0.0
        for code, pos in positions.items():
            df = data_by_code[code]
            if dt in df.index:
                row = df.loc[dt]
                close_px = float(row["Close"])
                stock_val += pos.shares * close_px

                # 추세 이탈 또는 최대 보유기간 체크 -> 익일 시가 청산 예약
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

        # 익일 매수 후보 스캔 (t일 종가 기준 확정 지표만 사용)
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

            if strategy_name == "Strategy_A_VCP_Breakout":
                # 정배열 + 거래대금 2.5배 폭발 + 20일 신고가 종가 돌파 + 건강한 양봉(+3%~+20%)
                if (
                    pd.notna(row["ma60"])
                    and pd.notna(row["avg_tval20_prev"])
                    and pd.notna(row["high20_prev"])
                    and row["avg_tval20_prev"] > 0
                ):
                    tval_ratio = tval / float(row["avg_tval20_prev"])
                    ret1d = float(row["ret_1d"]) if pd.notna(row["ret_1d"]) else 0.0
                    if (
                        c > float(row["ma20"]) > float(row["ma60"])
                        and float(row["ma20_slope5"]) > 0
                        and tval_ratio >= 2.5
                        and c >= float(row["high20_prev"])
                        and c > o
                        and 0.03 <= ret1d <= 0.20
                    ):
                        candidates.append((code, tval_ratio))

            elif strategy_name == "Strategy_B_BB_Squeeze":
                # 직전 5일 내 BB 폭 60일 하위 25% 압축 + BB 상단 돌파 + 거래량 2배 급증 + 3% 이상 강양봉 + 중기 상승
                if (
                    pd.notna(row["bb_upper"])
                    and pd.notna(row["ma60"])
                    and pd.notna(row["avg_vol20_prev"])
                    and pd.notna(row["recent_squeeze"])
                    and row["avg_vol20_prev"] > 0
                ):
                    vol_ratio = float(row["Volume"]) / float(row["avg_vol20_prev"])
                    body_ret = float(row["body_ret"])
                    ret1d = float(row["ret_1d"]) if pd.notna(row["ret_1d"]) else 0.0
                    if (
                        float(row["recent_squeeze"]) >= 1.0
                        and c > float(row["bb_upper"])
                        and c > float(row["ma60"])
                        and vol_ratio >= 2.0
                        and body_ret >= 0.03
                        and ret1d <= 0.22
                    ):
                        candidates.append((code, vol_ratio))

        # 거래대금/거래량 폭발 강도 상위 순 정렬
        candidates.sort(key=lambda x: x[1], reverse=True)
        pending_entries = candidates[: MAX_POSITIONS * 2]

    equity_df = pd.DataFrame(daily_logs)
    audit = {
        "max_observed_total_weight": float(max_observed_total_weight),
        "max_observed_single_entry_weight": float(max_observed_single_entry_weight),
        "min_observed_cash": float(min_observed_cash),
        "min_observed_equity": float(min_observed_equity),
        "stop_loss_pct": float(stop_loss_pct),
        "trailing_stop_pct": float(trailing_stop_pct),
        "round_trip_cost": float(ROUND_TRIP_COST if apply_costs else 0.0),
    }
    return equity_df, trades, audit


def run_strategy_c_larry(
    data_by_code: Dict[str, pd.DataFrame],
    trading_dates: List[pd.Timestamp],
    apply_costs: bool = True,
    k: float = 0.5,
    stop_loss_pct: float = 0.07,
) -> Tuple[pd.DataFrame, List[TradeRecord], Dict]:
    """
    전략 C (래리 윌리엄스 KRX 변동성 돌파 변형)
    - t-1일 마감 기준: 거래대금 >= 300억 + 정배열 추세 필터(Close > MA20 & Close > MA60) 통과 종목 중 거래대금 상위 5종목 선정
    - t일 장중: target = Open[t] + k * (High[t-1] - Low[t-1]) 돌파 시 진입 (최대 5종목, 각 20%)
    - t일 장중 -7% 하드 손절 체크
    - 미손절 포지션은 익일(t+1) 시가 전량 청산
    """
    buy_c = BUY_COST if apply_costs else 0.0
    sell_c = SELL_COST if apply_costs else 0.0

    cash = INITIAL_CAPITAL
    positions: Dict[str, Position] = {}
    trades: List[TradeRecord] = []
    daily_logs = []

    max_observed_total_weight = 0.0
    max_observed_single_entry_weight = 0.0
    min_observed_cash = INITIAL_CAPITAL
    min_observed_equity = INITIAL_CAPITAL

    for i, dt in enumerate(trading_dates):
        # (1) 전일 진입 포지션: 당일(t) 시가 전량 청산
        for code in list(positions.keys()):
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
            gross_ret = (exit_px / pos.entry_price) - 1.0
            net_ret = (proceeds / cost_basis) - 1.0
            cash += proceeds
            trades.append(TradeRecord(
                code=code,
                entry_date=pos.entry_date.strftime("%Y-%m-%d"),
                exit_date=dt.strftime("%Y-%m-%d"),
                entry_price=pos.entry_price,
                exit_price=exit_px,
                shares=pos.shares,
                weight_at_entry=pos.weight_at_entry,
                gross_ret=gross_ret,
                net_ret=net_ret,
                pnl=pnl,
                exit_reason="next_open_exit",
                holding_days=1,
            ))
            del positions[code]

        # (2) t-1일 기준 거래대금 상위 5종목 선정 (모두 t-1 확정 데이터 사용)
        ranked = []
        for code, df in data_by_code.items():
            if code in positions or dt not in df.index:
                continue
            row = df.loc[dt]
            if row["is_frozen"]:
                continue
            prev_tval = row["prev_tval"]
            prev_range = row["prev_range"]
            prev_close = row["prev_close"]
            if (
                pd.notna(prev_tval)
                and pd.notna(prev_range)
                and pd.notna(prev_close)
                and prev_tval >= 30_000_000_000.0
                and prev_close >= MIN_PRICE
                and prev_range > 0
                and (prev_range / prev_close) >= 0.02
                and bool(row["prev_close_gt_ma20"])
                and bool(row["prev_close_gt_ma60"])
            ):
                ranked.append((code, float(prev_tval)))

        ranked.sort(key=lambda x: x[1], reverse=True)
        watchlist = [c for c, _ in ranked[:MAX_POSITIONS]]

        # (3) 장중 변동성 돌파 진입 및 당일 손절 체크
        mtm_open = cash + sum(
            pos.shares * float(data_by_code[c].loc[dt, "Open"])
            if dt in data_by_code[c].index else pos.shares * pos.entry_price
            for c, pos in positions.items()
        )
        target_alloc = mtm_open * MAX_SINGLE_WEIGHT

        for code in watchlist:
            if len(positions) >= MAX_POSITIONS:
                break
            df = data_by_code[code]
            row = df.loc[dt]
            open_px = float(row["Open"])
            high_px = float(row["High"])
            low_px = float(row["Low"])
            prev_range = float(row["prev_range"])
            target_px = open_px + k * prev_range

            if high_px >= target_px and open_px < target_px:
                alloc = min(target_alloc, cash)
                shares = int(alloc // (target_px * (1.0 + buy_c)))
                if shares <= 0:
                    continue
                total_cost = shares * target_px * (1.0 + buy_c)
                if total_cost > cash:
                    continue

                cash -= total_cost
                entry_w = total_cost / mtm_open if mtm_open > 0 else 0.0
                max_observed_single_entry_weight = max(max_observed_single_entry_weight, entry_w)

                # 당일 장중 급락 손절(-7%) 체크 (보수적 가정: 돌파 후 저가가 손절가 이하면 당일 손절)
                hard_stop_px = target_px * (1.0 - stop_loss_pct)
                if low_px <= hard_stop_px:
                    proceeds = shares * hard_stop_px * (1.0 - sell_c)
                    pnl = proceeds - total_cost
                    cash += proceeds
                    trades.append(TradeRecord(
                        code=code,
                        entry_date=dt.strftime("%Y-%m-%d"),
                        exit_date=dt.strftime("%Y-%m-%d"),
                        entry_price=target_px,
                        exit_price=hard_stop_px,
                        shares=shares,
                        weight_at_entry=entry_w,
                        gross_ret=(hard_stop_px / target_px) - 1.0,
                        net_ret=(proceeds / total_cost) - 1.0,
                        pnl=pnl,
                        exit_reason="intraday_hard_stop",
                        holding_days=0,
                    ))
                else:
                    positions[code] = Position(
                        code=code,
                        entry_date=dt,
                        entry_price=target_px,
                        shares=shares,
                        weight_at_entry=entry_w,
                        prev_highest_high=high_px,
                        holding_days=1,
                    )

        # (4) 종가 기준 평가
        stock_val = 0.0
        for code, pos in positions.items():
            df = data_by_code[code]
            px = float(df.loc[dt, "Close"]) if dt in df.index else pos.entry_price
            stock_val += pos.shares * px

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

    equity_df = pd.DataFrame(daily_logs)
    audit = {
        "max_observed_total_weight": float(max_observed_total_weight),
        "max_observed_single_entry_weight": float(max_observed_single_entry_weight),
        "min_observed_cash": float(min_observed_cash),
        "min_observed_equity": float(min_observed_equity),
        "stop_loss_pct": float(stop_loss_pct),
        "trailing_stop_pct": 0.0,
        "round_trip_cost": float(ROUND_TRIP_COST if apply_costs else 0.0),
    }
    return equity_df, trades, audit


# ==============================================================================
# 5. 성과 지표 산출 (Full / IS / OOS)
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

    peak = np.maximum.accumulate(eq)
    dd = (eq / peak) - 1.0
    mdd = float(dd.min())

    # 해당 구간 진입 거래 필터
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

    avg_exposure = float(sub["invested_weight"].mean())

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
        "avg_exposure": float(avg_exposure),
        "final_equity": float(eq[-1]),
    }


def calc_benchmark_metrics(bench_df: pd.DataFrame, code: str, start_date: str, end_date: str) -> Dict:
    sub = bench_df[(bench_df["code"] == code) & (bench_df["date"] >= start_date) & (bench_df["date"] <= end_date)].copy()
    if len(sub) < 2:
        return {}
    c = sub["Close"].values
    total_ret = (c[-1] / c[0]) - 1.0
    years = len(sub) / 252.0
    cagr = (c[-1] / c[0]) ** (1.0 / years) - 1.0
    dr = pd.Series(c).pct_change().dropna()
    sharpe = (dr.mean() / dr.std()) * np.sqrt(252) if dr.std() > 0 else 0.0
    mdd = float((c / np.maximum.accumulate(c) - 1.0).min())
    return {
        "total_return": float(total_ret),
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "mdd": float(mdd),
    }


# ==============================================================================
# 6. Zero Look-Ahead 교란 검증 (미래 데이터 +500% 변조 시 과거 IS 자산 차이 = 0)
# ==============================================================================
def verify_zero_lookahead(
    ohlcv: pd.DataFrame,
    trading_dates: List[pd.Timestamp],
    baseline_is_equities: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """OOS 구간(2024-07-01 이후) 주가를 +500% 폭등 교란했을 때, IS 구간 일별 자산 곡선의 최대 절대 오차 검증"""
    perturbed = ohlcv.copy()
    mask = perturbed["date"] >= OOS_START
    for col in ["Open", "High", "Low", "Close"]:
        perturbed.loc[mask, col] = perturbed.loc[mask, col] * 6.0

    p_data = compute_indicators(perturbed)
    diffs = {}

    for strat_key, runner in [
        ("Strategy_A_VCP_Breakout", lambda: run_strategy_ab("Strategy_A_VCP_Breakout", p_data, trading_dates, True, 0.08, 0.12, 30, "ma10")),
        ("Strategy_B_BB_Squeeze", lambda: run_strategy_ab("Strategy_B_BB_Squeeze", p_data, trading_dates, True, 0.07, 0.10, 20, "ma20")),
        ("Strategy_C_Larry_VolBreakout", lambda: run_strategy_c_larry(p_data, trading_dates, True, 0.5, 0.07)),
    ]:
        eq_df, _, _ = runner()
        p_is_eq = eq_df[eq_df["date"] <= IS_END]["equity"].values
        base_is_eq = baseline_is_equities[strat_key]
        max_diff = float(np.max(np.abs(p_is_eq - base_is_eq)))
        diffs[strat_key] = max_diff
    return diffs


# ==============================================================================
# 7. 메인 실행
# ==============================================================================
def main():
    ohlcv, meta, bench = sync_data_to_duckdb(force_refresh=False)
    print(f"[Indicators] Computing causal indicators for {ohlcv['code'].nunique()} tickers...")
    data_by_code = compute_indicators(ohlcv)

    # 전체 거래일 캘린더 (벤치마크 KS11 기준, 없으면 전체 날짜 합집합)
    if not bench.empty and "KS11" in bench["code"].values:
        all_dates = pd.to_datetime(bench[bench["code"] == "KS11"]["date"].unique())
    else:
        all_dates = pd.to_datetime(ohlcv["date"].unique())
    trading_dates = sorted([d for d in all_dates if pd.Timestamp(BT_START) <= d <= pd.Timestamp(DATA_END)])
    print(f"[Backtest] Period: {trading_dates[0].strftime('%Y-%m-%d')} ~ {trading_dates[-1].strftime('%Y-%m-%d')} ({len(trading_dates)} trading days)")

    strategies_spec = [
        ("Strategy_A_VCP_Breakout", lambda cost: run_strategy_ab("Strategy_A_VCP_Breakout", data_by_code, trading_dates, cost, 0.08, 0.12, 30, "ma10")),
        ("Strategy_B_BB_Squeeze", lambda cost: run_strategy_ab("Strategy_B_BB_Squeeze", data_by_code, trading_dates, cost, 0.07, 0.10, 20, "ma20")),
        ("Strategy_C_Larry_VolBreakout", lambda cost: run_strategy_c_larry(data_by_code, trading_dates, cost, 0.5, 0.07)),
    ]

    results = {
        "meta": {
            "universe_size": int(ohlcv["code"].nunique()),
            "kospi_count": int(meta[meta["code"].isin(ohlcv["code"].unique())]["market"].eq("KOSPI").sum()),
            "kosdaq_count": int(meta[meta["code"].isin(ohlcv["code"].unique())]["market"].eq("KOSDAQ").sum()),
            "bt_start": trading_dates[0].strftime("%Y-%m-%d"),
            "is_end": IS_END,
            "oos_start": OOS_START,
            "bt_end": trading_dates[-1].strftime("%Y-%m-%d"),
            "trading_days": len(trading_dates),
            "initial_capital": INITIAL_CAPITAL,
            "max_positions": MAX_POSITIONS,
            "max_single_weight": MAX_SINGLE_WEIGHT,
            "buy_cost": BUY_COST,
            "sell_cost": SELL_COST,
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
        "strategies": {},
        "lookahead_perturbation_max_diff": {},
    }

    baseline_is_equities = {}

    for name, runner in strategies_spec:
        print(f"\n--- Running {name} ---")
        eq_net, trades_net, audit_net = runner(True)
        eq_gross, trades_gross, audit_gross = runner(False)

        baseline_is_equities[name] = eq_net[eq_net["date"] <= IS_END]["equity"].values

        # 일별 로그 CSV 저장
        eq_net.to_csv(BASE_DIR / f"equity_{name}_net.csv", index=False)

        # 청산 사유 분포
        reason_counts = {}
        for t in trades_net:
            reason_counts[t.exit_reason] = reason_counts.get(t.exit_reason, 0) + 1

        res_obj = {
            "net": {
                "full": calc_metrics(eq_net, trades_net, BT_START, DATA_END),
                "is": calc_metrics(eq_net, trades_net, BT_START, IS_END),
                "oos": calc_metrics(eq_net, trades_net, OOS_START, DATA_END),
                "audit": audit_net,
                "exit_reasons": reason_counts,
            },
            "gross": {
                "full": calc_metrics(eq_gross, trades_gross, BT_START, DATA_END),
                "is": calc_metrics(eq_gross, trades_gross, BT_START, IS_END),
                "oos": calc_metrics(eq_gross, trades_gross, OOS_START, DATA_END),
            },
        }
        results["strategies"][name] = res_obj

        nf = res_obj["net"]["full"]
        ni = res_obj["net"]["is"]
        no = res_obj["net"]["oos"]
        gf = res_obj["gross"]["full"]
        print(
            f"  [NET FULL]   CAGR={nf['cagr']*100:+.2f}%, Tot={nf['total_return']*100:+.2f}%, "
            f"Sharpe={nf['sharpe']:.2f}, MDD={nf['mdd']*100:.2f}%, Trades={nf['num_trades']}, Win={nf['win_rate']*100:.1f}%"
        )
        print(
            f"  [GROSS FULL] CAGR={gf['cagr']*100:+.2f}%, Tot={gf['total_return']*100:+.2f}%, "
            f"Sharpe={gf['sharpe']:.2f}, MDD={gf['mdd']*100:.2f}%"
        )
        print(
            f"  [NET IS]     CAGR={ni['cagr']*100:+.2f}%, Sharpe={ni['sharpe']:.2f}, MDD={ni['mdd']*100:.2f}%, Trades={ni['num_trades']}"
        )
        print(
            f"  [NET OOS]    CAGR={no['cagr']*100:+.2f}%, Sharpe={no['sharpe']:.2f}, MDD={no['mdd']*100:.2f}%, Trades={no['num_trades']}"
        )

    print("\n[Look-Ahead Audit] Perturbing OOS prices by +500% and verifying IS equity invariance...")
    la_diffs = verify_zero_lookahead(ohlcv, trading_dates, baseline_is_equities)
    results["lookahead_perturbation_max_diff"] = la_diffs
    for k, v in la_diffs.items():
        print(f"  {k}: max IS diff = {v:.10f}")

    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[Done] Results saved to {RESULTS_JSON_PATH}")


if __name__ == "__main__":
    main()
