#!/usr/bin/env python3
"""
Invest_God Phase 2 — 1억 원 통합 가상투자(Paper Trading) 1개월 시범운행 원장 엔진
(live/paper_portfolio_engine.py)

초기 원금 100,000,000원 (마스터 v2 85:15):
  ① 단기 코인 Turbo-1  : 28,330,000원 (live/state/upbit_paper_state.json 실시간 동기화)
  ② 중기 KRX D-3       : 28,330,000원 (D-3 돌파/손절/청산 가상 체결 + 현금 대기 시 연 2.5% RP 이자 일할)
  ③ 장기 코어 L-3      : 28,330,000원 (나스닥100/코스피200/BTC/골드/단기채 실제 시세 일별 시가평가)
  ④ 하이닉스 2배 로켓  : 15,010,000원 (MA20>MA120 시 하이닉스 2배 일간수익률, SAFE_RP 시 연 2.5% RP 일할)

영속 상태 파일 3종:
  - live/state/paper_portfolio_state.json
  - live/state/paper_equity_history.csv
  - live/state/paper_trades_log.csv

※ Idempotent 설계: 같은 날(KST) 여러 번 실행해도 이자·거래가 중복 누적되지 않고 최신 시세 시가평가만 갱신합니다.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
STATE_DIR = BASE_DIR / "state"
OUTPUT_DIR = BASE_DIR / "output"
LOG_DIR = BASE_DIR / "logs"

SIGNALS_PATH = OUTPUT_DIR / "latest_signals.json"
UPBIT_STATE_PATH = STATE_DIR / "upbit_paper_state.json"
UPBIT_STATUS_PATH = OUTPUT_DIR / "latest_upbit_status.json"

PAPER_STATE_PATH = STATE_DIR / "paper_portfolio_state.json"
EQUITY_CSV_PATH = STATE_DIR / "paper_equity_history.csv"
TRADES_CSV_PATH = STATE_DIR / "paper_trades_log.csv"

KST = dt.timezone(dt.timedelta(hours=9))

INITIAL_ALLOC = {
    "crypto_short_turbo1": 28_330_000.0,
    "krx_medium_d3": 28_330_000.0,
    "longterm_core_l3": 28_330_000.0,
    "skh_rocket_2x": 15_010_000.0,
}
INITIAL_TOTAL_KRW = 100_000_000.0
RP_ANNUAL_RATE = 0.025
ETF_ONEWAY_FEE = 0.00075
BTC_ONEWAY_FEE = 0.0018
KRX_BUY_FEE = 0.00265
KRX_SELL_FEE = 0.00465
ROCKET_SWITCH_FEE = 0.0015
ROCKET_ANNUAL_DRAG = 0.02

L3_ASSET_META = {
    "NASDAQ100": {"ticker": "133690.KS", "name": "TIGER 미국나스닥100", "fee": ETF_ONEWAY_FEE, "source": "yfinance"},
    "KOSPI200": {"ticker": "069500.KS", "name": "KODEX 200", "fee": ETF_ONEWAY_FEE, "source": "yfinance"},
    "BTC_KRW": {"ticker": "KRW-BTC", "name": "비트코인 (KRW-BTC)", "fee": BTC_ONEWAY_FEE, "source": "upbit"},
    "GOLD": {"ticker": "132030.KS", "name": "KODEX 골드선물(H)", "fee": ETF_ONEWAY_FEE, "source": "yfinance"},
    "SHORT_BOND": {"ticker": "153130.KS", "name": "KODEX 단기채권", "fee": ETF_ONEWAY_FEE, "source": "yfinance"},
}


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("paper_portfolio_engine")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh = logging.FileHandler(LOG_DIR / "paper_portfolio.log", encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


def fetch_yf_latest_price(ticker: str) -> Optional[float]:
    try:
        df = yf.download(ticker, period="7d", interval="1d", progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        closes = df["Close"].dropna()
        if closes.empty:
            return None
        return float(closes.iloc[-1])
    except Exception:
        return None


def fetch_upbit_btc_price() -> Optional[float]:
    try:
        resp = requests.get("https://api.upbit.com/v1/ticker", params={"markets": "KRW-BTC"}, timeout=10)
        resp.raise_for_status()
        return float(resp.json()[0]["trade_price"])
    except Exception:
        return None


def get_l3_live_prices(signals: Dict[str, Any]) -> Dict[str, float]:
    """L-3 5개 자산 최신 종가 수집 (signals JSON 우선 활용 + 골드/단기채 yfinance 조회)."""
    prices: Dict[str, float] = {}
    l3_sig = signals.get("buckets", {}).get("longterm_core_l3", {})
    for a in l3_sig.get("offensive_assets", []):
        if a.get("status") == "OK" and a.get("close"):
            prices[a["id"]] = float(a["close"])

    for aid, meta in L3_ASSET_META.items():
        if aid in prices:
            continue
        if meta["source"] == "yfinance":
            px = fetch_yf_latest_price(meta["ticker"])
            if px:
                prices[aid] = px
        elif meta["source"] == "upbit":
            px = fetch_upbit_btc_price()
            if px:
                prices[aid] = px
    return prices


def append_trades_csv(trades: List[Dict[str, Any]]) -> None:
    if not trades:
        return
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["timestamp_kst", "date", "bucket", "action", "ticker", "name", "qty", "price_krw", "amount_krw", "fee_krw", "reason"]
    file_exists = TRADES_CSV_PATH.exists() and TRADES_CSV_PATH.stat().st_size > 0
    with TRADES_CSV_PATH.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if not file_exists:
            w.writeheader()
        for t in trades:
            w.writerow({k: t.get(k, "") for k in cols})


def upsert_equity_history_csv(row: Dict[str, Any]) -> None:
    """같은 날짜(date)가 이미 있으면 덮어쓰고(Idempotent), 없으면 추가."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    cols = [
        "date",
        "day_num",
        "crypto_krw",
        "krx_d3_krw",
        "longterm_l3_krw",
        "rocket_krw",
        "total_equity_krw",
        "cum_return_pct",
        "hwm_krw",
        "drawdown_pct",
    ]
    rows: List[Dict[str, Any]] = []
    if EQUITY_CSV_PATH.exists() and EQUITY_CSV_PATH.stat().st_size > 0:
        with EQUITY_CSV_PATH.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get("date") != row["date"]:
                    rows.append(r)
    rows.append({k: row[k] for k in cols})
    rows.sort(key=lambda x: x["date"])

    with EQUITY_CSV_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def init_state(today_str: str) -> Dict[str, Any]:
    return {
        "mode": "DRY_RUN_PAPER_100M",
        "dry_run": True,
        "trial_days": 30,
        "start_date": today_str,
        "last_updated_date": today_str,
        "day_num": 1,
        "initial_capital_krw": INITIAL_TOTAL_KRW,
        "hwm_krw": INITIAL_TOTAL_KRW,
        "total_equity_krw": INITIAL_TOTAL_KRW,
        "buckets": {
            "crypto_short_turbo1": {
                "name": "① 단기 코인 (Turbo-1)",
                "initial_krw": INITIAL_ALLOC["crypto_short_turbo1"],
                "cash_krw": INITIAL_ALLOC["crypto_short_turbo1"],
                "equity_krw": INITIAL_ALLOC["crypto_short_turbo1"],
                "holdings_summary": "업비트 KRW 현금 100% 대기",
                "positions": {},
            },
            "krx_medium_d3": {
                "name": "② 중기 KRX (D-3)",
                "initial_krw": INITIAL_ALLOC["krx_medium_d3"],
                "cash_krw": INITIAL_ALLOC["krx_medium_d3"],
                "equity_krw": INITIAL_ALLOC["krx_medium_d3"],
                "last_interest_date": today_str,
                "last_action_date": None,
                "holdings_summary": "현금/RP(연 2.5%) 100% 대기",
                "positions": {},
            },
            "longterm_core_l3": {
                "name": "③ 장기 코어 (L-3)",
                "initial_krw": INITIAL_ALLOC["longterm_core_l3"],
                "cash_krw": INITIAL_ALLOC["longterm_core_l3"],
                "equity_krw": INITIAL_ALLOC["longterm_core_l3"],
                "initialized": False,
                "last_rebal_month": None,
                "holdings_summary": "미배분",
                "positions": {},
            },
            "skh_rocket_2x": {
                "name": "④ 하이닉스 2배 로켓",
                "initial_krw": INITIAL_ALLOC["skh_rocket_2x"],
                "cash_krw": INITIAL_ALLOC["skh_rocket_2x"],
                "equity_krw": INITIAL_ALLOC["skh_rocket_2x"],
                "mode": "SAFE_RP",
                "last_skh_close": None,
                "last_eval_date": today_str,
                "holdings_summary": "단기채/RP(연 2.5%) 100% 대피",
            },
        },
    }


def main() -> int:
    logger = setup_logging()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    signals = json.loads(SIGNALS_PATH.read_text(encoding="utf-8")) if SIGNALS_PATH.exists() else {}
    upbit_state = json.loads(UPBIT_STATE_PATH.read_text(encoding="utf-8")) if UPBIT_STATE_PATH.exists() else {}

    now_kst = dt.datetime.now(KST)
    today_str = now_kst.strftime("%Y-%m-%d")
    ts_kst = now_kst.strftime("%Y-%m-%d %H:%M:%S KST")

    if PAPER_STATE_PATH.exists():
        state = json.loads(PAPER_STATE_PATH.read_text(encoding="utf-8"))
    else:
        state = init_state(today_str)

    # 안전 잠금 보장
    dry_run = bool(cfg.get("dry_run", True)) or (os.environ.get("DRY_RUN", "true").lower() == "true")
    state["dry_run"] = dry_run

    start_dt = dt.date.fromisoformat(state["start_date"])
    today_dt = dt.date.fromisoformat(today_str)
    day_num = max(1, (today_dt - start_dt).days + 1)
    state["day_num"] = day_num

    new_trades: List[Dict[str, Any]] = []

    # =========================================================================
    # ① 단기 코인 (Turbo-1) — upbit_paper_state.json 실평가액 동기화
    # =========================================================================
    b_crypto = state["buckets"]["crypto_short_turbo1"]
    if upbit_state:
        b_crypto["cash_krw"] = round(float(upbit_state.get("cash_krw", b_crypto["initial_krw"])), 2)
        b_crypto["equity_krw"] = round(float(upbit_state.get("equity_krw", b_crypto["initial_krw"])), 2)
        u_pos = upbit_state.get("positions", {})
        b_crypto["positions"] = u_pos
        if u_pos:
            b_crypto["holdings_summary"] = "보유: " + ", ".join(u_pos.keys())
        else:
            b_crypto["holdings_summary"] = "업비트 KRW 현금 100% 방어 대기"

    # =========================================================================
    # ② 중기 KRX (D-3) — 일할 RP 이자(현금 대기분) + 돌파 후보 가상 진입/청산
    # =========================================================================
    b_d3 = state["buckets"]["krx_medium_d3"]
    last_int_dt = dt.date.fromisoformat(b_d3.get("last_interest_date", today_str))
    days_elapsed_d3 = max(0, (today_dt - last_int_dt).days)
    if days_elapsed_d3 > 0 and float(b_d3["cash_krw"]) > 0:
        interest = float(b_d3["cash_krw"]) * (RP_ANNUAL_RATE / 365.0) * days_elapsed_d3
        b_d3["cash_krw"] = round(float(b_d3["cash_krw"]) + interest, 2)
        b_d3["last_interest_date"] = today_str

    # 기존 D-3 보유 포지션 시가평가 및 손절/트레일링 체크
    d3_pos: Dict[str, Any] = b_d3.get("positions", {})
    d3_pos_val = 0.0
    for code in list(d3_pos.keys()):
        pos = d3_pos[code]
        px = fetch_yf_latest_price(pos["yf_ticker"]) or float(pos["last_price"])
        pos["last_price"] = px
        pos["high_price"] = max(float(pos["high_price"]), px)
        hard_stop = float(pos["entry_price"]) * 0.92
        trail_stop = float(pos["high_price"]) * 0.85
        if px <= hard_stop or px <= trail_stop:
            gross = float(pos["qty"]) * px
            fee = gross * KRX_SELL_FEE
            net = gross - fee
            b_d3["cash_krw"] = round(float(b_d3["cash_krw"]) + net, 2)
            reason = "D3_HARD_STOP_-8%" if px <= hard_stop else "D3_TRAILING_-15%"
            new_trades.append(
                {
                    "timestamp_kst": ts_kst,
                    "date": today_str,
                    "bucket": "krx_medium_d3",
                    "action": "PAPER_SELL",
                    "ticker": pos["yf_ticker"],
                    "name": pos["name"],
                    "qty": pos["qty"],
                    "price_krw": round(px, 0),
                    "amount_krw": round(net, 0),
                    "fee_krw": round(fee, 0),
                    "reason": reason,
                }
            )
            del d3_pos[code]
        else:
            d3_pos_val += float(pos["qty"]) * px

    # 신규 D-3 돌파 후보 가상 편입 (하루 1회, 최대 5종목, 종목당 20%)
    d3_sig = signals.get("buckets", {}).get("krx_medium_d3", {})
    if b_d3.get("last_action_date") != today_str:
        cands = d3_sig.get("breakout_candidates", [])
        total_d3_now = float(b_d3["cash_krw"]) + d3_pos_val
        for c in cands:
            if len(d3_pos) >= 5:
                break
            code = c["code"]
            if code in d3_pos:
                continue
            alloc = min(total_d3_now * 0.20, float(b_d3["cash_krw"]))
            px = float(c["close"])
            if alloc < px or px <= 0:
                continue
            qty = int(alloc // (px * (1.0 + KRX_BUY_FEE)))
            if qty <= 0:
                continue
            cost = qty * px
            fee = cost * KRX_BUY_FEE
            b_d3["cash_krw"] = round(float(b_d3["cash_krw"]) - (cost + fee), 2)
            yf_t = code + (".KS" if c["market"] == "KOSPI" else ".KQ")
            d3_pos[code] = {
                "name": c["name"],
                "yf_ticker": yf_t,
                "qty": qty,
                "entry_price": px,
                "last_price": px,
                "high_price": px,
                "entry_date": today_str,
            }
            d3_pos_val += qty * px
            new_trades.append(
                {
                    "timestamp_kst": ts_kst,
                    "date": today_str,
                    "bucket": "krx_medium_d3",
                    "action": "PAPER_BUY",
                    "ticker": yf_t,
                    "name": c["name"],
                    "qty": qty,
                    "price_krw": round(px, 0),
                    "amount_krw": round(cost, 0),
                    "fee_krw": round(fee, 0),
                    "reason": "D3_BREAKOUT_SIGNAL",
                }
            )
        b_d3["last_action_date"] = today_str

    b_d3["positions"] = d3_pos
    b_d3["equity_krw"] = round(float(b_d3["cash_krw"]) + d3_pos_val, 2)
    if d3_pos:
        b_d3["holdings_summary"] = "보유: " + ", ".join(f"{v['name']}({v['qty']}주)" for v in d3_pos.values())
    else:
        b_d3["holdings_summary"] = "현금/RP(연 2.5%) 100% 방어 대기"

    # =========================================================================
    # ③ 장기 코어 (L-3) — 실제 ETF/BTC 시세로 Day 1 즉시 편입 + 일별 시가평가
    # =========================================================================
    b_l3 = state["buckets"]["longterm_core_l3"]
    l3_sig = signals.get("buckets", {}).get("longterm_core_l3", {})
    target_weights_pct: Dict[str, float] = l3_sig.get("target_weights_pct", {})
    l3_prices = get_l3_live_prices(signals)
    cur_ym = today_str[:7]

    # Day 1 최초 편입 OR 월이 바뀐 첫 실행 시 L-3 목표 비중으로 리밸런싱
    need_l3_rebal = (not b_l3.get("initialized", False)) or (b_l3.get("last_rebal_month") != cur_ym and today_dt.day <= 5)

    if need_l3_rebal and target_weights_pct:
        # 1) 기존 포지션 전량 가상 청산 후 목표 비중으로 재배분 (깔끔한 월간 리밸런싱)
        for aid, pos in list(b_l3.get("positions", {}).items()):
            px = l3_prices.get(aid, float(pos["last_price"]))
            gross = float(pos["qty"]) * px
            fee = gross * L3_ASSET_META[aid]["fee"]
            b_l3["cash_krw"] = round(float(b_l3["cash_krw"]) + (gross - fee), 2)
            new_trades.append(
                {
                    "timestamp_kst": ts_kst,
                    "date": today_str,
                    "bucket": "longterm_core_l3",
                    "action": "PAPER_SELL",
                    "ticker": L3_ASSET_META[aid]["ticker"],
                    "name": L3_ASSET_META[aid]["name"],
                    "qty": round(float(pos["qty"]), 6),
                    "price_krw": round(px, 0),
                    "amount_krw": round(gross - fee, 0),
                    "fee_krw": round(fee, 0),
                    "reason": "L3_MONTHLY_REBAL_EXIT",
                }
            )
        b_l3["positions"] = {}

        # 2) 목표 비중에 따라 실제 시세로 매수
        avail_cash = float(b_l3["cash_krw"])
        for aid, w_pct in target_weights_pct.items():
            w = float(w_pct) / 100.0
            if w <= 0.0 or aid not in L3_ASSET_META:
                continue
            px = l3_prices.get(aid)
            if not px or px <= 0:
                continue
            meta = L3_ASSET_META[aid]
            alloc_budget = avail_cash * w
            fee_r = meta["fee"]
            if aid == "BTC_KRW":
                qty = (alloc_budget * (1.0 - fee_r)) / px
                cost = alloc_budget * (1.0 - fee_r)
                fee = alloc_budget * fee_r
                spend = alloc_budget
            else:
                qty = int(alloc_budget // (px * (1.0 + fee_r)))
                cost = qty * px
                fee = cost * fee_r
                spend = cost + fee
            if qty <= 0 or spend > float(b_l3["cash_krw"]) + 1e-6:
                continue
            b_l3["cash_krw"] = round(max(0.0, float(b_l3["cash_krw"]) - spend), 2)
            b_l3["positions"][aid] = {
                "ticker": meta["ticker"],
                "name": meta["name"],
                "qty": round(qty, 6) if aid == "BTC_KRW" else int(qty),
                "entry_price": px,
                "last_price": px,
                "weight_target_pct": w_pct,
            }
            new_trades.append(
                {
                    "timestamp_kst": ts_kst,
                    "date": today_str,
                    "bucket": "longterm_core_l3",
                    "action": "PAPER_BUY",
                    "ticker": meta["ticker"],
                    "name": meta["name"],
                    "qty": round(qty, 6) if aid == "BTC_KRW" else int(qty),
                    "price_krw": round(px, 0),
                    "amount_krw": round(cost, 0),
                    "fee_krw": round(fee, 0),
                    "reason": "L3_DAY1_INITIAL_ALLOC" if not b_l3.get("initialized") else "L3_MONTHLY_REBAL_BUY",
                }
            )
        b_l3["initialized"] = True
        b_l3["last_rebal_month"] = cur_ym

    # L-3 일별 시가평가
    l3_pos_val = 0.0
    l3_desc = []
    for aid, pos in b_l3.get("positions", {}).items():
        px = l3_prices.get(aid, float(pos["last_price"]))
        pos["last_price"] = px
        val = float(pos["qty"]) * px
        l3_pos_val += val
        unit = "BTC" if aid == "BTC_KRW" else "주"
        l3_desc.append(f"{pos['name']} {pos['qty']}{unit}")
    b_l3["equity_krw"] = round(float(b_l3["cash_krw"]) + l3_pos_val, 2)
    b_l3["holdings_summary"] = " + ".join(l3_desc) if l3_desc else "현금 100%"

    # =========================================================================
    # ④ 하이닉스 2배 로켓 — MA20 > MA120 시 2배 일간수익률, SAFE_RP 시 연 2.5% RP
    # =========================================================================
    b_rk = state["buckets"]["skh_rocket_2x"]
    rk_sig = signals.get("buckets", {}).get("skh_rocket_2x", {})
    target_mode = rk_sig.get("signal", "SAFE_RP")
    if target_mode not in ("ROCKET_ON", "SAFE_RP"):
        target_mode = "SAFE_RP"
    skh_close = float(rk_sig.get("close", 0) or 0)

    last_eval_dt = dt.date.fromisoformat(b_rk.get("last_eval_date", today_str))
    days_elapsed_rk = max(0, (today_dt - last_eval_dt).days)

    if days_elapsed_rk > 0:
        prev_mode = b_rk.get("mode", "SAFE_RP")
        if prev_mode == "SAFE_RP":
            interest = float(b_rk["equity_krw"]) * (RP_ANNUAL_RATE / 365.0) * days_elapsed_rk
            b_rk["equity_krw"] = round(float(b_rk["equity_krw"]) + interest, 2)
        elif prev_mode == "ROCKET_ON" and b_rk.get("last_skh_close") and skh_close > 0:
            daily_ret = skh_close / float(b_rk["last_skh_close"]) - 1.0
            lev_ret = 2.0 * daily_ret - (ROCKET_ANNUAL_DRAG / 252.0)
            b_rk["equity_krw"] = round(max(0.0, float(b_rk["equity_krw"]) * (1.0 + lev_ret)), 2)

        # 레짐 전환 시 편도 전환 비용 0.15% 반영
        if target_mode != prev_mode:
            fee = float(b_rk["equity_krw"]) * ROCKET_SWITCH_FEE
            b_rk["equity_krw"] = round(float(b_rk["equity_krw"]) - fee, 2)
            new_trades.append(
                {
                    "timestamp_kst": ts_kst,
                    "date": today_str,
                    "bucket": "skh_rocket_2x",
                    "action": "PAPER_SWITCH",
                    "ticker": "000660.KS",
                    "name": "SK하이닉스 2배 로켓",
                    "qty": 1,
                    "price_krw": round(skh_close, 0),
                    "amount_krw": round(b_rk["equity_krw"], 0),
                    "fee_krw": round(fee, 0),
                    "reason": f"ROCKET_SWITCH_{prev_mode}_TO_{target_mode}",
                }
            )
        b_rk["mode"] = target_mode
        b_rk["last_eval_date"] = today_str
    else:
        # Day 1 최초 모드 동기화 (전환비용 없음)
        b_rk["mode"] = target_mode

    if skh_close > 0:
        b_rk["last_skh_close"] = skh_close
    b_rk["cash_krw"] = b_rk["equity_krw"] if b_rk["mode"] == "SAFE_RP" else 0.0
    b_rk["holdings_summary"] = (
        f"SK하이닉스 2배 레버리지 보유 (기준가 {skh_close:,.0f}원)"
        if b_rk["mode"] == "ROCKET_ON"
        else "단기채/RP(연 2.5%) 100% 안전 대피"
    )

    # =========================================================================
    # 통합 회계 합산 및 영속 저장
    # =========================================================================
    total_equity = round(
        float(b_crypto["equity_krw"])
        + float(b_d3["equity_krw"])
        + float(b_l3["equity_krw"])
        + float(b_rk["equity_krw"]),
        2,
    )
    state["total_equity_krw"] = total_equity
    state["hwm_krw"] = round(max(float(state["hwm_krw"]), total_equity), 2)
    state["last_updated_date"] = today_str
    state["last_updated_kst"] = ts_kst

    cum_ret_pct = (total_equity / INITIAL_TOTAL_KRW - 1.0) * 100.0
    dd_pct = (total_equity / float(state["hwm_krw"]) - 1.0) * 100.0 if state["hwm_krw"] > 0 else 0.0
    state["cum_return_pct"] = round(cum_ret_pct, 4)
    state["cum_pnl_krw"] = round(total_equity - INITIAL_TOTAL_KRW, 0)
    state["drawdown_pct"] = round(dd_pct, 4)

    PAPER_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    append_trades_csv(new_trades)
    upsert_equity_history_csv(
        {
            "date": today_str,
            "day_num": day_num,
            "crypto_krw": round(b_crypto["equity_krw"], 0),
            "krx_d3_krw": round(b_d3["equity_krw"], 0),
            "longterm_l3_krw": round(b_l3["equity_krw"], 0),
            "rocket_krw": round(b_rk["equity_krw"], 0),
            "total_equity_krw": round(total_equity, 0),
            "cum_return_pct": round(cum_ret_pct, 4),
            "hwm_krw": round(state["hwm_krw"], 0),
            "drawdown_pct": round(dd_pct, 4),
        }
    )

    logger.info(f"=== Paper Portfolio Day {day_num}/{state['trial_days']} ({today_str}) ===")
    logger.info(f"  총 평가액: {total_equity:,.0f}원 (손익 {total_equity - INITIAL_TOTAL_KRW:+,.0f}원 / {cum_ret_pct:+.3f}% / MDD {dd_pct:.3f}%)")
    logger.info(f"  ① 단기코인: {b_crypto['equity_krw']:,.0f}원 | ② 중기KRX: {b_d3['equity_krw']:,.0f}원 | ③ 장기코어: {b_l3['equity_krw']:,.0f}원 | ④ 로켓: {b_rk['equity_krw']:,.0f}원")
    logger.info(f"  금일 신규 가상 체결: {len(new_trades)}건")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
