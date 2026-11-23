#!/usr/bin/env python3
"""
Invest_God Phase 2 — 업비트(Upbit) KRW 4시간봉 자동매매 엔진 (upbit_turbo1_bot.py)
전략: Turbo-1 (M3_R15) 현물 1x 터틀 급등 돌파
- 레버리지 0% (현물 1x), 1회 리스크 -1.5%, 단일 코인 상한 33%, 최대 4종목, HWM -20% 절반 감속
- 일일 손실 -5% 초과 또는 API 치명 오류 시 Kill-Switch 자동 발동 (신규 진입 전면 차단)
- DRY_RUN=true 기본 안전모드: 실제 업비트 주문 API 절대 미호출, 가상 장부(live/state/upbit_paper_state.json)만 갱신
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import sys
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jwt
import numpy as np
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
STATE_DIR = BASE_DIR / "state"
OUTPUT_DIR = BASE_DIR / "output"
LOG_DIR = BASE_DIR / "logs"

PAPER_STATE_PATH = STATE_DIR / "upbit_paper_state.json"
STATUS_OUT_PATH = OUTPUT_DIR / "latest_upbit_status.json"

KST = dt.timezone(dt.timedelta(hours=9))
UPBIT_API_BASE = "https://api.upbit.com/v1"


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("upbit_turbo1_bot")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh = logging.FileHandler(LOG_DIR / "upbit_bot.log", encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


class UpbitClient:
    """업비트 공식 REST API + JWT 서명 클라이언트 (DRY_RUN 안전 가드 내장)."""

    def __init__(self, dry_run: bool = True, access_key: str = "", secret_key: str = ""):
        self.dry_run = dry_run
        self.access_key = access_key
        self.secret_key = secret_key
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def _auth_headers(self, query_params: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        if not self.access_key or not self.secret_key:
            raise RuntimeError("UPBIT_ACCESS_KEY / UPBIT_SECRET_KEY 환경변수가 설정되지 않았습니다.")
        payload: Dict[str, Any] = {
            "access_key": self.access_key,
            "nonce": str(uuid.uuid4()),
        }
        if query_params:
            query_string = urllib.parse.urlencode(query_params).encode("utf-8")
            m = hashlib.sha512()
            m.update(query_string)
            payload["query_hash"] = m.hexdigest()
            payload["query_hash_alg"] = "SHA512"
        token = jwt.encode(payload, self.secret_key, algorithm="HS256")
        if isinstance(token, bytes):
            token = token.decode("utf-8")
        return {"Authorization": f"Bearer {token}"}

    def fetch_4h_candles(self, market: str, total_count: int = 650) -> pd.DataFrame:
        """업비트 공개 API로 4시간봉(minutes/240) 최대 total_count개 페이지네이션 수집."""
        collected: List[Dict[str, Any]] = []
        to_param: Optional[str] = None
        remaining = total_count

        while remaining > 0:
            batch = min(200, remaining)
            params: Dict[str, Any] = {"market": market, "count": batch}
            if to_param:
                params["to"] = to_param
            resp = self.session.get(f"{UPBIT_API_BASE}/candles/minutes/240", params=params, timeout=15)
            resp.raise_for_status()
            rows = resp.json()
            if not rows:
                break
            collected.extend(rows)
            oldest_utc = rows[-1]["candle_date_time_utc"]
            to_param = oldest_utc.replace("T", " ")
            remaining -= len(rows)
            if len(rows) < batch:
                break
            time.sleep(0.14)  # 업비트 초당 10회 제한 준수

        if not collected:
            raise ValueError(f"No 4h candles fetched for {market}")

        records = []
        for r in collected:
            records.append(
                {
                    "timestamp_utc": pd.to_datetime(r["candle_date_time_utc"], utc=True),
                    "open": float(r["opening_price"]),
                    "high": float(r["high_price"]),
                    "low": float(r["low_price"]),
                    "close": float(r["trade_price"]),
                    "volume": float(r["candle_acc_trade_volume"]),
                }
            )
        df = pd.DataFrame(records).drop_duplicates(subset=["timestamp_utc"]).sort_values("timestamp_utc").reset_index(drop=True)
        return df

    def get_accounts(self) -> List[Dict[str, Any]]:
        if self.dry_run:
            raise RuntimeError("get_accounts() should not be called in DRY_RUN mode.")
        headers = self._auth_headers()
        resp = self.session.get(f"{UPBIT_API_BASE}/accounts", headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def place_market_buy(self, market: str, krw_amount: float) -> Dict[str, Any]:
        """시장가 매수 (ord_type='price'). DRY_RUN=true 시 절대 API 미호출."""
        if self.dry_run:
            return {"dry_run": True, "side": "bid", "market": market, "krw_amount": round(krw_amount, 2)}
        params = {
            "market": market,
            "side": "bid",
            "price": str(int(krw_amount)),
            "ord_type": "price",
        }
        headers = self._auth_headers(params)
        resp = self.session.post(f"{UPBIT_API_BASE}/orders", json=params, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def place_market_sell(self, market: str, volume: float) -> Dict[str, Any]:
        """시장가 매도 (ord_type='market'). DRY_RUN=true 시 절대 API 미호출."""
        if self.dry_run:
            return {"dry_run": True, "side": "ask", "market": market, "volume": volume}
        params = {
            "market": market,
            "side": "ask",
            "volume": f"{volume:.8f}",
            "ord_type": "market",
        }
        headers = self._auth_headers(params)
        resp = self.session.post(f"{UPBIT_API_BASE}/orders", json=params, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.json()


def compute_indicators(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    out["ema50"] = out["close"].ewm(span=int(params["ema_fast"]), adjust=False).mean()
    out["ema200"] = out["close"].ewm(span=int(params["ema_slow"]), adjust=False).mean()
    out["ema600"] = out["close"].ewm(span=int(params["btc_regime_ema"]), adjust=False).mean()

    prev_close = out["close"].shift(1)
    tr = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_n = int(params["atr_period"])
    out["atr14"] = tr.ewm(alpha=1.0 / atr_n, adjust=False).mean()

    # Zero Look-Ahead: 직전 N봉 기준 돈치안/거래량 (shift(1))
    out["donchian_high_48"] = out["high"].rolling(int(params["donchian_entry"])).max().shift(1)
    out["donchian_low_24"] = out["low"].rolling(int(params["donchian_exit"])).min().shift(1)
    out["vol_ma20"] = out["volume"].rolling(int(params["vol_ma"])).mean().shift(1)
    return out


def select_closed_candles(df: pd.DataFrame, now_utc: dt.datetime) -> Tuple[pd.DataFrame, Optional[float]]:
    """현재 진행 중인 미완성 4시간봉은 신호 계산에서 제외 (Zero Look-Ahead)."""
    latest_open_ts = df["timestamp_utc"].iloc[-1]
    latest_close_ts = latest_open_ts + pd.Timedelta(hours=4)
    live_price = float(df["close"].iloc[-1])
    if now_utc < latest_close_ts.to_pydatetime():
        return df.iloc[:-1].copy(), live_price
    return df.copy(), live_price


def load_or_init_paper_state(initial_capital: float, today_utc: str) -> Dict[str, Any]:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if PAPER_STATE_PATH.exists():
        state = json.loads(PAPER_STATE_PATH.read_text(encoding="utf-8"))
        if state.get("day_start_date_utc") != today_utc:
            state["day_start_date_utc"] = today_utc
            state["day_start_equity_krw"] = state.get("equity_krw", initial_capital)
        return state
    return {
        "mode": "DRY_RUN_PAPER",
        "initial_capital_krw": initial_capital,
        "cash_krw": initial_capital,
        "equity_krw": initial_capital,
        "hwm_krw": initial_capital,
        "day_start_date_utc": today_utc,
        "day_start_equity_krw": initial_capital,
        "positions": {},
        "trade_history": [],
        "last_run_utc": None,
    }


def main() -> int:
    logger = setup_logging()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    guard = cfg["risk_guardrails"]
    tp = cfg["turbo1_params"]
    coins: List[str] = cfg["upbit_coins"]

    # 안전 기본값: config 또는 환경변수 중 하나라도 true면 DRY_RUN
    dry_run_cfg = bool(cfg.get("dry_run", True))
    dry_run_env = os.environ.get("DRY_RUN", "true").lower() == "true"
    dry_run = dry_run_cfg or dry_run_env

    now_utc = dt.datetime.now(dt.timezone.utc)
    now_kst = now_utc.astimezone(KST)
    today_utc_str = now_utc.strftime("%Y-%m-%d")

    logger.info(f"=== Upbit Turbo-1 Bot Start (DRY_RUN={dry_run}) ===")

    client = UpbitClient(
        dry_run=dry_run,
        access_key=os.environ.get("UPBIT_ACCESS_KEY", ""),
        secret_key=os.environ.get("UPBIT_SECRET_KEY", ""),
    )

    state = load_or_init_paper_state(float(tp["initial_paper_capital_krw"]), today_utc_str)
    fee_rate = float(guard["fee_slippage_oneway"])

    # 1. 8개 코인 4시간봉 650개 수집 & 지표 계산
    coin_frames: Dict[str, pd.DataFrame] = {}
    live_prices: Dict[str, float] = {}
    api_errors: List[str] = []

    for market in coins:
        try:
            raw_df = client.fetch_4h_candles(market, total_count=int(tp["candle_fetch_count"]))
            closed_df, live_px = select_closed_candles(raw_df, now_utc)
            if len(closed_df) < 600:
                raise ValueError(f"Only {len(closed_df)} closed 4h candles (< 600)")
            coin_frames[market] = compute_indicators(closed_df, tp)
            live_prices[market] = live_px
        except Exception as e:
            logger.error(f"[API ERROR] {market}: {e}")
            api_errors.append(f"{market}: {e}")

    # 2. BTC 100일(600봉) EMA 레짐 판정
    btc_regime_ok = False
    btc_info: Dict[str, Any] = {}
    if "KRW-BTC" in coin_frames:
        btc_last = coin_frames["KRW-BTC"].iloc[-1]
        btc_close = float(btc_last["close"])
        btc_ema600 = float(btc_last["ema600"])
        btc_regime_ok = btc_close > btc_ema600
        btc_info = {
            "closed_candle_utc": btc_last["timestamp_utc"].strftime("%Y-%m-%d %H:%M UTC"),
            "close_krw": round(btc_close, 0),
            "live_price_krw": round(live_prices["KRW-BTC"], 0),
            "ema600_krw": round(btc_ema600, 0),
            "gap_ema600_pct": round((btc_close / btc_ema600 - 1.0) * 100.0, 2),
            "regime_ok": btc_regime_ok,
        }

    # 3. 현재 가상/실전 평가액 계산 및 Kill-Switch / HWM 감속 체크
    positions: Dict[str, Dict[str, Any]] = state["positions"]
    pos_value_krw = 0.0
    for mkt, pos in positions.items():
        px = live_prices.get(mkt, pos["entry_price"])
        pos_value_krw += float(pos["qty"]) * px

    total_equity = float(state["cash_krw"]) + pos_value_krw
    state["equity_krw"] = round(total_equity, 2)
    state["hwm_krw"] = round(max(float(state["hwm_krw"]), total_equity), 2)

    hwm_dd = (total_equity / float(state["hwm_krw"]) - 1.0) if state["hwm_krw"] > 0 else 0.0
    day_start_eq = float(state.get("day_start_equity_krw", total_equity))
    daily_pnl_pct = (total_equity / day_start_eq - 1.0) if day_start_eq > 0 else 0.0

    hwm_throttle_active = hwm_dd <= float(guard["hwm_throttle_dd"])
    effective_risk = float(guard["risk_per_trade"]) * (0.5 if hwm_throttle_active else 1.0)

    kill_switch_reasons: List[str] = []
    if daily_pnl_pct <= float(guard["daily_loss_kill_switch"]):
        kill_switch_reasons.append(f"일일 손실 한도 초과 ({daily_pnl_pct*100:.2f}% <= {guard['daily_loss_kill_switch']*100:.1f}%)")
    if api_errors:
        kill_switch_reasons.append(f"API 수집 오류 {len(api_errors)}건 발생")
    kill_switch_triggered = len(kill_switch_reasons) > 0

    actions_taken: List[Dict[str, Any]] = []

    # 4. 보유 포지션 청산 조건 체크 (하드스톱 2*ATR / 트레일링 3*ATR / 24봉 돈치안 저점)
    for mkt in list(positions.keys()):
        if mkt not in coin_frames:
            continue
        pos = positions[mkt]
        row = coin_frames[mkt].iloc[-1]
        low_px = float(row["low"])
        high_px = float(row["high"])
        close_px = float(row["close"])
        atr14 = float(row["atr14"])
        dc_low_24 = float(row["donchian_low_24"])

        pos["highest_high"] = max(float(pos["highest_high"]), high_px, live_prices.get(mkt, close_px))
        hard_stop_px = float(pos["entry_price"]) - float(tp["hard_stop_atr"]) * float(pos["entry_atr"])
        trail_stop_px = float(pos["highest_high"]) - float(tp["trailing_stop_atr"]) * atr14

        exit_reason = None
        if low_px <= hard_stop_px or live_prices.get(mkt, close_px) <= hard_stop_px:
            exit_reason = f"HARD_STOP_2ATR ({hard_stop_px:,.0f})"
        elif low_px <= trail_stop_px or live_prices.get(mkt, close_px) <= trail_stop_px:
            exit_reason = f"TRAILING_STOP_3ATR ({trail_stop_px:,.0f})"
        elif close_px < dc_low_24:
            exit_reason = f"DONCHIAN_24_EXIT ({dc_low_24:,.0f})"

        if exit_reason:
            exec_px = live_prices.get(mkt, close_px)
            qty = float(pos["qty"])
            client.place_market_sell(mkt, qty)
            proceeds = qty * exec_px * (1.0 - fee_rate)
            pnl_krw = proceeds - float(pos["cost_krw"])
            state["cash_krw"] = round(float(state["cash_krw"]) + proceeds, 2)
            del positions[mkt]
            rec = {
                "time_utc": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "market": mkt,
                "action": "DRY_SELL" if dry_run else "LIVE_SELL",
                "reason": exit_reason,
                "price_krw": round(exec_px, 2),
                "qty": round(qty, 8),
                "pnl_krw": round(pnl_krw, 0),
            }
            actions_taken.append(rec)
            state["trade_history"].append(rec)
            logger.info(f"[EXIT] {rec}")

    # 5. 8개 코인 스캔 및 신규 진입 평가
    coin_signals: List[Dict[str, Any]] = []
    entry_candidates: List[Dict[str, Any]] = []

    for market in coins:
        if market not in coin_frames:
            coin_signals.append({"market": market, "status": "API_ERROR"})
            continue
        row = coin_frames[market].iloc[-1]
        close_px = float(row["close"])
        ema50 = float(row["ema50"])
        ema200 = float(row["ema200"])
        dc_hi48 = float(row["donchian_high_48"])
        dc_lo24 = float(row["donchian_low_24"])
        vol = float(row["volume"])
        vol_ma20 = float(row["vol_ma20"])
        atr14 = float(row["atr14"])

        trend_ok = ema50 > ema200
        breakout_ok = close_px > dc_hi48
        vol_ratio = (vol / vol_ma20) if vol_ma20 > 0 else 0.0
        vol_ok = vol_ratio >= float(tp["vol_mult"])
        entry_signal = btc_regime_ok and trend_ok and breakout_ok and vol_ok

        stop_dist_pct = (float(tp["hard_stop_atr"]) * atr14) / close_px if close_px > 0 else 0.05
        raw_weight = (effective_risk / stop_dist_pct) if stop_dist_pct > 0 else 0.0
        target_weight = min(float(guard["single_coin_cap"]), raw_weight)

        sig_info = {
            "market": market,
            "status": "OK",
            "closed_candle_utc": row["timestamp_utc"].strftime("%Y-%m-%d %H:%M UTC"),
            "close_krw": round(close_px, 2),
            "live_price_krw": round(live_prices[market], 2),
            "ema50": round(ema50, 2),
            "ema200": round(ema200, 2),
            "donchian_high_48": round(dc_hi48, 2),
            "donchian_low_24": round(dc_lo24, 2),
            "atr14": round(atr14, 2),
            "vol_ratio": round(vol_ratio, 2),
            "trend_ok": trend_ok,
            "breakout_ok": breakout_ok,
            "vol_ok": vol_ok,
            "entry_signal": entry_signal,
            "holding": market in positions,
            "target_weight_pct": round(target_weight * 100.0, 2),
        }
        coin_signals.append(sig_info)

        if entry_signal and (market not in positions):
            entry_candidates.append({**sig_info, "target_weight": target_weight})

    # 진입 실행 (Kill-Switch 미발동 & 최대 4종목 & 현금 범위 내 1x)
    entry_candidates.sort(key=lambda x: x["vol_ratio"], reverse=True)
    if not kill_switch_triggered and btc_regime_ok:
        for cand in entry_candidates:
            if len(positions) >= int(guard["max_positions"]):
                break
            mkt = cand["market"]
            alloc_krw = min(total_equity * cand["target_weight"], float(state["cash_krw"]))
            if alloc_krw < 10000:  # 업비트 최소 주문금액 5,000원 이상 여유
                continue
            exec_px = live_prices[mkt]
            client.place_market_buy(mkt, alloc_krw)
            qty = (alloc_krw * (1.0 - fee_rate)) / exec_px
            state["cash_krw"] = round(float(state["cash_krw"]) - alloc_krw, 2)
            positions[mkt] = {
                "entry_time_utc": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "entry_price": round(exec_px, 2),
                "entry_atr": round(cand["atr14"], 2),
                "highest_high": round(exec_px, 2),
                "qty": round(qty, 8),
                "cost_krw": round(alloc_krw, 2),
            }
            rec = {
                "time_utc": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "market": mkt,
                "action": "DRY_BUY" if dry_run else "LIVE_BUY",
                "reason": f"TURBO1_BREAKOUT (vol={cand['vol_ratio']}x)",
                "price_krw": round(exec_px, 2),
                "krw_amount": round(alloc_krw, 0),
                "qty": round(qty, 8),
            }
            actions_taken.append(rec)
            state["trade_history"].append(rec)
            logger.info(f"[ENTRY] {rec}")

    # 최종 상태 저장
    state["last_run_utc"] = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
    PAPER_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    if kill_switch_triggered:
        traffic = "RED"
        summary = f"킬스위치 발동 ({'; '.join(kill_switch_reasons)}) — 신규 진입 중단"
    elif not btc_regime_ok:
        traffic = "RED"
        gap = btc_info.get("gap_ema600_pct", 0.0)
        summary = f"BTC 100일선(600 EMA) 하회 ({gap:+.2f}%) — 하락장 자동 현금화 방어 모드 (보유 {len(positions)}/4)"
    else:
        entry_cnt = sum(1 for c in coin_signals if c.get("entry_signal"))
        traffic = "GREEN" if (len(positions) > 0 or entry_cnt > 0) else "YELLOW"
        summary = f"BTC 상승 레짐 ON (+{btc_info.get('gap_ema600_pct', 0):.2f}%) | 보유 {len(positions)}/4 | 신규 돌파 {entry_cnt}종목"

    status_payload = {
        "strategy_id": "CRYPTO-SHORT-M3-TURBO1",
        "generated_at_kst": now_kst.strftime("%Y-%m-%d %H:%M:%S KST"),
        "generated_at_utc": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "dry_run": dry_run,
        "traffic_light": traffic,
        "summary": summary,
        "btc_regime": btc_info,
        "risk_state": {
            "leverage": guard["leverage"],
            "effective_risk_pct": round(effective_risk * 100.0, 2),
            "hwm_dd_pct": round(hwm_dd * 100.0, 2),
            "hwm_throttle_active": hwm_throttle_active,
            "daily_pnl_pct": round(daily_pnl_pct * 100.0, 2),
            "kill_switch_triggered": kill_switch_triggered,
            "kill_switch_reasons": kill_switch_reasons,
        },
        "paper_account": {
            "equity_krw": state["equity_krw"],
            "cash_krw": state["cash_krw"],
            "hwm_krw": state["hwm_krw"],
            "open_positions_count": len(positions),
            "max_positions": guard["max_positions"],
            "positions": positions,
        },
        "actions_this_run": actions_taken,
        "coins": coin_signals,
    }

    STATUS_OUT_PATH.write_text(json.dumps(status_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Saved Upbit bot status to {STATUS_OUT_PATH}")
    logger.info(f"  [TURBO-1] {traffic} | {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
