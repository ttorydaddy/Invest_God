#!/usr/bin/env python3
"""
Invest_God Phase 2 — 일일 통합 시그널 스캐너 (daily_signal_engine.py)
- KRX 중기 (D-3): 코스피/코스닥 MA60 & 5일 급락 레짐 + 실시간 공개 시세 주도주 돌파 스캔
- 장기 코어 (L-3): 나스닥100(133690), 코스피200(069500), BTC(KRW-BTC) 120일선 레짐 & 목표 비중
- 하이닉스 2배 로켓 (20/120): SK하이닉스(000660) MA20 vs MA120 골든/데드크로스 판정
- 분기 리밸런싱(85:15) 알림

※ 실시간 공개 시세(Yahoo Finance, Upbit Public API)만 사용하며, 로컬 과거 DB로 fallback 하지 않습니다.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
UNIVERSE_PATH = BASE_DIR / "krx_universe.json"
OUTPUT_DIR = BASE_DIR / "output"
LOG_DIR = BASE_DIR / "logs"

KST = dt.timezone(dt.timedelta(hours=9))


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("daily_signal_engine")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh = logging.FileHandler(LOG_DIR / "daily_signal.log", encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


def normalize_yf_df(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance 단일 티커 DataFrame의 MultiIndex 컬럼을 안전하게 평탄화."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] for c in out.columns]
    out = out.dropna(subset=["Close"])
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out.sort_index()


def fetch_yf_single(ticker: str, period: str = "1y") -> pd.DataFrame:
    raw = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=False)
    return normalize_yf_df(raw)


def fetch_upbit_daily_candles(market: str = "KRW-BTC", count: int = 160) -> pd.DataFrame:
    url = "https://api.upbit.com/v1/candles/days"
    params = {"market": market, "count": min(count, 200)}
    headers = {"Accept": "application/json"}
    resp = requests.get(url, params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    rows = resp.json()
    if not rows:
        raise ValueError(f"Empty Upbit daily candles for {market}")
    records = []
    for r in rows:
        records.append(
            {
                "Date": pd.to_datetime(r["candle_date_time_kst"][:10]),
                "Open": float(r["opening_price"]),
                "High": float(r["high_price"]),
                "Low": float(r["low_price"]),
                "Close": float(r["trade_price"]),
                "Volume": float(r["candle_acc_trade_volume"]),
            }
        )
    df = pd.DataFrame(records).drop_duplicates(subset=["Date"]).set_index("Date").sort_index()
    return df


def analyze_krx_d3(cfg: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    p = cfg["krx_d3_params"]
    ma_n = int(p["market_ma_period"])
    crash_lim = float(p["market_5d_crash_limit"])

    # 1. 시장 지수 레짐 판정 (KOSPI / KOSDAQ)
    regime: Dict[str, Any] = {}
    for label, ticker in [("KOSPI", p["kospi_ticker"]), ("KOSDAQ", p["kosdaq_ticker"])]:
        try:
            df = fetch_yf_single(ticker, period="1y")
            if len(df) < ma_n + 5:
                raise ValueError(f"Insufficient bars ({len(df)}) for {ticker}")
            close = float(df["Close"].iloc[-1])
            ma60 = float(df["Close"].rolling(ma_n).mean().iloc[-1])
            ret_5d = float(df["Close"].iloc[-1] / df["Close"].iloc[-6] - 1.0)
            above_ma60 = close > ma60
            no_crash = ret_5d > crash_lim
            passed = above_ma60 and no_crash
            regime[label] = {
                "status": "OK",
                "ticker": ticker,
                "as_of_date": df.index[-1].strftime("%Y-%m-%d"),
                "close": round(close, 2),
                "ma60": round(ma60, 2),
                "gap_ma60_pct": round((close / ma60 - 1.0) * 100.0, 2),
                "ret_5d_pct": round(ret_5d * 100.0, 2),
                "above_ma60": above_ma60,
                "no_5d_crash": no_crash,
                "regime_pass": passed,
                "signal": "RISK_ON" if passed else "REST_DEFENSE",
            }
        except Exception as e:
            logger.error(f"[D-3] Benchmark fetch failed for {label} ({ticker}): {e}")
            regime[label] = {"status": "ERROR", "ticker": ticker, "error": str(e), "regime_pass": False, "signal": "ERROR"}

    # 2. 실시간 공개 시세 기반 KRX 주도주 돌파 스캔
    universe: List[Dict[str, Any]] = []
    if UNIVERSE_PATH.exists():
        universe = json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))
    else:
        logger.warning("[D-3] krx_universe.json not found; skipping stock scan.")

    tickers = [u["yf_ticker"] for u in universe]
    meta_map = {u["yf_ticker"]: u for u in universe}

    candidates: List[Dict[str, Any]] = []
    watchlist: List[Dict[str, Any]] = []
    scanned_ok = 0
    scan_failed = 0
    latest_stock_date: Optional[str] = None

    if tickers:
        logger.info(f"[D-3] Downloading live daily OHLCV for {len(tickers)} liquid KRX stocks via yfinance...")
        try:
            raw_batch = yf.download(
                tickers=tickers,
                period="6mo",
                interval="1d",
                group_by="ticker",
                threads=True,
                progress=False,
                auto_adjust=False,
            )
            for t in tickers:
                meta = meta_map[t]
                try:
                    if isinstance(raw_batch.columns, pd.MultiIndex) and t in raw_batch.columns.get_level_values(0):
                        sub = raw_batch[t].dropna(subset=["Close"]).sort_index()
                    else:
                        scan_failed += 1
                        continue
                    if len(sub) < 65:
                        scan_failed += 1
                        continue
                    scanned_ok += 1
                    last_dt = sub.index[-1].strftime("%Y-%m-%d")
                    if latest_stock_date is None or last_dt > latest_stock_date:
                        latest_stock_date = last_dt

                    c = sub["Close"].astype(float)
                    o = sub["Open"].astype(float)
                    h = sub["High"].astype(float)
                    v = sub["Volume"].astype(float)
                    tval = c * v

                    close_now = float(c.iloc[-1])
                    open_now = float(o.iloc[-1])
                    prev_close = float(c.iloc[-2])
                    tval_now = float(tval.iloc[-1])
                    tval_ma20 = float(tval.iloc[-21:-1].mean())
                    vol_mult = (tval_now / tval_ma20) if tval_ma20 > 0 else 0.0

                    ma20_series = c.rolling(20).mean()
                    ma60_series = c.rolling(60).mean()
                    ma20_now = float(ma20_series.iloc[-1])
                    ma20_5d_ago = float(ma20_series.iloc[-6])
                    ma60_now = float(ma60_series.iloc[-1])

                    prev_20d_high = float(h.iloc[-21:-1].max())
                    daily_ret = (close_now / prev_close - 1.0) if prev_close > 0 else 0.0
                    # Yahoo Finance가 당일 장마감 직후 Open=0으로 주는 경우 방어 처리
                    is_bullish = (close_now > open_now) if open_now > 0 else (close_now > prev_close)

                    mkt = meta["market"]
                    mkt_pass = bool(regime.get(mkt, {}).get("regime_pass", False))

                    cond_liquidity = (close_now >= p["min_price_krw"]) and (tval_now >= p["min_trading_value_krw"])
                    cond_trend = (close_now > ma20_now > ma60_now) and (ma20_now > ma20_5d_ago)
                    cond_vol = vol_mult >= p["vol_value_mult"]
                    cond_breakout = close_now > prev_20d_high
                    cond_candle = is_bullish and (p["min_daily_ret"] <= daily_ret <= p["max_daily_ret"])

                    row_info = {
                        "code": meta["code"],
                        "name": meta["name"],
                        "market": mkt,
                        "as_of_date": last_dt,
                        "close": int(round(close_now)),
                        "daily_ret_pct": round(daily_ret * 100.0, 2),
                        "trading_value_bkrw": round(tval_now / 1e8, 1),  # 억원
                        "vol_surge_ratio": round(vol_mult, 2),
                        "prev_20d_high": int(round(prev_20d_high)),
                        "ma20": int(round(ma20_now)),
                        "ma60": int(round(ma60_now)),
                        "market_regime_pass": mkt_pass,
                    }

                    if mkt_pass and cond_liquidity and cond_trend and cond_vol and cond_breakout and cond_candle:
                        row_info["note"] = "기술적 돌파 통과 — HTS/MTS에서 외인+기관 순매수(>=+2.0%, 3일누적>0) 최종 확인 필요"
                        candidates.append(row_info)
                    elif cond_liquidity and cond_trend and vol_mult >= 1.5 and daily_ret >= 0.02:
                        row_info["missing"] = [
                            k
                            for k, ok in [
                                ("시장레짐", mkt_pass),
                                ("거래대금2.5x", cond_vol),
                                ("20일신고가", cond_breakout),
                                ("양봉+3~20%", cond_candle),
                            ]
                            if not ok
                        ]
                        watchlist.append(row_info)
                except Exception:
                    scan_failed += 1
        except Exception as e:
            logger.error(f"[D-3] Batch yfinance scan failed: {e}")

    candidates.sort(key=lambda x: x["vol_surge_ratio"], reverse=True)
    watchlist.sort(key=lambda x: x["vol_surge_ratio"], reverse=True)

    kospi_ok = regime.get("KOSPI", {}).get("regime_pass", False)
    kosdaq_ok = regime.get("KOSDAQ", {}).get("regime_pass", False)
    if kospi_ok and kosdaq_ok:
        traffic = "GREEN"
        summary = f"코스피·코스닥 모두 상승 레짐 (기술적 돌파 후보 {len(candidates)}종목)"
    elif kospi_ok or kosdaq_ok:
        traffic = "YELLOW"
        on_mkt = "KOSPI" if kospi_ok else "KOSDAQ"
        summary = f"{on_mkt}만 상승 레짐 (돌파 후보 {len(candidates)}종목)"
    else:
        traffic = "RED"
        summary = "코스피·코스닥 모두 하락/경계 레짐 — 신규 진입 중단 (기존 종목 손절·트레일링만 관리)"

    return {
        "strategy_id": "KRX-MEDIUM-D3",
        "traffic_light": traffic,
        "summary": summary,
        "market_regime": regime,
        "live_scan_stats": {
            "universe_size": len(tickers),
            "scanned_ok": scanned_ok,
            "scan_failed": scan_failed,
            "latest_stock_date": latest_stock_date,
            "data_source": "yfinance live (no local fallback)",
        },
        "breakout_candidates": candidates[:10],
        "near_watchlist": watchlist[:8],
        "investor_flow_note": "외국인·기관 확정 수급(>=+2.0% 및 3일 누적>0%)은 16:00 KST 이후 증권사 조건검색식에서 최종 교차 확인하세요.",
    }


def analyze_longterm_l3(cfg: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    p = cfg["longterm_l3_params"]
    ma_n = int(p["ma_period"])

    assets_result: List[Dict[str, Any]] = []
    target_weights: Dict[str, float] = {}
    safe_pool = 0.0
    on_count = 0

    for asset in p["offensive_assets"]:
        try:
            if asset["source"] == "yfinance":
                df = fetch_yf_single(asset["ticker"], period="1y")
            elif asset["source"] == "upbit":
                df = fetch_upbit_daily_candles(asset["ticker"], count=160)
            else:
                raise ValueError(f"Unknown source {asset['source']}")

            if len(df) < ma_n:
                raise ValueError(f"Insufficient bars ({len(df)}) < MA{ma_n}")

            close = float(df["Close"].iloc[-1])
            ma120 = float(df["Close"].rolling(ma_n).mean().iloc[-1])
            above = close > ma120
            w = float(asset["base_weight"]) if above else 0.0
            if above:
                on_count += 1
            else:
                safe_pool += float(asset["base_weight"])

            target_weights[asset["id"]] = round(w * 100.0, 2)
            assets_result.append(
                {
                    "id": asset["id"],
                    "name": asset["name"],
                    "ticker": asset["ticker"],
                    "status": "OK",
                    "as_of_date": df.index[-1].strftime("%Y-%m-%d"),
                    "close": round(close, 2),
                    "ma120": round(ma120, 2),
                    "gap_ma120_pct": round((close / ma120 - 1.0) * 100.0, 2),
                    "regime": "RISK_ON" if above else "SAFE_HAVEN",
                    "base_weight_pct": round(float(asset["base_weight"]) * 100.0, 2),
                    "target_weight_pct": round(w * 100.0, 2),
                }
            )
        except Exception as e:
            logger.error(f"[L-3] Failed fetching {asset['ticker']}: {e}")
            safe_pool += float(asset["base_weight"])
            target_weights[asset["id"]] = 0.0
            assets_result.append(
                {
                    "id": asset["id"],
                    "name": asset["name"],
                    "ticker": asset["ticker"],
                    "status": "ERROR",
                    "error": str(e),
                    "regime": "ERROR_SAFE_FALLBACK",
                    "base_weight_pct": round(float(asset["base_weight"]) * 100.0, 2),
                    "target_weight_pct": 0.0,
                }
            )

    defensive_alloc: List[Dict[str, Any]] = []
    for d in p["defensive_assets"]:
        dw = round(safe_pool * float(d["split"]) * 100.0, 2)
        target_weights[d["id"]] = dw
        defensive_alloc.append(
            {
                "id": d["id"],
                "name": d["name"],
                "ticker": d["ticker"],
                "target_weight_pct": dw,
            }
        )

    if on_count == 3:
        traffic = "GREEN"
        summary = "3종 공격 자산 전원 120일선 상회 — 풀 공격 배분 (나스닥 40% / 코스피 30% / BTC 30%)"
    elif on_count > 0:
        traffic = "YELLOW"
        summary = f"공격 자산 {on_count}/3종 120일선 상회 — 이탈 비중 {round(safe_pool*100, 1)}% 금·단기채 50:50 대피"
    else:
        traffic = "RED"
        summary = "전원 120일선 이탈 — 100% 방어 대피 (KODEX 골드 50% + KODEX 단기채권 50%)"

    return {
        "strategy_id": "LONGTERM-CORE-L3",
        "traffic_light": traffic,
        "summary": summary,
        "offensive_assets": assets_result,
        "defensive_allocation": defensive_alloc,
        "target_weights_pct": target_weights,
        "rebalance_schedule": "매월 마지막 거래일 종가 확인 → 익월 첫 거래일 시가 리밸런싱",
    }


def analyze_skh_rocket(cfg: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    p = cfg["rocket_params"]
    fast_n = int(p["ma_fast"])
    slow_n = int(p["ma_slow"])
    try:
        df = fetch_yf_single(p["ticker"], period="1y")
        if len(df) < slow_n:
            raise ValueError(f"Insufficient bars ({len(df)}) < MA{slow_n}")
        close = float(df["Close"].iloc[-1])
        ma20 = float(df["Close"].rolling(fast_n).mean().iloc[-1])
        ma120 = float(df["Close"].rolling(slow_n).mean().iloc[-1])
        golden = ma20 > ma120
        spread_pct = (ma20 / ma120 - 1.0) * 100.0
        return {
            "strategy_id": "SKH-ROCKET-20-120",
            "status": "OK",
            "name": p["name"],
            "ticker": p["ticker"],
            "as_of_date": df.index[-1].strftime("%Y-%m-%d"),
            "close": int(round(close)),
            "ma20": int(round(ma20)),
            "ma120": int(round(ma120)),
            "spread_ma20_vs_ma120_pct": round(spread_pct, 2),
            "signal": "ROCKET_ON" if golden else "SAFE_RP",
            "action_kr": f"2배 레버리지 보유 ({p['on_instrument']})" if golden else f"단기채/RP 대피 ({p['off_instrument']})",
            "traffic_light": "GREEN" if golden else "RED",
            "summary": (
                f"골든크로스 유지 (MA20 {int(round(ma20)):,}원 > MA120 {int(round(ma120)):,}원, +{spread_pct:.2f}%) → 2배 레버리지 보유"
                if golden
                else f"데드크로스 상태 (MA20 {int(round(ma20)):,}원 <= MA120 {int(round(ma120)):,}원, {spread_pct:.2f}%) → 단기채/RP 전액 대피"
            ),
        }
    except Exception as e:
        logger.error(f"[ROCKET] Failed fetching {p['ticker']}: {e}")
        return {
            "strategy_id": "SKH-ROCKET-20-120",
            "status": "ERROR",
            "ticker": p["ticker"],
            "error": str(e),
            "signal": "ERROR",
            "traffic_light": "RED",
            "summary": f"시세 조회 실패: {e}",
        }


def compute_quarterly_rebal_status(now_kst: dt.datetime, cfg: Dict[str, Any]) -> Dict[str, Any]:
    months = cfg["rebalancing"]["months"]  # [1, 4, 7, 10]
    today = now_kst.date()

    # 현재 분기 첫날 및 다음 분기 첫날 계산
    next_candidates = []
    for y in [today.year, today.year + 1]:
        for m in months:
            d = dt.date(y, m, 1)
            if d >= today:
                next_candidates.append(d)
    next_q_start = min(next_candidates)
    days_until_next = (next_q_start - today).days

    # 오늘이 분기 초(1·4·7·10월 1~5일)이거나 3일 이내인지 판정
    is_quarter_start_window = (today.month in months and today.day <= 5) or (days_until_next <= 3)

    # 월말 체크 (L-3 월간 리밸런싱 알림용)
    next_month_first = dt.date(today.year + (1 if today.month == 12 else 0), 1 if today.month == 12 else today.month + 1, 1)
    days_to_month_end = (next_month_first - today).days - 1
    is_month_end_window = days_to_month_end <= 2

    return {
        "as_of_date_kst": today.isoformat(),
        "is_rebalance_alert": is_quarter_start_window,
        "next_quarter_start": next_q_start.isoformat(),
        "days_until_next_quarter": days_until_next,
        "is_l3_month_end_window": is_month_end_window,
        "days_to_month_end": days_to_month_end,
        "target_split": "본진 85.0% (단기 28.33% / 중기 28.33% / 장기 28.33%) : 로켓 15.0%",
        "checklist": [
            "총 평가자산(단기 코인 + 중기 KRX + 장기 코어 + 로켓) 합산",
            "로켓 비중 > 15%면 초과분 매도 → 본진 3개 버킷에 1/3씩 수확",
            "로켓 비중 < 15%면 부족분 보충 (단, 하이닉스 데드크로스 중이면 RP/단기채로 채움)",
            "본진 3개 버킷이 각각 28.33% 균등 배분되도록 미세 조정",
        ],
    }


def main() -> int:
    logger = setup_logging()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    now_kst = dt.datetime.now(KST)

    logger.info("=== Invest_God Daily Signal Engine Start ===")
    dry_run_env = os.environ.get("DRY_RUN", str(cfg.get("dry_run", True))).lower() == "true"

    krx_d3 = analyze_krx_d3(cfg, logger)
    longterm_l3 = analyze_longterm_l3(cfg, logger)
    rocket = analyze_skh_rocket(cfg, logger)
    rebal = compute_quarterly_rebal_status(now_kst, cfg)

    payload = {
        "generated_at_kst": now_kst.strftime("%Y-%m-%d %H:%M:%S KST"),
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "dry_run": dry_run_env,
        "portfolio_version": cfg["portfolio_version"],
        "portfolio_weights_pct": cfg["portfolio_weights_display_pct"],
        "buckets": {
            "krx_medium_d3": krx_d3,
            "longterm_core_l3": longterm_l3,
            "skh_rocket_2x": rocket,
        },
        "quarterly_rebalancing": rebal,
    }

    out_path = OUTPUT_DIR / "latest_signals.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Saved daily signals to {out_path}")
    logger.info(f"  [D-3] {krx_d3['traffic_light']} | {krx_d3['summary']}")
    logger.info(f"  [L-3] {longterm_l3['traffic_light']} | {longterm_l3['summary']}")
    logger.info(f"  [ROCKET] {rocket['traffic_light']} | {rocket['summary']}")
    logger.info(f"  [REBAL] Alert={rebal['is_rebalance_alert']} | Next={rebal['next_quarter_start']} (D-{rebal['days_until_next_quarter']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
