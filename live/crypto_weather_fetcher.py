#!/usr/bin/env python3
"""
Invest_God Phase 2 — 코인 수급 기상청 데이터 수집기 (crypto_weather_fetcher.py)
※ 순수 대시보드 참고용 — 자동매매 룰·비중·가상투자 계산에 일절 개입하지 않습니다.

무료 공개 데이터 3종:
  1) 스테이블코인 총시총 추세 (DefiLlama): https://stablecoins.llama.fi/stablecoincharts/all
  2) 미국 BTC 현물 ETF (IBIT) 기관 수급 추세 (yfinance)
  3) 크립토 공포·탐욕 지수 (Alternative.me): https://api.alternative.me/fng/?limit=7

출력: live/output/latest_crypto_weather.json
※ 모든 외부 호출은 timeout=10 + try/except로 격리하여 장애 시에도 본진 워크플로에 영향 0%.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import requests
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUT_PATH = OUTPUT_DIR / "latest_crypto_weather.json"
KST = dt.timezone(dt.timedelta(hours=9))


def fetch_stablecoin_mcap() -> Dict[str, Any]:
    try:
        resp = requests.get("https://stablecoins.llama.fi/stablecoincharts/all", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list) or len(data) < 31:
            raise ValueError("DefiLlama 응답 데이터 부족")

        def total_usd(row: Dict[str, Any]) -> float:
            circ = row.get("totalCirculatingUSD", {})
            return sum(float(v) for v in circ.values() if isinstance(v, (int, float)))

        now_usd = total_usd(data[-1])
        d7_usd = total_usd(data[-8])
        d30_usd = total_usd(data[-31])

        now_b = now_usd / 1e9
        chg_7d_pct = ((now_usd / d7_usd) - 1.0) * 100.0 if d7_usd > 0 else 0.0
        chg_30d_pct = ((now_usd / d30_usd) - 1.0) * 100.0 if d30_usd > 0 else 0.0
        chg_7d_b = (now_usd - d7_usd) / 1e9
        chg_30d_b = (now_usd - d30_usd) / 1e9

        as_of = dt.datetime.fromtimestamp(int(data[-1]["date"]), tz=dt.timezone.utc).strftime("%Y-%m-%d")

        if chg_7d_pct >= 0.5 and chg_30d_pct >= 1.0:
            light, label = "GREEN", "유동성 유입 🟢"
            comment = f"스테이블코인 실탄이 7일 {chg_7d_b:+.2f}B$ / 30일 {chg_30d_b:+.2f}B$ 증가하며 코인 시장 대기 자금이 유입 중입니다."
        elif chg_7d_pct <= -0.5 or chg_30d_pct <= -1.0:
            light, label = "RED", "유동성 이탈 🔴"
            comment = f"스테이블코인 시총이 7일 {chg_7d_b:+.2f}B$ / 30일 {chg_30d_b:+.2f}B$ 감소하며 시장 유동성이 빠져나가는 중입니다."
        else:
            light, label = "YELLOW", "보합 🟡"
            comment = f"스테이블코인 총시총 ${now_b:,.2f}B 부근에서 뚜렷한 유입·이탈 없이 횡보 중입니다."

        return {
            "status": "OK",
            "as_of_date": as_of,
            "mcap_billion_usd": round(now_b, 2),
            "chg_7d_pct": round(chg_7d_pct, 2),
            "chg_30d_pct": round(chg_30d_pct, 2),
            "chg_7d_billion_usd": round(chg_7d_b, 2),
            "chg_30d_billion_usd": round(chg_30d_b, 2),
            "traffic_light": light,
            "label": label,
            "comment_kr": comment,
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "error": str(e),
            "traffic_light": "YELLOW",
            "label": "조회 지연 🟡",
            "comment_kr": "DefiLlama 공개 API 일시 지연 — 자동매매에는 영향 없음",
        }


def fetch_ibit_etf() -> Dict[str, Any]:
    try:
        df = yf.download("IBIT", period="4mo", interval="1d", progress=False, auto_adjust=False, timeout=10)
        if df is None or df.empty:
            raise ValueError("IBIT 시세 없음")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.dropna(subset=["Close"]).sort_index()
        if len(df) < 25:
            raise ValueError(f"IBIT 데이터 부족 ({len(df)}일)")

        close = df["Close"].astype(float)
        vol = df["Volume"].astype(float)
        dvol = close * vol

        last_px = float(close.iloc[-1])
        ma20 = float(close.rolling(20).mean().iloc[-1])
        gap_ma20_pct = ((last_px / ma20) - 1.0) * 100.0
        ret_5d_pct = ((last_px / float(close.iloc[-6])) - 1.0) * 100.0
        avg_dvol_5d_b = float(dvol.iloc[-5:].mean()) / 1e9
        avg_dvol_20d_b = float(dvol.iloc[-20:].mean()) / 1e9
        vol_ratio = (avg_dvol_5d_b / avg_dvol_20d_b) if avg_dvol_20d_b > 0 else 1.0
        as_of = df.index[-1].strftime("%Y-%m-%d")

        if last_px > ma20 and ret_5d_pct > 0:
            light, label = "GREEN", "기관 강세 🟢"
            comment = f"IBIT가 20일선(+{gap_ma20_pct:.2f}%) 위에서 5일 {ret_5d_pct:+.2f}% 상승하며 미국 기관 수급이 우호적입니다."
        elif last_px < ma20 and ret_5d_pct < 0:
            light, label = "RED", "기관 약세 🔴"
            comment = f"IBIT가 20일선({gap_ma20_pct:+.2f}%) 아래로 밀리며 최근 5일 {ret_5d_pct:+.2f}% 약세 흐름입니다."
        else:
            light, label = "YELLOW", "관망 🟡"
            comment = f"IBIT가 20일선 대비 {gap_ma20_pct:+.2f}%(5일 {ret_5d_pct:+.2f}%)로 기관 수급 방향성 탐색 구간입니다."

        return {
            "status": "OK",
            "ticker": "IBIT",
            "as_of_date": as_of,
            "close_usd": round(last_px, 2),
            "ma20_usd": round(ma20, 2),
            "gap_ma20_pct": round(gap_ma20_pct, 2),
            "ret_5d_pct": round(ret_5d_pct, 2),
            "avg_dvol_5d_billion_usd": round(avg_dvol_5d_b, 2),
            "vol_ratio_5d_vs_20d": round(vol_ratio, 2),
            "traffic_light": light,
            "label": label,
            "comment_kr": comment,
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "ticker": "IBIT",
            "error": str(e),
            "traffic_light": "YELLOW",
            "label": "조회 지연 🟡",
            "comment_kr": "Yahoo Finance IBIT 조회 일시 지연 — 자동매매에는 영향 없음",
        }


def fetch_fear_and_greed() -> Dict[str, Any]:
    try:
        resp = requests.get("https://api.alternative.me/fng/?limit=7", timeout=10)
        resp.raise_for_status()
        rows = resp.json().get("data", [])
        if not rows:
            raise ValueError("Alternative.me 응답 비어 있음")

        score = int(rows[0]["value"])
        score_7d = int(rows[-1]["value"])
        diff_7d = score - score_7d
        as_of = dt.datetime.fromtimestamp(int(rows[0]["timestamp"]), tz=dt.timezone.utc).strftime("%Y-%m-%d")

        if score <= 24:
            grade_kr, light = "극단적 공포", "RED"
            comment = f"공포·탐욕 {score}점(극단적 공포, 7일 대비 {diff_7d:+d}p) — 시장 투심 극도 위축 구간입니다."
        elif score <= 44:
            grade_kr, light = "공포", "YELLOW"
            comment = f"공포·탐욕 {score}점(공포, 7일 대비 {diff_7d:+d}p) — 경계심이 우세한 구간입니다."
        elif score <= 54:
            grade_kr, light = "중립", "YELLOW"
            comment = f"공포·탐욕 {score}점(중립, 7일 대비 {diff_7d:+d}p) — 투자 심리가 균형 상태입니다."
        elif score <= 74:
            grade_kr, light = "탐욕", "GREEN"
            comment = f"공포·탐욕 {score}점(탐욕, 7일 대비 {diff_7d:+d}p) — 매수 심리가 활발한 추세 구간입니다."
        else:
            grade_kr, light = "극단적 탐욕", "RED"
            comment = f"공포·탐욕 {score}점(극단적 탐욕, 7일 대비 {diff_7d:+d}p) — 단기 과열 경계 구간입니다."

        return {
            "status": "OK",
            "as_of_date": as_of,
            "score": score,
            "score_7d_ago": score_7d,
            "diff_7d": diff_7d,
            "grade_kr": grade_kr,
            "raw_classification": rows[0].get("value_classification", ""),
            "traffic_light": light,
            "label": f"{grade_kr} ({score}점)",
            "comment_kr": comment,
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "error": str(e),
            "traffic_light": "YELLOW",
            "label": "조회 지연 🟡",
            "comment_kr": "Alternative.me 공개 API 일시 지연 — 자동매매에는 영향 없음",
        }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    now_kst = dt.datetime.now(KST)

    stable = fetch_stablecoin_mcap()
    ibit = fetch_ibit_etf()
    fng = fetch_fear_and_greed()

    payload = {
        "generated_at_kst": now_kst.strftime("%Y-%m-%d %H:%M:%S KST"),
        "disclaimer": "참고용 공개 수급 지표 — 자동매매 룰(Turbo-1/D-3/L-3/로켓)에 일절 미개입",
        "stablecoin_mcap": stable,
        "ibit_etf": ibit,
        "fear_and_greed": fng,
    }

    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Saved Crypto Weather snapshot: {OUT_PATH}")
    print(f"  ① 스테이블코인: {stable.get('label')} | ${stable.get('mcap_billion_usd', '-')}B (7d {stable.get('chg_7d_pct', '-')}% / 30d {stable.get('chg_30d_pct', '-')}%)")
    print(f"  ② 미국 BTC ETF (IBIT): {ibit.get('label')} | ${ibit.get('close_usd', '-')} (MA20 {ibit.get('gap_ma20_pct', '-')}% / 5d {ibit.get('ret_5d_pct', '-')}%)")
    print(f"  ③ 공포·탐욕 지수: {fng.get('label')} (7일 대비 {fng.get('diff_7d', '-')}p)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
