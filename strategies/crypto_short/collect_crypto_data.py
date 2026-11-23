"""
Invest_God Phase 1-B: 가상자산 4시간봉(4h) 실데이터 수집 및 DuckDB 캐싱 스크립트
- 데이터 출처: Binance Spot 공개 REST API (https://data-api.binance.vision/api/v3/klines)
- 대상 유니버스: 유동성 상위 10개 주요 코인 (USDT 현물 마켓)
  BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, XRPUSDT, ADAUSDT, DOGEUSDT, AVAXUSDT, LINKUSDT, LTCUSDT
- 수집 기간: 2020-11-01 00:00 UTC ~ 2026-11-21 20:00 UTC (워밍업 2개월 포함, 본 백테스트는 2021-01-01 ~ 2026-11-21)
- 저장 경로: strategies/crypto_short/crypto_short_data.duckdb
"""

import datetime
import time
from pathlib import Path
from typing import List, Tuple

import duckdb
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "crypto_short_data.duckdb"

API_URL = "https://data-api.binance.vision/api/v3/klines"
INTERVAL = "4h"
SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "LTCUSDT",
]

# 2020-11-01 00:00:00 UTC ~ 2026-11-21 20:00:00 UTC (마지막 완성 4h 봉)
START_UTC = datetime.datetime(2020, 11, 1, 0, 0, tzinfo=datetime.timezone.utc)
END_UTC = datetime.datetime(2026, 11, 21, 20, 0, tzinfo=datetime.timezone.utc)


def fetch_symbol_klines(symbol: str, start_ms: int, end_ms: int) -> List[Tuple]:
    rows = []
    cur_ms = start_ms
    session = requests.Session()
    while cur_ms <= end_ms:
        params = {
            "symbol": symbol,
            "interval": INTERVAL,
            "startTime": cur_ms,
            "endTime": end_ms,
            "limit": 1000,
        }
        for attempt in range(4):
            try:
                resp = session.get(API_URL, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as exc:
                if attempt == 3:
                    raise RuntimeError(f"Failed fetching {symbol} at {cur_ms}: {exc}") from exc
                time.sleep(1.5 * (attempt + 1))

        if not data:
            break

        for k in data:
            open_ms = int(k[0])
            if open_ms > end_ms:
                break
            dt_utc = datetime.datetime.fromtimestamp(open_ms / 1000.0, tz=datetime.timezone.utc).replace(tzinfo=None)
            rows.append(
                (
                    symbol,
                    dt_utc,
                    float(k[1]),  # open
                    float(k[2]),  # high
                    float(k[3]),  # low
                    float(k[4]),  # close
                    float(k[5]),  # volume (base)
                    float(k[7]),  # quote_volume (USDT)
                )
            )

        last_open_ms = int(data[-1][0])
        if last_open_ms <= cur_ms:
            break
        cur_ms = last_open_ms + 4 * 3600 * 1000
        if len(data) < 1000:
            break
        time.sleep(0.08)

    return rows


def main() -> None:
    start_ms = int(START_UTC.timestamp() * 1000)
    end_ms = int(END_UTC.timestamp() * 1000)

    expected_bars = int((end_ms - start_ms) / (4 * 3600 * 1000)) + 1
    print(f"[INFO] Collecting Binance {INTERVAL} OHLCV ({START_UTC.strftime('%Y-%m-%d')} ~ {END_UTC.strftime('%Y-%m-%d %H:%M')} UTC)")
    print(f"[INFO] Expected theoretical 4h bars per symbol: {expected_bars}")

    if DB_PATH.exists():
        DB_PATH.unlink()

    con = duckdb.connect(str(DB_PATH))
    con.execute(
        """
        CREATE TABLE ohlcv_4h (
            symbol VARCHAR,
            open_time_utc TIMESTAMP,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            quote_volume DOUBLE,
            PRIMARY KEY (symbol, open_time_utc)
        )
        """
    )
    con.execute(
        """
        CREATE TABLE data_meta (
            symbol VARCHAR PRIMARY KEY,
            source VARCHAR,
            interval VARCHAR,
            first_bar_utc TIMESTAMP,
            last_bar_utc TIMESTAMP,
            bar_count INTEGER,
            missing_bars INTEGER
        )
        """
    )

    for sym in SYMBOLS:
        t0 = time.time()
        rows = fetch_symbol_klines(sym, start_ms, end_ms)
        df = pd.DataFrame(
            rows,
            columns=[
                "symbol",
                "open_time_utc",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quote_volume",
            ],
        ).drop_duplicates(subset=["symbol", "open_time_utc"])
        df = df.sort_values("open_time_utc").reset_index(drop=True)

        con.register("tmp_df", df)
        con.execute("INSERT INTO ohlcv_4h SELECT * FROM tmp_df")
        con.unregister("tmp_df")

        first_bar = df["open_time_utc"].iloc[0]
        last_bar = df["open_time_utc"].iloc[-1]
        bar_count = len(df)
        missing_bars = expected_bars - bar_count

        con.execute(
            "INSERT INTO data_meta VALUES (?, ?, ?, ?, ?, ?, ?)",
            [sym, API_URL, INTERVAL, first_bar, last_bar, bar_count, missing_bars],
        )
        print(
            f"  - {sym:8s}: {bar_count:5d} bars | {first_bar} ~ {last_bar} | "
            f"missing={missing_bars:2d} | {time.time() - t0:.1f}s"
        )

    total_rows = con.execute("SELECT COUNT(*) FROM ohlcv_4h").fetchone()[0]
    con.close()
    print(f"[DONE] Saved {total_rows:,} rows to {DB_PATH}")


if __name__ == "__main__":
    main()
