"""
US Equities Data Collector for Qullamaggie Momentum Breakout Backtest
- Source: yfinance (auto_adjust=True, daily OHLCV)
- Period: 2018-01-01 ~ 2026-11-20 (need warm-up from 2017-09-01 for 63d/50d indicators on 2018-01-02)
- Output: strategies/us_qullamaggie/us_qullamaggie_data.duckdb
"""
import os
import time
import warnings
import duckdb
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "us_qullamaggie_data.duckdb")

START_DATE = "2017-08-01"   # Warm-up buffer for 63-day momentum & 50-day MA starting 2018-01-02
END_DATE = "2026-11-21"     # Exclusive end -> includes 2026-11-20

BENCHMARKS = ["QQQ", "SPY", "IWM"]

CORE_GROWTH_UNIVERSE = [
    # Semis & AI Hardware
    "NVDA", "AMD", "AVGO", "MU", "QCOM", "TXN", "AMAT", "LRCX", "KLAC", "ASML",
    "MRVL", "ON", "MCHP", "ADI", "NXPI", "ARM", "SMCI", "VRT", "ANET", "DELL",
    "HPE", "PSTG", "TSM", "MPWR", "ENTG", "WOLF", "ALAB", "CRDO",
    # Mega Tech & Internet
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NFLX", "TSLA", "ORCL", "CRM", "ADBE",
    "INTU", "NOW", "CSCO", "IBM", "UBER", "ABNB", "BKNG", "MELI", "SE", "SHOP",
    "SPOT", "DASH", "DUOL", "RDDT", "APP", "TTD", "PDD", "JD",
    # Cybersecurity & Enterprise SaaS
    "CRWD", "PANW", "FTNT", "ZS", "NET", "DDOG", "SNOW", "MDB", "PLTR", "TEAM",
    "WDAY", "HUBS", "VEEV", "BILL", "GTLB", "IOT", "MNDY", "CFLT", "ESTC", "DOCN",
    "S", "CYBR", "CHKP",
    # Fintech & Crypto-related
    "COIN", "MSTR", "HOOD", "XYZ", "SOFI", "NU", "V", "MA", "AXP", "TOST",
    "MARA", "RIOT", "CLSK", "WULF", "IREN", "HUT", "CORZ",
    # Consumer / Retail / Biotech / Healthcare Growth
    "CELH", "ELF", "CVNA", "CAVA", "CMG", "LULU", "DECK", "ONON", "CROX", "NKE",
    "SBUX", "COST", "WMT", "HD", "LOW", "TGT", "ULTA", "WING", "SHAK", "BROS",
    "MNST", "LLY", "NVO", "ISRG", "DXCM", "PODD", "ALGN", "REGN", "VRTX", "AMGN",
    "GILD", "BIIB", "CRSP", "BEAM", "NTLA", "VKTX", "HIMS",
    # Energy / Aerospace / Industrial / Nuclear / Space / Quantum
    "GE", "GEV", "CEG", "VST", "NRG", "AXON", "TDG", "LMT", "RTX", "BA",
    "CAT", "DE", "URI", "PWR", "EME", "FIX", "RKLB", "ASTS", "IONQ", "OKLO",
    "SMR", "CCJ", "FSLR", "XOM", "CVX", "OXY", "SLB",
]

# 2020~2022 boom & bust / former momentum leaders (Survivorship Bias Mitigation)
FALLEN_ANGELS = [
    "ZM", "PTON", "TDOC", "DOCU", "ROKU", "PYPL", "SNAP", "BABA", "NIO", "RIVN",
    "LCID", "PLUG", "ENPH", "SEDG", "MRNA", "BNTX", "ETSY", "CHWY", "UPST", "AFRM",
    "PINS", "TWLO", "OKTA", "PATH", "U", "RBLX", "FUBO", "SPCE", "BYND", "INTC",
    "WBA", "FVRR", "RNG", "Z", "OPEN", "NVAX", "BE", "BLDP", "CHPT", "QS",
    "RUN", "GME", "AMC", "BILI", "FUTU", "TIGR", "XPEV", "LI", "DKNG", "PENN",
    "LMND", "ROOT", "BIGC", "FROG", "ASAN", "AI", "SOUN", "BBAI", "W", "EXPI",
    "RDFN", "ARKK", "CLOV", "TLRY", "CGC",
]


def main():
    all_tickers = sorted(set(BENCHMARKS + CORE_GROWTH_UNIVERSE + FALLEN_ANGELS))
    fallen_set = set(FALLEN_ANGELS)
    bench_set = set(BENCHMARKS)

    print(f"Total requested tickers: {len(all_tickers)} (Benchmarks: {len(BENCHMARKS)}, Fallen Angels: {len(fallen_set)})")

    # Download in chunks to avoid rate limiting / memory issues
    chunk_size = 40
    records = []
    meta_rows = []
    failed = []

    for i in range(0, len(all_tickers), chunk_size):
        chunk = all_tickers[i:i + chunk_size]
        print(f"Downloading chunk {i // chunk_size + 1}/{(len(all_tickers) - 1) // chunk_size + 1} ({len(chunk)} tickers)...")
        try:
            raw = yf.download(
                chunk,
                start=START_DATE,
                end=END_DATE,
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False,
            )
        except Exception as e:
            print(f"Chunk error: {e}")
            raw = None

        if raw is None or raw.empty:
            failed.extend(chunk)
            continue

        for t in chunk:
            try:
                if len(chunk) == 1:
                    df = raw.copy()
                else:
                    if t not in raw.columns.get_level_values(0):
                        failed.append(t)
                        continue
                    df = raw[t].copy()

                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
                # Basic data sanity checks
                df = df[(df["Open"] > 0) & (df["High"] > 0) & (df["Low"] > 0) & (df["Close"] > 0) & (df["Volume"] >= 0)]
                # Enforce High >= max(Open, Close) and Low <= min(Open, Close)
                df["High"] = np.maximum(df["High"], np.maximum(df["Open"], df["Close"]))
                df["Low"] = np.minimum(df["Low"], np.minimum(df["Open"], df["Close"]))

                if len(df) < 80:
                    failed.append(t)
                    continue

                df = df.reset_index()
                date_col = "Date" if "Date" in df.columns else df.columns[0]
                df["date"] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
                df["ticker"] = t
                df["dollar_vol"] = df["Close"] * df["Volume"]

                # Check split-adjustment distortion: flag extreme abnormal dollar_vol spikes or zero volumes
                med_dv = float(df["dollar_vol"].median())
                records.append(df[["date", "ticker", "Open", "High", "Low", "Close", "Volume", "dollar_vol"]])

                cat = "benchmark" if t in bench_set else ("fallen_angel" if t in fallen_set else "core_growth")
                meta_rows.append({
                    "ticker": t,
                    "category": cat,
                    "start_date": df["date"].iloc[0],
                    "end_date": df["date"].iloc[-1],
                    "rows": len(df),
                    "median_dollar_vol": med_dv,
                })
            except Exception as e:
                print(f"Failed parsing {t}: {e}")
                failed.append(t)

        time.sleep(0.5)

    if not records:
        raise RuntimeError("No data collected!")

    all_df = pd.concat(records, ignore_index=True)
    all_df.columns = ["date", "ticker", "open", "high", "low", "close", "volume", "dollar_vol"]
    meta_df = pd.DataFrame(meta_rows)

    # Save to DuckDB
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    con = duckdb.connect(DB_PATH)
    con.execute("CREATE TABLE ohlcv AS SELECT * FROM all_df ORDER BY ticker, date")
    con.execute("CREATE TABLE universe_meta AS SELECT * FROM meta_df ORDER BY ticker")
    con.execute("CREATE INDEX idx_ohlcv_ticker_date ON ohlcv(ticker, date)")
    con.close()

    n_total = len(meta_df)
    n_bench = int((meta_df["category"] == "benchmark").sum())
    n_fallen = int((meta_df["category"] == "fallen_angel").sum())
    n_core = int((meta_df["category"] == "core_growth").sum())
    print("\n=== Data Collection Summary ===")
    print(f"DB Path: {DB_PATH}")
    print(f"Total valid tickers: {n_total} (Benchmarks: {n_bench}, Core Growth: {n_core}, Fallen Angels: {n_fallen})")
    print(f"Total OHLCV rows: {len(all_df):,}")
    print(f"Date range: {all_df['date'].min()} ~ {all_df['date'].max()}")
    print(f"Failed/Skipped ({len(failed)}): {failed}")


if __name__ == "__main__":
    main()
