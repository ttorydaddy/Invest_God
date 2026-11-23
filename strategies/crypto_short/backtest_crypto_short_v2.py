"""
Invest_God Phase 1-B-2: 가상자산 단기 버킷 C-2 개량 4종(M-1~M-4) 비교 백테스트 엔진
- 데이터: 기존 strategies/crypto_short/crypto_short_data.duckdb (Binance 4h, 10종목, 132,720행, 결측 0건)
- 기간: Full 2021-01-01 ~ 2026-11-21 | IS 2021-01-01 ~ 2024-06-30 (60%) | OOS 2024-07-01 ~ 2026-11-21 (40%)
- 공통 불변 제약:
  * 왕복 거래비용 0.36% (편도 0.18% = 수수료 0.10% + 급변동 슬리피지 0.08%)
  * 레버리지 0% (현물 1배, cash >= 0, 총 비중 <= 100%, 단일 코인 진입 비중 <= 24.5%)
  * 보수적 체결: t-1 종가 확정 신호 -> t 시가 진입, 동시 터치 시 손절 무조건 우선, 갭하락 시 min(open, stop)
  * OOS 주가 +500% 강제 교란 시 IS 오차 0.00000000 증명
- 비교 모델 4종:
  * M-1: 대안① — -3.0% 타이트 손절 고수 + 대형 3종목(BTC/ETH/BNB) 압축 + 48봉 돈치안 저회전
  * M-2: 대안② — 터틀 2xATR 변동성 손절 + 1회 계좌 리스크 -0.7% 고정 포지션 사이징 + 3xATR 트레일링 (4h, 10종목)
  * M-3: 대안② 강화 — M-2 + BTC 100일(600봉) EMA 레짐 필터 + 계좌 HWM -15% 시 리스크 절반 감속(-0.35%)
  * M-4: 참고 비교 — 일봉(1d) 리샘플링 초저회전 터틀 추세추종 (20일 돌파, 2xATR 손절, 1회 리스크 -1.0%, BTC 100일 EMA)
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import duckdb
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "crypto_short_data.duckdb"
RESULTS_JSON_PATH = BASE_DIR / "backtest_crypto_v2_results.json"

ALL_SYMBOLS = [
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
MAJOR_3_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

EVAL_START = pd.Timestamp("2021-01-01 00:00:00")
IS_END = pd.Timestamp("2024-06-30 20:00:00")
OOS_START = pd.Timestamp("2024-07-01 00:00:00")
EVAL_END = pd.Timestamp("2026-11-21 20:00:00")

EVAL_START_1D = pd.Timestamp("2021-01-01")
IS_END_1D = pd.Timestamp("2024-06-30")
OOS_START_1D = pd.Timestamp("2024-07-01")
EVAL_END_1D = pd.Timestamp("2026-11-21")

INITIAL_CAPITAL = 100_000.0
SINGLE_ENTRY_WEIGHT_CAP = 0.245  # strictly <= 25%
NET_SIDE_COST = 0.0018           # 0.18% per side => 0.36% round-trip
ROUND_TRIP_COST = NET_SIDE_COST * 2.0


@dataclass
class ModelConfig:
    code: str
    name: str
    timeframe: str             # "4h" or "1d"
    symbols: List[str]
    max_positions: int
    sizing_mode: str           # "fixed_weight" or "turtle_risk"
    fixed_stop_pct: float      # used if sizing_mode == "fixed_weight"
    atr_stop_mult: float       # used if sizing_mode == "turtle_risk"
    atr_trail_mult: float
    risk_per_trade: float      # target equity risk per trade (e.g. 0.007)
    use_btc_ema600: bool       # 100d EMA regime filter
    use_dd_throttle: bool      # HWM -15% -> half risk
    donch_entry: int
    donch_exit: int
    vol_mult: float


MODELS = [
    ModelConfig(
        code="M1_Major3_Tight3pct",
        name="M-1: Major 3 (BTC/ETH/BNB) + 48-Bar Donchian + Tight -3.0% Stop (4h)",
        timeframe="4h",
        symbols=MAJOR_3_SYMBOLS,
        max_positions=3,
        sizing_mode="fixed_weight",
        fixed_stop_pct=0.030,
        atr_stop_mult=0.0,
        atr_trail_mult=2.0,
        risk_per_trade=0.0075,
        use_btc_ema600=False,
        use_dd_throttle=False,
        donch_entry=48,
        donch_exit=24,
        vol_mult=1.2,
    ),
    ModelConfig(
        code="M2_Turtle_2ATR_Risk07",
        name="M-2: 10-Coin Turtle 2xATR Stop + Fixed -0.7% Account Risk + 3xATR Trail (4h)",
        timeframe="4h",
        symbols=ALL_SYMBOLS,
        max_positions=4,
        sizing_mode="turtle_risk",
        fixed_stop_pct=0.0,
        atr_stop_mult=2.0,
        atr_trail_mult=3.0,
        risk_per_trade=0.007,
        use_btc_ema600=False,
        use_dd_throttle=False,
        donch_entry=48,
        donch_exit=24,
        vol_mult=1.3,
    ),
    ModelConfig(
        code="M3_Turtle_Regime_DDThrottle",
        name="M-3: M-2 + BTC 100d(600-bar) EMA Regime + HWM -15% Risk Halving (4h)",
        timeframe="4h",
        symbols=ALL_SYMBOLS,
        max_positions=4,
        sizing_mode="turtle_risk",
        fixed_stop_pct=0.0,
        atr_stop_mult=2.0,
        atr_trail_mult=3.0,
        risk_per_trade=0.007,
        use_btc_ema600=True,
        use_dd_throttle=True,
        donch_entry=48,
        donch_exit=24,
        vol_mult=1.3,
    ),
    ModelConfig(
        code="M4_Daily1D_Turtle_LowTurn",
        name="M-4: 1D Resampled Ultra-Low-Turnover Turtle (20d Donchian, 2xATR, -1.0% Risk)",
        timeframe="1d",
        symbols=ALL_SYMBOLS,
        max_positions=4,
        sizing_mode="turtle_risk",
        fixed_stop_pct=0.0,
        atr_stop_mult=2.0,
        atr_trail_mult=2.5,
        risk_per_trade=0.010,
        use_btc_ema600=True,  # on 1d, EMA(100) = 100-day EMA
        use_dd_throttle=False,
        donch_entry=20,
        donch_exit=10,
        vol_mult=1.1,
    ),
]


def load_data_4h(perturb_oos_mult: float = 1.0) -> Tuple[pd.DatetimeIndex, Dict[str, Dict[str, np.ndarray]]]:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df_all = con.execute(
        """
        SELECT symbol, open_time_utc, open, high, low, close, volume
        FROM ohlcv_4h
        ORDER BY open_time_utc, symbol
        """
    ).fetchdf()
    con.close()

    df_all["open_time_utc"] = pd.to_datetime(df_all["open_time_utc"])
    timeline = pd.date_range(start=df_all["open_time_utc"].min(), end=EVAL_END, freq="4h")
    oos_mask = np.asarray(timeline >= OOS_START, dtype=bool)

    data: Dict[str, Dict[str, np.ndarray]] = {}
    for sym in ALL_SYMBOLS:
        sdf = df_all[df_all["symbol"] == sym].set_index("open_time_utc").reindex(timeline)
        tradable = sdf["close"].notna().to_numpy(dtype=bool)

        o = sdf["open"].to_numpy(dtype=np.float64, copy=True)
        h = sdf["high"].to_numpy(dtype=np.float64, copy=True)
        l = sdf["low"].to_numpy(dtype=np.float64, copy=True)
        c = sdf["close"].to_numpy(dtype=np.float64, copy=True)
        v = sdf["volume"].to_numpy(dtype=np.float64, copy=True)

        if perturb_oos_mult != 1.0:
            o[oos_mask] *= perturb_oos_mult
            h[oos_mask] *= perturb_oos_mult
            l[oos_mask] *= perturb_oos_mult
            c[oos_mask] *= perturb_oos_mult

        c_ffill = pd.Series(c).ffill().to_numpy(dtype=np.float64)
        o_ffill = np.where(np.isnan(o), c_ffill, o)

        s_c = pd.Series(c_ffill)
        s_h = pd.Series(np.where(np.isnan(h), c_ffill, h))
        s_l = pd.Series(np.where(np.isnan(l), c_ffill, l))
        s_v = pd.Series(np.where(np.isnan(v), 0.0, v))

        prev_c = s_c.shift(1)
        tr = pd.concat([(s_h - s_l), (s_h - prev_c).abs(), (s_l - prev_c).abs()], axis=1).max(axis=1)
        atr14 = tr.ewm(alpha=1.0 / 14.0, adjust=False).mean().to_numpy()

        data[sym] = {
            "tradable": tradable,
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": v,
            "close_ffill": c_ffill,
            "open_ffill": o_ffill,
            "ema50": s_c.ewm(span=50, adjust=False).mean().to_numpy(),
            "ema200": s_c.ewm(span=200, adjust=False).mean().to_numpy(),
            "ema600": s_c.ewm(span=600, adjust=False).mean().to_numpy(),
            "atr14": atr14,
            "donch_h48": s_h.rolling(48).max().to_numpy(),
            "donch_l24": s_l.rolling(24).min().to_numpy(),
            "vol_sma20": s_v.rolling(20).mean().to_numpy(),
        }
    return timeline, data


def load_data_1d(perturb_oos_mult: float = 1.0) -> Tuple[pd.DatetimeIndex, Dict[str, Dict[str, np.ndarray]]]:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df_all = con.execute(
        """
        SELECT symbol, open_time_utc, open, high, low, close, volume
        FROM ohlcv_4h
        ORDER BY open_time_utc, symbol
        """
    ).fetchdf()
    con.close()

    df_all["open_time_utc"] = pd.to_datetime(df_all["open_time_utc"])
    df_all["date"] = df_all["open_time_utc"].dt.floor("D")

    daily_agg = (
        df_all.groupby(["symbol", "date"], as_index=False)
        .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum"))
        .sort_values(["date", "symbol"])
    )

    timeline = pd.date_range(start=daily_agg["date"].min(), end=EVAL_END_1D, freq="D")
    oos_mask = np.asarray(timeline >= OOS_START_1D, dtype=bool)

    data: Dict[str, Dict[str, np.ndarray]] = {}
    for sym in ALL_SYMBOLS:
        sdf = daily_agg[daily_agg["symbol"] == sym].set_index("date").reindex(timeline)
        tradable = sdf["close"].notna().to_numpy(dtype=bool)

        o = sdf["open"].to_numpy(dtype=np.float64, copy=True)
        h = sdf["high"].to_numpy(dtype=np.float64, copy=True)
        l = sdf["low"].to_numpy(dtype=np.float64, copy=True)
        c = sdf["close"].to_numpy(dtype=np.float64, copy=True)
        v = sdf["volume"].to_numpy(dtype=np.float64, copy=True)

        if perturb_oos_mult != 1.0:
            o[oos_mask] *= perturb_oos_mult
            h[oos_mask] *= perturb_oos_mult
            l[oos_mask] *= perturb_oos_mult
            c[oos_mask] *= perturb_oos_mult

        c_ffill = pd.Series(c).ffill().to_numpy(dtype=np.float64)
        o_ffill = np.where(np.isnan(o), c_ffill, o)

        s_c = pd.Series(c_ffill)
        s_h = pd.Series(np.where(np.isnan(h), c_ffill, h))
        s_l = pd.Series(np.where(np.isnan(l), c_ffill, l))
        s_v = pd.Series(np.where(np.isnan(v), 0.0, v))

        prev_c = s_c.shift(1)
        tr = pd.concat([(s_h - s_l), (s_h - prev_c).abs(), (s_l - prev_c).abs()], axis=1).max(axis=1)
        atr14 = tr.ewm(alpha=1.0 / 14.0, adjust=False).mean().to_numpy()

        data[sym] = {
            "tradable": tradable,
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": v,
            "close_ffill": c_ffill,
            "open_ffill": o_ffill,
            "ema50": s_c.ewm(span=50, adjust=False).mean().to_numpy(),
            "ema100": s_c.ewm(span=100, adjust=False).mean().to_numpy(),
            "ema200": s_c.ewm(span=200, adjust=False).mean().to_numpy(),
            "atr14": atr14,
            "donch_h20": s_h.rolling(20).max().to_numpy(),
            "donch_l10": s_l.rolling(10).min().to_numpy(),
            "vol_sma20": s_v.rolling(20).mean().to_numpy(),
        }
    return timeline, data


def run_model(
    cfg: ModelConfig,
    timeline: pd.DatetimeIndex,
    data: Dict[str, Dict[str, np.ndarray]],
    side_cost: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    is_1d = cfg.timeframe == "1d"
    eval_start = EVAL_START_1D if is_1d else EVAL_START
    is_end = IS_END_1D if is_1d else IS_END

    start_idx = int(np.searchsorted(timeline, eval_start))
    end_idx = int(len(timeline) - 1)

    cash = INITIAL_CAPITAL
    hwm_equity = INITIAL_CAPITAL
    positions: Dict[str, Dict] = {}
    trades: List[Dict] = []
    daily_rows: List[Dict] = []

    min_observed_cash = INITIAL_CAPITAL
    min_observed_equity = INITIAL_CAPITAL
    max_observed_total_weight = 0.0
    max_observed_single_entry_weight = 0.0
    max_observed_positions = 0
    dd_throttle_bars = 0

    btc = data["BTCUSDT"]
    btc_start_price = btc["close_ffill"][start_idx]

    donch_h_key = "donch_h20" if is_1d else "donch_h48"
    donch_l_key = "donch_l10" if is_1d else "donch_l24"

    for t in range(start_idx, end_idx + 1):
        ts = timeline[t]
        prev = t - 1
        prev2 = t - 2

        # ------------------------------------------------------------------
        # STEP A: Open of bar t — Channel Exit (signal from bar t-1 close)
        # ------------------------------------------------------------------
        for sym in list(positions.keys()):
            pos = positions[sym]
            sd = data[sym]
            if not sd["tradable"][t]:
                continue

            if sd["close_ffill"][prev] < sd[donch_l_key][prev2]:
                raw_exit = sd["open"][t]
                exec_exit = raw_exit * (1.0 - side_cost)
                proceeds = pos["qty"] * exec_exit
                net_pnl = proceeds - pos["cost_basis"]
                net_ret = net_pnl / pos["cost_basis"]
                acct_impact = net_pnl / pos["entry_equity"]
                cash += proceeds

                trades.append(
                    {
                        "symbol": sym,
                        "entry_time_utc": str(pos["entry_time"]),
                        "exit_time_utc": str(ts),
                        "raw_entry_price": round(pos["raw_entry_price"], 6),
                        "exec_entry_price": round(pos["exec_entry_price"], 6),
                        "raw_exit_price": round(raw_exit, 6),
                        "exec_exit_price": round(exec_exit, 6),
                        "entry_weight": round(pos["entry_weight"], 5),
                        "stop_dist_pct": round(pos["stop_dist_pct"], 5),
                        "entry_notional": round(pos["cost_basis"], 2),
                        "exit_proceeds": round(proceeds, 2),
                        "net_pnl": round(net_pnl, 2),
                        "net_return": round(net_ret, 6),
                        "account_impact_pct": round(acct_impact, 6),
                        "bars_held": pos["bars_held"],
                        "exit_reason": "CHANNEL_EXIT",
                        "period": "IS" if pos["entry_time"] <= is_end else "OOS",
                    }
                )
                del positions[sym]

        # ------------------------------------------------------------------
        # STEP B: Open of bar t — Regime Filter, DD Throttle & New Entries
        # ------------------------------------------------------------------
        eq_open = cash + sum(pos["qty"] * data[sym]["open_ffill"][t] for sym, pos in positions.items())
        cur_dd = eq_open / hwm_equity - 1.0 if hwm_equity > 0 else 0.0
        throttle_on = cfg.use_dd_throttle and (cur_dd <= -0.15)
        if throttle_on:
            dd_throttle_bars += 1

        eff_risk = (cfg.risk_per_trade * 0.5) if throttle_on else cfg.risk_per_trade
        eff_weight_cap = (SINGLE_ENTRY_WEIGHT_CAP * 0.5) if throttle_on else SINGLE_ENTRY_WEIGHT_CAP

        avail_slots = cfg.max_positions - len(positions)
        if avail_slots > 0:
            btc_c_p = btc["close_ffill"][prev]
            if is_1d:
                btc_ok = btc_c_p > btc["ema100"][prev]
            else:
                btc_ok = btc_c_p > btc["ema200"][prev]
                if cfg.use_btc_ema600:
                    btc_ok = btc_ok and (btc_c_p > btc["ema600"][prev])

            if btc_ok:
                candidates: List[Tuple[float, str, float]] = []
                for sym in cfg.symbols:
                    if sym in positions:
                        continue
                    sd = data[sym]
                    if not (sd["tradable"][t] and sd["tradable"][prev]):
                        continue

                    c_p = sd["close"][prev]
                    v_p = sd["volume"][prev]
                    vsma_p = sd["vol_sma20"][prev]
                    ema50_p = sd["ema50"][prev]
                    ema200_p = sd["ema200"][prev]
                    atr_p = sd["atr14"][prev]
                    dh_p2 = sd[donch_h_key][prev2]

                    if any(math.isnan(x) for x in [c_p, v_p, vsma_p, ema50_p, ema200_p, atr_p, dh_p2]):
                        continue
                    if vsma_p <= 0 or atr_p <= 0 or c_p <= 0:
                        continue

                    trend_ok = c_p > ema50_p
                    if cfg.code == "M1_Major3_Tight3pct":
                        trend_ok = trend_ok and (ema50_p > ema200_p)

                    if trend_ok and (c_p > dh_p2) and (v_p > cfg.vol_mult * vsma_p):
                        score = v_p / vsma_p
                        atr_pct = atr_p / c_p
                        candidates.append((score, sym, atr_pct))

                if candidates:
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    for _, sym, atr_pct in candidates[:avail_slots]:
                        if cfg.sizing_mode == "fixed_weight":
                            stop_dist_pct = cfg.fixed_stop_pct
                            w = eff_weight_cap
                        else:
                            min_stop = 0.03 if is_1d else 0.02
                            stop_dist_pct = max(min_stop, cfg.atr_stop_mult * atr_pct)
                            w = min(eff_weight_cap, eff_risk / stop_dist_pct)

                        target_notional = min(w * eq_open, cash)
                        if target_notional < 100.0:
                            continue

                        raw_entry = data[sym]["open"][t]
                        exec_entry = raw_entry * (1.0 + side_cost)
                        qty = target_notional / exec_entry
                        cash -= target_notional
                        if cash < 0.0 and cash > -1e-6:
                            cash = 0.0

                        entry_w = target_notional / eq_open
                        if entry_w > max_observed_single_entry_weight:
                            max_observed_single_entry_weight = entry_w

                        hard_stop = raw_entry * (1.0 - stop_dist_pct)
                        positions[sym] = {
                            "entry_time": ts,
                            "entry_equity": eq_open,
                            "entry_weight": entry_w,
                            "raw_entry_price": raw_entry,
                            "exec_entry_price": exec_entry,
                            "qty": qty,
                            "cost_basis": target_notional,
                            "stop_dist_pct": stop_dist_pct,
                            "entry_atr_pct": atr_pct,
                            "hard_stop": hard_stop,
                            "trail_stop": 0.0,
                            "highest_close": raw_entry,
                            "bars_held": 0,
                        }

        # ------------------------------------------------------------------
        # STEP C: Intra-bar t — Conservative Hard / Trailing Stop Check
        # ------------------------------------------------------------------
        for sym in list(positions.keys()):
            pos = positions[sym]
            sd = data[sym]
            if not sd["tradable"][t]:
                continue

            bar_open = sd["open"][t]
            bar_low = sd["low"][t]
            effective_stop = max(pos["hard_stop"], pos["trail_stop"])

            if bar_low <= effective_stop:
                raw_exit = min(bar_open, effective_stop)
                exit_reason = "HARD_STOP" if effective_stop <= pos["hard_stop"] + 1e-12 else "TRAIL_STOP"
                exec_exit = raw_exit * (1.0 - side_cost)
                proceeds = pos["qty"] * exec_exit
                net_pnl = proceeds - pos["cost_basis"]
                net_ret = net_pnl / pos["cost_basis"]
                acct_impact = net_pnl / pos["entry_equity"]
                cash += proceeds

                trades.append(
                    {
                        "symbol": sym,
                        "entry_time_utc": str(pos["entry_time"]),
                        "exit_time_utc": str(ts),
                        "raw_entry_price": round(pos["raw_entry_price"], 6),
                        "exec_entry_price": round(pos["exec_entry_price"], 6),
                        "raw_exit_price": round(raw_exit, 6),
                        "exec_exit_price": round(exec_exit, 6),
                        "entry_weight": round(pos["entry_weight"], 5),
                        "stop_dist_pct": round(pos["stop_dist_pct"], 5),
                        "entry_notional": round(pos["cost_basis"], 2),
                        "exit_proceeds": round(proceeds, 2),
                        "net_pnl": round(net_pnl, 2),
                        "net_return": round(net_ret, 6),
                        "account_impact_pct": round(acct_impact, 6),
                        "bars_held": pos["bars_held"] + 1,
                        "exit_reason": exit_reason,
                        "period": "IS" if pos["entry_time"] <= is_end else "OOS",
                    }
                )
                del positions[sym]

        # ------------------------------------------------------------------
        # STEP D: Close of bar t — Update Trailing Stop, HWM & Mark-to-Market
        # ------------------------------------------------------------------
        invested_val = 0.0
        max_pos_val = 0.0
        for sym, pos in positions.items():
            sd = data[sym]
            c_t = sd["close_ffill"][t]
            pos["bars_held"] += 1
            if c_t > pos["highest_close"]:
                pos["highest_close"] = c_t

            # Activate trailing stop only after meaningful favorable move (causal close update)
            trigger_mult = 1.03 if cfg.sizing_mode == "fixed_weight" else (1.0 + 1.5 * pos["entry_atr_pct"])
            if pos["highest_close"] >= pos["raw_entry_price"] * trigger_mult:
                atr_t = sd["atr14"][t]
                if not math.isnan(atr_t) and atr_t > 0:
                    cand_trail = pos["highest_close"] - cfg.atr_trail_mult * atr_t
                    pos["trail_stop"] = max(pos["trail_stop"], pos["hard_stop"], cand_trail)

            val = pos["qty"] * c_t
            invested_val += val
            if val > max_pos_val:
                max_pos_val = val

        equity = cash + invested_val
        if equity > hwm_equity:
            hwm_equity = equity

        inv_weight = invested_val / equity if equity > 0 else 0.0
        max_single_w = max_pos_val / equity if equity > 0 else 0.0

        if cash < min_observed_cash:
            min_observed_cash = cash
        if equity < min_observed_equity:
            min_observed_equity = equity
        if inv_weight > max_observed_total_weight:
            max_observed_total_weight = inv_weight
        if len(positions) > max_observed_positions:
            max_observed_positions = len(positions)

        if is_1d or (ts.hour == 20):
            btc_eq = INITIAL_CAPITAL * (btc["close_ffill"][t] / btc_start_price)
            daily_rows.append(
                {
                    "date": str(ts.date()),
                    "equity": round(equity, 4),
                    "cash": round(cash, 4),
                    "invested_value": round(invested_val, 4),
                    "invested_weight": round(inv_weight, 6),
                    "max_single_weight": round(max_single_w, 6),
                    "num_positions": int(len(positions)),
                    "throttle_active": int(throttle_on),
                    "btc_bh_equity": round(btc_eq, 4),
                }
            )

    daily_df = pd.DataFrame(daily_rows)
    trades_df = pd.DataFrame(trades)
    worst_trade_acct_loss = float(trades_df["account_impact_pct"].min()) if len(trades_df) > 0 else 0.0

    audit = {
        "timeframe": cfg.timeframe,
        "round_trip_cost": side_cost * 2.0,
        "max_positions_cap": cfg.max_positions,
        "single_entry_weight_cap": SINGLE_ENTRY_WEIGHT_CAP,
        "min_observed_cash": round(float(min_observed_cash), 6),
        "min_observed_equity": round(float(min_observed_equity), 6),
        "max_observed_total_weight": round(float(max_observed_total_weight), 6),
        "max_observed_single_entry_weight": round(float(max_observed_single_entry_weight), 6),
        "max_observed_positions": int(max_observed_positions),
        "worst_single_trade_account_pct": round(worst_trade_acct_loss, 6),
        "dd_throttle_bars": int(dd_throttle_bars),
    }
    return daily_df, trades_df, audit


def compute_metrics(daily_df: pd.DataFrame, trades_df: pd.DataFrame, col: str = "equity") -> Dict:
    if len(daily_df) < 2:
        return {}
    eq = daily_df[col].to_numpy(dtype=np.float64)
    dates = pd.to_datetime(daily_df["date"])
    total_return = float(eq[-1] / eq[0] - 1.0)
    n_days = max(1, (dates.iloc[-1] - dates.iloc[0]).days)
    cagr = float((eq[-1] / eq[0]) ** (365.0 / n_days) - 1.0) if eq[-1] > 0 else -1.0

    daily_ret = pd.Series(eq).pct_change().dropna().to_numpy()
    mean_r = float(np.mean(daily_ret))
    std_r = float(np.std(daily_ret, ddof=1)) if len(daily_ret) > 1 else 0.0
    sharpe = float((mean_r / std_r) * math.sqrt(365.0)) if std_r > 1e-12 else 0.0

    down = daily_ret[daily_ret < 0.0]
    down_std = float(np.std(down, ddof=1)) if len(down) > 1 else 0.0
    sortino = float((mean_r / down_std) * math.sqrt(365.0)) if down_std > 1e-12 else 0.0

    running_max = np.maximum.accumulate(eq)
    dd = eq / running_max - 1.0
    mdd = float(np.min(dd))
    calmar = float(cagr / abs(mdd)) if abs(mdd) > 1e-12 else 0.0

    res = {
        "start_date": str(dates.iloc[0].date()),
        "end_date": str(dates.iloc[-1].date()),
        "calendar_days": int(n_days),
        "start_equity": round(float(eq[0]), 2),
        "end_equity": round(float(eq[-1]), 2),
        "krw_1eok_final_eok": round(float(eq[-1] / eq[0]), 4),
        "total_return": round(total_return, 6),
        "cagr": round(cagr, 6),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "mdd": round(mdd, 6),
        "calmar": round(calmar, 4),
    }

    if trades_df is not None and len(trades_df) > 0:
        wins = trades_df[trades_df["net_return"] > 0]
        losses = trades_df[trades_df["net_return"] <= 0]
        gp = float(wins["net_pnl"].sum())
        gl = abs(float(losses["net_pnl"].sum()))
        pf = float(gp / gl) if gl > 1e-9 else (99.0 if gp > 0 else 0.0)
        sl_hits = int((trades_df["exit_reason"] == "HARD_STOP").sum())
        res.update(
            {
                "num_trades": int(len(trades_df)),
                "win_rate": round(float(len(wins) / len(trades_df)), 4),
                "profit_factor": round(pf, 4),
                "avg_loss_account_pct": round(float(losses["account_impact_pct"].mean()), 6) if len(losses) > 0 else 0.0,
                "max_loss_account_pct": round(float(trades_df["account_impact_pct"].min()), 6),
                "avg_win_account_pct": round(float(wins["account_impact_pct"].mean()), 6) if len(wins) > 0 else 0.0,
                "avg_entry_weight": round(float(trades_df["entry_weight"].mean()), 4),
                "hard_stop_rate": round(float(sl_hits / len(trades_df)), 4),
            }
        )
    elif trades_df is not None:
        res.update(
            {
                "num_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_loss_account_pct": 0.0,
                "max_loss_account_pct": 0.0,
                "avg_win_account_pct": 0.0,
                "avg_entry_weight": 0.0,
                "hard_stop_rate": 0.0,
            }
        )
    return res


def compute_yearly_returns(daily_df: pd.DataFrame, col: str = "equity") -> Dict[str, float]:
    df = daily_df.copy()
    df["dt"] = pd.to_datetime(df["date"])
    df["year"] = df["dt"].dt.year
    yearly = {}
    prev_end = float(df[col].iloc[0])
    for yr, grp in df.groupby("year"):
        end_val = float(grp[col].iloc[-1])
        yearly[str(yr)] = round(end_val / prev_end - 1.0, 6)
        prev_end = end_val
    return yearly


def split_metrics(daily_df: pd.DataFrame, trades_df: pd.DataFrame, col: str = "equity") -> Dict:
    dates = pd.to_datetime(daily_df["date"])
    is_df = daily_df[dates <= pd.Timestamp("2024-06-30")].reset_index(drop=True)
    oos_df = daily_df[dates >= pd.Timestamp("2024-07-01")].reset_index(drop=True)

    if trades_df is not None and len(trades_df) > 0:
        is_tr = trades_df[trades_df["period"] == "IS"].reset_index(drop=True)
        oos_tr = trades_df[trades_df["period"] == "OOS"].reset_index(drop=True)
    else:
        is_tr = trades_df
        oos_tr = trades_df

    return {
        "full": compute_metrics(daily_df, trades_df, col=col),
        "is": compute_metrics(is_df, is_tr, col=col),
        "oos": compute_metrics(oos_df, oos_tr, col=col),
        "yearly_returns": compute_yearly_returns(daily_df, col=col),
    }


def main() -> None:
    print("[INFO] Loading 4h & 1d resampled Binance Spot data from DuckDB...")
    tl_4h, data_4h = load_data_4h(perturb_oos_mult=1.0)
    _, data_4h_pert = load_data_4h(perturb_oos_mult=6.0)

    tl_1d, data_1d = load_data_1d(perturb_oos_mult=1.0)
    _, data_1d_pert = load_data_1d(perturb_oos_mult=6.0)

    results = {
        "meta": {
            "db_path": str(DB_PATH),
            "eval_period": "2021-01-01 ~ 2026-11-21",
            "is_period": "2021-01-01 ~ 2024-06-30",
            "oos_period": "2024-07-01 ~ 2026-11-21",
            "net_round_trip_cost": ROUND_TRIP_COST,
            "single_entry_weight_cap": SINGLE_ENTRY_WEIGHT_CAP,
        },
        "benchmark_btc_bh": {},
        "models": {},
    }

    btc_done = False

    for cfg in MODELS:
        print(f"\n[RUN] {cfg.name}")
        tl = tl_1d if cfg.timeframe == "1d" else tl_4h
        dt = data_1d if cfg.timeframe == "1d" else data_4h
        dt_pert = data_1d_pert if cfg.timeframe == "1d" else data_4h_pert

        daily_net, trades_net, audit_net = run_model(cfg, tl, dt, side_cost=NET_SIDE_COST)
        daily_gross, trades_gross, audit_gross = run_model(cfg, tl, dt, side_cost=0.0)
        daily_pert, _, _ = run_model(cfg, tl, dt_pert, side_cost=NET_SIDE_COST)

        is_mask = pd.to_datetime(daily_net["date"]) <= pd.Timestamp("2024-06-30")
        la_max_diff = float(
            np.max(np.abs(daily_net.loc[is_mask, "equity"].to_numpy() - daily_pert.loc[is_mask, "equity"].to_numpy()))
        )

        eq_path = BASE_DIR / f"equity_crypto_v2_{cfg.code}_net.csv"
        tr_path = BASE_DIR / f"trades_crypto_v2_{cfg.code}_net.csv"
        daily_net.to_csv(eq_path, index=False)
        trades_net.to_csv(tr_path, index=False)

        if not btc_done:
            results["benchmark_btc_bh"] = split_metrics(daily_net, None, col="btc_bh_equity")
            btc_done = True

        net_m = split_metrics(daily_net, trades_net, col="equity")
        gross_m = split_metrics(daily_gross, trades_gross, col="equity")

        results["models"][cfg.code] = {
            "name": cfg.name,
            "timeframe": cfg.timeframe,
            "lookahead_is_max_abs_diff": la_max_diff,
            "net": {**net_m, "audit": audit_net},
            "gross": {**gross_m, "audit": audit_gross},
        }

        nf = net_m["full"]
        ni = net_m["is"]
        no = net_m["oos"]
        gf = gross_m["full"]
        print(
            f"  NET  Full : Ret={nf['total_return']*100:+7.2f}% (1억->{nf['krw_1eok_final_eok']:.2f}억) | "
            f"CAGR={nf['cagr']*100:+6.2f}% | Sharpe={nf['sharpe']:+5.2f} | MDD={nf['mdd']*100:+6.2f}% | "
            f"Tr={nf['num_trades']} | Win={nf['win_rate']*100:4.1f}% | "
            f"AvgAcctLoss={nf['avg_loss_account_pct']*100:+.2f}% | MaxAcctLoss={nf['max_loss_account_pct']*100:+.2f}%"
        )
        print(
            f"  NET  IS   : Ret={ni['total_return']*100:+7.2f}% | CAGR={ni['cagr']*100:+6.2f}% | "
            f"Sharpe={ni['sharpe']:+5.2f} | MDD={ni['mdd']*100:+6.2f}% | Tr={ni['num_trades']}"
        )
        print(
            f"  NET  OOS  : Ret={no['total_return']*100:+7.2f}% | CAGR={no['cagr']*100:+6.2f}% | "
            f"Sharpe={no['sharpe']:+5.2f} | MDD={no['mdd']*100:+6.2f}% | Tr={no['num_trades']}"
        )
        print(
            f"  GROSS Full: Ret={gf['total_return']*100:+7.2f}% | CAGR={gf['cagr']*100:+6.2f}% | "
            f"Sharpe={gf['sharpe']:+5.2f} | LookAhead_Diff={la_max_diff:.8f}"
        )
        print(f"  Yearly Net: { {k: f'{v*100:+.1f}%' for k, v in net_m['yearly_returns'].items()} }")

    bf = results["benchmark_btc_bh"]["full"]
    bi = results["benchmark_btc_bh"]["is"]
    bo = results["benchmark_btc_bh"]["oos"]
    print("\n[BENCHMARK] BTC Buy & Hold")
    print(f"  Full: Ret={bf['total_return']*100:+7.2f}% (1억->{bf['krw_1eok_final_eok']:.2f}억) | CAGR={bf['cagr']*100:+6.2f}% | Sharpe={bf['sharpe']:+5.2f} | MDD={bf['mdd']*100:+6.2f}%")
    print(f"  IS  : Ret={bi['total_return']*100:+7.2f}% | CAGR={bi['cagr']*100:+6.2f}% | Sharpe={bi['sharpe']:+5.2f} | MDD={bi['mdd']*100:+6.2f}%")
    print(f"  OOS : Ret={bo['total_return']*100:+7.2f}% | CAGR={bo['cagr']*100:+6.2f}% | Sharpe={bo['sharpe']:+5.2f} | MDD={bo['mdd']*100:+6.2f}%")
    print(f"  Yearly BTC: { {k: f'{v*100:+.1f}%' for k, v in results['benchmark_btc_bh']['yearly_returns'].items()} }")

    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[DONE] Saved summary JSON to {RESULTS_JSON_PATH}")


if __name__ == "__main__":
    main()
