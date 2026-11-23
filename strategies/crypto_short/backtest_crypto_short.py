"""
Invest_God Phase 1-B: 가상자산(Crypto 24h) 타이트 손절(-2.5% ~ -3.0%) 단기 퀀트 전략 3종 백테스트 엔진
- 데이터: Binance Spot 4시간봉(4h) 실데이터 10종목 (crypto_short_data.duckdb)
- 기간: Full 2021-01-01 ~ 2026-11-21 | IS 2021-01-01 ~ 2024-06-30 (60%) | OOS 2024-07-01 ~ 2026-11-21 (40%)
- 현실 제약:
  * 레버리지 0% (현물 1배, cash >= 0, 총 비중 <= 100%)
  * 단일 코인 진입 비중 <= 24.5% (<= 25% 한도, 최대 동시 보유 4코인)
  * 하드 손절: C-1(-2.5%), C-2(-3.0%), C-3(-2.5%) — 동시 터치 시 손절 무조건 우선 체결
  * 거래비용: 왕복 0.36% (편도 0.18% = 수수료 0.10% + 급변동 슬리피지 0.08%)
  * 서킷브레이커: 3연패 또는 24시간 롤링 손실 -4.0% 도달 시 24시간(6봉) 신규 진입 중단
  * Zero Look-Ahead: t-1 종가 확정 신호 -> t 시가 진입, OOS +500% 교란 테스트 IS 오차 0.000000 증명
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "crypto_short_data.duckdb"
RESULTS_JSON_PATH = BASE_DIR / "backtest_crypto_results.json"

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

EVAL_START = pd.Timestamp("2021-01-01 00:00:00")
IS_END = pd.Timestamp("2024-06-30 20:00:00")
OOS_START = pd.Timestamp("2024-07-01 00:00:00")
EVAL_END = pd.Timestamp("2026-11-21 20:00:00")

INITIAL_CAPITAL = 100_000.0
MAX_POSITIONS = 4
SINGLE_ENTRY_WEIGHT = 0.245  # <= 25% cap strictly enforced
NET_SIDE_COST = 0.0018       # 0.18% per side => 0.36% round-trip (fee 0.10% + slippage 0.08%)
ROUND_TRIP_COST = NET_SIDE_COST * 2.0
COOLDOWN_BARS = 6            # 24 hours (6 x 4h bars)
CONSEC_LOSS_LIMIT = 3
DAILY_DD_LIMIT = -0.04       # -4.0% rolling 24h drawdown triggers cooldown


@dataclass
class StrategyConfig:
    code: str
    name: str
    stop_loss_pct: float
    take_profit_pct: Optional[float]
    max_hold_bars: int
    use_trailing: bool


STRATEGIES = [
    StrategyConfig(
        code="C1_VolBreakout",
        name="C-1: Volatility Breakout + Trend Filter + Tight Stop (-2.5%)",
        stop_loss_pct=0.025,
        take_profit_pct=0.060,
        max_hold_bars=12,  # 48h
        use_trailing=False,
    ),
    StrategyConfig(
        code="C2_DonchianVol",
        name="C-2: Donchian 20-Bar Breakout + Volume Surge + Tight Stop (-3.0%)",
        stop_loss_pct=0.030,
        take_profit_pct=None,
        max_hold_bars=18,  # 72h
        use_trailing=True,
    ),
    StrategyConfig(
        code="C3_UptrendMeanRev",
        name="C-3: Uptrend Short-Term Oversold Mean-Reversion + Tight Stop (-2.5%)",
        stop_loss_pct=0.025,
        take_profit_pct=0.050,
        max_hold_bars=12,  # 48h
        use_trailing=False,
    ),
]


def load_market_data(perturb_oos_mult: float = 1.0) -> Tuple[pd.DatetimeIndex, Dict[str, Dict[str, np.ndarray]], Dict]:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df_all = con.execute(
        """
        SELECT symbol, open_time_utc, open, high, low, close, volume, quote_volume
        FROM ohlcv_4h
        ORDER BY open_time_utc, symbol
        """
    ).fetchdf()
    meta_df = con.execute("SELECT * FROM data_meta ORDER BY symbol").fetchdf()
    con.close()

    df_all["open_time_utc"] = pd.to_datetime(df_all["open_time_utc"])
    timeline = pd.date_range(
        start=df_all["open_time_utc"].min(),
        end=EVAL_END,
        freq="4h",
    )

    oos_mask = np.asarray(timeline >= OOS_START, dtype=bool)

    data: Dict[str, Dict[str, np.ndarray]] = {}
    for sym in SYMBOLS:
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

        # Forward-filled close ONLY for mark-to-market equity valuation on exchange maintenance gaps
        c_ffill = pd.Series(c).ffill().to_numpy(dtype=np.float64)
        o_ffill = np.where(np.isnan(o), c_ffill, o)

        # Strictly causal indicators computed on forward-filled close/ohlc
        s_close = pd.Series(c_ffill)
        s_high = pd.Series(np.where(np.isnan(h), c_ffill, h))
        s_low = pd.Series(np.where(np.isnan(l), c_ffill, l))
        s_vol = pd.Series(np.where(np.isnan(v), 0.0, v))

        ema50 = s_close.ewm(span=50, adjust=False).mean().to_numpy()
        ema100 = s_close.ewm(span=100, adjust=False).mean().to_numpy()
        ema200 = s_close.ewm(span=200, adjust=False).mean().to_numpy()

        prev_c = s_close.shift(1)
        tr = pd.concat(
            [
                (s_high - s_low),
                (s_high - prev_c).abs(),
                (s_low - prev_c).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr14 = tr.ewm(alpha=1.0 / 14.0, adjust=False).mean().to_numpy()

        delta = s_close.diff()
        gain = delta.clip(lower=0.0).ewm(alpha=1.0 / 14.0, adjust=False).mean()
        loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / 14.0, adjust=False).mean()
        rs = gain / loss.replace(0.0, np.nan)
        rsi14 = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0).to_numpy()

        bb_mid = s_close.rolling(20).mean().to_numpy()
        bb_std = s_close.rolling(20).std(ddof=0).to_numpy()
        bb_low = bb_mid - 2.0 * bb_std

        donch_high20 = s_high.rolling(20).max().to_numpy()
        donch_low10 = s_low.rolling(10).min().to_numpy()
        vol_sma20 = s_vol.rolling(20).mean().to_numpy()

        data[sym] = {
            "tradable": tradable,
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": v,
            "close_ffill": c_ffill,
            "open_ffill": o_ffill,
            "ema50": ema50,
            "ema100": ema100,
            "ema200": ema200,
            "atr14": atr14,
            "rsi14": rsi14,
            "bb_mid": bb_mid,
            "bb_low": bb_low,
            "donch_high20": donch_high20,
            "donch_low10": donch_low10,
            "vol_sma20": vol_sma20,
        }

    meta_info = {
        "source": str(meta_df["source"].iloc[0]),
        "interval": str(meta_df["interval"].iloc[0]),
        "universe": SYMBOLS,
        "total_timeline_bars": int(len(timeline)),
        "eval_start_utc": str(EVAL_START),
        "is_end_utc": str(IS_END),
        "oos_start_utc": str(OOS_START),
        "eval_end_utc": str(EVAL_END),
        "missing_bars_by_symbol": {
            row["symbol"]: int(row["missing_bars"]) for _, row in meta_df.iterrows()
        },
    }
    return timeline, data, meta_info


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
        gross_profit = float(wins["net_pnl"].sum())
        gross_loss = abs(float(losses["net_pnl"].sum()))
        pf = float(gross_profit / gross_loss) if gross_loss > 1e-9 else (99.0 if gross_profit > 0 else 0.0)
        sl_hits = int((trades_df["exit_reason"] == "HARD_STOP").sum())
        res.update(
            {
                "num_trades": int(len(trades_df)),
                "win_rate": round(float(len(wins) / len(trades_df)), 4),
                "profit_factor": round(pf, 4),
                "avg_win_pct": round(float(wins["net_return"].mean()), 5) if len(wins) > 0 else 0.0,
                "avg_loss_pct": round(float(losses["net_return"].mean()), 5) if len(losses) > 0 else 0.0,
                "avg_hold_hours": round(float(trades_df["bars_held"].mean() * 4.0), 2),
                "hard_stop_hits": sl_hits,
                "hard_stop_rate": round(float(sl_hits / len(trades_df)), 4),
            }
        )
    elif trades_df is not None:
        res.update(
            {
                "num_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_win_pct": 0.0,
                "avg_loss_pct": 0.0,
                "avg_hold_hours": 0.0,
                "hard_stop_hits": 0,
                "hard_stop_rate": 0.0,
            }
        )
    return res


def run_strategy(
    cfg: StrategyConfig,
    timeline: pd.DatetimeIndex,
    data: Dict[str, Dict[str, np.ndarray]],
    side_cost: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    start_idx = int(np.searchsorted(timeline, EVAL_START))
    end_idx = int(len(timeline) - 1)

    cash = INITIAL_CAPITAL
    positions: Dict[str, Dict] = {}
    trades: List[Dict] = []
    daily_rows: List[Dict] = []

    consec_losses = 0
    cooldown_until_bar = -1
    equity_history: List[float] = []

    # Bar-level worst-case audit trackers (across EVERY 4h bar!)
    min_observed_cash = INITIAL_CAPITAL
    min_observed_equity = INITIAL_CAPITAL
    max_observed_total_weight = 0.0
    max_observed_single_entry_weight = 0.0
    max_observed_single_mtm_weight = 0.0
    max_observed_positions = 0
    cooldown_triggers = 0

    btc = data["BTCUSDT"]
    btc_start_price = btc["close_ffill"][start_idx]

    for t in range(start_idx, end_idx + 1):
        ts = timeline[t]
        prev = t - 1
        prev2 = t - 2

        # ------------------------------------------------------------------
        # STEP A: Open of bar t — Signal-based exits & Time-stop exits
        # ------------------------------------------------------------------
        for sym in list(positions.keys()):
            pos = positions[sym]
            sd = data[sym]
            if not sd["tradable"][t]:
                continue

            exit_reason = None
            if pos["bars_held"] >= cfg.max_hold_bars:
                exit_reason = "TIME_STOP"
            elif cfg.code == "C2_DonchianVol":
                if sd["close_ffill"][prev] < sd["donch_low10"][prev2]:
                    exit_reason = "CHANNEL_EXIT"
            elif cfg.code == "C3_UptrendMeanRev":
                if (sd["close_ffill"][prev] >= sd["bb_mid"][prev]) or (sd["rsi14"][prev] >= 55.0):
                    exit_reason = "MEANREV_EXIT"

            if exit_reason is not None:
                raw_exit = sd["open"][t]
                exec_exit = raw_exit * (1.0 - side_cost)
                proceeds = pos["qty"] * exec_exit
                net_pnl = proceeds - pos["cost_basis"]
                net_ret = net_pnl / pos["cost_basis"]
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
                        "qty": round(pos["qty"], 8),
                        "entry_notional": round(pos["cost_basis"], 2),
                        "exit_proceeds": round(proceeds, 2),
                        "net_pnl": round(net_pnl, 2),
                        "net_return": round(net_ret, 6),
                        "bars_held": pos["bars_held"],
                        "exit_reason": exit_reason,
                        "period": "IS" if pos["entry_time"] <= IS_END else "OOS",
                    }
                )
                del positions[sym]

                if net_pnl < 0:
                    consec_losses += 1
                    if consec_losses >= CONSEC_LOSS_LIMIT:
                        cooldown_until_bar = max(cooldown_until_bar, t + COOLDOWN_BARS)
                        consec_losses = 0
                        cooldown_triggers += 1
                else:
                    consec_losses = 0

        # ------------------------------------------------------------------
        # STEP B: Open of bar t — Circuit breaker check & New Entries
        # ------------------------------------------------------------------
        eq_open = cash + sum(pos["qty"] * data[sym]["open_ffill"][t] for sym, pos in positions.items())
        if len(equity_history) >= 6:
            ret_24h = eq_open / equity_history[-6] - 1.0
            if ret_24h <= DAILY_DD_LIMIT and t >= cooldown_until_bar:
                cooldown_until_bar = t + COOLDOWN_BARS
                cooldown_triggers += 1

        in_cooldown = t < cooldown_until_bar
        avail_slots = MAX_POSITIONS - len(positions)

        if (not in_cooldown) and (avail_slots > 0):
            candidates: List[Tuple[float, str]] = []
            btc_c_prev = btc["close_ffill"][prev]
            btc_ema100_prev = btc["ema100"][prev]
            btc_ema200_prev = btc["ema200"][prev]

            for sym in SYMBOLS:
                if sym in positions:
                    continue
                sd = data[sym]
                if not (sd["tradable"][t] and sd["tradable"][prev]):
                    continue

                c_p = sd["close"][prev]
                o_p = sd["open"][prev]
                v_p = sd["volume"][prev]
                vsma_p = sd["vol_sma20"][prev]
                atr_p2 = sd["atr14"][prev2]
                ema50_p = sd["ema50"][prev]
                ema200_p = sd["ema200"][prev]
                rsi_p = sd["rsi14"][prev]
                bb_low_p = sd["bb_low"][prev]
                dhigh_p2 = sd["donch_high20"][prev2]

                if any(math.isnan(x) for x in [c_p, o_p, v_p, vsma_p, atr_p2, ema50_p, ema200_p, rsi_p, bb_low_p, dhigh_p2]):
                    continue
                if vsma_p <= 0 or atr_p2 <= 0:
                    continue

                if cfg.code == "C1_VolBreakout":
                    if (
                        btc_c_prev > btc_ema200_prev
                        and c_p > ema50_p
                        and ema50_p > ema200_p
                        and c_p > (o_p + 0.5 * atr_p2)
                        and v_p > 1.2 * vsma_p
                    ):
                        score = (c_p - o_p) / atr_p2
                        candidates.append((score, sym))

                elif cfg.code == "C2_DonchianVol":
                    if (
                        btc_c_prev > btc_ema100_prev
                        and c_p > ema50_p
                        and c_p > dhigh_p2
                        and v_p > 1.5 * vsma_p
                    ):
                        score = v_p / vsma_p
                        candidates.append((score, sym))

                elif cfg.code == "C3_UptrendMeanRev":
                    if (
                        btc_c_prev > btc_ema200_prev
                        and c_p > ema200_p
                        and rsi_p < 35.0
                        and c_p < bb_low_p
                    ):
                        score = -rsi_p  # most oversold first
                        candidates.append((score, sym))

            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                for _, sym in candidates[:avail_slots]:
                    target_notional = min(SINGLE_ENTRY_WEIGHT * eq_open, cash)
                    if target_notional < 100.0:
                        continue
                    raw_entry = data[sym]["open"][t]
                    exec_entry = raw_entry * (1.0 + side_cost)
                    qty = target_notional / exec_entry
                    cash -= target_notional
                    # Floating point safety clamp (never negative)
                    if cash < 0.0 and cash > -1e-6:
                        cash = 0.0

                    entry_w = target_notional / eq_open
                    if entry_w > max_observed_single_entry_weight:
                        max_observed_single_entry_weight = entry_w

                    hard_stop = raw_entry * (1.0 - cfg.stop_loss_pct)
                    tp_price = raw_entry * (1.0 + cfg.take_profit_pct) if cfg.take_profit_pct is not None else None

                    positions[sym] = {
                        "entry_time": ts,
                        "raw_entry_price": raw_entry,
                        "exec_entry_price": exec_entry,
                        "qty": qty,
                        "cost_basis": target_notional,
                        "hard_stop": hard_stop,
                        "trail_stop": 0.0,
                        "tp_price": tp_price,
                        "highest_close": raw_entry,
                        "bars_held": 0,
                    }

        # ------------------------------------------------------------------
        # STEP C: Intra-bar t — Conservative Stop-Loss First, Then Take-Profit
        # ------------------------------------------------------------------
        for sym in list(positions.keys()):
            pos = positions[sym]
            sd = data[sym]
            if not sd["tradable"][t]:
                continue

            bar_open = sd["open"][t]
            bar_high = sd["high"][t]
            bar_low = sd["low"][t]

            effective_stop = max(pos["hard_stop"], pos["trail_stop"])
            exit_reason = None
            raw_exit = None

            # Priority 1: Hard/Trailing Stop ALWAYS checked before Take-Profit!
            if bar_low <= effective_stop:
                raw_exit = min(bar_open, effective_stop)
                exit_reason = "HARD_STOP" if effective_stop <= pos["hard_stop"] + 1e-12 else "TRAIL_STOP"
            elif pos["tp_price"] is not None and bar_high >= pos["tp_price"]:
                raw_exit = pos["tp_price"]
                exit_reason = "TAKE_PROFIT"

            if exit_reason is not None:
                exec_exit = raw_exit * (1.0 - side_cost)
                proceeds = pos["qty"] * exec_exit
                net_pnl = proceeds - pos["cost_basis"]
                net_ret = net_pnl / pos["cost_basis"]
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
                        "qty": round(pos["qty"], 8),
                        "entry_notional": round(pos["cost_basis"], 2),
                        "exit_proceeds": round(proceeds, 2),
                        "net_pnl": round(net_pnl, 2),
                        "net_return": round(net_ret, 6),
                        "bars_held": pos["bars_held"] + 1,
                        "exit_reason": exit_reason,
                        "period": "IS" if pos["entry_time"] <= IS_END else "OOS",
                    }
                )
                del positions[sym]

                if net_pnl < 0:
                    consec_losses += 1
                    if consec_losses >= CONSEC_LOSS_LIMIT:
                        cooldown_until_bar = max(cooldown_until_bar, t + COOLDOWN_BARS)
                        consec_losses = 0
                        cooldown_triggers += 1
                else:
                    consec_losses = 0

        # ------------------------------------------------------------------
        # STEP D: Close of bar t — Update Trailing Stops & Mark-to-Market
        # ------------------------------------------------------------------
        invested_val = 0.0
        max_pos_val = 0.0
        for sym, pos in positions.items():
            sd = data[sym]
            c_t = sd["close_ffill"][t]
            pos["bars_held"] += 1
            if c_t > pos["highest_close"]:
                pos["highest_close"] = c_t

            # Trailing stop updated strictly at bar close -> tested on NEXT bar!
            if cfg.use_trailing and pos["highest_close"] >= pos["raw_entry_price"] * 1.02:
                atr_t = sd["atr14"][t]
                if not math.isnan(atr_t) and atr_t > 0:
                    cand_trail = pos["highest_close"] - 1.5 * atr_t
                    pos["trail_stop"] = max(pos["trail_stop"], pos["hard_stop"], cand_trail)

            val = pos["qty"] * c_t
            invested_val += val
            if val > max_pos_val:
                max_pos_val = val

        equity = cash + invested_val
        equity_history.append(equity)
        inv_weight = invested_val / equity if equity > 0 else 0.0
        max_single_w = max_pos_val / equity if equity > 0 else 0.0

        if cash < min_observed_cash:
            min_observed_cash = cash
        if equity < min_observed_equity:
            min_observed_equity = equity
        if inv_weight > max_observed_total_weight:
            max_observed_total_weight = inv_weight
        if max_single_w > max_observed_single_mtm_weight:
            max_observed_single_mtm_weight = max_single_w
        if len(positions) > max_observed_positions:
            max_observed_positions = len(positions)

        # Daily snapshot at 20:00 UTC bar (closes at 00:00 UTC end of day)
        if ts.hour == 20:
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
                    "cooldown_active": int(in_cooldown),
                    "btc_bh_equity": round(btc_eq, 4),
                }
            )

    daily_df = pd.DataFrame(daily_rows)
    trades_df = pd.DataFrame(trades)
    audit = {
        "stop_loss_pct": cfg.stop_loss_pct,
        "round_trip_cost": side_cost * 2.0,
        "max_positions_cap": MAX_POSITIONS,
        "single_entry_weight_cap": SINGLE_ENTRY_WEIGHT,
        "min_observed_cash": round(float(min_observed_cash), 6),
        "min_observed_equity": round(float(min_observed_equity), 6),
        "max_observed_total_weight": round(float(max_observed_total_weight), 6),
        "max_observed_single_entry_weight": round(float(max_observed_single_entry_weight), 6),
        "max_observed_single_mtm_weight": round(float(max_observed_single_mtm_weight), 6),
        "max_observed_positions": int(max_observed_positions),
        "cooldown_triggers": int(cooldown_triggers),
    }
    return daily_df, trades_df, audit


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
    }


def main() -> None:
    print("[INFO] Loading real 4h Binance Spot data from DuckDB...")
    timeline, data, meta_info = load_market_data(perturb_oos_mult=1.0)
    print(f"[INFO] Loaded {len(SYMBOLS)} symbols, {len(timeline)} 4h bars.")

    print("[INFO] Loading OOS +500% perturbed dataset for Look-Ahead Bias verification...")
    _, data_perturbed, _ = load_market_data(perturb_oos_mult=6.0)

    results = {
        "meta": {
            **meta_info,
            "initial_capital_usdt": INITIAL_CAPITAL,
            "max_positions": MAX_POSITIONS,
            "single_entry_weight_cap": SINGLE_ENTRY_WEIGHT,
            "net_round_trip_cost": ROUND_TRIP_COST,
            "cooldown_bars_4h": COOLDOWN_BARS,
            "consec_loss_limit": CONSEC_LOSS_LIMIT,
            "daily_dd_limit": DAILY_DD_LIMIT,
        },
        "benchmark_btc_bh": {},
        "strategies": {},
    }

    btc_bench_computed = False

    for cfg in STRATEGIES:
        print(f"\n[RUN] {cfg.name}")
        daily_net, trades_net, audit_net = run_strategy(cfg, timeline, data, side_cost=NET_SIDE_COST)
        daily_gross, trades_gross, audit_gross = run_strategy(cfg, timeline, data, side_cost=0.0)

        # Look-ahead perturbation test on Net run
        daily_pert, _, _ = run_strategy(cfg, timeline, data_perturbed, side_cost=NET_SIDE_COST)
        is_mask = pd.to_datetime(daily_net["date"]) <= pd.Timestamp("2024-06-30")
        la_max_diff = float(
            np.max(np.abs(daily_net.loc[is_mask, "equity"].to_numpy() - daily_pert.loc[is_mask, "equity"].to_numpy()))
        )

        # Save raw evidence CSVs
        eq_path = BASE_DIR / f"equity_crypto_{cfg.code}_net.csv"
        tr_path = BASE_DIR / f"trades_crypto_{cfg.code}_net.csv"
        daily_net.to_csv(eq_path, index=False)
        trades_net.to_csv(tr_path, index=False)

        if not btc_bench_computed:
            results["benchmark_btc_bh"] = split_metrics(daily_net, None, col="btc_bh_equity")
            btc_bench_computed = True

        net_metrics = split_metrics(daily_net, trades_net, col="equity")
        gross_metrics = split_metrics(daily_gross, trades_gross, col="equity")

        results["strategies"][cfg.code] = {
            "name": cfg.name,
            "config": {
                "stop_loss_pct": cfg.stop_loss_pct,
                "take_profit_pct": cfg.take_profit_pct,
                "max_hold_bars": cfg.max_hold_bars,
                "use_trailing": cfg.use_trailing,
            },
            "lookahead_is_max_abs_diff": la_max_diff,
            "net": {**net_metrics, "audit": audit_net},
            "gross": {**gross_metrics, "audit": audit_gross},
        }

        nf = net_metrics["full"]
        ni = net_metrics["is"]
        no = net_metrics["oos"]
        gf = gross_metrics["full"]
        print(
            f"  NET  Full : Ret={nf['total_return']*100:+6.2f}% | CAGR={nf['cagr']*100:+6.2f}% | "
            f"Sharpe={nf['sharpe']:+5.2f} | MDD={nf['mdd']*100:+6.2f}% | Trades={nf['num_trades']} | "
            f"Win={nf['win_rate']*100:4.1f}% | SL_Hit={nf['hard_stop_rate']*100:4.1f}%"
        )
        print(
            f"  NET  IS   : Ret={ni['total_return']*100:+6.2f}% | CAGR={ni['cagr']*100:+6.2f}% | "
            f"Sharpe={ni['sharpe']:+5.2f} | MDD={ni['mdd']*100:+6.2f}% | Trades={ni['num_trades']}"
        )
        print(
            f"  NET  OOS  : Ret={no['total_return']*100:+6.2f}% | CAGR={no['cagr']*100:+6.2f}% | "
            f"Sharpe={no['sharpe']:+5.2f} | MDD={no['mdd']*100:+6.2f}% | Trades={no['num_trades']}"
        )
        print(
            f"  GROSS Full: Ret={gf['total_return']*100:+6.2f}% | CAGR={gf['cagr']*100:+6.2f}% | "
            f"Sharpe={gf['sharpe']:+5.2f} | LookAhead_Diff={la_max_diff:.8f}"
        )

    bf = results["benchmark_btc_bh"]["full"]
    bi = results["benchmark_btc_bh"]["is"]
    bo = results["benchmark_btc_bh"]["oos"]
    print("\n[BENCHMARK] BTC Buy & Hold")
    print(f"  Full: Ret={bf['total_return']*100:+6.2f}% | CAGR={bf['cagr']*100:+6.2f}% | Sharpe={bf['sharpe']:+5.2f} | MDD={bf['mdd']*100:+6.2f}%")
    print(f"  IS  : Ret={bi['total_return']*100:+6.2f}% | CAGR={bi['cagr']*100:+6.2f}% | Sharpe={bi['sharpe']:+5.2f} | MDD={bi['mdd']*100:+6.2f}%")
    print(f"  OOS : Ret={bo['total_return']*100:+6.2f}% | CAGR={bo['cagr']*100:+6.2f}% | Sharpe={bo['sharpe']:+5.2f} | MDD={bo['mdd']*100:+6.2f}%")

    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[DONE] Saved summary JSON to {RESULTS_JSON_PATH}")


if __name__ == "__main__":
    main()
