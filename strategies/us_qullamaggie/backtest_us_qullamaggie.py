"""
US Equities Qullamaggie Momentum Breakout Backtest Engine
- Strict Zero Look-Ahead Bias (all signals/stops/sizing use t-1 finalized data)
- Conservative same-day stop-out & gap-down open fill
- Zero leverage (1x cash only: cash >= 0, invested_weight <= 100%)
- Base cost: 0.25% round-trip (0.125% one-way)
- Stress test: 0.50% round-trip (0.25% one-way)
- Built-in +500% OOS Look-Ahead Perturbation Test (IS diff must be 0.000000)

Outputs in strategies/us_qullamaggie/:
  - equity_Q-1.csv, equity_Q-2.csv, equity_Q-3.csv, equity_Q-4.csv
  - trades_Q-1.csv, trades_Q-2.csv, trades_Q-3.csv, trades_Q-4.csv
  - results_us_qullamaggie.json
"""
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "us_qullamaggie_data.duckdb")

INITIAL_CAPITAL = 100_000.0
SIM_START = "2018-01-02"
IS_END = "2023-12-31"
OOS_START = "2024-01-01"
SIM_END = "2026-11-20"

BENCHMARKS = {"QQQ", "SPY", "IWM"}


@dataclass
class ModelConfig:
    name: str
    label: str
    trail_ma: str               # "ma10" or "ma20"
    risk_pct: float             # e.g., 0.01
    max_single_weight: float    # e.g., 0.20
    max_positions: int          # e.g., 6
    throttle_dd: Optional[float] = None  # e.g., -0.15 -> half risk
    partial_days: int = 4
    partial_frac: float = 1.0 / 3.0


@dataclass
class Position:
    ticker: str
    entry_date: str
    entry_price: float
    initial_shares: int
    shares: int
    initial_stop: float
    stop_price: float
    entry_cost_total: float     # total cash spent at entry (including buy fee)
    days_held: int = 0
    partial_taken: bool = False
    partial_date: str = ""
    partial_price: float = 0.0
    partial_shares: int = 0
    partial_proceeds: float = 0.0
    pending_partial_sell: bool = False
    pending_exit_all: bool = False


def load_data_from_duckdb(oos_perturb_mult: float = 1.0) -> Tuple[List[str], Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Load OHLCV from DuckDB and compute strictly causal indicators per ticker.
    If oos_perturb_mult != 1.0, multiply all OHLC prices for date >= OOS_START by oos_perturb_mult
    BEFORE computing indicators (used for +500% OOS Look-Ahead Bias perturbation test).
    """
    con = duckdb.connect(DB_PATH, read_only=True)
    df_all = con.execute("SELECT date, ticker, open, high, low, close, volume FROM ohlcv ORDER BY ticker, date").df()
    meta_df = con.execute("SELECT * FROM universe_meta").df()
    con.close()

    if oos_perturb_mult != 1.0:
        mask = df_all["date"] >= OOS_START
        for col in ["open", "high", "low", "close"]:
            df_all.loc[mask, col] = df_all.loc[mask, col] * oos_perturb_mult

    df_all["dollar_vol"] = df_all["close"] * df_all["volume"]

    # Master trading calendar from QQQ within [SIM_START, SIM_END]
    qqq_raw = df_all[df_all["ticker"] == "QQQ"].sort_values("date").reset_index(drop=True)
    trading_dates = qqq_raw[(qqq_raw["date"] >= SIM_START) & (qqq_raw["date"] <= SIM_END)]["date"].tolist()

    # Precompute strictly causal indicators per ticker
    data_map: Dict[str, pd.DataFrame] = {}
    for ticker, grp in df_all.groupby("ticker", sort=False):
        g = grp.sort_values("date").reset_index(drop=True)
        c = g["close"]
        h = g["high"]
        l = g["low"]

        g["ma10"] = c.rolling(10, min_periods=10).mean()
        g["ma20"] = c.rolling(20, min_periods=20).mean()
        g["ma50"] = c.rolling(50, min_periods=50).mean()
        g["adr20"] = ((h / l) - 1.0).rolling(20, min_periods=20).mean()
        g["adv20"] = g["dollar_vol"].rolling(20, min_periods=20).mean()
        g["ret63"] = c / c.shift(63) - 1.0
        g["ret21"] = c / c.shift(21) - 1.0
        g["box_high_5"] = h.rolling(5, min_periods=5).max()

        # Shift(1) columns represent t-1 finalized values available BEFORE Day t opens
        g["prev_close"] = c.shift(1)
        g["prev_low"] = l.shift(1)
        g["prev_ma10"] = g["ma10"].shift(1)
        g["prev_ma20"] = g["ma20"].shift(1)
        g["prev_ma50"] = g["ma50"].shift(1)
        g["prev_adr20"] = g["adr20"].shift(1)
        g["prev_adv20"] = g["adv20"].shift(1)
        g["prev_ret63"] = g["ret63"].shift(1)
        g["prev_ret21"] = g["ret21"].shift(1)
        g["prev_box_high_5"] = g["box_high_5"].shift(1)

        g = g.set_index("date")
        data_map[ticker] = g

    return trading_dates, data_map, meta_df


def build_daily_lookup(trading_dates: List[str], data_map: Dict[str, pd.DataFrame]):
    """
    Build fast per-date dictionaries for simulation:
      - day_bars[date][ticker] = tuple of values needed for day t
      - day_candidates[date] = list of tickers passing all t-1 setup filters, sorted by prev_ret63 desc
      - qqq_filter[date] = bool (QQQ prev_close > QQQ prev_ma20)
    All setup filters strictly use t-1 finalized values!
    """
    date_set = set(trading_dates)
    day_bars: Dict[str, Dict[str, dict]] = {d: {} for d in trading_dates}
    day_candidates: Dict[str, List[Tuple[float, str]]] = {d: [] for d in trading_dates}
    qqq_filter: Dict[str, bool] = {}

    qqq_df = data_map["QQQ"]
    for d in trading_dates:
        row = qqq_df.loc[d]
        qqq_filter[d] = bool(
            pd.notna(row["prev_close"])
            and pd.notna(row["prev_ma20"])
            and (row["prev_close"] > row["prev_ma20"])
        )

    for ticker, df in data_map.items():
        if ticker in BENCHMARKS:
            continue
        sub = df.loc[df.index.intersection(date_set)]
        if sub.empty:
            continue

        # Vectorized t-1 setup filter (100% causal, uses only prev_* columns)
        cond = (
            sub["prev_close"].notna()
            & sub["prev_ma50"].notna()
            & sub["prev_ret63"].notna()
            & sub["prev_adr20"].notna()
            & sub["prev_adv20"].notna()
            & sub["prev_box_high_5"].notna()
            & (sub["prev_close"] >= 5.0)
            & (sub["prev_adv20"] >= 20_000_000.0)
            & (sub["prev_adr20"] >= 0.04)
            & (sub["prev_ret63"] >= 0.25)
            & (sub["prev_ret21"] > 0.0)
            & (sub["prev_close"] > sub["prev_ma10"])
            & (sub["prev_ma10"] > sub["prev_ma20"])
            & (sub["prev_ma20"] > sub["prev_ma50"])
            & (((sub["prev_close"] - sub["prev_ma20"]) / sub["prev_close"]) <= 2.0 * sub["prev_adr20"])
            & (((sub["prev_box_high_5"] - sub["prev_close"]) / sub["prev_close"]) <= 1.5 * sub["prev_adr20"])
        )

        # Store bars for fast access
        records = sub[["open", "high", "low", "close", "ma10", "ma20", "prev_close", "prev_low", "prev_adr20", "prev_box_high_5", "prev_ret63"]].to_dict("index")
        for d, rec in records.items():
            day_bars[d][ticker] = rec

        # Store candidates passing t-1 filters
        for d in sub.index[cond]:
            day_candidates[d].append((float(records[d]["prev_ret63"]), ticker))

    # Sort candidates by 3-month momentum descending (tie-break by ticker for determinism)
    for d in trading_dates:
        day_candidates[d].sort(key=lambda x: (-x[0], x[1]))

    return day_bars, day_candidates, qqq_filter


def run_backtest(
    cfg: ModelConfig,
    trading_dates: List[str],
    day_bars: Dict[str, Dict[str, dict]],
    day_candidates: Dict[str, List[Tuple[float, str]]],
    qqq_filter: Dict[str, bool],
    round_trip_cost: float = 0.0025,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute day-by-day Qullamaggie simulation with strict cash accounting & zero look-ahead.
    """
    fee_one_way = round_trip_cost / 2.0
    cash = INITIAL_CAPITAL
    positions: Dict[str, Position] = {}
    last_known_price: Dict[str, float] = {}

    equity_prev = INITIAL_CAPITAL
    hwm_prev = INITIAL_CAPITAL

    equity_rows = []
    trade_rows = []

    def close_position_record(pos: Position, exit_date: str, exit_price: float, exit_reason: str):
        nonlocal cash
        proceeds_final = pos.shares * exit_price * (1.0 - fee_one_way)
        cash += proceeds_final

        total_proceeds = pos.partial_proceeds + proceeds_final
        net_pnl = total_proceeds - pos.entry_cost_total
        net_ret = net_pnl / pos.entry_cost_total

        avg_exit = (
            (pos.partial_shares * pos.partial_price + pos.shares * exit_price) / pos.initial_shares
            if pos.initial_shares > 0 else exit_price
        )

        trade_rows.append({
            "model": cfg.name,
            "ticker": pos.ticker,
            "entry_date": pos.entry_date,
            "exit_date": exit_date,
            "days_held": pos.days_held,
            "entry_price": round(pos.entry_price, 4),
            "initial_stop": round(pos.initial_stop, 4),
            "final_stop": round(pos.stop_price, 4),
            "exit_price": round(exit_price, 4),
            "avg_exit_price": round(avg_exit, 4),
            "initial_shares": pos.initial_shares,
            "partial_taken": pos.partial_taken,
            "partial_date": pos.partial_date,
            "partial_price": round(pos.partial_price, 4) if pos.partial_taken else 0.0,
            "net_return_pct": round(net_ret * 100.0, 4),
            "net_pnl": round(net_pnl, 2),
            "exit_reason": exit_reason,
            "period": "IS" if pos.entry_date <= IS_END else "OOS",
        })

    for d in trading_dates:
        bars = day_bars[d]

        # Update last known prices for open positions
        for t, bar in bars.items():
            last_known_price[t] = bar["close"]

        # ---------------------------------------------------------
        # STEP A: Scheduled Open Exits (decided from t-1 close)
        # ---------------------------------------------------------
        to_remove = []
        for t, pos in list(positions.items()):
            bar = bars.get(t)
            if bar is None:
                continue
            op = bar["open"]
            if pos.pending_exit_all:
                close_position_record(pos, d, op, f"TRAIL_{cfg.trail_ma.upper()}_EXIT")
                to_remove.append(t)
            elif pos.pending_partial_sell and not pos.partial_taken:
                sell_sh = int(math.floor(pos.shares * cfg.partial_frac))
                if sell_sh >= 1 and sell_sh < pos.shares:
                    part_proc = sell_sh * op * (1.0 - fee_one_way)
                    cash += part_proc
                    pos.shares -= sell_sh
                    pos.partial_taken = True
                    pos.partial_date = d
                    pos.partial_price = op
                    pos.partial_shares = sell_sh
                    pos.partial_proceeds = part_proc
                    # Raise stop to breakeven (entry_price)
                    pos.stop_price = max(pos.stop_price, pos.entry_price)
                pos.pending_partial_sell = False

        for t in to_remove:
            del positions[t]

        # ---------------------------------------------------------
        # STEP B: Intraday Hard / Breakeven Stop on Existing Positions
        # ---------------------------------------------------------
        to_remove = []
        for t, pos in list(positions.items()):
            bar = bars.get(t)
            if bar is None:
                continue
            if bar["low"] <= pos.stop_price:
                # Gap-down conservative fill: min(Open, stop_price)
                exit_px = min(bar["open"], pos.stop_price)
                reason = "BREAKEVEN_STOP" if pos.partial_taken else "INITIAL_STOP"
                close_position_record(pos, d, exit_px, reason)
                to_remove.append(t)

        for t in to_remove:
            del positions[t]

        # ---------------------------------------------------------
        # STEP C: New Breakout Entries (Day t Stop-Buy Orders)
        # ---------------------------------------------------------
        if qqq_filter[d] and len(positions) < cfg.max_positions:
            # Determine active risk_pct from t-1 equity & HWM (zero look-ahead)
            active_risk_pct = cfg.risk_pct
            if cfg.throttle_dd is not None and hwm_prev > 0:
                dd_prev = equity_prev / hwm_prev - 1.0
                if dd_prev <= cfg.throttle_dd:
                    active_risk_pct = cfg.risk_pct * 0.5

            for _, t in day_candidates[d]:
                if len(positions) >= cfg.max_positions:
                    break
                if t in positions:
                    continue
                bar = bars.get(t)
                if bar is None:
                    continue

                trigger = bar["prev_box_high_5"]
                op = bar["open"]
                hi = bar["high"]
                lo = bar["low"]
                prev_c = bar["prev_close"]
                prev_lo = bar["prev_low"]
                prev_adr = bar["prev_adr20"]

                # Breakout trigger check
                if hi < trigger:
                    continue

                # Gap-up chase filter: skip excessive gaps
                if op > trigger * (1.0 + 0.5 * prev_adr) or op > prev_c * 1.08:
                    continue

                # Conservative fill price
                fill_px = max(op, trigger)

                # Initial stop price (max 8% stop or 1.0 * ADR, floored at prev_low)
                max_stop_pct = min(1.0 * prev_adr, 0.08)
                stop_px = max(prev_lo, fill_px * (1.0 - max_stop_pct))

                # Sanity check: stop must be at least 1% below fill to avoid divide-by-tiny-risk
                risk_per_share = fill_px - stop_px
                if risk_per_share <= 0 or (risk_per_share / fill_px) < 0.01:
                    continue

                # Position sizing (strictly 1x cash only, based on t-1 equity)
                risk_dollars = equity_prev * active_risk_pct
                notional_by_risk = (risk_dollars / risk_per_share) * fill_px
                notional_by_cap = equity_prev * cfg.max_single_weight
                notional_by_cash = cash / (1.0 + fee_one_way)

                target_notional = min(notional_by_risk, notional_by_cap, notional_by_cash)
                shares = int(math.floor(target_notional / fill_px))

                if shares < 1 or (shares * fill_px) < (equity_prev * 0.01):
                    continue

                entry_cost = shares * fill_px * (1.0 + fee_one_way)
                if entry_cost > cash + 1e-9:
                    # Extra safety guard against floating-point edge cases
                    shares = int(math.floor((cash / (1.0 + fee_one_way)) / fill_px))
                    if shares < 1:
                        continue
                    entry_cost = shares * fill_px * (1.0 + fee_one_way)

                cash -= entry_cost
                pos = Position(
                    ticker=t,
                    entry_date=d,
                    entry_price=fill_px,
                    initial_shares=shares,
                    shares=shares,
                    initial_stop=stop_px,
                    stop_price=stop_px,
                    entry_cost_total=entry_cost,
                    days_held=1,
                )

                # CONSERVATIVE SAME-DAY STOP RULE:
                # If intraday Low[t] <= stop_price on entry day, assume worst-case order -> immediate stop-out at stop_px
                # (Cannot exit at Open[t] < stop_px because Open[t] occurred before intraday breakout to trigger)
                if lo <= stop_px:
                    close_position_record(pos, d, stop_px, "SAME_DAY_STOP")
                else:
                    positions[t] = pos

        # ---------------------------------------------------------
        # STEP D: End-of-Day Evaluation (for t+1 Open) & Mark-to-Market
        # ---------------------------------------------------------
        invested_val = 0.0
        for t, pos in positions.items():
            bar = bars.get(t)
            px = bar["close"] if bar is not None else last_known_price[t]
            invested_val += pos.shares * px

            if bar is not None:
                # Increment days held for pre-existing positions (new entries already have days_held=1)
                if pos.entry_date != d:
                    pos.days_held += 1

                # Check trailing MA exit for t+1 Open
                ma_val = bar[cfg.trail_ma]
                if pd.notna(ma_val) and px < ma_val:
                    pos.pending_exit_all = True
                # Check 4-day partial profit take for t+1 Open (if not exiting all)
                elif (not pos.partial_taken) and (pos.days_held >= cfg.partial_days) and (px > pos.entry_price):
                    pos.pending_partial_sell = True

        equity = cash + invested_val
        inv_weight = invested_val / equity if equity > 0 else 0.0

        # Strict financial invariants check
        assert cash >= -1e-6, f"Negative cash violation on {d}: {cash}"
        assert inv_weight <= 1.0 + 1e-6, f"Leverage violation on {d}: {inv_weight}"

        equity_rows.append({
            "date": d,
            "equity": round(equity, 2),
            "cash": round(cash, 2),
            "invested": round(invested_val, 2),
            "invested_weight": round(inv_weight, 6),
            "num_positions": len(positions),
            "qqq_filter": int(qqq_filter[d]),
        })

        equity_prev = equity
        hwm_prev = max(hwm_prev, equity)

    # Accounting identity verification: realized PnL + unrealized PnL == final_equity - INITIAL_CAPITAL
    realized_pnl = sum(r["net_pnl"] for r in trade_rows)
    unrealized_pnl = 0.0
    for t, pos in positions.items():
        mtm_val = pos.shares * last_known_price[t]
        unrealized_pnl += (pos.partial_proceeds + mtm_val - pos.entry_cost_total)
    final_eq = equity_rows[-1]["equity"]
    acct_err = abs((final_eq - INITIAL_CAPITAL) - (realized_pnl + unrealized_pnl))
    assert acct_err < 1.0, f"Accounting mismatch for {cfg.name}: err={acct_err:.4f}"

    eq_df = pd.DataFrame(equity_rows)
    tr_df = pd.DataFrame(trade_rows)
    return eq_df, tr_df


def compute_metrics(eq_series: pd.Series, dates: pd.Series, trades_df: Optional[pd.DataFrame] = None) -> dict:
    """Compute CAGR (365.25 calendar days), Sharpe (252d), MDD, Total Return, Trade Stats."""
    if len(eq_series) < 2:
        return {}
    eq = eq_series.astype(float).values
    d0 = pd.to_datetime(dates.iloc[0])
    d1 = pd.to_datetime(dates.iloc[-1])
    cal_days = max((d1 - d0).days, 1)

    total_ret = eq[-1] / eq[0] - 1.0
    cagr = (eq[-1] / eq[0]) ** (365.25 / cal_days) - 1.0

    daily_ret = pd.Series(eq).pct_change().dropna()
    vol_ann = float(daily_ret.std(ddof=1) * np.sqrt(252)) if len(daily_ret) > 1 else 0.0
    sharpe = float((daily_ret.mean() / daily_ret.std(ddof=1)) * np.sqrt(252)) if (len(daily_ret) > 1 and daily_ret.std(ddof=1) > 0) else 0.0

    cummax = np.maximum.accumulate(eq)
    dd = eq / cummax - 1.0
    mdd = float(dd.min())

    res = {
        "start_date": str(dates.iloc[0]),
        "end_date": str(dates.iloc[-1]),
        "calendar_days": int(cal_days),
        "start_equity": round(float(eq[0]), 2),
        "end_equity": round(float(eq[-1]), 2),
        "total_return_pct": round(total_ret * 100.0, 2),
        "cagr_pct": round(cagr * 100.0, 2),
        "ann_vol_pct": round(vol_ann * 100.0, 2),
        "sharpe": round(sharpe, 3),
        "mdd_pct": round(mdd * 100.0, 2),
        "calmar": round(cagr / abs(mdd), 3) if mdd < 0 else 0.0,
    }

    if trades_df is not None and not trades_df.empty:
        n = len(trades_df)
        wins = trades_df[trades_df["net_pnl"] > 0]
        losses = trades_df[trades_df["net_pnl"] <= 0]
        win_rate = len(wins) / n if n > 0 else 0.0
        avg_win_pct = float(wins["net_return_pct"].mean()) if len(wins) > 0 else 0.0
        avg_loss_pct = float(losses["net_return_pct"].mean()) if len(losses) > 0 else 0.0
        payoff_ratio = abs(avg_win_pct / avg_loss_pct) if avg_loss_pct < 0 else 0.0
        gross_profit = float(wins["net_pnl"].sum()) if len(wins) > 0 else 0.0
        gross_loss = abs(float(losses["net_pnl"].sum())) if len(losses) > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        same_day_stops = int((trades_df["exit_reason"] == "SAME_DAY_STOP").sum())
        partial_cnt = int(trades_df["partial_taken"].sum())

        res.update({
            "num_trades": int(n),
            "win_rate_pct": round(win_rate * 100.0, 2),
            "avg_win_pct": round(avg_win_pct, 2),
            "avg_loss_pct": round(avg_loss_pct, 2),
            "payoff_ratio": round(payoff_ratio, 3),
            "profit_factor": round(profit_factor, 3),
            "avg_days_held": round(float(trades_df["days_held"].mean()), 1),
            "same_day_stops": same_day_stops,
            "partial_tp_count": partial_cnt,
        })
    return res


def compute_yearly_returns(eq_df: pd.DataFrame) -> Dict[str, float]:
    df = eq_df.copy()
    df["year"] = df["date"].str.slice(0, 4)
    yearly = {}
    prev_end_eq = INITIAL_CAPITAL
    for yr, grp in df.groupby("year", sort=True):
        end_eq = float(grp["equity"].iloc[-1])
        ret = (end_eq / prev_end_eq - 1.0) * 100.0
        yearly[yr] = round(ret, 2)
        prev_end_eq = end_eq
    return yearly


def evaluate_full_is_oos(eq_df: pd.DataFrame, tr_df: Optional[pd.DataFrame] = None) -> dict:
    is_eq = eq_df[eq_df["date"] <= IS_END].reset_index(drop=True)
    oos_eq = eq_df[eq_df["date"] >= OOS_START].reset_index(drop=True)

    is_tr = tr_df[tr_df["entry_date"] <= IS_END] if (tr_df is not None and not tr_df.empty) else None
    oos_tr = tr_df[tr_df["entry_date"] >= OOS_START] if (tr_df is not None and not tr_df.empty) else None

    return {
        "full": compute_metrics(eq_df["equity"], eq_df["date"], tr_df),
        "is": compute_metrics(is_eq["equity"], is_eq["date"], is_tr),
        "oos": compute_metrics(oos_eq["equity"], oos_eq["date"], oos_tr),
        "yearly_returns_pct": compute_yearly_returns(eq_df),
        "avg_invested_weight_pct": round(float(eq_df["invested_weight"].mean() * 100.0), 2) if "invested_weight" in eq_df.columns else 100.0,
        "max_invested_weight_pct": round(float(eq_df["invested_weight"].max() * 100.0), 2) if "invested_weight" in eq_df.columns else 100.0,
        "min_cash": round(float(eq_df["cash"].min()), 2) if "cash" in eq_df.columns else 0.0,
    }


def main():
    print("Loading data from DuckDB...")
    trading_dates, data_map, meta_df = load_data_from_duckdb(oos_perturb_mult=1.0)
    day_bars, day_candidates, qqq_filter = build_daily_lookup(trading_dates, data_map)

    # Benchmark: QQQ Buy & Hold
    qqq_df = data_map["QQQ"].loc[trading_dates].reset_index()
    qqq_eq = pd.DataFrame({
        "date": qqq_df["date"],
        "equity": INITIAL_CAPITAL * (qqq_df["close"] / qqq_df["close"].iloc[0]),
        "cash": 0.0,
        "invested_weight": 1.0,
    })
    qqq_stats = evaluate_full_is_oos(qqq_eq, None)

    # Step 1: Run Q-1 (MA10) and Q-2 (MA20) to select IS winner for Q-3 and Q-4
    cfg_q1 = ModelConfig(
        name="Q-1", label="정석 MA10 (리스크 1.0%, 최대 6종목)",
        trail_ma="ma10", risk_pct=0.01, max_single_weight=0.20, max_positions=6,
    )
    cfg_q2 = ModelConfig(
        name="Q-2", label="정석 MA20 (리스크 1.0%, 최대 6종목)",
        trail_ma="ma20", risk_pct=0.01, max_single_weight=0.20, max_positions=6,
    )

    eq_q1, tr_q1 = run_backtest(cfg_q1, trading_dates, day_bars, day_candidates, qqq_filter, round_trip_cost=0.0025)
    eq_q2, tr_q2 = run_backtest(cfg_q2, trading_dates, day_bars, day_candidates, qqq_filter, round_trip_cost=0.0025)

    res_q1 = evaluate_full_is_oos(eq_q1, tr_q1)
    res_q2 = evaluate_full_is_oos(eq_q2, tr_q2)

    # Select IS winner strictly based on IS Sharpe (tie-break: IS CAGR) — ZERO OOS peek!
    is_winner_ma = "ma10" if (res_q1["is"]["sharpe"], res_q1["is"]["cagr_pct"]) >= (res_q2["is"]["sharpe"], res_q2["is"]["cagr_pct"]) else "ma20"
    is_winner_name = "Q-1" if is_winner_ma == "ma10" else "Q-2"
    print(f"IS Winner selected strictly on IS (2018-2023): {is_winner_name} ({is_winner_ma.upper()}) | Q-1 IS Sharpe={res_q1['is']['sharpe']}, CAGR={res_q1['is']['cagr_pct']}% vs Q-2 IS Sharpe={res_q2['is']['sharpe']}, CAGR={res_q2['is']['cagr_pct']}%")

    cfg_q3 = ModelConfig(
        name="Q-3", label=f"방어 스로틀링 ({is_winner_ma.upper()} + HWM -15% 시 리스크 절반)",
        trail_ma=is_winner_ma, risk_pct=0.01, max_single_weight=0.20, max_positions=6, throttle_dd=-0.15,
    )
    cfg_q4 = ModelConfig(
        name="Q-4", label=f"공격형 리스크 1.5% ({is_winner_ma.upper()}, 단일 25%, 최대 5종목)",
        trail_ma=is_winner_ma, risk_pct=0.015, max_single_weight=0.25, max_positions=5,
    )

    eq_q3, tr_q3 = run_backtest(cfg_q3, trading_dates, day_bars, day_candidates, qqq_filter, round_trip_cost=0.0025)
    eq_q4, tr_q4 = run_backtest(cfg_q4, trading_dates, day_bars, day_candidates, qqq_filter, round_trip_cost=0.0025)

    res_q3 = evaluate_full_is_oos(eq_q3, tr_q3)
    res_q4 = evaluate_full_is_oos(eq_q4, tr_q4)

    configs = [cfg_q1, cfg_q2, cfg_q3, cfg_q4]
    outputs = {
        "Q-1": (eq_q1, tr_q1, res_q1),
        "Q-2": (eq_q2, tr_q2, res_q2),
        "Q-3": (eq_q3, tr_q3, res_q3),
        "Q-4": (eq_q4, tr_q4, res_q4),
    }

    # Save CSV files (4 models)
    for name, (eq_df, tr_df, _) in outputs.items():
        eq_df.to_csv(os.path.join(BASE_DIR, f"equity_{name}.csv"), index=False)
        tr_df.to_csv(os.path.join(BASE_DIR, f"trades_{name}.csv"), index=False)

    # Step 2: 2x Cost Stress Test (0.50% round-trip)
    print("\nRunning 2x Transaction Cost Stress Test (0.50% round-trip)...")
    stress_results = {}
    for cfg in configs:
        eq_s, tr_s = run_backtest(cfg, trading_dates, day_bars, day_candidates, qqq_filter, round_trip_cost=0.0050)
        stress_results[cfg.name] = evaluate_full_is_oos(eq_s, tr_s)

    # Step 3: +500% OOS Perturbation Test (Look-Ahead Bias Proof)
    print("Running +500% OOS Perturbation Test (Look-Ahead Bias Check)...")
    p_dates, p_data_map, _ = load_data_from_duckdb(oos_perturb_mult=6.0)
    p_bars, p_cands, p_qqq = build_daily_lookup(p_dates, p_data_map)
    perturbation_report = {}
    for cfg in configs:
        eq_orig = outputs[cfg.name][0]
        eq_pert, _ = run_backtest(cfg, p_dates, p_bars, p_cands, p_qqq, round_trip_cost=0.0025)
        is_orig = eq_orig[eq_orig["date"] <= IS_END]["equity"].values
        is_pert = eq_pert[eq_pert["date"] <= IS_END]["equity"].values
        max_abs_diff = float(np.max(np.abs(is_orig - is_pert)))
        perturbation_report[cfg.name] = {
            "is_max_abs_equity_diff": max_abs_diff,
            "passed": bool(max_abs_diff == 0.0),
        }
        assert max_abs_diff == 0.0, f"LOOK-AHEAD BIAS DETECTED in {cfg.name}: diff={max_abs_diff}"

    # Top 5 Best & Worst Trades for primary models (Q-1, Q-2, Q-4)
    top_trades = {}
    for name, (_, tr_df, _) in outputs.items():
        if tr_df.empty:
            continue
        best5 = tr_df.sort_values("net_return_pct", ascending=False).head(5).to_dict("records")
        worst5 = tr_df.sort_values("net_return_pct", ascending=True).head(5).to_dict("records")
        top_trades[name] = {"best_5": best5, "worst_5": worst5}

    summary = {
        "meta": {
            "sim_start": SIM_START,
            "is_end": IS_END,
            "oos_start": OOS_START,
            "sim_end": SIM_END,
            "initial_capital": INITIAL_CAPITAL,
            "base_round_trip_cost_pct": 0.25,
            "stress_round_trip_cost_pct": 0.50,
            "universe_total_tickers": int(len(meta_df)),
            "universe_core_growth": int((meta_df["category"] == "core_growth").sum()),
            "universe_fallen_angels": int((meta_df["category"] == "fallen_angel").sum()),
            "universe_benchmarks": int((meta_df["category"] == "benchmark").sum()),
            "is_winner_selected_for_q3_q4": is_winner_name,
            "is_winner_trail_ma": is_winner_ma,
        },
        "benchmark_qqq": qqq_stats,
        "models_base_025pct": {
            cfg.name: {
                "label": cfg.label,
                "config": {
                    "trail_ma": cfg.trail_ma,
                    "risk_pct": cfg.risk_pct,
                    "max_single_weight": cfg.max_single_weight,
                    "max_positions": cfg.max_positions,
                    "throttle_dd": cfg.throttle_dd,
                },
                **outputs[cfg.name][2],
            }
            for cfg in configs
        },
        "models_stress_050pct": stress_results,
        "lookahead_perturbation_test_500pct": perturbation_report,
        "top_trades": top_trades,
    }

    out_json = os.path.join(BASE_DIR, "results_us_qullamaggie.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== BACKTEST RESULTS SUMMARY (Base 0.25% RT Cost) ===")
    print(f"{'Model':<10} | {'Full CAGR':>9} {'Full Shp':>8} {'Full MDD':>9} | {'IS CAGR':>8} {'IS Shp':>7} {'IS MDD':>8} | {'OOS CAGR':>8} {'OOS Shp':>7} {'OOS MDD':>8} | {'Trades':>6} {'Win%':>6} {'Payoff':>6}")
    print("-" * 125)
    b = qqq_stats
    print(f"{'QQQ B&H':<10} | {b['full']['cagr_pct']:>8.2f}% {b['full']['sharpe']:>8.3f} {b['full']['mdd_pct']:>8.2f}% | {b['is']['cagr_pct']:>7.2f}% {b['is']['sharpe']:>7.3f} {b['is']['mdd_pct']:>7.2f}% | {b['oos']['cagr_pct']:>7.2f}% {b['oos']['sharpe']:>7.3f} {b['oos']['mdd_pct']:>7.2f}% | {'-':>6} {'-':>6} {'-':>6}")
    for cfg in configs:
        r = outputs[cfg.name][2]
        print(
            f"{cfg.name:<10} | {r['full']['cagr_pct']:>8.2f}% {r['full']['sharpe']:>8.3f} {r['full']['mdd_pct']:>8.2f}% | "
            f"{r['is']['cagr_pct']:>7.2f}% {r['is']['sharpe']:>7.3f} {r['is']['mdd_pct']:>7.2f}% | "
            f"{r['oos']['cagr_pct']:>7.2f}% {r['oos']['sharpe']:>7.3f} {r['oos']['mdd_pct']:>7.2f}% | "
            f"{r['full']['num_trades']:>6d} {r['full']['win_rate_pct']:>5.1f}% {r['full']['payoff_ratio']:>6.2f}"
        )

    print("\n=== 2x COST STRESS TEST (0.50% RT Cost) ===")
    for cfg in configs:
        s = stress_results[cfg.name]
        print(f"{cfg.name:<10} | Full CAGR: {s['full']['cagr_pct']:>6.2f}% (Shp {s['full']['sharpe']:.3f}, MDD {s['full']['mdd_pct']:.2f}%) | IS CAGR: {s['is']['cagr_pct']:>6.2f}% | OOS CAGR: {s['oos']['cagr_pct']:>6.2f}%")

    print("\n=== +500% OOS LOOK-AHEAD PERTURBATION TEST ===")
    for name, rep in perturbation_report.items():
        print(f"{name}: max_abs_diff = {rep['is_max_abs_equity_diff']:.6f} -> {'PASS' if rep['passed'] else 'FAIL'}")

    print(f"\nSaved results to: {out_json}")


if __name__ == "__main__":
    main()
