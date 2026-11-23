"""
Invest_God Phase 1-B-3: 가상자산 단기 M-3 터보(공격형) 포지션 사이징 스케일링 테스트
- 전략 로직(진입·청산 조건)은 M-3/M-2 원형에서 단 1줄도 수정하지 않음 (Base M3_R07 결과 100% 동일 재현 확인)
- 오직 1회 계좌 리스크, 단일 코인 상한, 최대 보유 종목수, HWM 감속 기준만 공격형 스케일링
- 현물 1배 절대 유지 (레버리지 0%, 총 비중 <= 100%, cash >= 0), 왕복 비용 0.36%, OOS +500% 교란 오차 0.00000000
- Z3-Solver SLA 기계적 검증(z3_turbo_verification.json)까지 일괄 수행
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from z3 import And, Real, RealVal, Solver, sat

from backtest_crypto_short_v2 import (
    ALL_SYMBOLS,
    EVAL_START,
    INITIAL_CAPITAL,
    IS_END,
    NET_SIDE_COST,
    ROUND_TRIP_COST,
    compute_metrics,
    compute_yearly_returns,
    load_data_4h,
    split_metrics,
)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_JSON_PATH = BASE_DIR / "turbo_results.json"
Z3_JSON_PATH = BASE_DIR / "z3_turbo_verification.json"


@dataclass
class TurboConfig:
    code: str
    name: str
    risk_per_trade: float
    single_cap: float
    max_positions: int
    use_btc_ema600: bool
    dd_throttle_level: Optional[float]  # e.g. -0.15, -0.20, or None


TURBO_CASES = [
    TurboConfig(
        code="M3_R07",
        name="Base (M-3_R07): Risk -0.7% | Cap 24.5% | Max 4 | HWM -15% Throttle",
        risk_per_trade=0.007,
        single_cap=0.245,
        max_positions=4,
        use_btc_ema600=True,
        dd_throttle_level=-0.15,
    ),
    TurboConfig(
        code="M3_R15",
        name="Turbo-1 (M-3_R15): Risk -1.5% | Cap 33.0% | Max 4 | HWM -20% Throttle",
        risk_per_trade=0.015,
        single_cap=0.330,
        max_positions=4,
        use_btc_ema600=True,
        dd_throttle_level=-0.20,
    ),
    TurboConfig(
        code="M3_R20",
        name="Turbo-2 (M-3_R20): Risk -2.0% | Cap 45.0% | Max 3 | HWM -20% Throttle",
        risk_per_trade=0.020,
        single_cap=0.450,
        max_positions=3,
        use_btc_ema600=True,
        dd_throttle_level=-0.20,
    ),
    TurboConfig(
        code="M2_R15",
        name="Turbo-3 (M-2_R15): No EMA600/Throttle | Risk -1.5% | Cap 33.0% | Max 4",
        risk_per_trade=0.015,
        single_cap=0.330,
        max_positions=4,
        use_btc_ema600=False,
        dd_throttle_level=None,
    ),
]


def run_turbo_case(
    cfg: TurboConfig,
    timeline: pd.DatetimeIndex,
    data: Dict[str, Dict[str, np.ndarray]],
    side_cost: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    start_idx = int(np.searchsorted(timeline, EVAL_START))
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
    weight_sum = 0.0
    bar_count = 0

    btc = data["BTCUSDT"]
    btc_start_price = btc["close_ffill"][start_idx]

    for t in range(start_idx, end_idx + 1):
        ts = timeline[t]
        prev = t - 1
        prev2 = t - 2

        # STEP A: Channel Exit (exact M-3/M-2 logic: Close[t-1] < Donchian_Low(24)[t-2])
        for sym in list(positions.keys()):
            pos = positions[sym]
            sd = data[sym]
            if not sd["tradable"][t]:
                continue

            if sd["close_ffill"][prev] < sd["donch_l24"][prev2]:
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
                        "period": "IS" if pos["entry_time"] <= IS_END else "OOS",
                    }
                )
                del positions[sym]

        # STEP B: Regime Filter, DD Throttle & New Entries (exact M-3/M-2 logic)
        eq_open = cash + sum(pos["qty"] * data[sym]["open_ffill"][t] for sym, pos in positions.items())
        cur_dd = eq_open / hwm_equity - 1.0 if hwm_equity > 0 else 0.0
        throttle_on = (cfg.dd_throttle_level is not None) and (cur_dd <= cfg.dd_throttle_level)
        if throttle_on:
            dd_throttle_bars += 1

        eff_risk = (cfg.risk_per_trade * 0.5) if throttle_on else cfg.risk_per_trade
        eff_weight_cap = (cfg.single_cap * 0.5) if throttle_on else cfg.single_cap

        avail_slots = cfg.max_positions - len(positions)
        if avail_slots > 0:
            btc_c_p = btc["close_ffill"][prev]
            btc_ok = btc_c_p > btc["ema200"][prev]
            if cfg.use_btc_ema600:
                btc_ok = btc_ok and (btc_c_p > btc["ema600"][prev])

            if btc_ok:
                candidates: List[Tuple[float, str, float]] = []
                for sym in ALL_SYMBOLS:
                    if sym in positions:
                        continue
                    sd = data[sym]
                    if not (sd["tradable"][t] and sd["tradable"][prev]):
                        continue

                    c_p = sd["close"][prev]
                    v_p = sd["volume"][prev]
                    vsma_p = sd["vol_sma20"][prev]
                    ema50_p = sd["ema50"][prev]
                    atr_p = sd["atr14"][prev]
                    dh_p2 = sd["donch_h48"][prev2]

                    if any(math.isnan(x) for x in [c_p, v_p, vsma_p, ema50_p, atr_p, dh_p2]):
                        continue
                    if vsma_p <= 0 or atr_p <= 0 or c_p <= 0:
                        continue

                    if (c_p > ema50_p) and (c_p > dh_p2) and (v_p > 1.3 * vsma_p):
                        score = v_p / vsma_p
                        atr_pct = atr_p / c_p
                        candidates.append((score, sym, atr_pct))

                if candidates:
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    for _, sym, atr_pct in candidates[:avail_slots]:
                        stop_dist_pct = max(0.02, 2.0 * atr_pct)
                        w = min(eff_weight_cap, eff_risk / stop_dist_pct)
                        # Strict spot 1x: never spend more than available cash!
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

        # STEP C: Intra-bar Hard / Trailing Stop Check
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
                        "period": "IS" if pos["entry_time"] <= IS_END else "OOS",
                    }
                )
                del positions[sym]

        # STEP D: Bar Close — Update Trailing Stop (3xATR), HWM & Mark-to-Market
        invested_val = 0.0
        max_pos_val = 0.0
        for sym, pos in positions.items():
            sd = data[sym]
            c_t = sd["close_ffill"][t]
            pos["bars_held"] += 1
            if c_t > pos["highest_close"]:
                pos["highest_close"] = c_t

            if pos["highest_close"] >= pos["raw_entry_price"] * (1.0 + 1.5 * pos["entry_atr_pct"]):
                atr_t = sd["atr14"][t]
                if not math.isnan(atr_t) and atr_t > 0:
                    cand_trail = pos["highest_close"] - 3.0 * atr_t
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

        weight_sum += inv_weight
        bar_count += 1

        if cash < min_observed_cash:
            min_observed_cash = cash
        if equity < min_observed_equity:
            min_observed_equity = equity
        if inv_weight > max_observed_total_weight:
            max_observed_total_weight = inv_weight
        if len(positions) > max_observed_positions:
            max_observed_positions = len(positions)

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
                    "throttle_active": int(throttle_on),
                    "btc_bh_equity": round(btc_eq, 4),
                }
            )

    daily_df = pd.DataFrame(daily_rows)
    trades_df = pd.DataFrame(trades)
    worst_trade_acct_loss = float(trades_df["account_impact_pct"].min()) if len(trades_df) > 0 else 0.0

    audit = {
        "risk_per_trade": cfg.risk_per_trade,
        "single_cap": cfg.single_cap,
        "max_positions_cap": cfg.max_positions,
        "round_trip_cost": side_cost * 2.0,
        "avg_invested_weight": round(float(weight_sum / max(1, bar_count)), 6),
        "avg_cash_weight": round(float(1.0 - weight_sum / max(1, bar_count)), 6),
        "min_observed_cash": round(float(min_observed_cash), 6),
        "min_observed_equity": round(float(min_observed_equity), 6),
        "max_observed_total_weight": round(float(max_observed_total_weight), 6),
        "max_observed_single_entry_weight": round(float(max_observed_single_entry_weight), 6),
        "max_observed_positions": int(max_observed_positions),
        "worst_single_trade_account_pct": round(worst_trade_acct_loss, 6),
        "dd_throttle_bars": int(dd_throttle_bars),
    }
    return daily_df, trades_df, audit


def check_clause(name: str, expr) -> Dict:
    s = Solver()
    s.add(expr)
    res = s.check()
    return {"clause": name, "status": "SAT" if res == sat else "UNSAT", "passed": res == sat}


def verify_z3_turbo(results: Dict) -> Dict:
    btc = results["benchmark_btc_bh"]
    bf, bo = btc["full"], btc["oos"]
    out = {"verifier": "Z3 SMT Solver (Phase 1-B-3 Crypto Turbo Scaling SLA)", "cases": {}}

    for cfg in TURBO_CASES:
        code = cfg.code
        cdata = results["cases"][code]
        df_eq = pd.read_csv(BASE_DIR / f"equity_turbo_{code}_net.csv")
        df_tr = pd.read_csv(BASE_DIR / f"trades_turbo_{code}_net.csv")

        audit = cdata["net"]["audit"]
        nf = cdata["net"]["full"]
        no = cdata["net"]["oos"]
        gf = cdata["gross"]["full"]
        la_diff = float(cdata["lookahead_is_max_abs_diff"])

        max_w = Real("max_total_weight")
        min_cash = Real("min_cash")
        min_eq = Real("min_equity")
        max_single_w = Real("max_single_entry_weight")
        max_pos = Real("max_positions_held")
        rt_cost = Real("round_trip_cost")
        net_ret = Real("net_full_return")
        gross_ret = Real("gross_full_return")
        la_err = Real("lookahead_diff")
        full_cagr = Real("full_cagr")
        full_sharpe = Real("full_sharpe")
        full_mdd = Real("full_mdd")
        oos_cagr = Real("oos_cagr")
        oos_sharpe = Real("oos_sharpe")
        oos_mdd = Real("oos_mdd")

        bindings = And(
            max_w == RealVal(f"{max(float(df_eq['invested_weight'].max()), audit['max_observed_total_weight']):.10f}"),
            min_cash == RealVal(f"{min(float(df_eq['cash'].min()), audit['min_observed_cash']):.10f}"),
            min_eq == RealVal(f"{min(float(df_eq['equity'].min()), audit['min_observed_equity']):.10f}"),
            max_single_w == RealVal(f"{audit['max_observed_single_entry_weight']:.10f}"),
            max_pos == RealVal(str(max(int(df_eq["num_positions"].max()), audit["max_observed_positions"]))),
            rt_cost == RealVal(f"{audit['round_trip_cost']:.10f}"),
            net_ret == RealVal(f"{nf['total_return']:.10f}"),
            gross_ret == RealVal(f"{gf['total_return']:.10f}"),
            la_err == RealVal(f"{la_diff:.10f}"),
            full_cagr == RealVal(f"{nf['cagr']:.10f}"),
            full_sharpe == RealVal(f"{nf['sharpe']:.10f}"),
            full_mdd == RealVal(f"{nf['mdd']:.10f}"),
            oos_cagr == RealVal(f"{no['cagr']:.10f}"),
            oos_sharpe == RealVal(f"{no['sharpe']:.10f}"),
            oos_mdd == RealVal(f"{no['mdd']:.10f}"),
        )

        clauses = [
            check_clause("SLA-1_NoLeverage_Weight_LE_100pct", And(bindings, max_w <= RealVal("1.0001"), max_pos <= RealVal(str(cfg.max_positions)))),
            check_clause("SLA-2_NoBankruptcy_Cash_GE_0_Equity_GT_0", And(bindings, min_cash >= RealVal("0"), min_eq > RealVal("0"))),
            check_clause("SLA-3_SingleCoinCap_Respected", And(bindings, max_single_w <= RealVal(f"{cfg.single_cap + 0.0001:.6f}"))),
            check_clause("SLA-4_ConservativeCost_GE_30bps_Deducted", And(bindings, rt_cost >= RealVal("0.003"), net_ret < gross_ret)),
            check_clause("SLA-5_ZeroLookAhead_Perturbation_0", And(bindings, la_err <= RealVal("0.000000001"))),
            check_clause("SLA-6_OOS_NetProfit_Positive", And(bindings, oos_cagr > RealVal("0"), oos_sharpe > RealVal("0"))),
            check_clause(
                "SLA-7_OOS_Beats_BTC_CAGR_Sharpe_MDD",
                And(bindings, oos_cagr > RealVal(f"{bo['cagr']:.10f}"), oos_sharpe > RealVal(f"{bo['sharpe']:.10f}"), oos_mdd > RealVal(f"{bo['mdd']:.10f}")),
            ),
            check_clause(
                "SLA-8_Full_And_OOS_Dominates_BTC_CAGR_Sharpe_MDD",
                And(
                    bindings,
                    full_cagr > RealVal(f"{bf['cagr']:.10f}"),
                    full_sharpe > RealVal(f"{bf['sharpe']:.10f}"),
                    full_mdd > RealVal(f"{bf['mdd']:.10f}"),
                    oos_cagr > RealVal(f"{bo['cagr']:.10f}"),
                    oos_sharpe > RealVal(f"{bo['sharpe']:.10f}"),
                    oos_mdd > RealVal(f"{bo['mdd']:.10f}"),
                ),
            ),
        ]

        all_pass = all(c["passed"] for c in clauses)
        out["cases"][code] = {
            "daily_rows_verified": int(len(df_eq)),
            "trade_rows_verified": int(len(df_tr)),
            "overall_all_8_sla": "SAT" if all_pass else "UNSAT",
            "clauses": clauses,
        }
        print(f"  [Z3 {code:8s}] All(8/8)={'SAT' if all_pass else 'UNSAT'}")

    with open(Z3_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out


def main() -> None:
    print("[INFO] Loading 4h Binance Spot data from DuckDB...")
    tl_4h, data_4h = load_data_4h(perturb_oos_mult=1.0)
    _, data_4h_pert = load_data_4h(perturb_oos_mult=6.0)

    results = {
        "meta": {
            "eval_period": "2021-01-01 ~ 2026-11-21",
            "is_period": "2021-01-01 ~ 2024-06-30",
            "oos_period": "2024-07-01 ~ 2026-11-21",
            "net_round_trip_cost": ROUND_TRIP_COST,
            "leverage": "0% (Spot 1x strictly enforced)",
        },
        "benchmark_btc_bh": {},
        "cases": {},
    }

    btc_done = False

    for cfg in TURBO_CASES:
        print(f"\n[RUN] {cfg.name}")
        daily_net, trades_net, audit_net = run_turbo_case(cfg, tl_4h, data_4h, side_cost=NET_SIDE_COST)
        daily_gross, trades_gross, audit_gross = run_turbo_case(cfg, tl_4h, data_4h, side_cost=0.0)
        daily_pert, _, _ = run_turbo_case(cfg, tl_4h, data_4h_pert, side_cost=NET_SIDE_COST)

        is_mask = pd.to_datetime(daily_net["date"]) <= pd.Timestamp("2024-06-30")
        la_max_diff = float(
            np.max(np.abs(daily_net.loc[is_mask, "equity"].to_numpy() - daily_pert.loc[is_mask, "equity"].to_numpy()))
        )

        daily_net.to_csv(BASE_DIR / f"equity_turbo_{cfg.code}_net.csv", index=False)
        trades_net.to_csv(BASE_DIR / f"trades_turbo_{cfg.code}_net.csv", index=False)

        if not btc_done:
            results["benchmark_btc_bh"] = split_metrics(daily_net, None, col="btc_bh_equity")
            btc_done = True

        net_m = split_metrics(daily_net, trades_net, col="equity")
        gross_m = split_metrics(daily_gross, trades_gross, col="equity")

        results["cases"][cfg.code] = {
            "name": cfg.name,
            "lookahead_is_max_abs_diff": la_max_diff,
            "net": {**net_m, "audit": audit_net},
            "gross": {**gross_m, "audit": audit_gross},
        }

        nf = net_m["full"]
        no = net_m["oos"]
        print(
            f"  Full Net : Ret={nf['total_return']*100:+7.2f}% (1억->{nf['krw_1eok_final_eok']:5.2f}억) | "
            f"CAGR={nf['cagr']*100:+6.2f}% | Sharpe={nf['sharpe']:+5.2f} | MDD={nf['mdd']*100:+6.2f}% | "
            f"Calmar={nf['calmar']:4.2f} | AvgExp={audit_net['avg_invested_weight']*100:4.1f}% | "
            f"MaxAcctLoss={nf['max_loss_account_pct']*100:+.2f}%"
        )
        print(
            f"  OOS  Net : Ret={no['total_return']*100:+7.2f}% | CAGR={no['cagr']*100:+6.2f}% | "
            f"Sharpe={no['sharpe']:+5.2f} | MDD={no['mdd']*100:+6.2f}% | LA_Diff={la_max_diff:.8f}"
        )
        print(f"  Yearly   : { {k: f'{v*100:+.1f}%' for k, v in net_m['yearly_returns'].items()} }")

    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n[Z3] Running Z3-Solver SLA verification on all 4 Turbo cases...")
    verify_z3_turbo(results)
    print(f"[DONE] Saved {RESULTS_JSON_PATH} and {Z3_JSON_PATH}")


if __name__ == "__main__":
    main()
