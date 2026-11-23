#!/usr/bin/env python3
"""
Invest_God Phase 2 — Z3-Solver 기계적 SLA 검증기 (verify_phase2_z3_sla.py)
실제 운영 설정 파일 3종을 직접 파싱하여 제1번 명제(PC 꺼짐 클라우드 24/7 무인 동작,
레버리지 0%, 비중 합 100%, 리스크 한도, 기본 Dry-Run 안전모드)를 Z3 SMT Solver로 SAT 증명합니다.
- 파싱 대상:
  1) live/config.json
  2) .github/workflows/daily_signal_report.yml
  3) live/deploy/invest-god-upbit.timer
"""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Any, Dict

import yaml
from z3 import And, Bool, Int, Real, Solver, sat

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
CONFIG_PATH = BASE_DIR / "config.json"
WORKFLOW_PATH = ROOT_DIR / ".github" / "workflows" / "paper_trading_24_7.yml"
TIMER_PATH = BASE_DIR / "deploy" / "invest-god-upbit.timer"
PAPER_STATE_PATH = BASE_DIR / "state" / "paper_portfolio_state.json"
OUTPUT_PATH = BASE_DIR / "output" / "z3_phase2_sla.json"

KST = dt.timezone(dt.timedelta(hours=9))


def parse_systemd_timer(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    on_cal = ""
    persistent = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("OnCalendar="):
            on_cal = s.split("=", 1)[1].strip()
        elif s.startswith("Persistent="):
            persistent = s.split("=", 1)[1].strip().lower() == "true"

    # Parse hours & minute from "*-*-* 00,04,08,12,16,20:02:00 UTC"
    m = re.search(r"([0-9,]+):(\d{2}):\d{2}\s+UTC$", on_cal)
    hours = [int(h) for h in m.group(1).split(",")] if m else []
    minute = int(m.group(2)) if m else -1
    return {
        "raw_on_calendar": on_cal,
        "utc_hours": hours,
        "minute": minute,
        "is_utc": on_cal.endswith("UTC"),
        "persistent": persistent,
    }


def parse_github_workflow(path: Path) -> Dict[str, Any]:
    raw_text = path.read_text(encoding="utf-8")
    raw = yaml.safe_load(raw_text)
    # PyYAML parses unquoted "on:" as boolean True
    on_block = raw.get("on", raw.get(True, {}))
    schedules = on_block.get("schedule", [])
    crons = [s.get("cron", "") for s in schedules if isinstance(s, dict)]
    perms = raw.get("permissions", {})
    contents_perm = str(perms.get("contents", "")) if isinstance(perms, dict) else ""
    jobs = raw.get("jobs", {})
    job = jobs.get("paper-trading-24-7", next(iter(jobs.values()), {}))
    runs_on = str(job.get("runs-on", ""))
    env_dry_run = str(job.get("env", {}).get("DRY_RUN", ""))
    steps = job.get("steps", [])
    has_auto_commit = any(
        isinstance(st, dict)
        and "git commit" in str(st.get("run", ""))
        and "github-actions[bot]" in str(st.get("run", ""))
        and "[skip ci]" in str(st.get("run", ""))
        for st in steps
    )
    return {
        "crons": crons,
        "runs_on": runs_on,
        "env_dry_run": env_dry_run,
        "contents_permission": contents_perm,
        "has_auto_commit": has_auto_commit,
    }


def check_constraint(name: str, exprs) -> Dict[str, Any]:
    s = Solver()
    for e in exprs:
        s.add(e)
    res = s.check()
    return {
        "name": name,
        "result": "SAT" if res == sat else "UNSAT",
    }


def main() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    wf = parse_github_workflow(WORKFLOW_PATH)
    tm = parse_systemd_timer(TIMER_PATH)

    checks = []

    # ---------------------------------------------------------
    # SLA-1: PC 꺼짐 클라우드 24/7 무인 동작 (paper_trading_24_7.yml + .timer 직접 파싱)
    # ---------------------------------------------------------
    gh_cloud_runner = Bool("gh_cloud_runner")
    gh_cron_4h_ok = Bool("gh_cron_4h_crypto_utc")
    gh_cron_krx_ok = Bool("gh_cron_1620_kst_weekdays")
    gh_write_perm = Bool("gh_contents_write_perm")
    gh_auto_commit = Bool("gh_auto_commit_state")
    tm_utc_ok = Bool("timer_utc_4h_6runs")
    tm_persistent = Bool("timer_persistent")
    cfg_pc_off = Bool("config_pc_off_enabled")

    expected_hours = [0, 4, 8, 12, 16, 20]
    checks.append(
        check_constraint(
            "sla1_pc_off_cloud_24_7",
            [
                gh_cloud_runner == (wf["runs_on"] == "ubuntu-latest"),
                gh_cron_4h_ok == ("5 0,4,8,12,16,20 * * *" in wf["crons"]),
                gh_cron_krx_ok == ("20 7 * * 1-5" in wf["crons"]),
                gh_write_perm == (wf["contents_permission"] == "write"),
                gh_auto_commit == bool(wf["has_auto_commit"]),
                tm_utc_ok == (tm["is_utc"] and tm["utc_hours"] == expected_hours and tm["minute"] == 2),
                tm_persistent == tm["persistent"],
                cfg_pc_off == (
                    bool(cfg.get("cloud_24_7_enabled"))
                    and bool(cfg["schedules"]["daily_signal_report"]["pc_off_capable"])
                    and bool(cfg["schedules"]["upbit_turbo1_bot"]["pc_off_capable"])
                ),
                And(
                    gh_cloud_runner,
                    gh_cron_4h_ok,
                    gh_cron_krx_ok,
                    gh_write_perm,
                    gh_auto_commit,
                    tm_utc_ok,
                    tm_persistent,
                    cfg_pc_off,
                ),
            ],
        )
    )

    # ---------------------------------------------------------
    # SLA-2: 레버리지 0% (현물 1x)
    # ---------------------------------------------------------
    lev = Real("leverage")
    checks.append(
        check_constraint(
            "sla2_zero_leverage_spot_1x",
            [
                lev == float(cfg["risk_guardrails"]["leverage"]),
                lev == 1.0,
            ],
        )
    )

    # ---------------------------------------------------------
    # SLA-3: 마스터 v2 비중 합계 100.0% (85:15 정확 준수)
    # ---------------------------------------------------------
    pw = cfg["portfolio_weights"]
    w_c = Real("w_crypto")
    w_k = Real("w_krx")
    w_l = Real("w_longterm")
    w_r = Real("w_rocket")
    total_w = w_c + w_k + w_l + w_r
    core_w = w_c + w_k + w_l
    checks.append(
        check_constraint(
            "sla3_portfolio_weights_master_v2_100pct",
            [
                w_c == float(pw["crypto_short_turbo1"]),
                w_k == float(pw["krx_medium_d3"]),
                w_l == float(pw["longterm_core_l3"]),
                w_r == float(pw["skh_rocket_2x"]),
                total_w >= 0.999999,
                total_w <= 1.000001,
                w_r == 0.15,
                core_w >= 0.849999,
                core_w <= 0.850001,
                w_c >= 0.2833,
                w_c <= 0.2834,
                w_k >= 0.2833,
                w_k <= 0.2834,
                w_l >= 0.2833,
                w_l <= 0.2834,
            ],
        )
    )

    # ---------------------------------------------------------
    # SLA-4: 업비트 Turbo-1 리스크 가드레일 불변 준수
    # ---------------------------------------------------------
    g = cfg["risk_guardrails"]
    coins = cfg["upbit_coins"]
    rpt = Real("risk_per_trade")
    cap = Real("single_coin_cap")
    max_pos = Int("max_positions")
    hwm_dd = Real("hwm_throttle_dd")
    kill_dd = Real("daily_loss_kill_switch")
    n_coins = Int("num_upbit_krw_coins")
    all_krw = Bool("all_coins_krw_spot")

    checks.append(
        check_constraint(
            "sla4_turbo1_risk_guardrails",
            [
                rpt == float(g["risk_per_trade"]),
                cap == float(g["single_coin_cap"]),
                max_pos == int(g["max_positions"]),
                hwm_dd == float(g["hwm_throttle_dd"]),
                kill_dd == float(g["daily_loss_kill_switch"]),
                n_coins == len(coins),
                all_krw == all(c.startswith("KRW-") for c in coins),
                rpt == 0.015,
                cap == 0.33,
                max_pos == 4,
                hwm_dd == -0.20,
                kill_dd == -0.05,
                n_coins == 8,
                all_krw,
            ],
        )
    )

    # ---------------------------------------------------------
    # SLA-5: 기본 Dry-Run 안전모드 이중 잠금
    # ---------------------------------------------------------
    cfg_dry = Bool("config_dry_run_true")
    wf_dry = Bool("workflow_default_dry_run_true")
    checks.append(
        check_constraint(
            "sla5_default_dry_run_safety_lock",
            [
                cfg_dry == (cfg.get("dry_run") is True),
                wf_dry == ("'true'" in wf["env_dry_run"] or wf["env_dry_run"].lower() == "true"),
                And(cfg_dry, wf_dry),
            ],
        )
    )

    # ---------------------------------------------------------
    # SLA-6: 1억 원 통합 가상투자 회계 무결성 (paper_portfolio_state.json 직접 파싱)
    # ---------------------------------------------------------
    paper_parsed = None
    if PAPER_STATE_PATH.exists():
        ps = json.loads(PAPER_STATE_PATH.read_text(encoding="utf-8"))
        pb = ps.get("buckets", {})
        b_keys = ["crypto_short_turbo1", "krx_medium_d3", "longterm_core_l3", "skh_rocket_2x"]
        init_cap = Real("paper_initial_capital")
        tot_eq = Real("paper_total_equity")
        sum_init = sum(float(pb[k]["initial_krw"]) for k in b_keys)
        sum_eq = sum(float(pb[k]["equity_krw"]) for k in b_keys)
        min_cash = min(float(pb[k]["cash_krw"]) for k in b_keys)
        p_sum_init = Real("paper_sum_bucket_initials")
        p_sum_eq = Real("paper_sum_bucket_equities")
        p_min_cash = Real("paper_min_bucket_cash")
        p_dry = Bool("paper_dry_run_true")
        checks.append(
            check_constraint(
                "sla6_paper_portfolio_100m_accounting",
                [
                    init_cap == float(ps.get("initial_capital_krw", 0)),
                    tot_eq == float(ps.get("total_equity_krw", 0)),
                    p_sum_init == sum_init,
                    p_sum_eq == sum_eq,
                    p_min_cash == min_cash,
                    p_dry == (ps.get("dry_run") is True),
                    init_cap == 100_000_000.0,
                    p_sum_init == 100_000_000.0,
                    tot_eq > 0.0,
                    p_sum_eq - tot_eq >= -1.0,
                    p_sum_eq - tot_eq <= 1.0,
                    p_min_cash >= 0.0,
                    p_dry,
                ],
            )
        )
        paper_parsed = {
            "initial_capital_krw": ps.get("initial_capital_krw"),
            "total_equity_krw": ps.get("total_equity_krw"),
            "sum_bucket_equities_krw": round(sum_eq, 2),
            "min_bucket_cash_krw": round(min_cash, 2),
            "day_num": ps.get("day_num"),
            "dry_run": ps.get("dry_run"),
        }
    else:
        checks.append({"name": "sla6_paper_portfolio_100m_accounting", "result": "UNSAT"})

    all_sat = all(c["result"] == "SAT" for c in checks)
    report = {
        "verified_at_kst": dt.datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST"),
        "parsed_files": {
            "config_json": str(CONFIG_PATH),
            "github_workflow_yml": str(WORKFLOW_PATH),
            "systemd_timer": str(TIMER_PATH),
            "paper_portfolio_state": str(PAPER_STATE_PATH),
        },
        "parsed_evidence": {
            "github_workflow": wf,
            "systemd_timer": tm,
            "config_dry_run": cfg.get("dry_run"),
            "config_weights_sum": round(sum(float(v) for v in pw.values()), 10),
            "paper_portfolio": paper_parsed,
        },
        "checks": checks,
        "passed_count": sum(1 for c in checks if c["result"] == "SAT"),
        "total_count": len(checks),
        "overall_verdict": "SAT" if all_sat else "UNSAT",
    }

    OUTPUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"=== Z3-Solver Phase 2 SLA Verification: {report['overall_verdict']} ({report['passed_count']}/{report['total_count']}) ===")
    for c in checks:
        print(f"  - {c['name']}: {c['result']}")
    return 0 if all_sat else 1


if __name__ == "__main__":
    raise SystemExit(main())
