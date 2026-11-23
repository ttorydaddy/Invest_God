#!/usr/bin/env python3
"""
Invest_God Phase 2 — PC HTML 대시보드 생성기 (dashboard_generator.py)
- 1억 원 통합 가상투자 현황판 (Day N / 30, 4버킷 원장, 내장 SVG 수익 곡선, 가상 체결 내역)
- 4개 버킷 신호등 (단기 코인 Turbo-1 / 중기 KRX D-3 / 장기 코어 L-3 / 하이닉스 2배 로켓)
- 내일 아침 08:50 주문 체크리스트 & 분기 리밸런싱 알림
- 외부 CDN 의존성 0% (단일 HTML 파일, PC 오프라인 완벽 동작)
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
OUTPUT_DIR = BASE_DIR / "output"
STATE_DIR = BASE_DIR / "state"
DOCS_DIR = ROOT_DIR / "docs"

SIGNALS_PATH = OUTPUT_DIR / "latest_signals.json"
UPBIT_PATH = OUTPUT_DIR / "latest_upbit_status.json"
Z3_PATH = OUTPUT_DIR / "z3_phase2_sla.json"
WEATHER_PATH = OUTPUT_DIR / "latest_crypto_weather.json"
PAPER_STATE_PATH = STATE_DIR / "paper_portfolio_state.json"
EQUITY_CSV_PATH = STATE_DIR / "paper_equity_history.csv"
TRADES_CSV_PATH = STATE_DIR / "paper_trades_log.csv"
DASHBOARD_OUT = OUTPUT_DIR / "daily_dashboard.html"
PAGES_INDEX_OUT = DOCS_DIR / "index.html"
PAGES_NOJEKYLL = DOCS_DIR / ".nojekyll"


def badge_html(light: str, label: str = "") -> str:
    light = (light or "YELLOW").upper()
    mapping = {
        "GREEN": ("#10b981", "rgba(16,185,129,0.16)", "🟢 정상/상승"),
        "YELLOW": ("#f59e0b", "rgba(245,158,11,0.16)", "🟡 관망/부분방어"),
        "RED": ("#ef4444", "rgba(239,68,68,0.16)", "🔴 방어/대피"),
    }
    color, bg, default_text = mapping.get(light, mapping["YELLOW"])
    txt = html.escape(label or default_text)
    return (
        f'<span style="display:inline-flex;align-items:center;gap:6px;padding:4px 12px;'
        f'border-radius:999px;font-size:12px;font-weight:700;color:{color};background:{bg};'
        f'border:1px solid {color}55;">{txt}</span>'
    )


def fmt_krw(v: Any, signed: bool = False) -> str:
    try:
        f = float(v)
        return f"{f:+,.0f}원" if signed else f"{f:,.0f}원"
    except Exception:
        return "-"


def fmt_pct(v: Any, digits: int = 2) -> str:
    try:
        f = float(v)
        color = "#34d399" if f > 0 else ("#f87171" if f < 0 else "#94a3b8")
        return f'<span style="color:{color};font-weight:700;">{f:+.{digits}f}%</span>'
    except Exception:
        return "-"


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def build_equity_svg(history: List[Dict[str, str]], initial_krw: float = 100_000_000.0) -> str:
    """외부 CDN 없이 순수 내장 SVG로 일별 평가액 추이 차트를 생성합니다."""
    w, h = 760, 200
    pad_l, pad_r, pad_t, pad_b = 78, 24, 24, 34
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b

    if not history:
        return '<div style="padding:24px;text-align:center;color:#94a3b8;font-size:13px;">일별 평가액 데이터가 아직 없습니다.</div>'

    vals = [float(r["total_equity_krw"]) for r in history]
    dates = [r["date"] for r in history]
    min_v = min(min(vals), initial_krw)
    max_v = max(max(vals), initial_krw)
    span = max(max_v - min_v, initial_krw * 0.01)  # 최소 ±1% 스케일 확보
    y_min = min_v - span * 0.25
    y_max = max_v + span * 0.25

    def to_y(v: float) -> float:
        return pad_t + plot_h * (1.0 - (v - y_min) / (y_max - y_min))

    def to_x(i: int) -> float:
        if len(vals) == 1:
            return pad_l + plot_w * 0.5
        return pad_l + plot_w * (i / (len(vals) - 1))

    base_y = to_y(initial_krw)
    pts = [(to_x(i), to_y(v)) for i, v in enumerate(vals)]
    poly_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)

    area_svg = ""
    line_svg = ""
    if len(pts) >= 2:
        area_pts = f"{pts[0][0]:.1f},{pad_t+plot_h:.1f} {poly_str} {pts[-1][0]:.1f},{pad_t+plot_h:.1f}"
        area_svg = f'<polygon points="{area_pts}" fill="url(#eqGrad)" />'
        line_svg = f'<polyline points="{poly_str}" fill="none" stroke="#38bdf8" stroke-width="2.6" stroke-linejoin="round" />'

    dots_svg = ""
    for i, (x, y) in enumerate(pts):
        dots_svg += (
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="#38bdf8" stroke="#0b1120" stroke-width="2">'
            f'<title>{dates[i]}: {vals[i]:,.0f}원</title></circle>'
        )

    # X축 라벨 (최대 5개)
    x_labels = ""
    step = max(1, len(dates) // 5)
    idxs = sorted(set(list(range(0, len(dates), step)) + [len(dates) - 1]))
    for i in idxs:
        x_labels += f'<text x="{to_x(i):.1f}" y="{h - 10}" fill="#94a3b8" font-size="11" text-anchor="middle">{dates[i][5:]}</text>'

    # Y축 3단 눈금
    y_ticks = ""
    for frac in [0.0, 0.5, 1.0]:
        val = y_min + (y_max - y_min) * frac
        yy = to_y(val)
        y_ticks += (
            f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{w - pad_r}" y2="{yy:.1f}" stroke="#1f2937" stroke-dasharray="3,3" />'
            f'<text x="{pad_l - 8}" y="{yy + 4:.1f}" fill="#94a3b8" font-size="11" text-anchor="end">{val/1e8:.3f}억</text>'
        )

    day1_hint = (
        f'<text x="{pts[0][0]:.1f}" y="{pts[0][1] - 12:.1f}" fill="#e2e8f0" font-size="12" font-weight="700" text-anchor="middle">'
        f'Day 1 시작 ({vals[0]:,.0f}원)</text>'
        if len(vals) == 1
        else ""
    )

    return f"""<svg viewBox="0 0 {w} {h}" style="width:100%;height:auto;display:block;background:#0f172a;border-radius:12px;border:1px solid #1e293b;">
  <defs>
    <linearGradient id="eqGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.32" />
      <stop offset="100%" stop-color="#38bdf8" stop-opacity="0.02" />
    </linearGradient>
  </defs>
  {y_ticks}
  <line x1="{pad_l}" y1="{base_y:.1f}" x2="{w - pad_r}" y2="{base_y:.1f}" stroke="#f59e0b" stroke-width="1.2" stroke-dasharray="5,4" />
  <text x="{w - pad_r - 2}" y="{base_y - 5:.1f}" fill="#f59e0b" font-size="10" text-anchor="end">원금 1.000억 기준선</text>
  {area_svg}
  {line_svg}
  {dots_svg}
  {day1_hint}
  {x_labels}
</svg>"""


def build_paper_section(paper: Dict[str, Any], history: List[Dict[str, str]], trades: List[Dict[str, str]]) -> str:
    if not paper:
        return ""
    day_num = int(paper.get("day_num", 1))
    trial_days = int(paper.get("trial_days", 30))
    pct_prog = min(100.0, max(3.3, (day_num / max(1, trial_days)) * 100.0))

    init_krw = float(paper.get("initial_capital_krw", 100_000_000.0))
    tot_krw = float(paper.get("total_equity_krw", init_krw))
    pnl_krw = tot_krw - init_krw
    ret_pct = float(paper.get("cum_return_pct", 0.0))
    dd_pct = float(paper.get("drawdown_pct", 0.0))

    bucket_order = [
        ("crypto_short_turbo1", "🟠 ① 단기 코인 (Turbo-1)"),
        ("krx_medium_d3", "🔵 ② 중기 KRX (D-3)"),
        ("longterm_core_l3", "🟢 ③ 장기 코어 (L-3)"),
        ("skh_rocket_2x", "🩷 ④ 하이닉스 2배 로켓"),
    ]
    b_rows = ""
    for key, label in bucket_order:
        bk = paper.get("buckets", {}).get(key, {})
        b_init = float(bk.get("initial_krw", 0))
        b_eq = float(bk.get("equity_krw", 0))
        b_ret = ((b_eq / b_init - 1.0) * 100.0) if b_init > 0 else 0.0
        b_hold = bk.get("holdings_summary", "-")
        b_rows += (
            f"<tr>"
            f"<td><strong>{label}</strong></td>"
            f"<td>{fmt_krw(b_init)}</td>"
            f"<td><strong>{fmt_krw(b_eq)}</strong></td>"
            f"<td>{fmt_pct(b_ret, 3)}</td>"
            f"<td style='color:#cbd5e1;font-size:12px;'>{html.escape(b_hold)}</td>"
            f"</tr>"
        )

    trade_rows = ""
    for t in reversed(trades[-10:]):
        trade_rows += (
            f"<tr>"
            f"<td>{html.escape(t.get('date',''))}</td>"
            f"<td>{html.escape(t.get('action',''))}</td>"
            f"<td><strong>{html.escape(t.get('name',''))}</strong> ({html.escape(t.get('ticker',''))})</td>"
            f"<td>{html.escape(str(t.get('qty','')))}</td>"
            f"<td>{fmt_krw(t.get('price_krw',0))}</td>"
            f"<td>{fmt_krw(t.get('amount_krw',0))}</td>"
            f"<td style='color:#94a3b8;font-size:12px;'>{html.escape(t.get('reason',''))}</td>"
            f"</tr>"
        )
    if not trade_rows:
        trade_rows = '<tr><td colspan="7" style="text-align:center;color:#94a3b8;">아직 체결된 가상 매매 내역이 없습니다.</td></tr>'

    svg_chart = build_equity_svg(history, init_krw)
    pnl_color = "#34d399" if pnl_krw > 0 else ("#f87171" if pnl_krw < 0 else "#f1f5f9")

    return f"""
  <div class="card" style="margin-bottom:20px;border:1px solid #0284c7;background:linear-gradient(180deg,#0f172a 0%,#111827 100%);">
    <div class="card-head">
      <div>
        <h2 style="margin:0;font-size:20px;">💰 1억 원 통합 가상투자 현황판 (1개월 시범운행 Day {day_num} / {trial_days})</h2>
        <div style="font-size:12px;color:#94a3b8;margin-top:3px;">시작일: {html.escape(paper.get('start_date','-'))} | 최근 시가평가: {html.escape(paper.get('last_updated_kst','-'))} | 실제 시세 기반 수수료 반영 모의원장</div>
      </div>
      <div style="min-width:160px;">
        <div style="display:flex;justify-content:space-between;font-size:11px;color:#94a3b8;margin-bottom:4px;"><span>시범운행 진척도</span><strong>Day {day_num}/{trial_days}</strong></div>
        <div style="height:8px;background:#1e293b;border-radius:999px;overflow:hidden;"><div style="width:{pct_prog:.1f}%;height:100%;background:linear-gradient(90deg,#38bdf8,#10b981);"></div></div>
      </div>
    </div>

    <div class="kpi-row" style="grid-template-columns:repeat(auto-fit,minmax(180px,1fr));margin-top:14px;">
      <div class="kpi"><div class="kpi-label">총 통합 평가액</div><div class="kpi-val" style="color:#38bdf8;font-size:22px;">{fmt_krw(tot_krw)}</div></div>
      <div class="kpi"><div class="kpi-label">초기 가상 원금</div><div class="kpi-val" style="font-size:20px;">{fmt_krw(init_krw)}</div></div>
      <div class="kpi"><div class="kpi-label">누적 손익 (수익률)</div><div class="kpi-val" style="color:{pnl_color};font-size:19px;">{fmt_krw(pnl_krw, signed=True)} <span style="font-size:13px;">({ret_pct:+.3f}%)</span></div></div>
      <div class="kpi"><div class="kpi-label">현재 MDD (고점 대비)</div><div class="kpi-val" style="font-size:20px;">{fmt_pct(dd_pct, 3)}</div></div>
      <div class="kpi"><div class="kpi-label">운행 일수</div><div class="kpi-val" style="font-size:20px;">Day {day_num} / {trial_days}</div></div>
    </div>

    <div style="display:grid;grid-template-columns:1.25fr 1fr;gap:18px;margin-top:16px;">
      <div>
        <h4 style="margin:0 0 8px;font-size:14px;color:#e2e8f0;">📦 4개 버킷별 가상 원장 현황</h4>
        <table>
          <thead><tr><th>버킷</th><th>초기 원금</th><th>현재 평가액</th><th>수익률</th><th>현재 보유 자산</th></tr></thead>
          <tbody>{b_rows}</tbody>
        </table>
      </div>
      <div>
        <h4 style="margin:0 0 8px;font-size:14px;color:#e2e8f0;">📈 일별 총 평가액 추이 (내장 SVG)</h4>
        {svg_chart}
      </div>
    </div>

    <h4 style="margin:18px 0 8px;font-size:14px;color:#e2e8f0;">🧾 최근 가상 체결 내역 (paper_trades_log.csv)</h4>
    <table>
      <thead><tr><th>일자</th><th>구분</th><th>종목/자산</th><th>수량</th><th>체결가</th><th>체결금액</th><th>사유</th></tr></thead>
      <tbody>{trade_rows}</tbody>
    </table>
  </div>
"""


def build_checklist(signals: Dict[str, Any], upbit: Dict[str, Any]) -> List[str]:
    items: List[str] = []
    b = signals.get("buckets", {})
    d3 = b.get("krx_medium_d3", {})
    l3 = b.get("longterm_core_l3", {})
    rk = b.get("skh_rocket_2x", {})
    rb = signals.get("quarterly_rebalancing", {})

    if rk.get("signal") == "ROCKET_ON":
        items.append(f"[로켓 15%] SK하이닉스 골든크로스 유지 ({rk.get('spread_ma20_vs_ma120_pct', 0):+.2f}%) → SK하이닉스 2배 ETF/ETN 보유 유지")
    elif rk.get("signal") == "SAFE_RP":
        items.append("[로켓 15%] ⚠️ SK하이닉스 데드크로스! 내일 08:50 시가에 2배 ETF 전량 매도 → RP/단기채 전액 대피")

    cands = d3.get("breakout_candidates", [])
    if cands:
        names = ", ".join(f"{c['name']}({c['code']})" for c in cands[:5])
        items.append(f"[중기 D-3 28.33%] 기술적 돌파 후보 {len(cands)}종목 ({names}) → HTS에서 외인+기관 순매수(>=+2%, 3일누적>0) 확인 후 시가 갭 +15% 이하만 매수")
    else:
        items.append("[중기 D-3 28.33%] 오늘 신규 돌파 후보 0종목 → 기존 보유 종목 20일선 이탈 여부 및 -8% 손절 / -15% 트레일링 감시주문만 유지")

    if rb.get("is_l3_month_end_window"):
        w = l3.get("target_weights_pct", {})
        items.append(
            f"[장기 L-3 28.33%] 📅 월말 리밸런싱 구간(월말 D-{rb.get('days_to_month_end')}) → "
            f"나스닥 {w.get('NASDAQ100',0)}% / 코스피 {w.get('KOSPI200',0)}% / BTC {w.get('BTC_KRW',0)}% / "
            f"금 {w.get('GOLD',0)}% / 단기채 {w.get('SHORT_BOND',0)}% 목표 비중 조정"
        )
    else:
        items.append(f"[장기 L-3 28.33%] 월말 리밸런싱 대기 중 (월말까지 D-{rb.get('days_to_month_end', '-')}) — 현재 레짐 모니터링만 유지")

    if upbit:
        items.append(f"[단기 코인 Turbo-1 28.33%] 클라우드 봇 자동 감시 중 — {upbit.get('summary', '')}")

    if rb.get("is_rebalance_alert"):
        items.append("🚨 [분기 리밸런싱 실행일] 본진 85% : 로켓 15% 비중 기계적 복원 필요!")

    return items


def build_crypto_weather_section(weather: Dict[str, Any]) -> str:
    if not weather:
        return ""
    st = weather.get("stablecoin_mcap", {})
    ib = weather.get("ibit_etf", {})
    fg = weather.get("fear_and_greed", {})

    st_val = f"${st.get('mcap_billion_usd', 0):,.2f}B" if st.get("status") == "OK" else "조회 지연"
    st_sub = (
        f"7일 {fmt_pct(st.get('chg_7d_pct', 0))} ({st.get('chg_7d_billion_usd', 0):+.2f}B$) &nbsp;|&nbsp; "
        f"30일 {fmt_pct(st.get('chg_30d_pct', 0))} ({st.get('chg_30d_billion_usd', 0):+.2f}B$)"
        if st.get("status") == "OK"
        else "-"
    )

    ib_val = f"${ib.get('close_usd', 0):,.2f}" if ib.get("status") == "OK" else "조회 지연"
    ib_sub = (
        f"MA20 대비 {fmt_pct(ib.get('gap_ma20_pct', 0))} &nbsp;|&nbsp; "
        f"5일 {fmt_pct(ib.get('ret_5d_pct', 0))} &nbsp;|&nbsp; 거래대금 {ib.get('vol_ratio_5d_vs_20d', 1.0):.2f}x"
        if ib.get("status") == "OK"
        else "-"
    )

    fg_val = f"{fg.get('score', '-')} / 100" if fg.get("status") == "OK" else "조회 지연"
    fg_diff = fg.get("diff_7d", 0)
    fg_sub = (
        f"등급: <strong>{html.escape(fg.get('grade_kr','-'))}</strong> &nbsp;|&nbsp; "
        f"7일 전({fg.get('score_7d_ago','-')}점) 대비 <strong>{fg_diff:+d}p</strong>"
        if fg.get("status") == "OK"
        else "-"
    )

    return f"""
  <div class="card" style="margin-bottom:20px;border:1px solid #334155;">
    <div class="card-head">
      <div>
        <h3 class="card-title">🌤️ 코인 수급 기상청 (참고용 — 자동매매 룰 미개입)</h3>
        <div style="font-size:12px;color:#94a3b8;margin-top:2px;">갱신: {html.escape(weather.get('generated_at_kst','-'))} | 무료 공개 데이터 기반 거시 수급 참고 지표 (Turbo-1/D-3/L-3/로켓 매매 규칙에 일절 개입하지 않음)</div>
      </div>
      <span style="background:rgba(148,163,184,0.14);color:#cbd5e1;padding:4px 10px;border-radius:999px;font-size:11px;font-weight:600;">ℹ️ 순수 참고용</span>
    </div>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px;">
      <div style="background:#1e293b;border-radius:12px;padding:15px;border:1px solid #334155;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
          <span style="font-size:13px;font-weight:700;color:#e2e8f0;">💵 ① 스테이블코인 총시총 (실탄)</span>
          {badge_html(st.get('traffic_light','YELLOW'), st.get('label','-'))}
        </div>
        <div style="font-size:22px;font-weight:800;margin-bottom:4px;">{st_val}</div>
        <div style="font-size:12px;color:#cbd5e1;margin-bottom:8px;">{st_sub}</div>
        <div style="font-size:12px;color:#94a3b8;line-height:1.45;">{html.escape(st.get('comment_kr','-'))}</div>
      </div>

      <div style="background:#1e293b;border-radius:12px;padding:15px;border:1px solid #334155;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
          <span style="font-size:13px;font-weight:700;color:#e2e8f0;">🏦 ② 미국 BTC 현물 ETF (IBIT)</span>
          {badge_html(ib.get('traffic_light','YELLOW'), ib.get('label','-'))}
        </div>
        <div style="font-size:22px;font-weight:800;margin-bottom:4px;">{ib_val}</div>
        <div style="font-size:12px;color:#cbd5e1;margin-bottom:8px;">{ib_sub}</div>
        <div style="font-size:12px;color:#94a3b8;line-height:1.45;">{html.escape(ib.get('comment_kr','-'))}</div>
      </div>

      <div style="background:#1e293b;border-radius:12px;padding:15px;border:1px solid #334155;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
          <span style="font-size:13px;font-weight:700;color:#e2e8f0;">🌡️ ③ 크립토 공포·탐욕 지수</span>
          {badge_html(fg.get('traffic_light','YELLOW'), fg.get('label','-'))}
        </div>
        <div style="font-size:22px;font-weight:800;margin-bottom:4px;">{fg_val}</div>
        <div style="font-size:12px;color:#cbd5e1;margin-bottom:8px;">{fg_sub}</div>
        <div style="font-size:12px;color:#94a3b8;line-height:1.45;">{html.escape(fg.get('comment_kr','-'))}</div>
      </div>
    </div>
  </div>
"""


def render_html(
    signals: Dict[str, Any],
    upbit: Dict[str, Any],
    z3_sla: Dict[str, Any],
    paper: Dict[str, Any],
    history: List[Dict[str, str]],
    trades: List[Dict[str, str]],
    weather: Dict[str, Any],
) -> str:
    b = signals.get("buckets", {})
    d3 = b.get("krx_medium_d3", {})
    l3 = b.get("longterm_core_l3", {})
    rk = b.get("skh_rocket_2x", {})
    rb = signals.get("quarterly_rebalancing", {})

    dry_run = signals.get("dry_run", True)
    mode_badge = (
        '<span style="background:#1e3a8a;color:#93c5fd;padding:5px 12px;border-radius:8px;font-weight:700;font-size:12px;">🛡️ DRY-RUN 모의 안전모드</span>'
        if dry_run
        else '<span style="background:#7f1d1d;color:#fca5a5;padding:5px 12px;border-radius:8px;font-weight:700;font-size:12px;">⚡ LIVE 실전 모드</span>'
    )

    z3_verdict = z3_sla.get("overall_verdict", "PENDING")
    z3_cnt = f"{z3_sla.get('passed_count',0)}/{z3_sla.get('total_count',0)}"
    z3_badge = (
        f'<span style="background:#064e3b;color:#6ee7b7;padding:5px 12px;border-radius:8px;font-weight:700;font-size:12px;">✅ Z3 SLA: ALL SAT ({z3_cnt})</span>'
        if z3_verdict == "SAT"
        else f'<span style="background:#78350f;color:#fde68a;padding:5px 12px;border-radius:8px;font-weight:700;font-size:12px;">⚠️ Z3 SLA: {html.escape(str(z3_verdict))}</span>'
    )

    upbit_rows = ""
    for c in upbit.get("coins", []):
        if c.get("status") != "OK":
            upbit_rows += f"<tr><td>{html.escape(c.get('market',''))}</td><td colspan='7'>API 오류</td></tr>"
            continue
        sig_txt = "🚀 진입신호" if c.get("entry_signal") else ("📦 보유중" if c.get("holding") else "대기")
        upbit_rows += (
            f"<tr>"
            f"<td><strong>{html.escape(c['market'])}</strong></td>"
            f"<td>{fmt_krw(c['close_krw'])}</td>"
            f"<td>{'✅' if c['trend_ok'] else '❌'}</td>"
            f"<td>{'✅' if c['breakout_ok'] else '—'}</td>"
            f"<td>{c['vol_ratio']:.2f}x {'✅' if c['vol_ok'] else ''}</td>"
            f"<td>{fmt_krw(c['donchian_high_48'])}</td>"
            f"<td>{c['target_weight_pct']:.1f}%</td>"
            f"<td>{sig_txt}</td>"
            f"</tr>"
        )

    reg = d3.get("market_regime", {})
    regime_rows = ""
    for mkt_name in ["KOSPI", "KOSDAQ"]:
        r = reg.get(mkt_name, {})
        if r.get("status") == "OK":
            regime_rows += (
                f"<tr>"
                f"<td><strong>{mkt_name}</strong> ({r.get('as_of_date','')})</td>"
                f"<td>{r.get('close',0):,.2f}</td>"
                f"<td>{r.get('ma60',0):,.2f} ({fmt_pct(r.get('gap_ma60_pct',0))})</td>"
                f"<td>{fmt_pct(r.get('ret_5d_pct',0))}</td>"
                f"<td>{badge_html('GREEN' if r.get('regime_pass') else 'RED', '진입 허용' if r.get('regime_pass') else '휴식/방어')}</td>"
                f"</tr>"
            )

    cand_rows = ""
    for c in d3.get("breakout_candidates", []):
        cand_rows += (
            f"<tr><td><strong>{html.escape(c['name'])}</strong> ({c['code']})</td>"
            f"<td>{c['market']}</td><td>{c['close']:,}원</td><td>{fmt_pct(c['daily_ret_pct'])}</td>"
            f"<td>{c['trading_value_bkrw']:,.1f}억</td><td><strong>{c['vol_surge_ratio']:.2f}x</strong></td></tr>"
        )
    if not cand_rows:
        cand_rows = '<tr><td colspan="6" style="text-align:center;color:#94a3b8;">오늘 기술적 돌파 조건을 모두 만족한 종목이 없습니다.</td></tr>'

    l3_rows = ""
    for a in l3.get("offensive_assets", []):
        if a.get("status") == "OK":
            l3_rows += (
                f"<tr><td><strong>{html.escape(a['name'])}</strong> ({a['ticker']})</td>"
                f"<td>{a['as_of_date']}</td>"
                f"<td>{a['close']:,.0f}</td><td>{a['ma120']:,.0f} ({fmt_pct(a['gap_ma120_pct'])})</td>"
                f"<td>{badge_html('GREEN' if a['regime']=='RISK_ON' else 'RED', '보유' if a['regime']=='RISK_ON' else '금·채권 대피')}</td>"
                f"<td><strong>{a['target_weight_pct']:.1f}%</strong></td></tr>"
            )
    for d in l3.get("defensive_allocation", []):
        l3_rows += (
            f"<tr style='background:rgba(148,163,184,0.06);'><td>🛡️ {html.escape(d['name'])} ({d['ticker']})</td>"
            f"<td colspan='3' style='color:#94a3b8;'>이탈 공격자산 50:50 방어 대피처</td>"
            f"<td>{badge_html('YELLOW', '방어자산')}</td>"
            f"<td><strong>{d['target_weight_pct']:.1f}%</strong></td></tr>"
        )

    checklist_html = "".join(f"<li>{html.escape(item)}</li>" for item in build_checklist(signals, upbit))
    rebal_banner_bg = "rgba(239,68,68,0.18)" if rb.get("is_rebalance_alert") else "rgba(59,130,246,0.12)"
    rebal_border = "#ef4444" if rb.get("is_rebalance_alert") else "#3b82f6"
    rebal_title = (
        "🚨 분기 리밸런싱 실행 구간! (85:15 비중 복원 필요)"
        if rb.get("is_rebalance_alert")
        else f"📅 다음 분기 리밸런싱: {rb.get('next_quarter_start','-')} (D-{rb.get('days_until_next_quarter','-')})"
    )

    btc_reg = upbit.get("btc_regime", {})
    paper_acc = upbit.get("paper_account", {})
    risk_st = upbit.get("risk_state", {})
    paper_hero = build_paper_section(paper, history, trades)

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Invest_God Phase 2 — 1억 가상투자 & 일일 통합 퀀트 대시보드</title>
<style>
  :root {{
    --bg: #0b1120;
    --card: #111827;
    --card2: #1e293b;
    --border: #1f2937;
    --text: #f1f5f9;
    --muted: #94a3b8;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 24px; background: var(--bg); color: var(--text);
    font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    line-height: 1.55;
  }}
  .container {{ max-width: 1320px; margin: 0 auto; }}
  header {{
    display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;
    gap: 16px; padding-bottom: 18px; border-bottom: 1px solid var(--border); margin-bottom: 20px;
  }}
  h1 {{ margin: 0; font-size: 24px; letter-spacing: -0.02em; }}
  .sub {{ color: var(--muted); font-size: 13px; margin-top: 4px; }}
  .badges {{ display: flex; gap: 10px; flex-wrap: wrap; }}
  .alloc-bar {{ display: flex; height: 12px; border-radius: 999px; overflow: hidden; margin: 10px 0 6px; background: #1e293b; }}
  .alloc-legend {{ display: flex; flex-wrap: wrap; gap: 16px; font-size: 12px; color: var(--muted); margin-bottom: 20px; }}
  .grid-2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(540px, 1fr)); gap: 20px; margin-bottom: 20px; }}
  .card {{
    background: var(--card); border: 1px solid var(--border); border-radius: 16px;
    padding: 20px; box-shadow: 0 10px 25px rgba(0,0,0,0.25);
  }}
  .card-head {{ display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 12px; flex-wrap: wrap; }}
  .card-title {{ font-size: 17px; font-weight: 700; margin: 0; }}
  .summary {{ font-size: 14px; color: #cbd5e1; margin: 0 0 14px; padding: 10px 12px; background: rgba(30,41,59,0.65); border-radius: 10px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ padding: 9px 10px; text-align: left; border-bottom: 1px solid #1f2937; }}
  th {{ color: var(--muted); font-weight: 600; font-size: 12px; }}
  .kpi-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin-bottom: 14px; }}
  .kpi {{ background: var(--card2); border-radius: 10px; padding: 12px; }}
  .kpi-label {{ font-size: 11px; color: var(--muted); }}
  .kpi-val {{ font-size: 18px; font-weight: 700; margin-top: 2px; }}
  .checklist {{ background: linear-gradient(135deg, #0f172a, #172554); border: 1px solid #1d4ed8; border-radius: 16px; padding: 20px; margin-bottom: 20px; }}
  .checklist ul {{ margin: 10px 0 0; padding-left: 20px; }}
  .checklist li {{ margin-bottom: 8px; font-size: 14px; }}
  .rebal-box {{ background: {rebal_banner_bg}; border: 1px solid {rebal_border}; border-radius: 14px; padding: 16px 20px; margin-bottom: 20px; }}
  footer {{ text-align: center; color: var(--muted); font-size: 12px; margin-top: 28px; padding-top: 16px; border-top: 1px solid var(--border); }}
</style>
</head>
<body>
<div class="container">
  <header>
    <div>
      <h1>🚀 Invest_God Phase 2 — 마스터 v2 (85:15) 1억 가상투자 대시보드</h1>
      <div class="sub">생성 시각: {html.escape(signals.get('generated_at_kst','-'))} | PC HTML 대시보드 중심 1개월 시범운행</div>
    </div>
    <div class="badges">{mode_badge} {z3_badge}</div>
  </header>

  <div class="alloc-bar">
    <div style="width:28.33%;background:#f59e0b;"></div>
    <div style="width:28.33%;background:#3b82f6;"></div>
    <div style="width:28.34%;background:#10b981;"></div>
    <div style="width:15.00%;background:#ec4899;"></div>
  </div>
  <div class="alloc-legend">
    <span>🟠 단기 코인 Turbo-1: <strong>28.33%</strong></span>
    <span>🔵 중기 KRX D-3: <strong>28.33%</strong></span>
    <span>🟢 장기 코어 L-3: <strong>28.33%</strong></span>
    <span>🩷 하이닉스 2배 로켓: <strong>15.00%</strong></span>
  </div>

  {paper_hero}

  <div class="checklist">
    <h2 style="margin:0;font-size:17px;">📋 내일 아침(08:50) 시가 주문 & 오늘의 액션 체크리스트</h2>
    <ul>{checklist_html}</ul>
  </div>

  <div class="rebal-box">
    <strong>{html.escape(rebal_title)}</strong>
    <div style="font-size:13px;color:#cbd5e1;margin-top:4px;">목표 배분: {html.escape(rb.get('target_split',''))} (1·4·7·10월 첫 거래일 기계적 복원)</div>
  </div>

  <div class="grid-2">
    <div class="card">
      <div class="card-head">
        <h3 class="card-title">🩷 ④ 하이닉스 2배 로켓 (15.0%)</h3>
        {badge_html(rk.get('traffic_light','RED'), rk.get('signal','-'))}
      </div>
      <div class="summary">{html.escape(rk.get('summary','-'))}</div>
      <div class="kpi-row">
        <div class="kpi"><div class="kpi-label">SK하이닉스 종가 ({rk.get('as_of_date','-')})</div><div class="kpi-val">{fmt_krw(rk.get('close',0))}</div></div>
        <div class="kpi"><div class="kpi-label">MA20</div><div class="kpi-val">{fmt_krw(rk.get('ma20',0))}</div></div>
        <div class="kpi"><div class="kpi-label">MA120</div><div class="kpi-val">{fmt_krw(rk.get('ma120',0))}</div></div>
        <div class="kpi"><div class="kpi-label">MA20 vs MA120</div><div class="kpi-val">{fmt_pct(rk.get('spread_ma20_vs_ma120_pct',0))}</div></div>
      </div>
      <div style="font-size:13px;color:#cbd5e1;">권고 포지션: <strong>{html.escape(rk.get('action_kr','-'))}</strong></div>
    </div>

    <div class="card">
      <div class="card-head">
        <h3 class="card-title">🟢 ③ 장기 코어 L-3 (28.33%)</h3>
        {badge_html(l3.get('traffic_light','YELLOW'))}
      </div>
      <div class="summary">{html.escape(l3.get('summary','-'))}</div>
      <table>
        <thead><tr><th>자산</th><th>기준일</th><th>종가</th><th>120일선(괴리)</th><th>상태</th><th>목표비중</th></tr></thead>
        <tbody>{l3_rows}</tbody>
      </table>
    </div>
  </div>

  <div class="grid-2">
    <div class="card">
      <div class="card-head">
        <h3 class="card-title">🔵 ② 중기 KRX D-3 주도주 (28.33%)</h3>
        {badge_html(d3.get('traffic_light','YELLOW'))}
      </div>
      <div class="summary">{html.escape(d3.get('summary','-'))}</div>
      <table>
        <thead><tr><th>지수</th><th>종가</th><th>MA60(괴리)</th><th>5일 등락</th><th>레짐</th></tr></thead>
        <tbody>{regime_rows}</tbody>
      </table>
      <h4 style="margin:16px 0 8px;font-size:14px;">🚀 오늘의 기술적 돌파 후보 (실시간 스캔 {d3.get('live_scan_stats',{}).get('scanned_ok',0)}/{d3.get('live_scan_stats',{}).get('universe_size',0)}종목)</h4>
      <table>
        <thead><tr><th>종목</th><th>시장</th><th>종가</th><th>등락률</th><th>거래대금</th><th>거래배수</th></tr></thead>
        <tbody>{cand_rows}</tbody>
      </table>
      <div style="font-size:12px;color:#94a3b8;margin-top:10px;">※ {html.escape(d3.get('investor_flow_note',''))}</div>
    </div>

    <div class="card">
      <div class="card-head">
        <h3 class="card-title">🟠 ① 단기 코인 업비트 Turbo-1 (28.33%)</h3>
        {badge_html(upbit.get('traffic_light','YELLOW'))}
      </div>
      <div class="summary">{html.escape(upbit.get('summary','업비트 봇 상태 데이터 없음'))}</div>
      <div class="kpi-row">
        <div class="kpi"><div class="kpi-label">BTC 종가 / EMA600</div><div class="kpi-val" style="font-size:14px;">{fmt_krw(btc_reg.get('close_krw',0))}<br/><span style="color:#94a3b8;font-size:12px;">EMA600: {fmt_krw(btc_reg.get('ema600_krw',0))}</span></div></div>
        <div class="kpi"><div class="kpi-label">가상계좌 평가액</div><div class="kpi-val" style="font-size:15px;">{fmt_krw(paper_acc.get('equity_krw',0))}</div></div>
        <div class="kpi"><div class="kpi-label">보유 / 리스크</div><div class="kpi-val" style="font-size:15px;">{paper_acc.get('open_positions_count',0)}/{paper_acc.get('max_positions',4)}종목 ({risk_st.get('effective_risk_pct',1.5)}%)</div></div>
        <div class="kpi"><div class="kpi-label">킬스위치</div><div class="kpi-val" style="font-size:14px;">{'🔴 발동' if risk_st.get('kill_switch_triggered') else '🟢 정상'}</div></div>
      </div>
      <table>
        <thead><tr><th>마켓</th><th>4h 종가</th><th>추세</th><th>48돌파</th><th>거래량</th><th>48고점</th><th>비중</th><th>신호</th></tr></thead>
        <tbody>{upbit_rows}</tbody>
      </table>
      <div style="font-size:12px;color:#94a3b8;margin-top:8px;">최신 실행: {html.escape(upbit.get('generated_at_kst','-'))} | 레버리지: {risk_st.get('leverage',1.0)}x (현물 100%)</div>
    </div>
  </div>

  {build_crypto_weather_section(weather)}

  <footer>
    Invest_God Phase 2 Automated Quant System • 1억 원 1개월 가상투자 시범운행 • Zero Leverage • Default DRY_RUN Safe Mode
  </footer>
</div>
</body>
</html>
"""


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    signals = json.loads(SIGNALS_PATH.read_text(encoding="utf-8")) if SIGNALS_PATH.exists() else {}
    upbit = json.loads(UPBIT_PATH.read_text(encoding="utf-8")) if UPBIT_PATH.exists() else {}
    z3_sla = json.loads(Z3_PATH.read_text(encoding="utf-8")) if Z3_PATH.exists() else {}
    weather = json.loads(WEATHER_PATH.read_text(encoding="utf-8")) if WEATHER_PATH.exists() else {}
    paper = json.loads(PAPER_STATE_PATH.read_text(encoding="utf-8")) if PAPER_STATE_PATH.exists() else {}
    history = load_csv_rows(EQUITY_CSV_PATH)
    trades = load_csv_rows(TRADES_CSV_PATH)

    doc = render_html(signals, upbit, z3_sla, paper, history, trades, weather)
    DASHBOARD_OUT.write_text(doc, encoding="utf-8")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PAGES_INDEX_OUT.write_text(doc, encoding="utf-8")
    if not PAGES_NOJEKYLL.exists():
        PAGES_NOJEKYLL.write_text("", encoding="utf-8")
    print(f"[OK] Generated upgraded 100M Paper Trading dashboard HTML: {DASHBOARD_OUT}")
    print(f"[OK] Synced GitHub Pages dashboard: {PAGES_INDEX_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
