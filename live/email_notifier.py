#!/usr/bin/env python3
"""
Invest_God Phase 2 — 일일 이메일 리포터 (email_notifier.py)
- 환경변수: SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_APP_PASSWORD, NOTIFY_EMAIL
- 안전 가드: DRY_RUN=true 이거나 SMTP 환경변수가 하나라도 미설정이면
  절대 외부 이메일을 발송하지 않고 live/output/latest_email_preview.html 에 저장합니다.
"""

from __future__ import annotations

import html
import json
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
OUTPUT_DIR = BASE_DIR / "output"
SIGNALS_PATH = OUTPUT_DIR / "latest_signals.json"
UPBIT_PATH = OUTPUT_DIR / "latest_upbit_status.json"
PREVIEW_OUT = OUTPUT_DIR / "latest_email_preview.html"


def build_email_html(signals: Dict[str, Any], upbit: Dict[str, Any]) -> str:
    b = signals.get("buckets", {})
    d3 = b.get("krx_medium_d3", {})
    l3 = b.get("longterm_core_l3", {})
    rk = b.get("skh_rocket_2x", {})
    rb = signals.get("quarterly_rebalancing", {})

    def emoji(light: str) -> str:
        return {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}.get((light or "").upper(), "⚪")

    cands = d3.get("breakout_candidates", [])
    cand_txt = ", ".join(f"{c['name']}({c['code']}, {c['vol_surge_ratio']}x)" for c in cands[:5]) if cands else "없음"
    w = l3.get("target_weights_pct", {})

    return f"""<!DOCTYPE html>
<html lang="ko">
<head><meta charset="UTF-8" /></head>
<body style="font-family:Arial,'Malgun Gothic',sans-serif;background:#f8fafc;padding:20px;color:#0f172a;">
  <div style="max-width:680px;margin:0 auto;background:#ffffff;border-radius:12px;padding:24px;border:1px solid #e2e8f0;">
    <h2 style="margin:0 0 6px;color:#0f172a;">📊 Invest_God 마스터 v2 (85:15) 일일 시그널 리포트</h2>
    <p style="margin:0 0 18px;color:#64748b;font-size:13px;">기준 시각: {html.escape(signals.get('generated_at_kst','-'))} | 모드: {'DRY-RUN 모의' if signals.get('dry_run', True) else 'LIVE 실전'}</p>

    <table style="width:100%;border-collapse:collapse;font-size:14px;margin-bottom:18px;">
      <thead>
        <tr style="background:#f1f5f9;">
          <th style="padding:10px;text-align:left;border-bottom:2px solid #cbd5e1;">버킷 (비중)</th>
          <th style="padding:10px;text-align:center;border-bottom:2px solid #cbd5e1;">신호등</th>
          <th style="padding:10px;text-align:left;border-bottom:2px solid #cbd5e1;">오늘의 핵심 판정</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td style="padding:10px;border-bottom:1px solid #e2e8f0;"><strong>① 단기 코인 (28.33%)</strong></td>
          <td style="padding:10px;text-align:center;border-bottom:1px solid #e2e8f0;">{emoji(upbit.get('traffic_light','YELLOW'))}</td>
          <td style="padding:10px;border-bottom:1px solid #e2e8f0;">{html.escape(upbit.get('summary','업비트 봇 대기 중'))}</td>
        </tr>
        <tr>
          <td style="padding:10px;border-bottom:1px solid #e2e8f0;"><strong>② 중기 KRX D-3 (28.33%)</strong></td>
          <td style="padding:10px;text-align:center;border-bottom:1px solid #e2e8f0;">{emoji(d3.get('traffic_light','YELLOW'))}</td>
          <td style="padding:10px;border-bottom:1px solid #e2e8f0;">{html.escape(d3.get('summary','-'))}<br/><span style="color:#475569;font-size:12px;">돌파 후보: {html.escape(cand_txt)}</span></td>
        </tr>
        <tr>
          <td style="padding:10px;border-bottom:1px solid #e2e8f0;"><strong>③ 장기 코어 L-3 (28.33%)</strong></td>
          <td style="padding:10px;text-align:center;border-bottom:1px solid #e2e8f0;">{emoji(l3.get('traffic_light','YELLOW'))}</td>
          <td style="padding:10px;border-bottom:1px solid #e2e8f0;">{html.escape(l3.get('summary','-'))}<br/><span style="color:#475569;font-size:12px;">목표: 나스닥 {w.get('NASDAQ100',0)}% / 코스피 {w.get('KOSPI200',0)}% / BTC {w.get('BTC_KRW',0)}% / 금 {w.get('GOLD',0)}% / 단기채 {w.get('SHORT_BOND',0)}%</span></td>
        </tr>
        <tr>
          <td style="padding:10px;border-bottom:1px solid #e2e8f0;"><strong>④ 하이닉스 로켓 (15.0%)</strong></td>
          <td style="padding:10px;text-align:center;border-bottom:1px solid #e2e8f0;">{emoji(rk.get('traffic_light','RED'))}</td>
          <td style="padding:10px;border-bottom:1px solid #e2e8f0;">{html.escape(rk.get('summary','-'))}</td>
        </tr>
      </tbody>
    </table>

    <div style="background:#eff6ff;border-left:4px solid #3b82f6;padding:12px 14px;border-radius:6px;font-size:13px;">
      <strong>📅 분기 리밸런싱(85:15) 상태:</strong>
      {'🚨 실행 구간입니다! 본진 85% : 로켓 15% 비중을 복원하세요.' if rb.get('is_rebalance_alert') else f"다음 리밸런싱 {rb.get('next_quarter_start','-')} (D-{rb.get('days_until_next_quarter','-')})"}
    </div>

    <p style="margin:18px 0 0;font-size:11px;color:#94a3b8;text-align:center;">
      Invest_God Phase 2 Cloud Automated Report • 상세 차트는 PC HTML 대시보드(daily_dashboard.html)에서 확인하세요.
    </p>
  </div>
</body>
</html>
"""


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    signals = json.loads(SIGNALS_PATH.read_text(encoding="utf-8")) if SIGNALS_PATH.exists() else {}
    upbit = json.loads(UPBIT_PATH.read_text(encoding="utf-8")) if UPBIT_PATH.exists() else {}

    dry_run_cfg = bool(cfg.get("dry_run", True))
    dry_run_env = os.environ.get("DRY_RUN", "true").lower() == "true"
    dry_run = dry_run_cfg or dry_run_env

    email_html = build_email_html(signals, upbit)
    PREVIEW_OUT.write_text(email_html, encoding="utf-8")

    smtp_host = os.environ.get("SMTP_HOST", "").strip()
    smtp_port = os.environ.get("SMTP_PORT", "587").strip()
    smtp_user = os.environ.get("SMTP_USER", "").strip()
    smtp_pass = os.environ.get("SMTP_APP_PASSWORD", "").strip()
    notify_email = os.environ.get("NOTIFY_EMAIL", "").strip()

    missing = [k for k, v in [("SMTP_HOST", smtp_host), ("SMTP_USER", smtp_user), ("SMTP_APP_PASSWORD", smtp_pass), ("NOTIFY_EMAIL", notify_email)] if not v]

    if dry_run or missing:
        reason = "DRY_RUN=true 안전모드" if dry_run else f"SMTP 환경변수 미설정 ({', '.join(missing)})"
        print(f"[DRY-RUN] 외부 이메일 발송 생략 ({reason}) -> 미리보기 저장 완료: {PREVIEW_OUT}")
        return 0

    # 실전 발송 (DRY_RUN=false & 모든 SMTP 변수 설정 시에만 동작)
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[Invest_God] 마스터 v2 일일 시그널 ({signals.get('generated_at_kst', '')[:10]})"
    msg["From"] = smtp_user
    msg["To"] = notify_email
    msg.attach(MIMEText(email_html, "html", "utf-8"))

    with smtplib.SMTP(smtp_host, int(smtp_port), timeout=20) as server:
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(smtp_user, [notify_email], msg.as_string())

    print(f"[LIVE] 이메일 발송 완료 -> {notify_email}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
