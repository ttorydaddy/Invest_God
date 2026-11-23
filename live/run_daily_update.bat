@echo off
chcp 65001 > nul
setlocal

echo ============================================================
echo  Invest_God Phase 2 — 1억 원 가상투자 일일 업데이트 (DRY-RUN)
echo ============================================================
set DRY_RUN=true
cd /d "%~dp0"

echo [1/5] 업비트 4시간봉 Turbo-1 시세 스캔 (Dry-Run)...
python upbit_turbo1_bot.py
if errorlevel 1 echo [경고] 업비트 봇 실행 중 오류 발생

echo [2/5] KRX D-3 / 장기 L-3 / 하이닉스 로켓 일일 시그널 분석...
python daily_signal_engine.py
if errorlevel 1 echo [경고] 일일 시그널 엔진 실행 중 오류 발생

echo [3/5] 1억 원 통합 가상투자 원장 시가평가 업데이트...
python paper_portfolio_engine.py
if errorlevel 1 echo [경고] 가상투자 원장 업데이트 오류 발생

echo [4/5] Z3-Solver SLA 및 회계 무결성 검증...
python verify_phase2_z3_sla.py

echo [5/5] PC HTML 대시보드 생성 및 브라우저 열기...
python dashboard_generator.py

if exist "%~dp0output\daily_dashboard.html" (
    echo ✅ 완료! 브라우저에서 대시보드를 엽니다...
    start "" "%~dp0output\daily_dashboard.html"
) else (
    echo ❌ 대시보드 파일 생성에 실패했습니다.
)

echo.
pause
endlocal
