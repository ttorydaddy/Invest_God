#!/usr/bin/env bash
# Invest_God Phase 2 — Oracle Cloud Always Free VPS 원클릭 설치 스크립트
# 업비트 Open API 고정 IP 화이트리스트 대응 + systemd timer 4시간봉 24/7 무인 가동
set -euo pipefail

APP_DIR="/opt/invest_god"
SERVICE_USER="${SUDO_USER:-ubuntu}"

echo "============================================================"
echo " Invest_God Phase 2 — Oracle Cloud 고정 IP 24/7 설치 시작"
echo "============================================================"

# 1. 고정 공인 IP 확인 (업비트 Open API 화이트리스트 등록용)
PUBLIC_IP=$(curl -s --max-time 5 https://ifconfig.me || echo "UNKNOWN")
echo "[1/6] 이 서버의 고정 공인 IP: ${PUBLIC_IP}"
echo "      👉 업비트 Open API 관리 페이지에서 위 IP를 허용 IP로 등록하세요."

# 2. 시스템 패키지 및 Python3 가상환경 준비
echo "[2/6] Python 3 및 필수 패키지 설치..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-venv python3-pip git curl

# 3. 작업 디렉토리 구성
echo "[3/6] ${APP_DIR} 디렉토리 동기화..."
sudo mkdir -p "${APP_DIR}"
sudo chown -R "${SERVICE_USER}:${SERVICE_USER}" "${APP_DIR}"

if [ -d "./live" ]; then
  cp -r ./live "${APP_DIR}/"
else
  echo "❌ 현재 폴더에 live/ 디렉토리가 없습니다. 프로젝트 루트에서 실행하세요."
  exit 1
fi

# 4. Python 가상환경 및 의존성 설치
echo "[4/6] 가상환경(venv) 생성 및 requirements.txt 설치..."
python3 -m venv "${APP_DIR}/venv"
"${APP_DIR}/venv/bin/pip" install --upgrade pip -q
"${APP_DIR}/venv/bin/pip" install -r "${APP_DIR}/live/requirements.txt" -q

# 5. 안전 기본값 .env 파일 생성 (없을 때만)
ENV_FILE="${APP_DIR}/live/.env"
if [ ! -f "${ENV_FILE}" ]; then
  cat << 'EOF' > "${ENV_FILE}"
# Invest_God Upbit Bot 환경변수 (대표님 승인 전까지 DRY_RUN=true 유지!)
DRY_RUN=true
UPBIT_ACCESS_KEY=
UPBIT_SECRET_KEY=
EOF
  chmod 600 "${ENV_FILE}"
  echo "[5/6] ${ENV_FILE} 생성 완료 (기본값 DRY_RUN=true, 권한 600)"
else
  echo "[5/6] 기존 ${ENV_FILE} 유지"
fi

# 6. systemd service & timer 등록
echo "[6/6] systemd service & timer 등록 및 활성화..."
sudo sed "s|User=ubuntu|User=${SERVICE_USER}|g" "${APP_DIR}/live/deploy/invest-god-upbit.service" | sudo tee /etc/systemd/system/invest-god-upbit.service > /dev/null
sudo cp "${APP_DIR}/live/deploy/invest-god-upbit.timer" /etc/systemd/system/invest-god-upbit.timer

sudo systemctl daemon-reload
sudo systemctl enable --now invest-god-upbit.timer

echo "============================================================"
echo " ✅ 설치 완료! 다음 실행 예정 시각 확인:"
systemctl list-timers invest-god-upbit.timer --no-pager
echo "------------------------------------------------------------"
echo " 수동 Dry-Run 테스트 명령:"
echo "   sudo systemctl start invest-god-upbit.service"
echo "   journalctl -u invest-god-upbit.service -n 50 --no-pager"
echo "============================================================"
