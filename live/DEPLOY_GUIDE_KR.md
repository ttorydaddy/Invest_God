# Invest_God Phase 2 — PC 꺼짐 24/7 클라우드 배포 가이드 (대표님용)

대표님 PC가 **완전히 꺼져 있어도** 365일 24시간 자동으로 돌아가는 **무료·고정IP 이원화 클라우드 아키텍처**입니다.

| 담당 업무 | 클라우드 서버 | 비용 | 실행 주기 | 왜 이 서버를 쓰나요? |
| :--- | :--- | :---: | :--- | :--- |
| **KRX 중기(`D-3`) + 장기(`L-3`) + 하이닉스 로켓 + 이메일 리포트 + HTML 대시보드** | **GitHub Actions** (`ubuntu-latest`) | **무료** | 평일 `16:20 KST` (`07:20 UTC`) | 고정 IP 불필요, 완전 무인 스케줄링 + 대시보드 HTML 아티팩트 자동 보관 |
| **업비트 KRW 4시간봉 자동매매 (`Turbo-1`)** | **Oracle Cloud Always Free VPS** (`systemd timer`) | **평생 무료** | 매일 `00:02, 04:02, 08:02, 12:02, 16:02, 20:02 UTC` | **업비트 Open API는 고정 IP 화이트리스트 등록이 필수**이기 때문 (GitHub Actions는 매번 IP가 바뀜) |

---

## 1단계: GitHub Actions 일일 이메일 & 대시보드 자동화 (5분 완료)

1. 이 프로젝트(`Invest_God`)를 대표님의 **Private GitHub 저장소**에 올립니다.
2. GitHub 저장소 상단 메뉴 → **`Settings` → `Secrets and variables` → `Actions`**로 이동합니다.
3. **`Secrets` 탭**에 아래 5개 이메일 정보를 등록합니다 (Gmail 기준: Google 계정 → 보안 → 2단계 인증 → 앱 비밀번호 16자리 발급).
   - `SMTP_HOST`: `smtp.gmail.com`
   - `SMTP_PORT`: `587`
   - `SMTP_USER`: 대표님 Gmail 주소 (예: `myemail@gmail.com`)
   - `SMTP_APP_PASSWORD`: 발급받은 16자리 앱 비밀번호
   - `NOTIFY_EMAIL`: 리포트를 받을 이메일 주소
4. **`Variables` 탭** 확인:
   - 처음 테스트할 때는 `DRY_RUN` 변수를 안 넣거나 `true`로 둡니다 (외부 이메일 발송 없이 미리보기만 생성).
   - 실제 이메일을 받고 싶을 때 `DRY_RUN` = `false`로 바꾸고 `live/config.json`의 `"dry_run": false`로 변경하면 매일 평일 16:20 KST에 이메일이 도착합니다.
5. **PC에서 대시보드 보기**: GitHub → `Actions` 탭 → 최신 실행 클릭 → `invest-god-daily-report` 다운로드 → `daily_dashboard.html` 더블클릭!

---

## 2단계: Oracle Cloud 무료 티어 고정 IP 업비트 24시간 봇 배포 (10분 완료)

1. [Oracle Cloud Free Tier](https://www.oracle.com/kr/cloud/free/)에 가입 후 **Always Free VM (Ubuntu 22.04/24.04)** 인스턴스 1대를 생성합니다. (한국 춘천/서울 리전 추천)
2. SSH로 서버에 접속한 뒤 프로젝트를 복사하고 원클릭 설치 스크립트를 실행합니다:
   ```bash
   cd ~/Invest_God
   chmod +x live/deploy/oracle_cloud_setup.sh
   ./live/deploy/oracle_cloud_setup.sh
   ```
3. 설치 스크립트가 화면에 출력해 주는 **고정 공인 IP**를 메모합니다.
4. **업비트 웹/앱 → 마이페이지 → Open API 관리**에서 새 API 키를 발급합니다:
   - 권한: **자산조회, 주문조회, 주문하기** (⚠️ **출금하기 권한은 절대 체크하지 마세요!**)
   - 허용 IP 주소: 3번에서 메모한 **Oracle Cloud 고정 공인 IP** 입력
5. 서버의 `/opt/invest_god/live/.env` 파일을 열어 키를 입력합니다:
   ```bash
   nano /opt/invest_god/live/.env
   ```
   ```ini
   DRY_RUN=true
   UPBIT_ACCESS_KEY=여기에_액세스키
   UPBIT_SECRET_KEY=여기에_시크릿키
   ```
6. **최소 1~2주간 `DRY_RUN=true` 모의 실행**으로 로그와 가상 장부(`/opt/invest_god/live/state/upbit_paper_state.json`)가 정상 동작하는지 확인하세요.
   ```bash
   sudo systemctl start invest-god-upbit.service
   journalctl -u invest-god-upbit.service -n 50 --no-pager
   ```

---

## 3단계: 실전(실돈) 전환 절차 (안전 이중 잠금 해제)

이 시스템은 실수로 실돈이 나가는 것을 막기 위해 **`config.json` + `.env` 이중 잠금**이 걸려 있습니다.
대표님이 실전 전환을 최종 결정하셨을 때만 아래 2곳을 모두 `false`로 바꾸세요:
1. `/opt/invest_god/live/config.json` → `"dry_run": false`
2. `/opt/invest_god/live/.env` → `DRY_RUN=false`

> **안전 가드레일(코드 내 불변 하드코딩)**:
> - 레버리지 `1.0x` (현물 전용), 1회 리스크 `-1.5%`, 단일 코인 상한 `33%`, 최대 4종목
> - 고점 대비 `-20%` 도달 시 1회 리스크 자동으로 절반(`-0.75%`) 감속
> - 일일 손실 `-5%` 초과 또는 API 오류 시 **Kill-Switch** 자동 발동하여 신규 매수 즉시 차단
