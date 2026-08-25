# LEAP Hand v1 실물 연결 및 시험 절차 (Ubuntu Linux / Windows)

이 문서는 Ubuntu Linux (및 Windows) 환경에서 DYNAMIXEL XC330 기반 16모터 LEAP Hand v1을 현재 프로젝트 코드로 점검하는 순서를 설명합니다. v2의 8모터 텐던형 LEAP Hand에는 이 코드를 사용하면 안 됩니다.

> **중요:** 실물 시험 중에는 손가락과 링크 주변에 사람의 손, 케이블, 공구를 두지 마세요. 소프트웨어 Torque OFF가 실패할 경우를 대비해 5V 전원을 즉시 물리적으로 차단할 수 있어야 합니다.

## 시험 단계 요약

다음 순서를 건너뛰지 않습니다.

```text
전원 및 배선 확인
→ 시리얼 포트(/dev/ttyUSB0)와 ID 0~15 확인
→ Torque OFF 연결 진단
→ 현재 자세 유지 Torque 시험
→ 모터별 편 손 영점 기록
→ 단일 관절 ±5° 저속 시험
→ 사람 손 캘리브레이션
→ MuJoCo 실시간 추종 확인
→ 실물 teleop 안전 기능 완성 후 통합 시험
```

## 1. 시험 전 준비

### 하드웨어

- LEAP Hand v1 16모터 버전인지 확인합니다.
- 손을 책상이나 프레임에 단단히 고정합니다.
- 손가락이 움직일 수 있는 공간에서 장애물을 제거합니다.
- 5V 전원을 즉시 차단할 수 있는 스위치 또는 커넥터를 준비합니다.
- LEAP Hand에 5V 전원과 Micro-USB 통신 케이블을 연결합니다.
- 가능하면 USB 허브와 여러 개의 연장 케이블을 사용하지 않습니다.

### 프로그램 (Ubuntu Linux)

터미널을 열고 프로젝트 폴더로 이동합니다.

```bash
cd ~/Desktop/IsaacSim/leap-hand
source .venv/bin/activate
```

### 시리얼 포트 권한 및 저지연 설정 (Ubuntu)

USB 변환기를 연결한 후 다음을 실행합니다.

```bash
# 1. 시리얼 포트 쓰기 권한 설정
sudo chmod 666 /dev/ttyUSB0
# (또는 sudo usermod -aG dialout $USER 후 재로그인)

# 2. FTDI USB 통신 지연 최소화 (1ms Low-Latency 모드)
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
```

## 2. 시리얼 포트와 모터 ID 확인

1. LEAP Hand에 5V 전원을 공급합니다.
2. Micro-USB 케이블을 컴퓨터에 연결합니다.
3. 시리얼 포트가 인식되었는지 확인합니다: `ls /dev/ttyUSB*`
4. DYNAMIXEL Wizard 또는 `leap_hand_hardware_check.py`로 모터 ID `0~15`가 모두 검색되는지 확인합니다.
5. DYNAMIXEL Wizard를 사용한 경우 **Wizard를 완전히 종료**합니다. (포트 점유 방지)

## 3. Torque OFF 연결 진단

처음 실행할 때는 `--torque-test`를 절대 추가하지 않습니다.

```bash
python leap_hand_hardware_check.py --port /dev/ttyUSB0
```

`--port` 인자를 생략해도 기본값으로 `/dev/ttyUSB0`이 사용됩니다.

이 명령은 다음 항목만 확인하고 Torque를 켜지 않습니다.

- ID `0~15`의 통신 응답과 모델 번호
- 현재 16개 모터 위치
- 모터 온도
- 입력 전압
- Hardware Error

정상 출력의 핵심 형태는 다음과 같습니다.

```text
Connected with torque OFF: COM13
Motor IDs/model numbers: {0: ..., 1: ..., ..., 15: ...}
Hardware errors: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Torque was never enabled.
Torque OFF; serial port closed.
```

다음 상황에서는 시험을 중단하고 원인을 먼저 해결합니다.

- ID가 하나라도 누락되거나 응답하지 않음
- Hardware Error가 하나라도 `0`이 아님
- 모터가 빨간색으로 점멸함
- 입력 전압이 비정상적이거나 모터별로 큰 차이가 남
- 연결만 했는데 손가락이 움직임
- 통신 오류가 반복됨

## 4. 현재 자세 유지 Torque 시험

손 주변을 비우고 전원 차단 수단을 손이 닿는 위치에 둡니다.

```bash
python leap_hand_hardware_check.py \
  --port /dev/ttyUSB0 \
  --current-limit 300 \
  --watchdog-ms 500 \
  --torque-test \
  --hold-seconds 2
```

실행 과정은 다음과 같습니다.

```text
현재 모터 위치 읽기
→ 현재 위치를 Goal Position으로 기록
→ Torque ON
→ 같은 목표를 2초 동안 재전송
→ Torque OFF
```

정상이라면 손가락이 튀지 않고 현재 자세에서 단단해졌다가 2초 후 다시 힘이 풀립니다.

다음 현상이 나타나면 즉시 5V 전원을 차단합니다.

- 손가락이 갑자기 튀거나 반대 방향으로 움직임
- 손이 심하게 떨림
- 링크끼리 강하게 누름
- 모터에서 비정상적인 소리나 냄새가 발생함
- 프로그램 종료 후에도 Torque가 유지됨
- `Torque-off was not acknowledged` 경고가 출력됨

## 5. 모터별 편 손 영점 기록

손을 완전히 편 중립 자세에 놓고, Torque OFF 상태에서 16개 모터의 raw 위치를 기록합니다. 이 값은 이후 프로젝트의 `0°` 명령이 해당 실물 손의 편 자세를 목표로 하게 만드는 모터별 offset입니다.

```bash
python leap_hand_motor_calibration.py --port /dev/ttyUSB0
```

프로그램은 모터 통신·온도·Hardware Error를 확인한 뒤 다음 입력을 기다립니다.

```text
Type RECORD to capture the open-pose calibration:
```

손을 의도한 편 자세로 고정한 뒤에만 대문자로 `RECORD`를 입력합니다. 이 과정은 Torque를 켜거나 모터를 움직이지 않고, 기본 31개 raw 위치 샘플의 중앙값을 기록합니다.

기본 저장 위치는 다음이며 개인 장비 데이터라 Git에서 제외됩니다.

```text
calibration/hardware_motors.yaml
```

이미 저장된 보정을 다시 기록하려면 명시적으로 `--force`를 붙입니다.

```bash
python leap_hand_motor_calibration.py --port /dev/ttyUSB0 --force
```

생성된 YAML에는 관절별 `motor_id`, `open_motor_radians`, `sign`이 들어 있습니다. `sign`은 처음에 모두 `1`이며, 다음 단일 관절 시험에서 특정 관절이 반대 방향으로 움직일 때만 해당 관절의 값을 `-1`로 바꿉니다.

```yaml
joints:
  index_pip_flex:
    motor_id: 2
    open_motor_radians: 3.141592654
    sign: 1
```

보정 파일을 적용할 때는 아래처럼 `--motor-calibration-file`을 추가합니다.

```bash
python leap_hand_hardware_check.py \
  --port /dev/ttyUSB0 \
  --motor-calibration-file calibration/hardware_motors.yaml
```

## 6. 단일 관절 저속 시험

전체 손을 한 번에 움직이지 않고 선택한 관절 하나만 현재 위치에서 `+5°` 이동했다가 원래 위치로 돌아오게 합니다.

첫 시험은 검지 PIP 관절, `+5°`, 최대 `30°/s`로 시작합니다.

```bash
python leap_hand_joint_test.py \
  --port /dev/ttyUSB0 \
  --joint index_pip_flex \
  --delta 5 \
  --max-joint-speed 30 \
  --current-limit 300 \
  --motor-calibration-file calibration/hardware_motors.yaml
```

프로그램은 계획을 출력한 뒤 다음 문구를 표시합니다.

```text
Clear the hand and type MOVE to enable torque (anything else cancels):
```

손 주변이 비어 있고 전원 차단 준비가 되어 있을 때만 대문자로 `MOVE`를 입력합니다. 다른 값을 입력하면 Torque를 켜지 않고 종료합니다.

정상 동작 순서는 다음과 같습니다.

```text
Torque OFF 상태에서 위치·온도·오류 확인
→ MOVE 입력 확인
→ 현재 위치를 다시 측정
→ 현재 위치로 Goal Position 초기화
→ Torque ON
→ 선택 관절만 +5° 저속 이동
→ 기본 1초 유지
→ 시작 위치로 저속 복귀
→ Torque OFF
```

반대 방향은 음수 delta로 확인합니다.

```bash
python leap_hand_joint_test.py \
  --port /dev/ttyUSB0 \
  --joint index_pip_flex \
  --delta -5 \
  --max-joint-speed 30 \
  --current-limit 300
```

단일 관절 시험에는 다음 하드 제한이 적용됩니다.

| 항목 | 기본값 | 허용 범위 |
|---|---:|---:|
| 상대 이동량 | `5°` | `0° 초과 ~ 15° 이하` |
| 최대 관절 속도 | `30°/s` | `0°/s 초과 ~ 60°/s 이하` |
| 전류 제한 | `300mA` | `1~300mA` |
| 유지 시간 | `1초` | `0~5초` |
| 명령 주기 | `50Hz` | `10~100Hz` |
| 최대 위치 추종 오차 | `15°` | `1~30°` |
| 최대 허용 온도 | `50°C` | `30~70°C` |

위치 추종 오차, 온도 또는 Hardware Error 조건을 위반하면 예외를 발생시키고 `finally`에서 Torque OFF를 요청합니다.

### 관절 이름

`--joint`에는 다음 이름 중 하나를 사용합니다.

```text
index_mcp_side       index_mcp_flex       index_pip_flex       index_dip_flex
middle_mcp_side      middle_mcp_flex      middle_pip_flex      middle_dip_flex
ring_mcp_side        ring_mcp_flex        ring_pip_flex        ring_dip_flex
thumb_cmc_side       thumb_cmc_flex       thumb_mcp_flex       thumb_ip_flex
```

권장 시험 순서는 검지, 중지, 약지, 엄지 순이며 각 관절을 `+5°`와 `-5°`로 따로 확인합니다. 방향이 예상과 다르면 전체 teleop으로 넘어가지 말고 모터 ID, 혼 조립 방향과 좌표 변환을 확인합니다.

## 7. 사람 손 캘리브레이션

실물 손의 전원과 Torque를 끈 상태에서 웹캠 캘리브레이션을 수행합니다.

```bash
python webcam_hand_tracking.py --profile jiwoo
```

1. 사람 손을 자연스럽게 완전히 펼칩니다.
2. `C`를 누르고 기본 1.5초 동안 유지합니다.
3. 같은 손으로 주먹을 완전히 쥡니다.
4. `F`를 누르고 기본 1.5초 동안 유지합니다.
5. 화면에 표시되는 16개 최종 각도의 방향과 범위를 확인합니다.

다른 사람은 별도 프로필을 사용합니다.

```bash
python webcam_hand_tracking.py --profile tester_name
```

## 8. MuJoCo 최종 확인

실물 teleop 전에 같은 프로필로 MuJoCo 추종을 다시 확인합니다.

```bash
python webcam_mujoco_teleop.py --profile jiwoo
```

확인 항목:

- 오른손만 MuJoCo 오른손을 제어함
- 편 손과 주먹 자세가 캘리브레이션 범위와 일치함
- 검지, 중지, 약지, 엄지의 방향과 순서가 실물 단일 관절 시험 결과와 일치함
- 가위와 주먹 자세에서 자기충돌 제한이 갑자기 손 전체를 펴지 않음
- 손을 카메라에서 치웠을 때 오래된 필터 상태가 초기화됨

## 9. 실물 teleop 전 필수 구현 상태

현재 저장소에는 다음 기능이 구현되어 있습니다.

- 실물 연결 및 Torque OFF 진단
- 현재 위치 유지 Torque 시험
- 단일 관절 상대각 저속 시험
- 관절 범위 제한
- 관절 속도 제한
- 500ms DYNAMIXEL Bus Watchdog
- 위치·속도·전류 피드백 읽기
- 온도·전압·Hardware Error 읽기
- 예외 발생 시 Torque OFF 요청

그러나 웹캠의 최종 관절각을 실물 모터로 보내는 실시간 teleop 실행 파일은 아직 만들지 않았습니다. 다음 기능이 구현되고 검증되기 전에는 `webcam_hand_tracking.py`의 `command_angles`를 실물 컨트롤러에 직접 연결하지 않습니다.

- 명시적인 ARM 키와 비상 정지 키
- 카메라 추적 손실 시간에 따른 Hold 및 Torque OFF
- 고정 주기 50~100Hz 모터 명령 루프
- 가장 최신 비전 목표만 사용하는 latest-wins 구조
- 실제 관절 위치와 명령 위치 차이의 연속 감시
- 전류·온도·Hardware Error의 실행 중 감시
- 통신 예외 또는 뷰어 종료 시 확실한 Torque OFF

권장 추적 손실 정책은 다음과 같습니다.

```text
0~200ms 손실     마지막 안전 목표 유지
200~500ms 손실   새 목표 전송 중단
500ms 이상       Torque OFF
```

## 10. 정상 종료 및 비상 정지

정상 종료 순서:

```text
새 목표 명령 중단
→ Torque OFF
→ 시리얼 포트 닫기
→ 5V 전원 차단
→ USB 케이블 분리
```

비정상 동작 시:

1. 가능하면 `Ctrl+C`로 프로그램을 중단합니다.
2. 움직임이 계속되거나 Torque OFF 응답이 없으면 즉시 5V 전원을 차단합니다.
3. 손가락을 억지로 잡아 멈추지 않습니다.
4. Hardware Error, 모터 ID, 배선, 혼 조립 방향을 다시 확인합니다.
5. 원인을 확인하기 전에는 Torque를 다시 켜지 않습니다.

Bus Watchdog가 작동하면 Goal Position 등의 목표 레지스터가 일시적으로 읽기 전용 상태가 될 수 있습니다. 다시 실행할 때 컨트롤러의 `configure()`가 Watchdog 상태를 초기화하지만, 오류가 반복되면 전원을 차단하고 통신 상태를 점검합니다.

## 11. 소프트웨어 자체 테스트

실물 연결 전에 전체 자동 테스트를 실행할 수 있습니다.

```bash
python -m unittest discover -s tests -v
```

명령행 옵션만 확인하려면 다음을 실행합니다.

```bash
python leap_hand_hardware_check.py --help
python leap_hand_motor_calibration.py --help
python leap_hand_joint_test.py --help
python webcam_hand_tracking.py --help
```

## 참고 자료

- [LEAP Hand v1 공식 API 및 하드웨어 설정](https://github.com/leap-hand/LEAP_Hand_API)
- [ROBOTIS XC330 공식 제어표](https://emanual.robotis.com/docs/en/dxl/x/xc330-m288/)

