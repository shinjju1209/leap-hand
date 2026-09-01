# LEAP Hand vision prototype

> 🎪 **전시회 / 부스 운영자 가이드**: 하드웨어 연결부터 3단계 사전 점검, 키오스크 앱 운영법은 [**BOOTH_OPERATOR_MANUAL.md**](BOOTH_OPERATOR_MANUAL.md)를 참고하세요.

노트북 웹캠과 MediaPipe Hand Landmarker로 손의 21개 랜드마크와 LEAP Hand에 대응하는 16개 사람 손 관절각을 실시간 확인하는 프로토타입입니다. 개인별 편 손/주먹 가동범위 보정 후 One Euro Filter와 데드밴드를 적용하고, 사람의 rock/paper/scissors 동작을 인식해 로봇 동작과 비교한 승패를 CSV에 기록합니다. MuJoCo 실시간 연동과 LEAP Hand v1 실물 제어용 안전 API도 포함하지만, 비전 스트림과 실물 모터의 직접 연동은 아직 활성화하지 않습니다.

## 1. 가상환경과 패키지 설치

### Linux (Ubuntu)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. MediaPipe 모델 다운로드

> 저장소에 `models/hand_landmarker.task`가 이미 포함되어 있지 않은 경우에만 다운로드합니다.

### Linux (Ubuntu)
```bash
mkdir -p models
curl -L -o models/hand_landmarker.task "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
```

### Windows (PowerShell)
```powershell
New-Item -ItemType Directory -Force models
Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" `
  -OutFile "models\hand_landmarker.task"
```

## 3. MuJoCo 오른손 모델

Google DeepMind MuJoCo Menagerie의 LEAP Hand 오른손 MJCF 모델과 필요한 메시를 `models/mujoco/leap_hand`에 포함합니다. 실행 장면 파일은 `models/mujoco/leap_hand/scene_right.xml`이며, 원본 모델의 MIT 라이선스와 README도 같은 폴더에 보존합니다.

`MujocoHandController`는 현재 비전 코드의 16개 각도 순서를 MuJoCo actuator 순서로 변환하고, degree→radian 변환과 관절 한계 제한을 수행합니다.

큐브 재배향 데모(10절)가 쓰는 장면은 이것과 별개입니다. MuJoCo Playground의
`scene_mjx_cube.xml`과 그 메시·텍스처를 `assets/reorient_scene/`에 함께
포함하므로(28개 파일, 8 MB) Playground 체크아웃을 따로 받을 필요가 없습니다.
두 모델은 목적이 다릅니다 — `scene_right.xml`은 실물 손과 일치해야 하는
디지털 트윈이고, `scene_mjx_cube.xml`은 정책이 학습된 환경입니다.

```python
from mujoco_hand_controller import MujocoHandController

controller = MujocoHandController()
controller.set_target_degrees(filtered_angles)
controller.step_for(1.0 / 30.0)
```

오른손 손가락을 검지, 중지, 약지, 엄지 순서로 하나씩 굽혔다 펴는 테스트:

```bash
python mujoco_finger_test.py
```

rock, paper, scissors 자세를 차례로 확인하는 선택적 MuJoCo 미리보기:

```bash
python -m rps.mujoco_demo
```

웹캠의 오른손 동작을 MuJoCo LEAP Hand가 실시간으로 따라 하게 하려면:

```bash
python webcam_mujoco_teleop.py --profile jiwoo
```

MediaPipe 영상 창과 MuJoCo 뷰어가 함께 열립니다. 영상 창에서 `MUJOCO: FOLLOWING RIGHT HAND`가 표시되면 `개인별 가동범위 보정 → One Euro → 데드밴드`를 거친 16개 각도가 매 프레임 MuJoCo 컨트롤러로 전달됩니다. 오른손 모델만 포함되어 있으므로 `Right`로 인식된 손만 제어하고, 손이 없거나 왼손만 보이면 마지막 목표 자세를 유지합니다. 처음 사용하는 프로필은 편 손의 `C` 보정과 완전히 쥔 주먹의 `F` 보정을 차례로 수행합니다. `Q`/`Esc`를 누르거나 MuJoCo 뷰어를 닫으면 통합 실행이 종료됩니다.

자기충돌 방지 기능은 기본으로 활성화됩니다. 실제 목표를 적용하기 전에 별도의 MuJoCo 상태에서 손 링크끼리 접촉하는지 예측하고, 접촉이 예상되면 **직전 안전 자세와 새 목표 자세 사이**를 이분 탐색하여 접촉 직전까지만 움직입니다. 충돌 때마다 전체 손을 0°로 축소하지 않으므로 갑자기 손이 펴지지 않습니다. 제한이 작동하면 영상에 `MUJOCO: SELF-COLLISION HOLD | 74% TARGET`처럼 표시됩니다. 이 검사는 손 내부의 자기접촉만 제한하므로 추후 물체나 바닥과의 접촉과 구분할 수 있습니다.

충돌 제한 전후를 비교하는 진단 목적에서만 다음 옵션으로 끌 수 있습니다.

```bash
python webcam_mujoco_teleop.py --profile jiwoo --no-collision-avoidance
```

기존 웹캠 실행 명령에 `--mujoco`를 추가해도 동일하게 실행할 수 있습니다.

```bash
python webcam_hand_tracking.py --profile jiwoo --mujoco
```

실시간 실물 LEAP Hand 하드웨어 Teleoperation:

```bash
python webcam_hardware_teleop.py --profile jiwoo
```

실물 로봇과 MuJoCo 시뮬레이션을 동시에 실시간 추종하려면:

```bash
python webcam_hand_tracking.py --profile jiwoo --hardware --mujoco
```

## 4. LEAP Hand v1 실물 API

`LeapHandHardwareController`는 DYNAMIXEL XC330 기반 16모터 LEAP Hand v1 전용입니다. v2의 8모터 텐던 구조에는 사용할 수 없습니다. 공식 LEAPsim 관절 범위를 적용하고, `0°=편 손`인 현재 프로젝트 각도를 실물 모터의 `π rad=편 손` 좌표로 변환합니다. 모든 목표 명령에는 기본 `120°/s`의 관절별 속도 제한이 적용되며, 통신이 잠시 멈췄다가 재개돼도 한 번에 허용되는 변화량은 기본 0.1초분(`12°`)으로 제한됩니다.

안전을 위해 객체를 생성하거나 `connect()`만 호출해서는 토크가 켜지지 않습니다.

```text
connect()       → 시리얼 포트 열기, ID 0~15 확인, Torque OFF
configure()     → current-based position mode, 300mA, PID, 500ms watchdog
enable_torque() → 현재 위치를 목표값으로 먼저 기록한 뒤 Torque ON
command_degrees → 공식 관절 범위로 제한한 16개 degree 명령 전송
close()         → 즉시 Torque OFF 후 포트 닫기
```

먼저 토크를 전혀 켜지 않는 연결·센서 점검을 실행합니다. 포트 기본값은 `/dev/ttyUSB0`입니다.

```bash
python leap_hand_hardware_check.py --port /dev/ttyUSB0
```

16개 ID, 현재 위치, 온도, 입력 전압과 hardware error가 모두 정상인 것을 확인한 뒤에만 현재 위치 유지 토크 테스트를 수행합니다. 이 테스트는 새로운 자세를 명령하지 않지만, 실행 전 손 주변과 기구 사이에서 사람의 손을 치우고 전원 차단 수단을 준비해야 합니다.

```bash
python leap_hand_hardware_check.py --port /dev/ttyUSB0 --torque-test
```

실물 손을 편 중립 자세에 놓고 모터별 영점을 기록하려면, Torque OFF 상태에서 다음을 실행합니다. 기록 파일은 기본적으로 `calibration/hardware_motors.yaml`에 저장되며 개인 장비 데이터이므로 Git에서 제외됩니다.

```bash
python leap_hand_motor_calibration.py --port /dev/ttyUSB0
```

손을 의도한 편 자세로 고정한 뒤 터미널에 정확히 `RECORD`를 입력합니다. 이 과정은 모터를 움직이거나 Torque를 켜지 않습니다. 생성된 파일은 각 관절의 모터 ID, 편 손 raw 위치, 방향(`sign`)을 저장합니다. 방향은 모두 처음에 `1`로 저장되므로, 아래 단일 관절 시험에서 반대 방향으로 움직이는 관절만 `sign: -1`로 수정합니다. 보정 파일을 사용할 때는 점검·시험 명령에 `--motor-calibration-file calibration/hardware_motors.yaml`를 추가합니다.

그다음에는 전체 손이 아니라 관절 하나만 현재 자세에서 기본 `+5°` 이동했다가 원래 자세로 복귀하는 저속 테스트를 수행합니다. 이 스크립트는 최대 전류를 300mA, 이동량을 15°, 속도를 60°/s로 제한하며, 실행 전 터미널에 정확히 `MOVE`를 입력해야 Torque가 켜집니다. 나머지 15개 관절은 시작 위치를 유지합니다.

```bash
python leap_hand_joint_test.py \
  --port /dev/ttyUSB0 \
  --joint index_pip_flex \
  --delta 5 \
  --max-joint-speed 30 \
  --motor-calibration-file calibration/hardware_motors.yaml
```

반대 방향은 `--delta -5`로 확인합니다. 테스트 도중 명령과 실제 위치 차이가 기본 15°를 넘거나, 모터 온도가 기본 50°C 이상이거나, hardware error가 보고되면 예외를 발생시키고 `finally`에서 즉시 Torque OFF를 요청합니다.

Python API 예시:

```python
from leap_hand_hardware_controller import LeapHandHardwareController

with LeapHandHardwareController("/dev/ttyUSB0", current_limit_milliamps=300) as hand:
    hand.configure()
    hand.enable_torque()
    hand.command_degrees(safe_angles_degrees)
    feedback = hand.read_feedback()
    health = hand.read_health()
```

어떤 예외나 `Ctrl+C`가 발생해도 context manager와 점검 스크립트의 `finally`에서 Torque OFF를 요청합니다. 통신 자체가 끊기는 경우에는 500ms DYNAMIXEL Bus Watchdog가 추가로 동작합니다. 실물 비전 연동은 추적 손실 fail-safe와 피드백 감시를 더한 뒤 이 API에 연결합니다.

속도 제한은 `--max-joint-speed 90`처럼 점검 CLI에서 조절할 수 있습니다. `command_degrees()`를 반복 호출할 때 실제 경과 시간만큼 목표가 전진하므로, 별도의 보간 코드를 작성하지 않아도 컨트롤러 내부에서 급격한 목표 변화가 제한됩니다.

## 5. 실행

```bash
python webcam_hand_tracking.py
```

- 종료: `Q` 또는 `Esc`
- 카메라가 열리지 않으면: `python webcam_hand_tracking.py --camera 1`
- 좌우 반전을 끄려면: `python webcam_hand_tracking.py --no-mirror`
- 필터를 끄려면: `python webcam_hand_tracking.py --no-filter`
- 필터 튜닝: `python webcam_hand_tracking.py --min-cutoff 0.5 --beta 0.08 --derivative-cutoff 1.0`
- 데드밴드 기본값 및 튜닝: `python webcam_hand_tracking.py --deadband 1.2`
- 데드밴드를 끄려면: `python webcam_hand_tracking.py --deadband 0`
- 첫 라운드의 로봇 동작 지정: `python webcam_hand_tracking.py --robot-move rock`
- RPS 판정 안정화 프레임 수: `python webcam_hand_tracking.py --gesture-stable-frames 4`
- RPS 임계값 튜닝: `python webcam_hand_tracking.py --gesture-extended-max 55 --gesture-curled-min 100`
- 검지+엄지 scissors 튜닝: `python webcam_hand_tracking.py --gesture-thumb-extended-max 70 --gesture-thumb-extended-min-span 0.75 --gesture-thumb-curled-max-span 0.55`
- 개인 프로필 선택: `python webcam_hand_tracking.py --profile jiwoo`
- 중립 자세 수집 시간 변경: `python webcam_hand_tracking.py --profile jiwoo --calibration-seconds 2.0`
- 저장된 중립 보정을 적용하지 않으려면: `python webcam_hand_tracking.py --no-neutral-calibration`

손의 좌·우 판정은 항상 반전하지 않은 원본 카메라 프레임에서 수행합니다. `--mirror`는 사용자에게 보여주는 화면과 랜드마크 위치만 거울처럼 반전합니다.

화면에 `TRACKING`, FPS, 왼손/오른손 신뢰도, 21개 랜드마크와 다음 16개 관절각이 도 단위로 표시됩니다. 각 행은 `원본 > 중립 보정 + One Euro + 데드밴드 결과` 순서입니다. 데드밴드는 마지막 출력 명령과의 차이가 설정값보다 작은 관절을 이전 값으로 유지합니다.

- 검지·중지·약지: `MCP side`, `MCP flex`, `PIP flex`, `DIP flex`
- 엄지: `CMC side`, `CMC flex`, `MCP flex`, `IP flex`

각도는 MediaPipe의 3D 월드 랜드마크로 계산합니다. 손을 0.5초 이상 놓치면 오래된 필터 상태는 자동 초기화됩니다.

## 6. 사람 RPS 인식과 승패 기록

RPS 분류는 회전과 손 크기에 덜 민감하도록 MediaPipe의 3D 랜드마크에서 엄지(`T`), 검지(`I`), 중지(`M`), 약지(`R`), 새끼손가락(`P`)의 관절 굽힘각을 사용합니다. 기본 rock/paper/scissors에서는 엄지 위치를 무시하고, 검지+엄지 scissors 변형에서만 엄지가 펴지고 손바닥에서 떨어져 있는지 추가로 확인합니다.

| 동작 | 엄지 | 검지 | 중지 | 약지 | 새끼손가락 |
|---|---|---|---|---|---|
| rock | 무관 | 굽힘 | 굽힘 | 굽힘 | 굽힘 |
| paper | 무관 | 폄 | 폄 | 폄 | 폄 |
| scissors (V) | 무관 | 폄 | 폄 | 굽힘 | 굽힘 |
| scissors (check) | 폄/벌림 | 폄 | 굽힘 | 굽힘 | 굽힘 |

- 기본값에서 합산 굽힘각 `55°` 이하는 `extended`, `100°` 이상은 `curled`, 그 사이는 `ambiguous`입니다.
- 화면의 `I:E 12` 같은 값은 `손가락:상태 굽힘각`입니다. `E/C/A`는 각각 extended/curled/ambiguous입니다. `Tspan`은 엄지 끝과 손바닥 사이 거리를 손바닥 너비로 정규화한 값이며 기본적으로 `0.75` 이상이어야 check-sign scissors의 엄지로 인정됩니다.
- 같은 동작이 기본 4프레임 연속으로 인식돼야 확정됩니다. 손을 바꾸는 중간 자세는 라운드로 기록하지 않습니다.
- 실행 중 `1`은 로봇 rock, `2`는 paper, `3`은 scissors 라운드를 시작합니다. 실제 로봇 코드에서는 숫자 키 대신 로봇이 명령한 동작을 `start_round()`에 전달하면 됩니다.
- 확정된 라운드는 기본적으로 `rps_results.csv`에 `robot_move`, `human_move`, 사람 기준 `win/loss/tie`, UTC 시각과 함께 추가됩니다. `--results-csv PATH`로 위치를 변경할 수 있습니다.

로봇 제어 코드 연결 지점:

```python
from rps.rounds import CsvRoundRecorder, RpsRoundSession

session = RpsRoundSession(CsvRoundRecorder("rps_results.csv"))
session.start_round(robot_move)  # robot_move는 로봇 코드가 이미 알고 있는 값

# 비전 코드가 여러 프레임 뒤 확정한 human_move를 전달합니다.
record = session.observe_confirmed_human_move(human_move)
if record is not None:
    print(record.human_result)
```

전체 자동 테스트:

```bash
python -m unittest discover -s tests -v
```

## 7. 개인별 2점 가동범위 캘리브레이션

1. `--profile`에 사용할 사람 이름을 지정해 프로그램을 실행합니다.
2. 보정할 손을 자연스럽게 완전히 편 채 웹캠에 보여줍니다.
3. 화면이 손을 인식한 상태에서 `C`를 누르고 기본 1.5초 동안 자세를 유지합니다.
4. 첫 보정이 끝나면 같은 손으로 주먹을 완전히 쥐고 `F`를 누른 뒤 다시 1.5초 동안 유지합니다.
5. 두 자세의 중앙값이 해당 사람과 `Right`/`Left` 손에 따로 저장됩니다.

굽힘 관절은 `현재값 - 편 손 / 주먹 - 편 손`으로 0~100% 굽힘 비율을 계산한 뒤 MuJoCo의 안전한 주먹 목표각으로 매핑합니다. 특정 PIP/DIP 랜드마크가 가려져 측정 범위가 너무 작으면 같은 손가락의 다른 관절 굽힘 비율을 사용해 보완합니다. 좌우 각도(`MCP side`, `CMC side`)는 기존처럼 편 손의 중립값만 빼서 양·음 방향을 모두 유지합니다.

이후 처리 순서는 `개인별 2점 가동범위 보정 → One Euro Filter → 데드밴드`입니다. 같은 프로필에서 `C`를 다시 수행하면 기존 주먹 범위는 무효화되므로, 이어서 `F`도 다시 수행해야 합니다.

여러 프로필과 양손 보정값은 기본적으로 `calibration/neutral_angles.json`에 함께 저장됩니다. 이 폴더는 개인 측정 데이터이므로 Git에서 제외됩니다. 다른 파일을 사용하려면 `--calibration-file calibration/demo_team.json`처럼 지정할 수 있습니다.

2점 보정 후에도 전체 굽힘을 조금 강화하거나 줄이고 싶으면 캘리브레이션을 다시 하지 않고 `--flexion-scale`을 조절할 수 있습니다. 기본값은 `1.0`입니다.

```bash
# 10% 더 굽히기
python webcam_mujoco_teleop.py --profile jiwoo --flexion-scale 1.1

# 10% 덜 굽히기
python webcam_mujoco_teleop.py --profile jiwoo --flexion-scale 0.9
```

## 8. 관절 구조와 16 DoF

검지, 중지, 약지는 손바닥에서 손끝 방향으로 다음 세 관절을 가집니다.

```text
손바닥 → MCP → PIP → DIP → 손끝
```

- `MCP`(Metacarpophalangeal): 손바닥과 손가락이 연결되는 큰 관절입니다. 굽힘/펴짐(`MCP flex`)과 좌우 벌림/모음(`MCP side`)의 2개 자유도를 사용합니다.
- `PIP`(Proximal Interphalangeal): 손가락 가운데 관절이며 굽힘/펴짐 1개 자유도를 사용합니다.
- `DIP`(Distal Interphalangeal): 손끝과 가장 가까운 관절이며 굽힘/펴짐 1개 자유도를 사용합니다.

따라서 검지, 중지, 약지는 물리적 관절이 각각 3개지만, MCP를 두 방향으로 제어하므로 손가락마다 4 DoF입니다.

```text
MCP side + MCP flex + PIP flex + DIP flex = 손가락당 4 DoF
```

엄지는 다른 관절 이름과 구조를 사용합니다.

```text
손바닥 → CMC → MCP → IP → 엄지 끝
```

- `CMC`(Carpometacarpal): 엄지와 손바닥이 연결되는 관절이며 좌우 움직임(`CMC side`)과 굽힘(`CMC flex`)의 2개 자유도를 사용합니다.
- `MCP`: 엄지의 가운데 굽힘 관절로 1개 자유도를 사용합니다.
- `IP`(Interphalangeal): 엄지 끝쪽 굽힘 관절로 1개 자유도를 사용합니다. 엄지는 PIP와 DIP 대신 하나의 IP 관절을 사용합니다.

전체 제어 자유도는 다음과 같습니다.

| 손가락 | 제어 관절 | DoF |
|---|---|---:|
| 검지 | MCP side, MCP flex, PIP flex, DIP flex | 4 |
| 중지 | MCP side, MCP flex, PIP flex, DIP flex | 4 |
| 약지 | MCP side, MCP flex, PIP flex, DIP flex | 4 |
| 엄지 | CMC side, CMC flex, MCP flex, IP flex | 4 |
| 합계 |  | 16 |

이 프로젝트에서는 모든 굽힘각이 `0°`에 가까운 상태를 손을 편 중립 자세로 사용합니다. `MCP side`와 `CMC side`의 `0°`는 좌우로 기울지 않은 중앙 자세를 뜻합니다.

## 9. Sim-to-Real 강화학습(RL) Policy 배포

기본 설정은 official `models/LeapHand.pth`와 이에 맞는 안정 파지 상태를 사용해 큐브를 +Z축으로 회전합니다. Isaac의 관절 순서를 MuJoCo/실물의 `ANGLE_NAMES` 순서로 변환하고, 학습 당시의 20 Hz phase와 `1/24 rad` target 적분 크기를 유지합니다.

### 1) 실행 명령

```bash
# 1. 기본 official Policy를 MuJoCo에서 실행
python rl_sim2real_deploy.py --mode mujoco

# GUI 없이 5초 자동 검증
python rl_sim2real_deploy.py --mode mujoco --headless --auto-arm --duration 5

# 2. 실물 하드웨어 단독 배포
python rl_sim2real_deploy.py --mode hardware \
  --port /dev/ttyUSB0 \
  --motor-calibration-file calibration/hardware_motors.yaml

# 3. 실물 하드웨어 + MuJoCo 시뮬레이터 동시 실행 (Digital Twin 모드)
python rl_sim2real_deploy.py --mode both --port /dev/ttyUSB0
```

실물 모드는 저장된 모터별 편 손 원점과 방향을 필수로 사용합니다. 실행 직후 빈손으로 도달 가능한 cube-loading pose로 이동하고 그 자세를 유지합니다. `Cube-loading pose ready and holding` 문구를 확인한 다음 큐브를 지정된 파지 위치에 놓고 `A`를 누르세요. 큐브 지지 없이 도달하지 못하는 `middle_mcp_flex`를 포함해 정확한 MuJoCo policy grasp로 이동한 뒤 실측 관절값으로 policy/GRU를 초기화합니다. 실제 관절이 초기 목표의 12° 이내여야 policy가 시작되며, 실행 중 추종 오차가 25°를 넘으면 자동으로 Torque OFF 됩니다.

### 2) 조작 단축키 (GUI 창)

| 키 | 기능 | 설명 |
|---|---|---|
| **`A`** | **ARM (Policy 실행 시작)** | 초기 파지 자세로 이동 후 실시간 Policy 추론 및 제어 시작 |
| **`D` / `Space`** | **DISARM (비상 정지)** | Policy 제어 중단 및 하드웨어 토크 즉시 해제 |
| **`R`** | **RESET** | 손·큐브·속도·policy history/GRU를 하나의 초기 상태로 복원 |
| **`1` / `2`** | **회전 방향 전환** | `1`: 학습된 +Z 방향, `2`: 실험적인 -Z 방향 |
| **`Q` / `Esc`** | **종료** | 안전하게 Disarm 및 포트 종료 |

### 3) 주요 옵션

- `--config configs/inhand_cube_rotation.yaml`: 기본 파지 자세 및 제어 파라미터 YAML 파일
- `--control-hz 20.0`: Policy 추론 및 제어 루프 주기 (기본 20Hz)
- `--action-scale 0.0416666667`: target 적분 스케일. official checkpoint는 학습값 `1/24` 권장
- `--ema-alpha 0.8`: Sim-to-Real 떨림 방지를 위한 Action 지수이동평균(EMA) 필터 계수
- `--current-limit 350`: 파지/조작용 전류 제한 (mA)

큐브가 낙하하면 손과 큐브만 순간 이동시키지 않고 policy의 history와 GRU까지 함께 초기화합니다. 이 checkpoint는 +Z 회전으로 학습되었으므로 `2`의 역방향 동작은 동일한 성능을 보장하지 않습니다.

## 10. 전시회 및 부스 운영용 종합 키오스크 UI (`booth_app.py`)

관람객 체험 및 전시회 시연을 위한 원터치 터치스크린/키보드 지원 종합 키오스크 애플리케이션입니다.

### 0) 다른 장비에서 실행하기

설치는 1절과 같습니다. 부스 앱이 추가로 요구하는 것은 없습니다 — 손 모델,
재배향 정책, MuJoCo 장면이 모두 저장소에 들어 있고 LFS 도 쓰지 않으므로
clone 후 `pip install -r requirements.txt` 면 끝입니다.

이 저장소는 **비공개**입니다. 받는 사람이 저장소에 초대되어 있어야 하고,
`gh auth login` 이나 SSH 키로 인증이 되어 있어야 합니다. 둘 중 하나라도
빠지면 clone 은 `repository not found` 로 실패합니다 — 비공개 저장소는
권한이 없을 때 존재 자체를 숨기므로, 오타와 권한 문제가 같은 메시지로
나옵니다.

```bash
# 저장소 소유자가 먼저 초대합니다
gh api -X PUT repos/shinjju1209/leap-hand/collaborators/<github-id> -f permission=pull

# 받는 쪽
gh auth login
git clone https://github.com/shinjju1209/leap-hand
cd leap-hand
git checkout feat/booth-shape-ui
# 이후 1절의 가상환경 설치 절차를 그대로 따릅니다

python booth_app.py --mode mujoco --no-hardware   # 하드웨어 없이 확인
```

계정을 붙이기 번거로운 전시장 노트북이라면 파일을 그대로 옮겨도 됩니다.
저장소가 자립적이므로 `.git` 없이도 실행됩니다.

```bash
cd ~/Projects
tar czf leap-hand-booth.tar.gz --exclude=.venv --exclude=.git leap-hand/
```

실물 손을 붙이려면 매번 두 가지를 해야 합니다. USB 를 다시 꽂으면 둘 다
초기화되므로, 부스를 열 때마다 반복합니다.

```bash
sudo chmod 666 /dev/ttyUSB0
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
```

`latency_timer` 를 빠뜨리기 쉬운데, 이것이 제어 주기를 직접 결정합니다. FTDI
기본값은 16 ms 여서 명령 한 번마다 그만큼 기다리게 되고, 손이 늦게 따라오거나
끊겨 보입니다. 1 로 낮추면 사라집니다.

권한 쪽은 로그인 그룹으로 한 번에 해결할 수도 있습니다. 이 경우 `chmod` 는
다시 필요 없지만 `latency_timer` 는 여전히 매번 설정해야 합니다.

```bash
sudo usermod -aG dialout $USER      # 로그아웃 후 다시 로그인
```

전체 하드웨어 절차와 사전 점검은
[hardware_procedure.md](hardware_procedure.md) 와
[BOOTH_OPERATOR_MANUAL.md](BOOTH_OPERATOR_MANUAL.md) 에 있습니다.

설치 확인은 테스트로 합니다. 하드웨어도 카메라도 필요 없습니다.

```bash
python -m unittest tests.test_booth_theme tests.test_cube_reorient tests.test_booth_app
```

> `requirements.txt` 의 `torch` 와 `onnxruntime` 은 부스 앱이 쓰지 않습니다.
> 9절의 RL 배포 스크립트용이므로, 부스만 돌릴 것이라면 설치에 실패해도
> 무방합니다 (해당 테스트 4개만 건너뜁니다).

### 1) 실행 방법

```bash
# 1. 시뮬레이션 모드 (하드웨어 없이 웹캠과 MuJoCo로 시연)
python booth_app.py --mode mujoco

# 2. 실물 하드웨어 연동 부스 운영
python booth_app.py --mode hardware --port /dev/ttyUSB0

# 3. 실물 하드웨어 + MuJoCo 3D 디지털 트윈 동시 실행
python booth_app.py --mode both --port /dev/ttyUSB0
```

카메라는 자동으로 찾습니다. 인덱스 0번이 열리기만 하고 프레임을 주지 않는
장치인 경우가 있어(멀티 노드 웹캠) 순서대로 훑어 실제로 화면이 나오는 것을
씁니다. 고정하고 싶으면 `--camera-id 1` 처럼 직접 지정하세요.

저사양 장비에서 화면이 버벅이면 그림자를 끄세요. 얇은 테두리는 남으므로
납작해질 뿐 깨지지는 않습니다.

```bash
python booth_app.py --mode hardware --port /dev/ttyUSB0 --no-shadows
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--mode` | `hardware` | `hardware` / `mujoco` / `both` |
| `--port` | `/dev/ttyUSB0` | LEAP Hand 시리얼 포트 |
| `--camera-id` | `0` | 우선 시도할 카메라. 실패하면 0~5를 훑습니다 |
| `--profile` | `jiwoo` | 뉴트럴 캘리브레이션 프로필 이름 |
| `--current-limit` | `350` | 모터 전류 제한 (mA) |
| `--no-shadows` | — | 그림자를 끕니다 (저사양용) |
| `--no-mujoco` | — | 디지털 트윈 비활성화 |
| `--no-hardware` | — | 실물 손 비활성화 |

### 2) 텔레오퍼레이션 캘리브레이션 (중요)

편 손만 잡아도 동작하지만, 두 자세를 다 잡으면 가동범위가 제대로 나옵니다.

1. `C` — 편 손. 이 자세가 0도의 기준(뉴트럴 오프셋)이 됩니다.
2. `F` — 주먹. 이 자세가 최대 굽힘이 되어, 사람 손의 가동범위를 로봇의
   가동범위로 늘려주는 배율이 정해집니다.
3. `A` — Arm. 이제 손을 따라갑니다.

`F` 를 건너뛰면 배율이 적용되지 않아 사람 손의 각도가 그대로 나갑니다. 이 저장소의
프로필로 재보면 주먹을 꽉 쥐어도 관절당 40~60도가 나오므로, 로봇 손이 사람만큼
깊게 쥐지는 않습니다.

보정 상태는 `calibration/neutral_calibration.json` 의 `profiles`(편 손)와
`closed_profiles`(주먹) 양쪽에 값이 있는지로 확인할 수 있습니다.

### 3) 지원 모드 및 기능

| 모드 | 주요 기능 | 지원 단축키 |
|---|---|---|
| **메인 홈 (Home)** | 4개 서브 모드 카드 선택 및 전체 시스템 상태 확인 | `1` (텔레옵), `2` (RPS), `3` (쇼케이스), `4` (큐브 재배향), `Q` (종료) |
| **🖐️ 텔레오퍼레이션** | 웹캠 실시간 1:1 손 추적, 스켈레톤 시각화, 원유로 떨림 필터링 | `C` (편 손 보정), `F` (주먹 보정), `A` (Arm/추적 시작), `D`/`Space` (Disarm), `R` (보정 리셋), `H` (홈) |
| **✂️ 가위바위보 대결** | **3-2-1 대형 카운트다운**, 로봇 무작위 수 출력, 관람객 손 제스처 실시간 인식, 승/패/무 판정, 스코어보드 통계 | `Space` (라운드 시작), `P` (연속 자동 대결 ON/OFF), `R` (점수 초기화), `H` (홈) |
| **🎭 제스처 쇼케이스** | **3D 시뮬레이터 내장 뷰포트**, 원클릭 포즈 시연 (바위/보/가위/편손/중지/엄지척/OK사인/가리키기/락앤롤) 및 **동적 웨이브/인사 애니메이션** | `1` (바위), `2` (보), `3` (가위), `4` (편 손), `5` (중지), `6` (엄지 척), `7` (OK 사인), `8` (가리키기), `9` (락앤롤), `W` (핑거 웨이브), `V` (손 인사), `H` (홈) |
| **🧊 인핸드 큐브 재배향** | 학습된 강화학습 정책이 손 안에서 큐브를 목표 자세로 돌립니다. 왼쪽이 목표, 오른쪽이 정책이 다루는 큐브이고, 자세 오차를 실시간으로 표시합니다. **시뮬레이션 전용이라 실물 손에는 어떤 명령도 나가지 않습니다.** | `1`~`3` (목표 +90° X/Y/Z), `4`~`6` (−90°), `R` (무작위 목표), `0` (기준 자세), `G` (현재 큐브 자세를 목표로), `A` (학습용 자동 목표 ON/OFF), `Space` (일시정지), `X` (큐브·손 리셋), `H` (홈) |

### 4) 큐브 재배향 모드 준비물

> 이 모드의 전체 문서는 **[CUBE_REORIENT.md](CUBE_REORIENT.md)** 에 있습니다.
> 조작법, 왜 시뮬레이션 전용인지(모델 비교 실측값 포함), 구조와 테스트를 다룹니다.

이 모드만 두 가지를 더 필요로 합니다. 둘 중 하나라도 없으면 해당 화면만 비활성화되고
나머지 세 모드는 그대로 동작합니다 (콘솔에 이유가 출력됩니다).

- **정책 파일**: `models/cube_reorient_policy.npz` (`--reorient-policy` 로 변경)
- **MuJoCo 장면**: `assets/reorient_scene/` 에 함께 들어 있어 별도 설치가
  필요 없습니다. 없을 때만 MuJoCo Playground 체크아웃을 찾습니다
  (`--playground-root`). `mujoco_playground` 를 임포트하지는 않으므로
  jax·ml_collections 는 어느 쪽이든 필요 없습니다.

```bash
# 기본 실행 (부스 전체)
python booth_app.py --mode mujoco

# 재배향 화면만 끄기
python booth_app.py --no-reorient

# 장착 각도가 모델과 다를 때 어떻게 되는지 보기 (중력을 기울입니다)
python booth_app.py --reorient-tilt-deg 20
```

> 정책은 손이 모델과 같은 각도로 장착된 상태를 전제로 학습되었고, 실물 손이 견딜 수 있는
> 것보다 빠른 관절 목표를 내보냅니다. 그래서 이 화면의 출력은 하드웨어로 가지 않습니다
> (`step_smooth_control` 이 이 화면에서 곧바로 빠져나오고, 테스트가 이를 강제합니다).
