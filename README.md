# LEAP Hand vision prototype

노트북 웹캠과 MediaPipe Hand Landmarker로 손의 21개 랜드마크와 LEAP Hand에 대응하는 16개 사람 손 관절각을 실시간 확인하는 프로토타입입니다. 개인별 편 손/주먹 가동범위 보정 후 One Euro Filter와 데드밴드를 적용하고, 사람의 rock/paper/scissors 동작을 인식해 로봇 동작과 비교한 승패를 CSV에 기록합니다. MuJoCo 실시간 연동과 LEAP Hand v1 실물 제어용 안전 API도 포함하지만, 비전 스트림과 실물 모터의 직접 연동은 아직 활성화하지 않습니다.

## 1. 가상환경과 패키지 설치

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. MediaPipe 모델 다운로드

```powershell
New-Item -ItemType Directory -Force models
Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" `
  -OutFile "models\hand_landmarker.task"
```

## 3. MuJoCo 오른손 모델

Google DeepMind MuJoCo Menagerie의 LEAP Hand 오른손 MJCF 모델과 필요한 메시를 `models/mujoco/leap_hand`에 포함합니다. 실행 장면 파일은 `models/mujoco/leap_hand/scene_right.xml`이며, 원본 모델의 MIT 라이선스와 README도 같은 폴더에 보존합니다.

`MujocoHandController`는 현재 비전 코드의 16개 각도 순서를 MuJoCo actuator 순서로 변환하고, degree→radian 변환과 관절 한계 제한을 수행합니다.

```python
from mujoco_hand_controller import MujocoHandController

controller = MujocoHandController()
controller.set_target_degrees(filtered_angles)
controller.step_for(1.0 / 30.0)
```

오른손 손가락을 검지, 중지, 약지, 엄지 순서로 하나씩 굽혔다 펴는 테스트:

```powershell
python mujoco_finger_test.py
```

rock, paper, scissors 자세를 차례로 확인하는 선택적 MuJoCo 미리보기:

```powershell
python mujoco_rps_demo.py
```

웹캠의 오른손 동작을 MuJoCo LEAP Hand가 실시간으로 따라 하게 하려면:

```powershell
.\.venv\Scripts\python.exe webcam_mujoco_teleop.py --profile jiwoo
```

MediaPipe 영상 창과 MuJoCo 뷰어가 함께 열립니다. 영상 창에서 `MUJOCO: FOLLOWING RIGHT HAND`가 표시되면 `개인별 가동범위 보정 → One Euro → 데드밴드`를 거친 16개 각도가 매 프레임 MuJoCo 컨트롤러로 전달됩니다. 오른손 모델만 포함되어 있으므로 `Right`로 인식된 손만 제어하고, 손이 없거나 왼손만 보이면 마지막 목표 자세를 유지합니다. 처음 사용하는 프로필은 편 손의 `C` 보정과 완전히 쥔 주먹의 `F` 보정을 차례로 수행합니다. `Q`/`Esc`를 누르거나 MuJoCo 뷰어를 닫으면 통합 실행이 종료됩니다.

자기충돌 방지 기능은 기본으로 활성화됩니다. 실제 목표를 적용하기 전에 별도의 MuJoCo 상태에서 손 링크끼리 접촉하는지 예측하고, 접촉이 예상되면 **직전 안전 자세와 새 목표 자세 사이**를 이분 탐색하여 접촉 직전까지만 움직입니다. 충돌 때마다 전체 손을 0°로 축소하지 않으므로 갑자기 손이 펴지지 않습니다. 제한이 작동하면 영상에 `MUJOCO: SELF-COLLISION HOLD | 74% TARGET`처럼 표시됩니다. 이 검사는 손 내부의 자기접촉만 제한하므로 추후 물체나 바닥과의 접촉과 구분할 수 있습니다.

충돌 제한 전후를 비교하는 진단 목적에서만 다음 옵션으로 끌 수 있습니다.

```powershell
.\.venv\Scripts\python.exe webcam_mujoco_teleop.py --profile jiwoo --no-collision-avoidance
```

기존 웹캠 실행 명령에 `--mujoco`를 추가해도 동일하게 실행할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe webcam_hand_tracking.py --profile jiwoo --mujoco
```

## 4. LEAP Hand v1 실물 API

`LeapHandHardwareController`는 DYNAMIXEL XC330 기반 16모터 LEAP Hand v1 전용입니다. v2의 8모터 텐던 구조에는 사용할 수 없습니다. 공식 LEAPsim 관절 범위를 적용하고, `0°=편 손`인 현재 프로젝트 각도를 실물 모터의 `π rad=편 손` 좌표로 변환합니다.

안전을 위해 객체를 생성하거나 `connect()`만 호출해서는 토크가 켜지지 않습니다.

```text
connect()       → COM 포트 열기, ID 0~15 확인, Torque OFF
configure()     → current-based position mode, 300mA, PID, 500ms watchdog
enable_torque() → 현재 위치를 목표값으로 먼저 기록한 뒤 Torque ON
command_degrees → 공식 관절 범위로 제한한 16개 degree 명령 전송
close()         → 즉시 Torque OFF 후 포트 닫기
```

먼저 토크를 전혀 켜지 않는 연결·센서 점검을 실행합니다. `COM13`은 Dynamixel Wizard에서 확인한 실제 포트로 바꿉니다. Wizard는 점검 전에 종료해야 합니다.

```powershell
.\.venv\Scripts\python.exe leap_hand_hardware_check.py --port COM13
```

16개 ID, 현재 위치, 온도, 입력 전압과 hardware error가 모두 정상인 것을 확인한 뒤에만 현재 위치 유지 토크 테스트를 수행합니다. 이 테스트는 새로운 자세를 명령하지 않지만, 실행 전 손 주변과 기구 사이에서 사람의 손을 치우고 전원 차단 수단을 준비해야 합니다.

```powershell
.\.venv\Scripts\python.exe leap_hand_hardware_check.py --port COM13 --torque-test
```

Python API 예시:

```python
from leap_hand_hardware_controller import LeapHandHardwareController

with LeapHandHardwareController("COM13", current_limit_milliamps=300) as hand:
    hand.configure()
    hand.enable_torque()
    hand.command_degrees(safe_angles_degrees)
    feedback = hand.read_feedback()
    health = hand.read_health()
```

어떤 예외나 `Ctrl+C`가 발생해도 context manager와 점검 스크립트의 `finally`에서 Torque OFF를 요청합니다. 통신 자체가 끊기는 경우에는 500ms DYNAMIXEL Bus Watchdog가 추가로 동작합니다. 실물 비전 연동은 속도 제한, 추적 손실 fail-safe와 피드백 감시를 더한 뒤 이 API에 연결합니다.

## 5. 실행

```powershell
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
from rps_rounds import CsvRoundRecorder, RpsRoundSession

session = RpsRoundSession(CsvRoundRecorder("rps_results.csv"))
session.start_round(robot_move)  # robot_move는 로봇 코드가 이미 알고 있는 값

# 비전 코드가 여러 프레임 뒤 확정한 human_move를 전달합니다.
record = session.observe_confirmed_human_move(human_move)
if record is not None:
    print(record.human_result)
```

전체 자동 테스트:

```powershell
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

```powershell
# 10% 더 굽히기
.\.venv\Scripts\python.exe webcam_mujoco_teleop.py --profile jiwoo --flexion-scale 1.1

# 10% 덜 굽히기
.\.venv\Scripts\python.exe webcam_mujoco_teleop.py --profile jiwoo --flexion-scale 0.9
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
