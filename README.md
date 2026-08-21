# LEAP Hand vision prototype

노트북 웹캠과 MediaPipe Hand Landmarker로 손의 21개 랜드마크와 LEAP Hand에 대응하는 16개 사람 손 관절각을 실시간 확인하는 프로토타입입니다. 사람의 rock/paper/scissors 동작을 인식하고, 로봇 코드가 알려 주는 동작과 비교해 승패를 CSV에 기록합니다. 관절각에는 One Euro Filter와 데드밴드를 적용하며 아직 LEAP Hand 모터에는 연결하지 않습니다.

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

## 4. 실행

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

손의 좌·우 판정은 항상 반전하지 않은 원본 카메라 프레임에서 수행합니다. `--mirror`는 사용자에게 보여주는 화면과 랜드마크 위치만 거울처럼 반전합니다.

화면에 `TRACKING`, FPS, 왼손/오른손 신뢰도, 21개 랜드마크와 다음 16개 관절각이 도 단위로 표시됩니다. 각 행은 `원본 > One Euro + 데드밴드 결과` 순서입니다. 데드밴드는 마지막 출력 명령과의 차이가 설정값보다 작은 관절을 이전 값으로 유지합니다.

- 검지·중지·약지: `MCP side`, `MCP flex`, `PIP flex`, `DIP flex`
- 엄지: `CMC side`, `CMC flex`, `MCP flex`, `IP flex`

각도는 MediaPipe의 3D 월드 랜드마크로 계산합니다. 손가락을 편 상태에서 굽힘각이 0도에 가까워지는 표현이며, 개인별 중립 자세 오프셋은 이후 캘리브레이션 단계에서 보정합니다. 손을 0.5초 이상 놓치면 오래된 필터 상태는 자동 초기화됩니다.

## 5. 사람 RPS 인식과 승패 기록

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

## 6. 관절 구조와 16 DoF

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
