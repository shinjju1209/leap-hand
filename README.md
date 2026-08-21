# LEAP Hand vision prototype

노트북 웹캠과 MediaPipe Hand Landmarker로 손의 21개 랜드마크와 LEAP Hand에 대응하는 16개 사람 손 관절각을 실시간 확인하는 프로토타입입니다. 관절각에는 One Euro Filter와 데드밴드를 적용하며 아직 LEAP Hand 모터에는 연결하지 않습니다.

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

손의 좌·우 판정은 항상 반전하지 않은 원본 카메라 프레임에서 수행합니다. `--mirror`는 사용자에게 보여주는 화면과 랜드마크 위치만 거울처럼 반전합니다.

화면에 `TRACKING`, FPS, 왼손/오른손 신뢰도, 21개 랜드마크와 다음 16개 관절각이 도 단위로 표시됩니다. 각 행은 `원본 > One Euro + 데드밴드 결과` 순서입니다. 데드밴드는 마지막 출력 명령과의 차이가 설정값보다 작은 관절을 이전 값으로 유지합니다.

- 검지·중지·약지: `MCP side`, `MCP flex`, `PIP flex`, `DIP flex`
- 엄지: `CMC side`, `CMC flex`, `MCP flex`, `IP flex`

각도는 MediaPipe의 3D 월드 랜드마크로 계산합니다. 손가락을 편 상태에서 굽힘각이 0도에 가까워지는 표현이며, 개인별 중립 자세 오프셋은 이후 캘리브레이션 단계에서 보정합니다. 손을 0.5초 이상 놓치면 오래된 필터 상태는 자동 초기화됩니다.

## 5. 관절 구조와 16 DoF

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
