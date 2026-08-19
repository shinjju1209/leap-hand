# LEAP Hand vision prototype

노트북 웹캠과 MediaPipe Hand Landmarker로 손의 21개 랜드마크와 LEAP Hand에 대응하는 16개 사람 손 관절각을 실시간 확인하는 프로토타입입니다. 관절각에는 One Euro Filter를 적용하며 아직 LEAP Hand 모터에는 연결하지 않습니다.

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

## 3. 실행

```powershell
python webcam_hand_tracking.py
```

- 종료: `Q` 또는 `Esc`
- 카메라가 열리지 않으면: `python webcam_hand_tracking.py --camera 1`
- 좌우 반전을 끄려면: `python webcam_hand_tracking.py --no-mirror`
- 필터를 끄려면: `python webcam_hand_tracking.py --no-filter`
- 필터 튜닝: `python webcam_hand_tracking.py --min-cutoff 1.2 --beta 0.05 --derivative-cutoff 1.0`

손의 좌·우 판정은 항상 반전하지 않은 원본 카메라 프레임에서 수행합니다. `--mirror`는 사용자에게 보여주는 화면과 랜드마크 위치만 거울처럼 반전합니다.

화면에 `TRACKING`, FPS, 왼손/오른손 신뢰도, 21개 랜드마크와 다음 16개 관절각이 도 단위로 표시됩니다. 각 행은 `원본 > One Euro 결과` 순서입니다.

- 검지·중지·약지: `MCP side`, `MCP flex`, `PIP flex`, `DIP flex`
- 엄지: `CMC side`, `CMC flex`, `MCP flex`, `IP flex`

각도는 MediaPipe의 3D 월드 랜드마크로 계산합니다. 손가락을 편 상태에서 굽힘각이 0도에 가까워지는 표현이며, 개인별 중립 자세 오프셋은 이후 캘리브레이션 단계에서 보정합니다. 손을 0.5초 이상 놓치면 오래된 필터 상태는 자동 초기화됩니다.
