# Cobot Fire Sim

Isaac Sim 기반 화재 대응 로봇 자동화 시뮬레이션

- **robot1** (팔 있음): 화재 감지 → 소화기 자동 파지 → 화재 위치 이동 → 투척
- **robot2** (순찰): 방 순찰(5→1→2→3→4→6→화장실) + YOLOv8 인명 탐지

---

## 환경

- Ubuntu 22.04 / ROS2 Humble
- Isaac Sim 5.1 (설치 경로: `/home/yoon/dev_ws/isaac_sim/`)
- Python 3.11 (Isaac Sim 내장)

---

## 설치 전 필수 준비

### 1. omniverse CDN 머티리얼 캐시 배치

USB로 전달받은 폴더를 다음 위치에 복사:

```
map/omniverse-content-production.s3.us-west-2.amazonaws.com/
```

없으면 맵 머티리얼이 회색으로 표시되지만 시뮬레이션은 동작합니다.

### 2. Isaac Sim 경로 확인

[run_stage5.sh](run_stage5.sh) 내 경로가 본인 환경과 맞는지 확인:

```bash
/home/yoon/dev_ws/isaac_sim/isaacsim/_build/linux-x86_64/release/python.sh
```

다른 경로라면 모든 `run_*.sh` 파일의 해당 줄을 수정.

### 3. ROS2 Nav2 설치 확인

```bash
sudo apt install ros-humble-nav2-bringup ros-humble-nav2-msgs
```

---

## 실행 방법

### 터미널 1 — Isaac Sim 시뮬레이션

```bash
cd ~/cobot_fire_sim
./run_stage5.sh
```

### 터미널 2 — Nav2 + ROS2 브릿지 (선택, 자율주행 필요 시)

```bash
cd ~/cobot_fire_sim
source /opt/ros/humble/setup.bash
ros2 launch run_multi_nav2.launch.py
```

---

## 스테이지별 실행

| 스크립트 | 내용 |
|---|---|
| `run_stage1.sh` | 로봇 2대 스폰 + 키보드 제어 |
| `run_stage2.sh` | + 화재 이펙트 (10초 점화, 15초 확산) + 소화기 Grasp(G)/투척(Q) |
| `run_stage4.sh` | + YOLOv8 인명 탐지 + 사람 스폰 |
| `run_stage5.sh` | + 완전 자동화 (Nav2 연동, 자율 순찰/진압) |

---

## 조작키 (수동 모드, robot1)

| 키 | 동작 |
|---|---|
| ↑ / NumPad8 | 전진 |
| ↓ / NumPad2 | 후진 |
| ← / NumPad4 / N | 좌회전 |
| → / NumPad6 / M | 우회전 |
| G | 소화기 파지 |
| Q | 소화기 투척 |

---

## 폴더 구조

```
cobot_fire_sim/
├── applications/
│   ├── main_simulation.py   # 메인 시뮬레이션 루프
│   ├── spot_agent.py        # 로봇 에이전트 (Grasp/YOLO/자동화)
│   ├── environment.py       # 맵/소화기/사람 스폰
│   ├── spot_policy.py       # SpotArm RL 정책
│   └── new_policy.py        # 정책 로더
├── map/                     # USD 맵 + 소화기 에셋
├── assets/                  # Spot 로봇 USD
├── policies/                # RL 학습 모델 (.pt)
├── yolov8n.pt               # YOLO 인명 탐지 모델
├── cmd_vel_udp_bridge.py    # ROS2 cmd_vel → UDP 변환
├── pose_file_to_ros_bridge.py  # Isaac pose → ROS2 odom/TF
├── run_multi_nav2.launch.py # Nav2 멀티로봇 런치
├── robot1_nav2_params.yaml
├── robot2_nav2_params.yaml
└── nav2_map.yaml / nav2_map.png
```
