#!/bin/bash
# ============================================================
# Stage 1 실행 스크립트
# 확인 항목:
#   - Isaac Sim 실행되고 맵 + 로봇 2대 뜨는지
#   - 방향키로 robot1 이동되는지
#   - RViz2에서 robot1/robot2 Lidar scan 보이는지
# ============================================================
set -e
cd "$(dirname "$0")"

echo "============================================"
echo " Stage 1: 2대 로봇 기본 동작 확인"
echo "============================================"
echo ""
echo "[터미널 1] Isaac Sim (지금 이 창에서 실행)"
echo "[터미널 2] ros2 launch run_multi_nav2.launch.py"
echo ""
echo "Isaac Sim 시작 중..."
/home/yoon/dev_ws/isaac_sim/isaacsim/_build/linux-x86_64/release/python.sh applications/main_simulation.py
