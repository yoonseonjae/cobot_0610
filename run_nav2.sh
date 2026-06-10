#!/bin/bash
# ============================================================
# 터미널 2: Nav2 + ROS2 브릿지
# run_stage5.sh 실행 후 Isaac Sim이 완전히 뜨면 이 스크립트 실행
# ============================================================
set -e
cd "$(dirname "$0")"

source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

echo "============================================"
echo " Nav2 + pose bridge 시작"
echo " (Isaac Sim이 먼저 실행 중이어야 합니다)"
echo "============================================"
echo ""

ros2 launch run_multi_nav2.launch.py
