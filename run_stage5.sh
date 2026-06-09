#!/bin/bash
# ============================================================
# Stage 5 실행 스크립트
# 확인 항목:
#   - 10초 후 화재 점화 → robot1 자동으로 소화기 위치(9.83, 0.5)로 Nav2 이동
#   - robot1 소화기 자동 파지 → 화재 위치(0.158, -4.084)로 이동 → 자동 투척
#   - robot2 초기화 직후 5번방→1번방→2번방→3번방→4번방→6번방→화장실 순환 순찰
#   - robot2 사람 발견 시 ALIGNING→APPROACHING→ARRIVED (YOLO)
#   - 소화기 화재 근처 바닥 충돌 시 가스 분출
# ============================================================
set -e
cd "$(dirname "$0")"

echo "============================================"
echo " Stage 5: 완전 자동화 시나리오"
echo "   robot1: 화재 감지 → 소화기 → 진압"
echo "   robot2: 방 순찰 + 인명 탐지"
echo "============================================"
echo ""
/home/yoon/dev_ws/isaac_sim/isaacsim/_build/linux-x86_64/release/python.sh applications/main_simulation.py
