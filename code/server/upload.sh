#!/usr/bin/env bash
# 노트북(Windows Git Bash) → 서버: 코드 · 평가 영상 · E0 가중치를 /home/ubuntu/work 에 올린다.
#   bash code/server/upload.sh <pem 경로> [ubuntu@machine.runyour.ai]
# 인공지능종합설계/ 폴더 구조 그대로 (code/common.py가 ../6주차/ 를 찾음).
set -euo pipefail
PEM=${1:?"pem 경로"}
HOST=${2:-ubuntu@machine.runyour.ai}
PROJ=$(cd "$(dirname "$0")/../../.." && pwd)          # 인공지능종합설계/
cd "$PROJ"
tar cf - --exclude="종합설계/results" --exclude="종합설계/benchmarks" --exclude="종합설계/발표자료" \
         --exclude="종합설계/labels/evidence" --exclude="*/__pycache__" --exclude="*.pem" \
         종합설계 "6주차/종설 데이터" "6주차/best_E0.pt" \
  | ssh -i "$PEM" "$HOST" 'mkdir -p /home/ubuntu/work && tar xf - -C /home/ubuntu/work &&
      find /home/ubuntu/work/종합설계 \( -name "*.sh" -o -name "*.py" \) -exec sed -i "s/\r$//" {} + &&
      du -sh /home/ubuntu/work/*'
