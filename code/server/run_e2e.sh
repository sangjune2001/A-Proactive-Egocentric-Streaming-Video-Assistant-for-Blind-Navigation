#!/usr/bin/env bash
# end-to-end 실행 (흉내값 없음): 영상 실제 속도 → YOLO E0 + ByteTrack(GPU) → 트리거 B0 → 분배기
#   → 즉시 경고 + VLM(vLLM) → 하이브리드 → 문장 → TTS(Supertonic) → 재생 큐 → 시연 영상(패널 포함)
#   bash code/server/run_e2e.sh <클립 ...>        예) bash code/server/run_e2e.sh c09_bike_on_tactile
# 환경변수: MODEL(qwen2_5-vl-7b) VOICE(M1) RUN(결과 폴더 이름)
# start_servers.sh 를 먼저. 각 클립은 tmux 세션 e2e 안에서 돈다 (ssh가 끊겨도 계속).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
FAST=${FAST:-/home/ubuntu/fast}
LOGS=/home/ubuntu/work/logs
MODEL=${MODEL:-qwen2_5-vl-7b}
VOICE=${VOICE:-M1}
RUN=${RUN:-e2e_a5000_yolo_b0_${MODEL//[.-]/}_${VOICE}}
CLIPS=${*:?"클립 이름 (예: c09_bike_on_tactile)"}
mkdir -p "$LOGS"
cat > "$LOGS/_e2e.sh" <<EOF
. $FAST/venv/rt/bin/activate
cd $ROOT/code/runtime
for c in $CLIPS; do
  PYTHONUNBUFFERED=1 python run_demo.py --clip \$c --perception yolo --tracks E0 --device 0 --trigger b0 \\
    --vlm server --model $MODEL --base-url http://localhost:8000/v1 \\
    --tts server --tts-url http://localhost:7788/v1 --tts-model supertonic-3 --tts-voice $VOICE \\
    --run $RUN --dump-vlm-inputs --render
  echo "EXIT \$c=\$?"
done
echo ALL_DONE
EOF
tmux kill-session -t e2e 2>/dev/null || true
tmux new -d -s e2e "bash $LOGS/_e2e.sh > $LOGS/e2e_$RUN.log 2>&1"
echo "실행 중: tmux attach -t e2e  ·  로그 $LOGS/e2e_$RUN.log  ·  결과 $ROOT/results/runtime/$RUN/"
