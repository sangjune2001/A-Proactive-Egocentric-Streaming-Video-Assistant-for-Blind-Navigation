#!/usr/bin/env bash
# G3 하이브리드 정식 (docs/16): 정답 시각 23개 × {7B, 3B} × 지시문 {v0 5칸, h2 2칸, h2g 2칸 · 움직임 지면 기준}, 3회 반복.
# 같은 서버 조건(GPU 0.80)에서 v0도 다시 돌려 지연을 공정하게 비교. 모델을 바꿀 때 vLLM을 다시 띄운다.
#   bash code/server/run_g3.sh            (tmux g3 안에서 실행 권장 — 약 15분)
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
FAST=${FAST:-/home/ubuntu/fast}
. "$FAST/venv/rt/bin/activate"
cd "$ROOT/code/p0"
for MODEL in qwen2_5-vl-7b qwen2_5-vl-3b; do
  UTIL=0.80; [ "$MODEL" = qwen2_5-vl-3b ] && UTIL=0.60
  bash "$ROOT/code/server/start_servers.sh" "$MODEL" "$UTIL" || exit 1
  for P in v0 h2 h2g; do
    python run_p0.py --events oracle --model "$MODEL" --frames F8 --bbox draw --repeat 3 --prompt "$P" \
      --run "g3_${MODEL}_${P}" > /dev/null
    echo "done $MODEL $P"
  done
done
python vlm_eval.py $(for m in qwen2_5-vl-7b qwen2_5-vl-3b; do for p in v0 h2 h2g; do echo -n "g3_${m}_${p} "; done; done) \
  --ref g3_qwen2_5-vl-7b_v0 --out g3_hybrid
echo G3_DONE
