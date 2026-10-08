#!/usr/bin/env bash
# vLLM(VLM) + Supertonic(TTS) 서버를 tmux로 띄우고 준비될 때까지 기다린다.
#   bash code/server/start_servers.sh [모델 키 qwen2_5-vl-7b] [GPU 비율 0.80]
# GPU 24 GB = vLLM(7B bf16 약 16 GB + KV) 0.80 + YOLO11s-seg 약 1–2 GB. 3B면 0.6 정도로 낮춰도 됨.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
FAST=${FAST:-/home/ubuntu/fast}
LOGS=/home/ubuntu/work/logs
MODEL=${1:-qwen2_5-vl-7b}
UTIL=${2:-0.80}
mkdir -p "$LOGS"
tmux kill-session -t vllm 2>/dev/null || true
tmux kill-session -t tts 2>/dev/null || true
tmux new -d -s vllm "export HF_HOME=$FAST/hf; . $FAST/venv/p0/bin/activate; cd $ROOT/code/p0; python serve_vllm.py $MODEL --gpu-util $UTIL > $LOGS/vllm.log 2>&1"
tmux new -d -s tts  ". $FAST/venv/tts/bin/activate; supertonic serve --model supertonic-3 --port 7788 > $LOGS/tts.log 2>&1"
echo "서버 준비 중 (vLLM :8000 · TTS :7788) ..."
for i in $(seq 1 120); do
  a=$(curl -s -m 3 -o /dev/null -w "%{http_code}" localhost:8000/v1/models || true)
  b=$(curl -s -m 3 -o /dev/null -w "%{http_code}" localhost:7788/v1/health || true)
  [ "$a" = 200 ] && [ "$b" = 200 ] && { echo "준비 완료 ($((i * 5)) s)"; nvidia-smi --query-gpu=memory.used,memory.total --format=csv; exit 0; }
  # 로그 글자로 판단하면 vLLM의 무해한 컴파일 캐시 경고("RuntimeError: Cubin ...")에 걸림 → 세션이 살아 있는지로 판단
  for s in vllm tts; do
    tmux has-session -t $s 2>/dev/null || { echo "$s 서버가 죽음:"; tail -n 20 "$LOGS/$s.log"; exit 1; }
  done
  sleep 5
done
echo "시간 초과"; exit 1
