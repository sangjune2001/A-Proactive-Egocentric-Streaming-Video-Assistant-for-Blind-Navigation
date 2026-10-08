#!/usr/bin/env bash
# 서버에서 한 번: apt 패키지 + venv 3개(p0 · rt · tts) + VLM 모델. 세 venv는 tmux에서 동시에 설치 (약 8–10분).
#   bash code/server/setup_all.sh [HF 모델 ...]       기본 Qwen/Qwen2.5-VL-7B-Instruct
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)            # 종합설계/
FAST=${FAST:-/home/ubuntu/fast}
LOGS=/home/ubuntu/work/logs
MODELS=${*:-Qwen/Qwen2.5-VL-7B-Instruct}
PY=/opt/conda/bin/python; [ -x "$PY" ] || PY=python3
mkdir -p "$FAST"/{venv,hf} "$LOGS"

sudo apt-get update -qq
sudo apt-get install -y -qq tmux libgl1 libglib2.0-0 fonts-nanum > /dev/null

cat > "$LOGS/_setup_rt.sh" <<EOF
set -e
$PY -m venv $FAST/venv/rt && . $FAST/venv/rt/bin/activate && pip install -q -U pip
pip install -q ultralytics opencv-python-headless numpy scipy pillow imageio-ffmpeg openai transformers lap
python -c "import torch, ultralytics, cv2; print('rt ok', torch.__version__, torch.cuda.is_available(), ultralytics.__version__)"
EOF
cat > "$LOGS/_setup_tts.sh" <<EOF
set -e
$PY -m venv $FAST/venv/tts && . $FAST/venv/tts/bin/activate && pip install -q -U pip
pip install -q supertonic fastapi uvicorn python-multipart
python -c "import supertonic; print('tts ok')"
EOF
tmux new -d -s setup_p0  "FAST=$FAST bash $ROOT/code/p0/setup_server.sh /home/ubuntu/work/vlm $MODELS > $LOGS/setup_p0.log 2>&1"
tmux new -d -s setup_rt  "bash $LOGS/_setup_rt.sh  > $LOGS/setup_rt.log 2>&1"
tmux new -d -s setup_tts "bash $LOGS/_setup_tts.sh > $LOGS/setup_tts.log 2>&1"

echo "설치 중 (tmux: setup_p0 · setup_rt · setup_tts) ..."
while tmux ls 2>/dev/null | grep -q "^setup_"; do sleep 10; done
grep -h -E "^vllm |^rt ok|^tts ok" "$LOGS"/setup_*.log || true
if grep -l -E "Traceback|Error" "$LOGS"/setup_*.log; then echo "↑ 오류 로그 확인"; exit 1; fi
du -sh "$FAST"/hf "$FAST"/venv/*
