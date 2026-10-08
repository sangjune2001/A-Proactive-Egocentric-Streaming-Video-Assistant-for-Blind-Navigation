#!/usr/bin/env bash
# A5000 서버에서 TTS 비교 환경 설치 (엔진마다 의존성이 부딪쳐서 conda 환경을 따로 만든다).
#   FAST=/home/ubuntu/fast bash code/tts_bench/setup_tts_server.sh
# 환경: $FAST/tts/<엔진>  ·  CosyVoice 코드: $FAST/CosyVoice  ·  모델: $FAST/hf (HF_HOME)
# 설치 방법은 각 공식 README 그대로 (CosyVoice: python 3.10 + requirements.txt, Chatterbox: pip chatterbox-tts).
set -euo pipefail
FAST=${FAST:-/home/ubuntu/fast}
CONDA=${CONDA:-/opt/conda/bin/conda}
export HF_HOME=$FAST/hf
mkdir -p "$FAST"/tts "$HF_HOME"

env_py() { echo "$FAST/tts/$1/bin/python"; }
mk() {   # mk <이름> <python 버전>
  [ -x "$(env_py "$1")" ] || "$CONDA" create -y -q -p "$FAST/tts/$1" "python=$2" > /dev/null
  "$(env_py "$1")" -m pip install -q -U pip
}

echo "== supertonic (ONNX, CPU — 온디바이스 기준) =="
mk supertonic 3.11
"$(env_py supertonic)" -m pip install -q "supertonic[serve]==1.3.1" numpy

echo "== melotts =="
mk melotts 3.11
"$(env_py melotts)" -m pip install -q "git+https://github.com/myshell-ai/MeloTTS.git"
"$(env_py melotts)" -m unidic download

echo "== mms-tts-kor =="
mk mms 3.11
"$(env_py mms)" -m pip install -q torch transformers uroman numpy

echo "== chatterbox multilingual =="
mk chatterbox 3.11
"$(env_py chatterbox)" -m pip install -q chatterbox-tts numpy

echo "== CosyVoice2 / Fun-CosyVoice3 (README: python 3.10) =="
mk cosyvoice 3.10
[ -d "$FAST/CosyVoice" ] || git clone --recursive -q https://github.com/FunAudioLLM/CosyVoice.git "$FAST/CosyVoice"
"$(env_py cosyvoice)" -m pip install -q -r "$FAST/CosyVoice/requirements.txt"
"$(env_py cosyvoice)" -m pip install -q huggingface_hub
"$(env_py cosyvoice)" - <<EOF
from huggingface_hub import snapshot_download
for repo, d in [("FunAudioLLM/CosyVoice2-0.5B", "CosyVoice2-0.5B"), ("FunAudioLLM/Fun-CosyVoice3-0.5B-2512", "Fun-CosyVoice3-0.5B")]:
    snapshot_download(repo, local_dir="$FAST/CosyVoice/pretrained_models/" + d)
    print("ok", repo)
EOF

echo "== 채점 (faster-whisper large-v3, GPU) =="
mk asr 3.11
"$(env_py asr)" -m pip install -q faster-whisper numpy nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"

nvidia-smi --query-gpu=name,memory.total --format=csv
du -sh "$FAST"/tts "$FAST"/hf "$FAST"/CosyVoice
