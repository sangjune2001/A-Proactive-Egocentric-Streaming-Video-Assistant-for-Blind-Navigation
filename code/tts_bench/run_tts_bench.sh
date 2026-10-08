#!/usr/bin/env bash
# A5000에서 TTS 비교 전체 실행 (setup_tts_server.sh 다음). 로컬 노트북 결과(sapi 등)는 results/tts_bench에 그대로 둔다.
#   FAST=/home/ubuntu/fast bash code/tts_bench/run_tts_bench.sh
# 같은 문장 세트(sentences.json) · 같은 참고 음성(prompt_ko.wav) · 같은 채점(Whisper large-v3 CER).
set -uo pipefail
FAST=${FAST:-/home/ubuntu/fast}
export HF_HOME=$FAST/hf
cd "$(dirname "$0")"
py() { "$FAST/tts/$1/bin/python" "${@:2}"; }
CV=$FAST/CosyVoice/pretrained_models

py supertonic synth.py supertonic --voice F1 --tag srv_supertonic_F1_cpu
py supertonic synth.py supertonic --voice M1 --tag srv_supertonic_M1_cpu
py melotts    synth.py melotts --speed 1.0 --device cuda --tag srv_melotts_cuda
py mms        synth.py mms --device cuda --tag srv_mms_cuda
py chatterbox synth.py chatterbox --device cuda --tag srv_chatterbox_cuda
py cosyvoice  synth.py cosyvoice --repo "$FAST/CosyVoice" --model-dir "$CV/CosyVoice2-0.5B" --tag srv_cosyvoice2
py cosyvoice  synth.py cosyvoice --repo "$FAST/CosyVoice" --model-dir "$CV/CosyVoice2-0.5B" --stream --tag srv_cosyvoice2_stream
py cosyvoice  synth.py cosyvoice --repo "$FAST/CosyVoice" --model-dir "$CV/Fun-CosyVoice3-0.5B" --tag srv_cosyvoice3
py cosyvoice  synth.py cosyvoice --repo "$FAST/CosyVoice" --model-dir "$CV/Fun-CosyVoice3-0.5B" --stream --tag srv_cosyvoice3_stream

# faster-whisper GPU는 pip로 받은 cuDNN · cuBLAS 경로가 필요 (faster-whisper README)
NV=$("$FAST/tts/asr/bin/python" -c "import os, nvidia.cublas.lib, nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ':' + os.path.dirname(nvidia.cudnn.lib.__file__))")
LD_LIBRARY_PATH=$NV:${LD_LIBRARY_PATH:-} py asr asr_eval.py --device cuda \
  srv_supertonic_F1_cpu srv_supertonic_M1_cpu srv_melotts_cuda srv_mms_cuda srv_chatterbox_cuda \
  srv_cosyvoice2 srv_cosyvoice2_stream srv_cosyvoice3 srv_cosyvoice3_stream
