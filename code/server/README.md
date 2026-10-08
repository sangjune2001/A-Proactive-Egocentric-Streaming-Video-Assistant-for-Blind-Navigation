# server — RunYourAI A5000에서 end-to-end 실행

머신: NVIDIA RTX A5000 24 GB · CPU 32코어 · RAM 125 GB · Ubuntu 22.04 컨테이너 (`/opt/conda` Python 3.12)

| 경로 | 무엇 | 반납하면 |
|---|---|---|
| `/home/ubuntu/fast` (로컬 디스크) | venv 3개 · HF 모델 캐시 — 빨라야 하는 것 | **사라짐** |
| `/home/ubuntu/work` (로컬 디스크) | 코드 · 영상 · 실행 결과 (작은 파일 쓰기가 빠름) | **사라짐** |
| `/home/ubuntu/runyourai/vlmcall` (NFS storage 246 GB) | 결과 · 로그 백업 (`sync_to_storage.sh`) | 남음 (가끔 빈 볼륨으로 옴) |

NFS는 작은 파일 쓰기가 매우 느려서 venv · 실행은 로컬에서 하고, 끝나면 결과만 storage로 복사한다.

## 순서

```bash
# 0) 노트북(Windows Git Bash)에서: 코드 · 영상 · 가중치 올리기 (약 2–3분)
bash code/server/upload.sh ~/.ssh/a5000.pem ubuntu@machine.runyour.ai

# 서버에서
cd /home/ubuntu/work/종합설계
bash code/server/setup_all.sh          # 1) 환경 3개 + 7B 모델 (약 8–10분, 처음 한 번)
bash code/server/start_servers.sh      # 2) vLLM(7B, GPU 80 %) + Supertonic TTS 서버 (약 2–3분)
bash code/server/run_e2e.sh c09_bike_on_tactile   # 3) end-to-end 실행 + 시연 영상 (영상 길이 + 약 2분)
bash code/server/sync_to_storage.sh    # 4) 결과 · 로그를 storage로
```

## 환경 (`setup_all.sh`)

| venv | 무엇 | 왜 따로 |
|---|---|---|
| `p0` | vLLM 0.31 + openai — VLM 서버 | vLLM이 torch 버전을 고정 |
| `rt` | ultralytics(YOLO) · opencv · openai · transformers — 실행 시스템 | YOLO · ByteTrack · 트리거 · 분배기 · 렌더 |
| `tts` | supertonic + fastapi · uvicorn · python-multipart — TTS 서버 | ONNX 런타임, `supertonic serve`에 웹 서버 패키지 필요 |

apt: `tmux` · `libgl1` · `libglib2.0-0` (opencv) · `fonts-nanum` (시연 영상 한글 자막)

## 겪은 문제와 해결 (스크립트에 반영됨)

| 문제 | 해결 |
|---|---|
| `libGL.so.1` 없음 (ultralytics가 opencv-python을 끌어옴) | `apt install libgl1 libglib2.0-0` |
| `supertonic serve`: fastapi · uvicorn · python-multipart 없음 | tts venv에 설치 |
| vLLM이 GPU 90 %를 잡으면 YOLO가 못 올라감 | `serve_vllm.py --gpu-util 0.80` (7B 약 16 GB + KV, 남는 약 4 GB에 YOLO) |
| flashinfer JIT 실패 (nvcc 13.4 ↔ 헤더 13.0) | `serve_vllm.py`가 `VLLM_USE_FLASHINFER_SAMPLER=0` |
| 긴 ssh 명령이 원격에서 끊기면 프로세스도 죽음 | 모든 실행을 tmux 세션에서 |
| Windows에서 올린 .sh · .py의 CRLF | `upload.sh`가 서버에서 `sed -i 's/\r$//'` |
| ssh 안에서 `pkill -f "vllm serve"` → 자기 셸까지 죽음 | `tmux kill-session -t vllm` 사용 |
