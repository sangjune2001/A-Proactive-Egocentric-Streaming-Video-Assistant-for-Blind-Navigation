# 서버 실행 (런유어AI A5000, 새 서버 기준)

1. 환경: uv로 python 3.12 → `/opt/conda/bin/python` 링크, venv 두 개만 (`~/fast/venv/rt`: ultralytics·transformers, `~/fast/venv/tts`: supertonic). VLM(vLLM)은 꺼서 설치 안 함.
2. SJ 실행 시스템: `git clone -b SJ ... ~/sj` 후 `../../sj_runtime_patch/` 패치 적용
   (`git apply sj_runtime_v5_9.patch` → `after_v5_9/*.py` → `patch_sj_v5_12.py` → `patch_sj_v5_20.py`).
   SJ 브랜치에는 푸시하지 않음 (서버 사본에만).
3. Heami 음성 (서버는 리눅스라 Heami 직접 불가):
   - 서버에서 문장 목록 뽑기: `dispatcher.all_warn_texts()` → texts.json
   - 노트북(Windows)에서 `powershell -File heami_pieces.ps1 texts.json 출력폴더 2` (속도 2, 문장당 약 8 ms)
   - 앞뒤 무음 자르고 `~/sj/results/runtime/tts_cache/sapi_heami/<md5(문장)[:16]>.wav` 로 업로드
   - `speech.py`의 SapiEngine은 리눅스에서 캐시만 쓰도록 패치(`HEAMI_CACHE_ONLY`)
4. `bash run_pass1.sh` 한 번 → `bash run_ver.sh test_N v5_29 versions/pass2_trigger_v5_29.py sapi`
