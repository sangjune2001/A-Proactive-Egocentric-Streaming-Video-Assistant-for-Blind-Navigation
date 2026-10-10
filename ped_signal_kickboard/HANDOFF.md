# AI Hub 보행신호등 / 킥보드 데이터 추출 — 인수인계 (2026-10-09 기준)

다른 에이전트(Codex 등)가 이 문서만 보고 바로 이어서 실행할 수 있도록 작성함.

## 1. 목표 (사용자 요구)
- AI Hub 데이터셋에서 **보행자 신호등(pedestrian traffic light)** 과 **킥보드(개인형 이동장치, PM)** 가 나오는 이미지 + 라벨만 골라서
- **구글 드라이브**(`내 드라이브/aihub_신호등_킥보드/`, 계정 용량 2TB, 여유 약 1.2TB)에 저장.
- 사용자는 **PC가 아니라 구글 쪽(Colab → 드라이브)에서 돌리길 원함.** PC(Windows, C: 여유 약 17GB, RAM 부족)는 최대한 쓰지 말 것.
- 신호가 바뀌는 과정(연속 프레임)이 있으면 더 좋음. 한국 신호등이면 AI Hub가 아니어도 됨.
- 사용자는 확인 질문을 싫어함("알아서 해줘"). 한국어로 짧게 보고.

## 2. 핵심 사실 (조사 완료)
- AI Hub API: `https://api.aihub.or.kr/down/0.6/{datasetkey}.do?fileSn={filekey}`, 헤더 `apikey: <키>`.
  응답은 tar이고 그 안에 실제 파일이 `.part0, .part1 ...` 로 쪼개져 들어 있음(순서대로 이어 붙이면 원본).
  - **Range(이어받기) 미지원**(항상 200 전체). 속도 PC 기준 약 3MB/s, 동시 다운로드하면 오히려 느려짐(합계 1MB/s).
  - 파일 트리(키 목록): `https://api.aihub.or.kr/info/{datasetkey}.do` (인증 불필요). `trees/*.txt` 에 저장해 둠.
  - 데이터셋 목록: `https://api.aihub.or.kr/info/dataset.do`
- **다운로드 최소 단위 = 파일키(웹사이트의 TS1, TL1 같은 묶음)**. 이미지 한 장/폴더 단위로는 못 받음 → 원천 묶음을 스트리밍으로 받으면서 필요한 이미지만 꺼냄(디스크에 원본 저장 안 함).
- API 키: `aihub_apikey.txt` (드라이브 `aihub_traffic_code/` 와 PC `C:\Users\windo\.aihub_apikey`). 승인된 데이터셋: 188, 71572, 71579, 71784, 71786, 614. **187은 미승인**(HTTP 502 "신청 및 승인 후").

### 데이터셋별 판정 규칙
| ID | 내용 | 라벨 형식 | 남기는 기준 | 비고 |
|---|---|---|---|---|
| 614 | 개인형 이동장치 안전 데이터 (킥보드 전용) | JSON, `annotations.PM[]` (bbox, PM_code), `annotations.environment[]`(polygon) | `annotations.PM` 비어있지 않음 | Validation 라벨 56,423장 전부 해당. VL1=56587(101MB), VS1=56588(39GB). Training TS1~4 각 ~100GB(라벨 TL1~4 56579~56582) |
| 188 | 신호등/도로표지판 인지(수도권) | JSON `annotation[]`, `class=traffic_light`, `type in {car, pedestrian, bus, ...}`, `attribute[{red,green,yellow,...: on/off}]` | traffic_light 중 `type == "pedestrian"` 이 하나라도 있음 | 라벨 tar와 원천 tar 파일명이 1:1. 묶음별 개수는 `rank.csv` (이미지의 약 15~25%가 해당). 1280_720 묶음이 GB당 밀도 최고 |
| 71579 | 신호등 신호정보 인지 영상 | JSON `objects[]`: `class_name in {vehicular_signal, pedestrian_signal, invisible_signal, unusual_signal}`, `attribute.signal`, `flags.v2`(신호 변화 시나리오) | pedestrian_signal 있는 **클립 전체** | 클립당 프레임 5장뿐(띄엄띄엄), 보행신호등이 작음 → 가치 낮음. Training(TS.z01/z02/zip 분할 zip 249GB)은 제외 결정 |
| 71784 | 생활도로 객체인식 | COCO류 JSON(`category` 키), PM = category_id 99 | (현재 계획에서 제외) | 킥보드가 이미지의 ~16%. 신호등에 보행/차량 구분 없음 |
| 187 | 신호등/표지판(수도권 외) | 188과 동일 형식 | type == pedestrian | **미승인** — 사용자가 AI Hub에서 신청하면 추가 가능 |
| 71572, 71786, 513, 522 | — | — | 제외 | 보행신호등/킥보드 라벨 없음 또는 신호색 없음 |
| 189 | 인도보행 영상 | bbox xml이 이미지와 같은 zip | 미정 | traffic_light(보행/차량 구분 없음), scooter 클래스 있음. 라벨 분리 안 돼서 사전 집계 불가 |

## 3. 현재 계획 (pipeline.py `build_jobs()` 순서)
1. 614 Validation (VS1 39GB) — **PC에서 약 20GB까지 진행 후 중단**, 저장된 이미지는 saved.txt에 기록되어 재실행 시 건너뜀(단, 원천은 처음부터 다시 받음).
2. 188 Validation 전체 (9개 tar, 약 37GB, 보행신호 이미지 약 2.2만)
3. 188 Training 중 `1280_720` 묶음만 (19개 tar, 약 110GB, 보행신호 이미지 약 10만)
4. 71579 Validation (VS.zip 31GB, 718클립×5장. 일부는 이미 저장됨)

## 4. 코드/파일 위치
- PC: `C:\Users\windo\aihub_traffic\`
- 드라이브: `내 드라이브/aihub_traffic_code/` (pipeline.py, trees/, keep/, state.json, saved.txt, aihub_apikey.txt, aihub_colab.ipynb, rank.csv, HANDOFF.md)
- 결과: `내 드라이브/aihub_신호등_킥보드/<데이터셋폴더>/<Validation|Training>/shard_*.zip`
  - zip(무압축, ~1GB) 안에 `images/...`, `labels/...` 원래 경로 유지.
  - 71579는 `clips_pedestrian_signal.csv`(클립별 보행신호 변화 순서)도 있음.
  - 초기에 낱개로 올라간 71579 라벨 json 약 1,052개가 `71579_신호등신호정보/Validation/labels/` 에 있음(중복 무해).

### pipeline.py 구조
- `parse_tree` → `build_jobs` (데이터셋/split별 라벨 파일키, 원천 파일키 그룹)
- `scan_labels`: 라벨 파일을 스트리밍으로 읽어 남길 이미지 stem 집합(keep) 생성 + 라벨을 shard에 저장. 결과 캐시 `keep/<ds>_<split>.json`.
- `extract_sources`: 원천을 스트리밍(`stream_unzip` 또는 tarfile `r|`)하며 stem이 keep에 있으면 shard에 저장.
- `Shards`: `<OUT>/<ds>/<split>/shard_*.zip.tmp` 에 쓰다가 1GB 넘으면 `.zip` 으로 rename.
- `saved.txt`: 저장 완료 파일 경로(posix). 재실행 시 중복 저장 방지. `state.json.done_sources`: 끝난 원천 그룹.
- 환경변수 `AIHUB_OUT` 지정 시 그 경로(드라이브 마운트)에 바로 쓰고 rclone 업로드 안 함(Colab 모드). 미지정 시 PC `staging/` + rclone 업로드 스레드.
- 재시도: 네트워크 끊기면 처음부터 다시 받되 이미 넘긴 바이트는 버림(Range 미지원 때문).

## 5. 실행 방법
**사용자 확인: AI Hub 데이터는 Colab에서 받을 수 없음 → PC에서 실행(아래 "PC") 하는 것이 현재 방식.**
2026-10-09 오후부터 PC에서 pipeline.py 실행 중(백그라운드, 창 숨김). 확인: `Get-Content C:\Users\windo\aihub_traffic\pipeline.log -Tail 5`,
프로세스: `Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ? CommandLine -match aihub_traffic`.

### 현재 병렬 실행 구성 (2026-10-09 15:50~)
병렬로 받으면 합계 속도가 늘어남(1개 ~7MB/s → 3개 합계 ~9MB/s). 작업별 프로세스(창 숨김):
| TAG | AIHUB_JOBS | 로그 | 진행기록 |
|---|---|---|---|
| 614 | 614 | pipeline_614.log | state_614.json, saved_614.txt |
| 188v | 188:Validation | pipeline_188v.log | state_188v.json, saved_188v.txt |
| 188t | 188:Training | pipeline_188t.log | state_188t.json, saved_188t.txt |
| 71579 | 71579 | pipeline_71579.log | state_71579.json, saved_71579.txt |
- 업로드는 `uploader.py` 하나만 담당(1분마다 rclone move staging → gdrive). TAG가 있는 작업은 업로드 안 함.
- 재시작 예: `$env:AIHUB_TAG='188t'; $env:AIHUB_JOBS='188:Training'; python -u -X utf8 pipeline.py`
- 작업을 강제 종료했으면 다시 띄우기 전에 `python -X utf8 migrate.py` 실행(미완성 .zip.tmp 삭제 + 모든 saved*.txt 정리).
  주의: 실행 중인 다른 작업의 .zip.tmp까지 지우므로 **모든 작업을 멈춘 뒤에만** 실행할 것.
- SAVED는 시작 시 모든 saved*.txt를 합쳐 읽음 → 같은 이미지 중복 저장 방지.
- (21:40~) **claim 방식**: 같은 AIHUB_JOBS를 여러 작업이 돌리면 `claims/<gid>` 파일을 먼저 만든 작업만 그 묶음을 받음 → 188 Training을 188t/188t2/188t3 세 줄로 나눠 받는 중.
  `switch_188t.py`(구버전 188t가 49279 끝내면 교체), `queue_after.py 13040 188t2=188:Training`, `queue_after.py 34532 188t3=188:Training`.
  묶음이 5번 실패하면 claim을 지워 다른 작업이 재시도 가능. 강제 종료로 claim만 남고 미완료면 `claims/`에서 해당 파일 삭제 후 재실행.
- PowerShell 스크립트(.ps1)에 한글 경로 쓰면 PS 5.1이 인코딩을 잘못 읽어 실패함 → 파이썬 사용.

### Colab (AI Hub 다운로드 불가로 사용 안 함 — 참고용)
1. 드라이브에서 `aihub_traffic_code/aihub_colab.ipynb` 를 Colab으로 열기.
2. 셀 순서대로 실행(드라이브 마운트 → 속도 확인 → `AIHUB_OUT=/content/drive/MyDrive/aihub_신호등_킥보드` 로 `python -u pipeline.py`).
3. 끊기면 "모두 실행"으로 재개. 로그: `aihub_traffic_code/pipeline.log`.
- 주의: Colab 세션은 최대 ~12시간, 브라우저 탭 닫으면 90분 내 끊김. 속도는 Colab(해외)에서 측정 필요 — PC(3MB/s)보다 느리면 PC로 되돌리는 것 고려.
- 진행 확인: `!tail -5 pipeline.log`, `!ls -la /content/drive/MyDrive/aihub_신호등_킥보드/*/*`

### PC (대안)
```
cd C:\Users\windo\aihub_traffic
python -u -X utf8 pipeline.py        # staging/ 에 쓰고 rclone으로 드라이브에 move
```
- rclone: `C:\Users\windo\AppData\Local\Microsoft\WinGet\Packages\Rclone.Rclone_Microsoft.Winget.Source_8wekyb3d8bbwe\rclone-v1.75.1-windows-amd64\rclone.exe`, 리모트 이름 `gdrive:` (설정 완료). 공용 client_id 퇴역 경고 있음 → 업로드 실패 시 개인 client_id 필요.
- PC ↔ Colab 전환 시: 한쪽을 멈추고 `migrate.py`(PC staging의 미완성 .zip.tmp 정리 + saved.txt 정리) 실행 후 state.json/saved.txt/keep/ 를 다른 쪽으로 복사. **두 곳에서 동시에 돌리지 말 것**(saved.txt 충돌).

## 5-1. YOLO 학습 — ★확정 사항(사용자 강하게 요구)★
- **기존 `aihub189_yolo/runs/seg10/weights/best.pt`(yolo11n-seg)를 그대로 이어서 학습.** detect로 바꾸거나 새 모델 쓰지 말 것.
- **클래스 10개 그대로**: person, bicycle, scooter, motorcycle, car, bus, other_vehicle, obstacle, stairs, traffic_light. 새 클래스 추가 금지.
- 새 데이터는 원래 라벨 중 해당 클래스만: 188/71579 신호등 전부 → traffic_light(9), 614 PM → scooter(2). **기존 모델로 재라벨링(pseudo-label) 하지 말 것.**
- bbox는 사각형 폴리곤으로 저장(seg 학습용). 사용 시 출력은 boxes만 씀.
- 스크립트: prepare_yolo.py → build_189.py(기존 189 2000장 포함) → train_seg.py (`YOLO(best.pt).train(...)`, resume 지원). 서버용 server_run.sh.
- 학습 장소: 런유어AI(몬드리안 AI) GPU 서버에 SSH — 접속 정보 사용자에게 받아야 함. 대안 Colab 노트북 train_ped_kick.ipynb.

### 런유어AI 서버 (2026-10-09 23:30~) — 다운로드+학습 모두 여기서
- 접속: `ssh -i C:\Users\windo\.ssh\runyourai_a5000.pem ubuntu@machine.runyour.ai` (RTX A5000 24GB, 디스크 ~250GB, sudo 가능, python3만 있음)
  긴 명령은 서버가 SSH를 끊으므로 항상 `nohup ... &`. 상태: `bash ~/work/code/status.sh`
- 서버에서 AI Hub 다운로드 20MB/s(단일), 3줄 합계 ~44MB/s → **PC 다운로드는 중단, 남은 188 묶음은 서버(server_download.sh: sv/st1/st2 + rclone 업로더)**.
  PC의 saved*/state*/keep/claims 를 서버 ~/work/code 로 복사해 이어감. rclone 설정은 PC %APPDATA%\rclone\rclone.conf 복사.
- `after_download.sh`가 다운로드 종료+업로드 완료를 기다렸다가 `server_run.sh`(드라이브→서버 복사, 데이터셋 생성, 학습, 결과 업로드) 자동 실행. 로그 ~/work/run.log
- 결과: 서버 ~/work/runs/seg10_plus_tl_scooter, 끝나면 드라이브 aihub_ped_kick_yolo/runs_server
- 추가 데이터 요청(사용자): Roboflow kdigital/electric-scooter-cd7hw(→scooter), 한국 보행신호등 데이터(→traffic_light). Roboflow는 API 키 필요.

- 추가 데이터(샘플 직접 확인한 한국 보행신호등만): Roboflow robot-hsuip, chanyoung/pedestrian-light-crosswalk, cap-8nhra, usrg2,
  s-workspace-ddokc → traffic_light / kdigital/electric-scooter-cd7hw → scooter. `rf_to_yolo.py`(키: ~/.roboflow_key). 제외 목록과 이유는 대화 기록 참고
  (대만 d2upo, 홍콩, 중국, 터키, 유럽, 인니, 스리랑카, 다국적 mix, cible은 대부분 한국이나 일부 외국 섞여 제외).
- 사용자 요구: **신호가 바뀌는 장면 강조** → `oversample.py`가 71579 변화 클립(157개, csv는 71579 Validation shard_..._095736 zip 안)의
  프레임을 train_list.txt에 5배 반복. 클래스는 10개 그대로라 상태(빨강/초록)는 예측 안 함.

### (아래는 이전 초안 — 무시: 4클래스 detect 계획은 폐기됨)
## 5-1-old. YOLO 학습 (Colab GPU, 2026-10-09 밤 준비)
- PC에 NVIDIA GPU 없음 → 학습은 Colab. 기존 모델 `aihub189_yolo/runs/seg10`(YOLO seg, 189 인도보행 2000장, 8클래스)과는 별개로
  **detect 모델 새로 학습**(새 데이터는 bbox만 있어 seg 불가).
- 클래스: 0 ped_red, 1 ped_green, 2 ped_off, 3 kickboard. 차량 신호등은 라벨 안 함(보행/차량 구분 학습).
  - 188 보행신호의 ~70%가 red/green 모두 off(옆/뒷면 또는 소등) → ped_off.
  - 614 PM bbox는 [x,y,w,h], 탑승자+킥보드 전체를 감쌈. PM_code(13,17,18,...) 의미 미확인 → 전부 kickboard.
- 변환: `prepare_yolo.py <shard루트> <출력> --max-188 30000 --max-188-val 3000 --max-614 20000`
  (실제 이미지가 있는 것만 해시 고정 샘플링. 188 Validation→val, 614는 video_id 10%→val, 71579→train)
  검증 그림: `show_yolo.py <ds> out.jpg` — 로컬 샘플로 박스 위치 확인 완료.
- 노트북: 드라이브 `aihub_traffic_code/train_ped_kick.ipynb` (GPU 런타임). yolo11s, imgsz 1280, epochs 30, save_period 1,
  결과 `aihub_ped_kick_yolo/runs/det1`, 데이터셋 캐시 `aihub_ped_kick_yolo/ds.zip`.
  **다운로드가 다 끝난 뒤 실행**할 것(캐시가 생기면 이후 데이터가 반영 안 됨 → 새 데이터 반영하려면 ds.zip 삭제 후 재실행).

## 6. 남은 일 / 아이디어
- Colab 속도 확인 후 계획 진행. 188 Training 나머지(1920_1200, 1920_1080 묶음, rank.csv 참고)는 필요 시 `build_jobs`의 `only` 조건 수정.
- 614 Training(약 400GB) 은 킥보드 더 필요할 때만.
- 187 승인되면 `rank_files.py 187` 로 묶음별 보행신호 수 집계 → 상위 묶음만 추가(build_jobs에 188과 같은 방식으로).
- 결과 검증: shard 하나 받아서 `show_clip.py`(71579 시각화) 또는 bbox 그려서 확인.
- 보조 스크립트: `rank_files.py`(묶음별 대상 개수 → rank.csv), `count_val.py`, `show_clip.py`, `inspect_traffic_light.py`(로컬 폴더 점검), `migrate.py`.
