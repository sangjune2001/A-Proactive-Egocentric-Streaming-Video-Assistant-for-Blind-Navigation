# insight_fm — YOLO11s-seg × Foundation Model 인코더 비교 실험

팀 Insight 시각장애인 보행 보조 서비스의 YOLO-seg에 foundation model(FM) 인코더를 붙이는 실험 코드.
몬드리안 RTX A5000(24GB) 한 대에서 **처음부터 끝까지 무인으로** 돌도록 만들었음.

## 실험 설계

| ID | 트랙 | 인코더 | 설명 |
|---|---|---|---|
| E0 | – | – | YOLO11s-seg baseline |
| A1 / B1 | fusion / distill | SigLIP2-B/16 | |
| A2a / B2a | fusion / distill | DINOv2-B/14 | |
| A2b / B2b | fusion / distill | DINOv3-B/16 | |
| A3a / B3a | fusion / distill | DINOv2-B + SigLIP2-B | 두 인코더 채널 결합 |
| A3b / B3b | fusion / distill | DINOv3-B + SigLIP2-B | 두 인코더 채널 결합 |
| A4 / B4 | fusion / distill | RADIOv2.5-B | 비상업 라이선스 |
| A5 / B5 | fusion / distill | C-RADIOv3-B | 상업 이용 가능 |
| A6 / B6 | fusion / distill | C-RADIOv4-SO400M | Tier 2 (마지막에) |

- **트랙 A (fusion):** FM feature를 YOLO 백본 출력 P3/P4/P5에 더함. 마지막 conv를 0으로 초기화해서, 학습 시작 시점은 사전학습된 YOLO11s-seg와 완전히 같음. 추론할 때도 FM이 돌아감 → 성능 상한 확인용.
- **트랙 B (distill):** FM은 teacher로만 씀. P3/P4/P5를 FM 공간으로 투영하는 작은 head를 두고 cosine 손실로 FM feature를 닮게 학습. **추론은 순수 YOLO11s-seg** → 그램 CPU에서 baseline과 같은 속도. `export_plain.py`로 head를 떼어냄.
- FM은 전부 **frozen**. 모든 인코더의 feature grid는 입력의 1/16(P4)에 맞춰짐 (DINOv2는 patch 14라 내부에서 560 입력으로 리사이즈).
- 공통 설정: YOLO11s-seg 사전학습 가중치, imgsz 640, `copy_paste=0.3`, AMP. batch는 단일 인코더 16, 두 인코더 결합 12, SO400M 8.
- 데이터 분할: **영상(폴더) 단위** 70/15/15. val/test에 모든 클래스가 최대한 들어가는 seed를 자동 선택.

### 단계 (`run_all.py --phase all`이 순서대로 실행)

| 단계 | 내용 |
|---|---|
| sanity | E0, A2b, B3b를 데이터 2%, 1 epoch로. 실패하면 거기서 멈춤 |
| pilot | E0 + A1~A5 + B1~B5 (15개), pilot 데이터(train에서 scooter·stairs·traffic_light가 있는 사진 전부 + 나머지 클래스는 사진 약 200장씩, val은 클래스당 사진 약 100장), 30 epoch, pilot val로 평가 |
| tier2 | A6, B6 (C-RADIOv4-SO400M), pilot 설정 |
| full | E0 + pilot 상위 fusion 2개 + 상위 distill 2개, 전체 데이터, 100 epoch(patience 20), seed 0/1/2, **test**로 평가 |
| final | `--phase final`(기본 E0, `--ids`로 지정). pilot train으로 100 epoch(patience 30), best는 pilot val로 고르고 **full test 13,721장**으로 평가. `--phase all`에는 포함 안 됨 |

## 실행 순서

```bash
# 0) 코드 올리기
scp -r insight_fm <서버>:~/  &&  ssh <서버>  &&  cd ~/insight_fm

# 1) rclone 설정 복사 (WSL에서 쓰던 것)
#    WSL에서:  scp ~/.config/rclone/rclone.conf <서버>:~/.config/rclone/rclone.conf

# 2) tmux 세션 열기 (세션이 끊겨도 계속 돌도록). 아래 명령은 전부 세션 안에서 실행
tmux new -s exp
cd ~/insight_fm

# 3) 세팅 — 기본값: 라벨 gdrive:aihub189_yolo/labels_all.jsonl, 이미지 gdrive:sideguide/polygon/P1~P14.zip + gdrive:sideguide/surface/S1.zip, 로컬 ~/sg
read -s -p "HF token: " HF_TOKEN && export HF_TOKEN; echo   # DINOv3는 HF 라이선스 동의 후 토큰 필요. history에 안 남음
bash setup.sh                                   # deps → data → hf → prefetch → selftest → build

# 4) 무인 실행
export RCLONE_REMOTE_RUNS=gdrive:sideguide/runs # epoch마다 Drive로 백업
python run_all.py --phase all 2>&1 | tee -a results/all.log
#   Ctrl+B, D 로 빠져나오기 / tmux attach -t exp 로 다시 보기
```

**중간에 끊겨도** 같은 명령을 다시 치면 됨. 끝난 실험은 건너뛰고, 중단된 실험은 `last.pt`에서 이어서 학습함.

자주 쓰는 명령:

```bash
python run_all.py --phase pilot --ids E0,B3b      # 일부만
python run_all.py --phase full --ids E0,A5,B3b    # full 대상 직접 지정
python run_all.py --phase final                   # 최종 학습: E0, 100 epoch, test 평가
python run_all.py --phase final --ids A5,B5       # 이어서 C-RADIOv3 fusion / distill
python run_all.py --summary pilot                 # 결과표만 다시 만들기
tail -f results/run_all.log                       # 진행 상황
touch results/HOLD_full                          # full 단계 직전에 멈추기 (지우고 다시 실행하면 이어서 진행)
```

## 데이터 분석

[`analysis/data_analysis.md`](analysis/data_analysis.md) (`python analyze_data.py --jsonl ~/sg/labels_all.jsonl --out analysis`로 재생성)

## Pilot + Tier 2 결과 (2026-10-01, RTX A5000)

설정 (1차 pilot): train 996장 / val 443장(클래스당 사진 약 200 / 100장. 다음 pilot부터는 scooter·stairs·traffic_light 사진 전부 포함, train 8,120장), 30 epoch, seed 0, imgsz 640. 지표는 pilot val 기준.
지연은 A5000에서 batch 1, FP16. 원본: `results/pilot_summary.csv`, 모델·로그: `gdrive:sideguide/runs/<ID>_s0/`.

| ID | 방식 | 인코더 | mask mAP50-95 | E0 대비 | mask mAP50 | box mAP50-95 | 지연 (ms) |
|---|---|---|---|---|---|---|---|
| **A6** | fusion | C-RADIOv4-SO400M | 0.358 | +0.074 | 0.629 | 0.459 | 47.8 |
| **A2a** | fusion | DINOv2-B | 0.336 | +0.052 | 0.604 | 0.433 | 22.3 |
| A2b | fusion | DINOv3-B | 0.332 | +0.049 | 0.591 | 0.426 | 27.1 |
| A3a | fusion | DINOv2-B + SigLIP2-B | 0.332 | +0.048 | 0.604 | 0.434 | 34.0 |
| **A5** | fusion | C-RADIOv3-B | 0.327 | +0.043 | 0.588 | 0.420 | 20.8 |
| A3b | fusion | DINOv3-B + SigLIP2-B | 0.326 | +0.042 | 0.583 | 0.425 | 46.9 |
| A4 | fusion | RADIOv2.5-B | 0.318 | +0.035 | 0.577 | 0.415 | 20.8 |
| A1 | fusion | SigLIP2-B | 0.313 | +0.030 | 0.572 | 0.408 | 28.6 |
| B5 | distill | C-RADIOv3-B | 0.286 | +0.003 | 0.519 | 0.367 | 13.1 |
| B4 | distill | RADIOv2.5-B | 0.285 | +0.001 | 0.530 | 0.372 | 10.7 |
| **E0** | baseline | – | 0.283 | 0 | 0.519 | 0.370 | 10.4 |
| B3a | distill | DINOv2-B + SigLIP2-B | 0.281 | -0.003 | 0.524 | 0.362 | 12.8 |
| B2a | distill | DINOv2-B | 0.280 | -0.003 | 0.512 | 0.362 | 10.4 |
| B6 | distill | C-RADIOv4-SO400M | 0.280 | -0.004 | 0.518 | 0.358 | 12.9 |
| B1 | distill | SigLIP2-B | 0.278 | -0.006 | 0.516 | 0.363 | 10.3 |
| B3b | distill | DINOv3-B + SigLIP2-B | 0.277 | -0.006 | 0.514 | 0.363 | 10.6 |
| B2b | distill | DINOv3-B | 0.276 | -0.007 | 0.511 | 0.358 | 10.6 |

클래스별 mask AP50-95:

| ID | person | bicycle | scooter | motorcycle | car | bus | other_vehicle | obstacle | stairs | traffic_light |
|---|---|---|---|---|---|---|---|---|---|---|
| A6 | 0.425 | 0.288 | 0.182 | 0.442 | 0.518 | 0.446 | 0.397 | 0.379 | 0.329 | 0.173 |
| A2a | 0.396 | 0.248 | 0.169 | 0.426 | 0.491 | 0.434 | 0.366 | 0.361 | 0.295 | 0.172 |
| A2b | 0.392 | 0.265 | 0.160 | 0.400 | 0.501 | 0.432 | 0.373 | 0.354 | 0.288 | 0.157 |
| A3a | 0.401 | 0.251 | 0.189 | 0.411 | 0.502 | 0.435 | 0.363 | 0.361 | 0.248 | 0.154 |
| A5 | 0.391 | 0.225 | 0.168 | 0.407 | 0.494 | 0.402 | 0.341 | 0.365 | 0.306 | 0.170 |
| A3b | 0.389 | 0.239 | 0.168 | 0.408 | 0.504 | 0.422 | 0.345 | 0.353 | 0.280 | 0.146 |
| A4 | 0.383 | 0.239 | 0.135 | 0.392 | 0.488 | 0.401 | 0.326 | 0.358 | 0.294 | 0.170 |
| A1 | 0.385 | 0.243 | 0.136 | 0.399 | 0.500 | 0.409 | 0.323 | 0.334 | 0.274 | 0.130 |
| B5 | 0.356 | 0.218 | 0.133 | 0.362 | 0.468 | 0.380 | 0.239 | 0.307 | 0.285 | 0.114 |
| B4 | 0.365 | 0.213 | 0.127 | 0.364 | 0.470 | 0.364 | 0.247 | 0.314 | 0.261 | 0.122 |
| E0 | 0.358 | 0.205 | 0.143 | 0.356 | 0.470 | 0.365 | 0.243 | 0.321 | 0.239 | 0.134 |
| B3a | 0.376 | 0.194 | 0.135 | 0.339 | 0.479 | 0.353 | 0.251 | 0.302 | 0.257 | 0.118 |
| B2a | 0.360 | 0.211 | 0.133 | 0.357 | 0.469 | 0.360 | 0.242 | 0.313 | 0.243 | 0.112 |
| B6 | 0.352 | 0.199 | 0.126 | 0.350 | 0.475 | 0.350 | 0.249 | 0.326 | 0.245 | 0.124 |
| B1 | 0.350 | 0.211 | 0.122 | 0.345 | 0.472 | 0.376 | 0.240 | 0.320 | 0.219 | 0.123 |
| B3b | 0.358 | 0.194 | 0.147 | 0.347 | 0.477 | 0.352 | 0.245 | 0.302 | 0.244 | 0.104 |
| B2b | 0.365 | 0.217 | 0.113 | 0.342 | 0.478 | 0.364 | 0.227 | 0.316 | 0.220 | 0.116 |

**요약**

- **fusion은 7개 모두 E0보다 높음** (+0.030 ~ +0.074, 상대 11~26%). 최고는 A6(SO400M) 0.358이지만 지연 4.6배, FM 4.3억 파라미터.
  ViT-B 중에서는 A2a(DINOv2) +0.052, A5(C-RADIOv3, 상업 이용 가능) +0.044, 둘 다 지연 약 2배.
- **두 인코더 결합(A3a/A3b)은 단일 인코더보다 낫지 않고 더 느림.**
- **distill은 8개 모두 E0와 ±0.008 이내로 효과 없음.** SO400M teacher(B6)도 마찬가지. 이 설정(996장, 30 epoch, λ=1)에서는 FM 지식이 전달되지 않음.
- fusion에서 많이 오른 클래스: other_vehicle(0.24 → 0.33~0.40), stairs, motorcycle, bus. scooter·traffic_light는 모든 모델에서 0.19 이하.

**주의**

- seed 1개, val 443장이라 ±0.01 이하 차이는 노이즈일 수 있음(distill끼리의 순위, A2b/A3a/A5 순위 등). fusion과 E0의 차이(+0.03 이상)는 뚜렷함.
- stairs(train 264개), scooter(197개)는 데이터가 매우 적어 클래스 AP 변동이 큼.
- E0는 클래스 이름 수정(7=obstacle, 8=stairs) 직전에 시작돼서, `done.json`의 obstacle/stairs 값을 사후에 맞바꿈(`done.json.bak`에 원본).
- full 단계는 아직 실행하지 않음(`results/HOLD_full`).

**어떤 모델이 좋은가 (pilot 기준 판단)**

목적이 시각장애인 보행 보조(실시간)라서 성능과 속도를 같이 봄.

| 기준 | 후보 | 근거 |
|---|---|---|
| 성능만 | A6 (C-RADIOv4-SO400M fusion) | mAP50-95 0.358로 최고. 지연 4.6배, FM 4.3억 파라미터라 실시간용으로는 무거움 |
| **성능·속도 균형 (1순위)** | **A2a (DINOv2-B fusion)** | E0 대비 +0.052, mAP50 0.604. 지연 약 2배 (10.4 → 22.3ms) |
| 공동 후보 | A5 (C-RADIOv3-B fusion) | +0.043, fusion 중 가장 빠름 (20.8ms) |
| 배포 용이성 | 해당 없음 | distill은 E0와 성능이 같아 지금은 고를 이유가 없음 |

- A2a와 A5는 차이가 0.009로 seed 1개 기준 노이즈 범위라 사실상 동점. 둘 다 상업 이용 가능(DINOv2 Apache 2.0, C-RADIOv3 상업 이용 허용).
- **가장 큰 변수는 배포 기기.** fusion은 추론 때도 ViT-B가 돌기 때문에 그램 CPU(OpenVINO)에서는 GPU보다 훨씬 더 느려질 수 있음.
  실시간 속도가 안 나오면 쓸 수 있는 건 E0뿐이고, 남은 방법은 distill 개선(추론 속도는 E0와 같음).

**다음 단계**

1. 그램에서 E0 / A2a / A5의 실제 FPS 측정 (GPU 서버 불필요). 모델: `gdrive:sideguide/runs/<ID>_s0/weights/best.pt`
2. 속도가 충분하면 → full을 **E0 + A2a + A5**로 축소해서 진행. 원래 설계(5개 모델 × seed 3 × 100 epoch, 전체 데이터)는 A5000 기준 약 400~600시간으로 남은 크레딧(약 200시간)을 넘음.
   예: E0 + A2a + A5, seed 1, 50 epoch → 약 60시간 (pilot 속도로 역산한 대략치)
3. 너무 느리면 → full보다 **distill 개선 pilot**이 먼저 (손실 가중치 λ 증가, 학습 길이 증가 등)

## 결과물

- `results/pilot_summary.csv`, `results/full_summary.csv`: mask mAP50-95(평균±표준편차), mAP50, **E0 대비 증감**, 클래스별 AP(scooter 등), A5000 batch 1 FP16 지연(ms)/FPS, 추론 시 쓰이는 FM 파라미터 수
- `runs/<phase>/<ID>_s<seed>/`: Ultralytics 기본 결과(학습 곡선, confusion matrix, `weights/best.pt`) + `done.json`(모든 지표)
- 그램 배포용 (distill 또는 baseline):

```bash
python export_plain.py runs/full/B3b_s0/weights/best.pt deploy/B3b_plain.pt
yolo export model=deploy/B3b_plain.pt format=openvino imgsz=640   # 그램에서 OpenVINO로
```

## 파일

| 파일 | 역할 |
|---|---|
| `setup.sh` | 패키지 설치, Drive에서 라벨·zip 받기, HF 로그인, 가중치 미리 받기, 셀프테스트, 데이터셋 빌드 |
| `extract_images.py` | Drive의 원본 zip을 **하나씩** 받아 라벨된 프레임만 640px로 저장하고 zip은 삭제 (중단돼도 이어서) |
| `build_dataset.py` | `labels_all.jsonl` + 640px 이미지 → YOLO-seg 폴더(full/pilot), 영상 단위 분할, 클래스 분포 출력 |
| `encoders.py` | FM 인코더 7종 로더 (SigLIP2, DINOv2, DINOv3, RADIOv2.5, C-RADIOv3/v4, 결합) |
| `fm_yolo.py` | YOLO11s-seg에 fusion / distill을 붙인 모델과 Trainer |
| `train_one.py` | 실험 1개: 학습(또는 이어서 학습) → 평가 → 속도 측정 → `done.json` |
| `run_all.py` | 전체 단계를 순서대로 실행하고 결과표를 만듦 |
| `export_plain.py` | distill 모델을 순수 YOLO11s-seg `.pt`로 변환 |
| `selftest.py` | 합성 데이터로 전체 흐름 검증 (CPU로도 됨) |

## 꼭 알아둘 것

1. **`build_dataset.py --inspect` 결과를 먼저 볼 것.** jsonl 필드명과 이미지 파일명 규칙을 자동으로 맞추게 해뒀지만, `image_found=False`가 나오면 키와 파일명이 안 맞는 것이니 멈추고 확인.
2. **디스크:** 원본 zip은 총 134GB지만 한 번에 하나(약 10GB)만 받고 지우니, 여유 공간은 zip 하나 + 추출 이미지 + 가중치 정도면 충분함. 이미지 추출이 중간에 끊기면 `STEP=data bash setup.sh`로 다시 실행하면 이어서 함.
3. **HF 접근 권한:** DINOv3(`facebook/dinov3-vitb16-pretrain-lvd1689m`)는 HF 페이지에서 라이선스 동의 후 토큰이 있어야 받아짐. `setup.sh`의 prefetch 단계에서 `FAIL`이 뜨는 인코더가 있으면 그 실험만 실패로 기록되고 나머지는 계속 돔.
4. **라이선스:** RADIOv2.5는 NVIDIA 비상업 라이선스, C-RADIO 계열은 상업 이용 가능.
5. **검증 범위:** 코드 전체(데이터 빌드 → fusion/distill 학습 → 평가 → 이어서 학습 → plain export)는 CPU 합성 데이터로 검증함. 실제 사전학습 가중치 로드와 GPU 속도는 서버의 `prefetch`와 `sanity` 단계에서 처음 확인됨.
6. **포함하지 않은 것:** 인코더 부분 unfreeze/LoRA(모든 실험이 frozen 기준), 객체 크기별(소형) AP. 필요하면 pilot 결과를 보고 추가.
7. **소요 시간:** sanity가 끝나면 `results/run_all.log`에 실험별 소요 시간이 찍힘. pilot 첫 실험(E0) 시간을 기준으로 나머지를 추정할 것. 두 인코더 결합과 SO400M은 E0보다 확실히 느림.
