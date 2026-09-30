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
| pilot | E0 + A1~A5 + B1~B5 (15개), pilot 데이터(train에서 클래스당 약 200장, val은 클래스당 약 100장), 30 epoch, pilot val로 평가 |
| tier2 | A6, B6 (C-RADIOv4-SO400M), pilot 설정 |
| full | E0 + pilot 상위 fusion 2개 + 상위 distill 2개, 전체 데이터, 100 epoch(patience 20), seed 0/1/2, **test**로 평가 |

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
python run_all.py --summary pilot                 # 결과표만 다시 만들기
tail -f results/run_all.log                       # 진행 상황
touch results/HOLD_full                          # full 단계 직전에 멈추기 (지우고 다시 실행하면 이어서 진행)
```

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
