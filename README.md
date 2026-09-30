# A Proactive Egocentric Streaming Video Assistant for Blind Navigation

시각장애인 보행 보조를 위한 **능동형 1인칭 스트리밍 영상 어시스턴트** — Team Insight 캡스톤 프로젝트.

스트리밍 영상의 매 프레임은 경량 모듈(YOLO-seg + Trigger)이 감시하고, 위험 이벤트가 감지될 때만 VLM을 호출해 음성(TTS)으로 알린다.

## 브랜치

| 브랜치 | 담당 | 내용 |
|---|---|---|
| `main` | 공통 | 통합본 |
| `detection` | 박주영 | YOLO11s-seg 학습, 데이터셋 재라벨링, Foundation Model 인코더 비교 실험 |

---

## `detection` 브랜치: YOLO11s-seg × Foundation Model 인코더 실험

### 목적

AI Hub 189 인도 보행 영상(Polygon)을 10개 클래스로 재라벨링한 데이터에서, Foundation Model(FM) 인코더를 YOLO11s-seg에 붙이면 분할 성능이 오르는지, 특히 데이터가 부족한 클래스(킥보드 등)가 개선되는지 확인한다.

**클래스 (10):** person, bicycle, scooter, motorcycle, car, bus, other_vehicle, obstacle, stairs, traffic_light

### 두 가지 방식

| 트랙 | 방식 | 추론 시 FM | 용도 |
|---|---|---|---|
| **A. Fusion** | FM feature를 YOLO 백본 P3/P4/P5에 더함 (zero-init adapter) | 사용 | 성능 상한 확인 |
| **B. Distill** | FM을 teacher로만 쓰고, YOLO feature가 FM feature를 닮도록 학습 | **미사용** (순수 YOLO11s-seg) | 실제 배포 (노트북 CPU) |

### 비교 인코더

| ID (A/B) | 인코더 | 비고 |
|---|---|---|
| E0 | – | YOLO11s-seg baseline |
| 1 | SigLIP2-B/16 | vision-language, dense feature 개선판 |
| 2a / 2b | DINOv2-B / DINOv3-B | 자기지도, 공간 정보에 강함 |
| 3a / 3b | DINOv2-B + SigLIP2-B / DINOv3-B + SigLIP2-B | 두 인코더 채널 결합 |
| 4 | RADIOv2.5-B | CLIP + DINOv2 + SAM 증류 (비상업 라이선스) |
| 5 | C-RADIOv3-B | 상업 이용 가능 |
| 6 | C-RADIOv4-SO400M | Tier 2 |

모든 FM은 frozen. 데이터는 **영상(폴더) 단위**로 train/val/test = 70/15/15 분할.

### 실험 단계

`sanity`(동작 확인) → `pilot`(클래스당 약 200장, 30 epoch, 15개 실험) → `tier2` → `full`(E0 + pilot 상위 fusion 2개 + distill 2개, 전체 데이터, 100 epoch, seed 3개, test 평가)

### 실행 (몬드리안 RTX A5000)

```bash
git clone -b detection https://github.com/sangjune2001/A-Proactive-Egocentric-Streaming-Video-Assistant-for-Blind-Navigation.git repo
tmux new -s exp                              # 여기까지 붙여넣고, 아래는 tmux 세션 안에서 실행
```

```bash
cd ~/repo/insight_fm
read -s -p "HF token: " HF_TOKEN && export HF_TOKEN; echo   # DINOv3는 HF에서 라이선스 동의 필요. history에 안 남음
bash setup.sh                                # 설치 → Drive zip에서 640px 이미지 추출 → 가중치 → 셀프테스트 → 데이터셋 빌드
export RCLONE_REMOTE_RUNS=gdrive:sideguide/runs
python run_all.py --phase all 2>&1 | tee -a results/all.log
#   Ctrl+B, D 로 빠져나오기 / tmux attach -t exp 로 다시 보기
```

토큰을 `export HF_TOKEN=hf_...`로 직접 치면 `~/.bash_history`에 평문으로 남으니 쓰지 말 것.

중단되어도 같은 명령을 다시 실행하면 끝난 실험은 건너뛰고, 중단된 실험은 이어서 학습한다.

### 결과

- `insight_fm/results/pilot_summary.csv`, `full_summary.csv`: mask mAP50-95(평균±표준편차), baseline 대비 증감, 클래스별 AP, 추론 지연
- 배포용 모델 (distill → 순수 YOLO11s-seg):
  ```bash
  python export_plain.py runs/full/B3b_s0/weights/best.pt deploy/model.pt
  yolo export model=deploy/model.pt format=openvino imgsz=640
  ```

### 폴더 구조

```
insight_fm/
├── setup.sh            # 서버 세팅 전체
├── extract_images.py   # Drive 원본 zip → 라벨된 프레임만 640px 이미지로 추출
├── build_dataset.py    # jsonl + 이미지 → YOLO-seg 데이터셋 (영상 단위 분할)
├── encoders.py         # FM 인코더 로더
├── fm_yolo.py          # YOLO11s-seg + fusion / distill 모델
├── train_one.py        # 실험 1개: 학습 → 평가 → 속도 측정
├── run_all.py          # 전체 실험 자동 실행 + 결과표
├── export_plain.py     # distill 모델 → 순수 YOLO11s-seg
└── selftest.py         # 합성 데이터로 코드 검증
```

자세한 설명과 주의사항은 [`insight_fm/README.md`](insight_fm/README.md) 참고.
