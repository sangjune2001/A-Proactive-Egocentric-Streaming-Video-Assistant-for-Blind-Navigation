# 인수인계 보고서: `detection` 브랜치 (YOLO11s-seg × Foundation Model)

> 이 문서만 읽고 작업을 이어받을 수 있도록 쓴 보고서다. 무엇을 했고, 무엇이 되고 안 되는지, 왜 그런지, 숫자는 어디서 나왔는지, 어떻게 다시 돌리는지를 모두 담았다.
> 숫자 옆의 경로는 그 숫자를 만든 파일이다. 같은 명령을 다시 실행하면 같은 숫자가 나온다.
>
> 작성: 2026-10-03. **⏳ 표시는 학습 진행 중이라 결과가 나오면 채울 부분**(5클래스 최종 학습 E0 / A5, 추론 속도).

---

## 0. 한 페이지 요약

| 질문 | 답 |
|---|---|
| 무엇을 하나 | 시각장애인 보행 보조용 분할 모델(YOLO11s-seg)에 Foundation Model(FM) 인코더를 붙이면 성능이 오르는지 실험 |
| 데이터 | AI Hub 「인도보행 영상」(dataSetSn 189). 라벨된 사진 92,772장, 영상 1,955개, 객체 581,529개 |
| 1차 결과 (10클래스, 소량 데이터) | **FM을 추론에도 쓰는 fusion은 7개 모두 baseline보다 좋음**(mask mAP50-95 0.283 → 0.313~0.358). **FM을 학습 때만 쓰는 distill은 효과 없음**(±0.008) |
| 지금 진행 중 | 5클래스(scooter, stairs, obstacle, other_vehicle, traffic_light) 전용 모델, 100 epoch, test 13,721장으로 최종 평가. baseline(E0) → C-RADIOv3 fusion(A5) 순서 (distill B5는 요청에 따라 제외) |
| mAP가 낮은 주원인 (데이터에서 확인됨) | ① **작은 객체**: traffic_light 89%가 박스 짧은 변 16px 미만 ② **라벨 노이즈**: stairs 라벨의 상당수가 실제로는 맨홀·가로수 보호판·연석 (AI Hub 원본 라벨 문제) ③ **극소량 클래스**: scooter 사진 224장(영상 124개), stairs 372장 ④ **obstacle은 생김새가 다른 15종을 합친 클래스**(기둥·나무·볼라드·입간판 등) |
| 문제가 아닌 것 | 흐림(흐린 객체 0.2%, 대부분 야간), 어두운 프레임(0.4%) |
| 가장 먼저 할 일 | ⏳ 최종 결과 확인 → stairs 재라벨링 여부 결정 → 배포 기기(노트북 CPU)에서 속도 실측 |

---

## 1. 프로젝트와 이 브랜치의 역할

- 레포: `github.com/sangjune2001/A-Proactive-Egocentric-Streaming-Video-Assistant-for-Blind-Navigation`
- 전체 시스템: 1인칭 스트리밍 영상 → 경량 모듈(YOLO-seg + Trigger)이 매 프레임 감시 → 위험할 때만 VLM 호출 → TTS로 안내.
- **이 브랜치(`detection`)는 그중 YOLO-seg 부분**: 데이터셋 구축(재라벨링), YOLO11s-seg 학습, FM 인코더 비교. `main`에는 아직 병합하지 않았다.
- 실시간 보조가 목적이므로 **성능과 추론 속도를 같이** 봐야 한다. 최종 배포는 노트북(그램) CPU + OpenVINO를 가정했다(`insight_fm/export_plain.py` 주석).

## 2. 데이터

### 2.1 출처와 위치

| 무엇 | 어디 | 비고 |
|---|---|---|
| 원본 영상 프레임 (Polygon) | `gdrive:sideguide/polygon/P1.zip` ~ `P14.zip` | zip당 약 10GB, 합계 약 134GB. 1920×1080 |
| 원본 노면 데이터 (Surface) | `gdrive:sideguide/surface/S1.zip` | stairs 라벨은 여기서만 나옴 |
| **학습용 라벨** | `gdrive:aihub189_yolo/labels_all.jsonl` (243MB) | 29종 원본 라벨을 10클래스로 재라벨링한 YOLO-seg 폴리곤. Colab에서 만듦 |
| 원본 라벨 개수표 | `gdrive:sideguide/class_counts.csv` | AI Hub 원본 라벨별 객체·사진·폴더 수 |
| 라벨을 만든 기존 모델 | `gdrive:aihub189_yolo/runs/seg10/` | 클래스 순서의 정답(7=obstacle, 8=stairs)은 이 모델의 `names`로 확인 |

`labels_all.jsonl` 한 줄 형식:
```json
{"k": "Polygon_0001__MP_SEL_P000002",            // 고유 키 = <영상 폴더>__<파일명>
 "z": "/content/drive/MyDrive/sideguide/polygon/P1.zip",  // 원본 zip (MyDrive/ 뒤가 Drive 경로)
 "m": "Polygon_0001/MP_SEL_P000002.jpg",          // zip 안 경로
 "l": ["7 0.28 0.00 0.29 0.34 ...", ...]}         // YOLO-seg: 클래스 x1 y1 x2 y2 ... (0~1 정규화)
```

### 2.2 클래스 (10개)와 원본 매핑

`0 person, 1 bicycle, 2 scooter, 3 motorcycle, 4 car, 5 bus, 6 other_vehicle, 7 obstacle, 8 stairs, 9 traffic_light`

- **obstacle = 원본 15종 합침**: pole 98,463 / tree_trunk 97,018 / bollard 37,266 / movable_signage 20,203 / potted_plant 8,031 / bench 6,215 / power_controller 5,262 / barricade 4,323 / stop 4,311 / traffic_light_controller 3,986 / chair 3,643 / fire_hydrant 2,593 / table 1,317 / kiosk 1,198 / parking_meter 92 → 합계 293,921 (학습 데이터 293,918과 일치)
- **other_vehicle** = truck 33,211 + carrier 1,479 + stroller 485 + wheelchair 204
- **stairs** = Surface의 `caution_zone[stairs]` 399개 (Polygon에는 계단 클래스가 없음)
- **안 쓰는 원본 라벨**: traffic_sign 38,918개(꽤 많음), dog 166, cat 51
- 전체 매핑표: [`insight_fm/analysis/data_analysis.md` 7절](insight_fm/analysis/data_analysis.md)

### 2.3 규모와 분할

| | 영상 | 사진 | 객체 |
|---|---|---|---|
| 전체 | 1,955 | 92,772 | 581,529 (사진당 6.3개) |
| train | 1,368 | 64,952 | 403,570 |
| val | 293 | 14,099 | 90,024 |
| test | 294 | 13,721 | 87,935 |

- **영상 단위** 70/15/15 분할: 같은 영상의 프레임이 train과 test에 섞이지 않는다(섞이면 성능이 부풀려짐).
- 50개 seed 중 val·test에 드문 클래스가 가장 많이 들어가는 seed(34)를 자동 선택. 10클래스·5클래스 빌드 모두 **같은 분할**을 쓴다(`build_dataset.py --classes`는 분할을 정한 뒤 클래스를 거름).

### 2.4 클래스별 개수: 객체 수 ≠ 사진 수

"객체 수"는 폴리곤 개수, "사진 수"는 그 클래스가 하나라도 있는 사진 수다. 신호등 3개가 있는 사진은 객체 3, 사진 1.

| 클래스 | 객체 | 사진 | 영상 | 사진당 객체 | train 객체 / 사진 | test 객체 / 사진 |
|---|---|---|---|---|---|---|
| obstacle | 293,918 | 82,745 | 1,840 | 3.55 | 203,896 / 57,977 | 43,540 / 12,123 |
| car | 147,131 | 55,957 | 1,800 | 2.63 | 102,448 / 39,117 | 22,566 / 8,307 |
| person | 47,192 | 26,149 | 1,691 | 1.80 | 32,735 / 18,008 | 7,174 / 4,054 |
| other_vehicle | 35,379 | 21,516 | 1,681 | 1.64 | 24,645 / 14,984 | 5,317 / 3,286 |
| traffic_light | 26,796 | 11,106 | 1,374 | 2.41 | 18,598 / 7,750 | 4,193 / 1,680 |
| bus | 11,321 | 6,484 | 1,323 | 1.75 | 7,526 / 4,331 | 1,978 / 1,138 |
| bicycle | 10,003 | 6,581 | 1,367 | 1.52 | 6,969 / 4,566 | 1,600 / 1,031 |
| motorcycle | 9,039 | 6,731 | 1,261 | 1.34 | 6,292 / 4,692 | 1,394 / 1,025 |
| **stairs** | **399** | **372** | 115 | 1.07 | 264 / 247 | 79 / 72 |
| **scooter** | **351** | **224** | 124 | 1.57 | 197 / 144 | 94 / 33 |

- AI Hub 원본 표(scooter 351개 / 224장 / 124폴더)와 정확히 일치한다. **scooter는 train에 144장뿐**이고 test에는 33장이라 test AP 변동이 크다.
- 가장 많은 obstacle과 가장 적은 scooter의 객체 수는 약 837배 차이.
- 출처: `insight_fm/analysis/class_stats_all.csv`, `class_stats_by_split.csv` (`python analyze_data.py`)

### 2.5 카메라와 촬영 높이

| 카메라 | 사진 | 비율 | 추정 높이 |
|---|---|---|---|
| 스마트폰 (`MP_SEL`, `MP_KSC`, `MP_TW`) | 53,003 | 57% | 가슴 높이 (약 1.3~1.5m, 손에 들고 촬영) |
| ZED 스테레오 (`ZED1`~`ZED4`) | 39,397 | 43% | 허리~가슴 아래 (약 0.8~0.9m, 거치 촬영 추정) |
| Surface 스마트폰 | 372 | 0.4% | 바닥을 향해 촬영 |

- 높이는 공식 수치가 아니라 **추정치**다. 사진 속 보행자(약 1.7m)와 지평선 위치를 비교해 계산했다(사진 2장 기준, 경사로에서는 오차 큼). AI Hub 소개 페이지에는 촬영 높이가 없고, 데이터 설명서 PDF에 있을 수 있다.
- 두 카메라 모두 약간 아래를 향해 화면 아래쪽 절반 가까이가 보도다. 예시: `insight_fm/analysis/camera_samples.jpg`
- **배포 시 카메라를 가슴 높이에 다는 것이 학습 데이터와 가장 잘 맞는다.** 머리·안경 높이에서는 시점이 달라 성능이 떨어질 수 있다.

## 3. 실제 이미지로 본 객체 상태

학습에 쓰는 640px 이미지에서 객체 58만 개를 하나씩 측정했다(`insight_fm/analyze_objects.py`, 결과 [`analysis/objects/object_quality.md`](insight_fm/analysis/objects/object_quality.md), 원본 측정값 `gdrive:sideguide/runs_final5/analysis/objects.csv.gz`).

| 클래스 | 박스 짧은 변 중앙값 | 아주 작음 (<16px) | 잘림 (가장자리에 닿음) | 흐림 | 어두움 (평균 밝기<50) |
|---|---|---|---|---|---|
| **traffic_light** | **8px** | **88.8%** | 6.8% | 0.1% | 26.9% |
| obstacle | 16px | 47.9% | 30.3% | 0.6% | 21.6% |
| person | 18px | 39.0% | 5.9% | 0.2% | 24.2% |
| scooter | 25px | 30.8% | 7.1% | 0% | 17.4% |
| other_vehicle | 34px | 16.6% | 17.4% | 0.1% | 15.3% |
| stairs | 70px | 15.8% | **65.4%** | 0% | 0% |
| car | 30px | 18.9% | 17.0% | 0% | 13.4% |

해석(예시 이미지를 직접 보고 확인함):

1. **작은 객체가 가장 큰 문제.** traffic_light는 절반이 짧은 변 8px 이하다(예: 14×4px, 7×3px). 640px 입력에서는 신호등 색도 구분하기 어렵다. obstacle도 기둥처럼 **폭 5~20px인 가늘고 긴 객체**가 많아 마스크를 정확히 맞추기 어렵다.
2. **잘림은 대부분 자연스러운 경우.** 기둥·나무처럼 키 큰 객체가 화면 위로 잘린 것이고(카메라가 낮아서), 보이는 부분은 제대로 라벨되어 있다. stairs의 65%가 잘린 것도 계단이 화면 가장자리에 크게 걸치기 때문이다. 성능 저하 요인으로는 작다.
3. **흐림은 문제가 아님.** 흐린 객체 926개(0.2%)는 대부분 야간·역광의 어두운 장면이다. 움직임 블러는 거의 없다. (흐림 지표는 객체 박스 안 Laplacian 분산 < 50. JPEG 노이즈 때문에 절대값보다는 "거의 없다"는 결론만 믿을 것.)
4. **"어두움"은 조명보다 물체 색.** 어두운 객체의 대부분은 검은 기둥·나무 줄기·그늘 속 물체다. 어두운 프레임 자체는 0.4%뿐이다.
5. 클래스별 실제 모습: `analysis/objects/sheet_<클래스>.jpg`, 문제 유형별: `problem_<tiny|truncated|blurry|dark>.jpg`

## 4. ⚠️ 라벨 품질 문제 (원본 AI Hub)

### 4.1 stairs 라벨의 상당수는 계단이 아니다

- 학습 데이터의 stairs 399개는 **원본 XML의 `caution_zone[stairs]` 399개와 좌표까지 전부 일치**한다(변환 코드는 정확함).
- 그런데 원본 이미지에 원본 폴리곤을 그려 보면 **실제 계단은 3분의 1 정도**이고, 나머지는 **가로수 보호판, 맨홀, 연석·턱, 차도 경계석**에 stairs가 붙어 있다. 즉 **AI Hub 원본 작업자의 라벨 오류**다.
- 증거 이미지: [`insight_fm/analysis/stairs_label_noise.jpg`](insight_fm/analysis/stairs_label_noise.jpg) (빨간 선 = 원본 stairs 폴리곤, 무작위 12장)
- 게다가 **Polygon 영상(전체의 99.6%)에는 계단 클래스 자체가 없어서**, 그 영상에 찍힌 계단은 배경으로 학습된다.
- 결론: **stairs는 데이터를 늘려도 좋아지지 않는다. 재라벨링이 먼저다.** 372장이라 직접 고치는 데 몇 시간이면 된다.
- 확인 스크립트: `insight_fm/tools/s1chk.py`(원본 XML 라벨 집계), `s1match.py`(학습 라벨 ↔ 원본 폴리곤 대조), `s1draw.py`(원본 폴리곤 그리기). S1.zip을 `~/sg/s1chk/`에 받아 두고 실행

### 4.2 기타 라벨 특성

- **obstacle = 15종 혼합**: 기둥·나무·볼라드·입간판·화분·벤치·키오스크… 생김새가 전혀 달라 하나의 클래스로 배우기 어렵다. 보행 보조 관점에서는 "부딪힐 수 있는 것"이라 합친 것으로 보이며, 성능을 올리려면 2~3개 그룹(기둥형 / 낮은 장애물 / 큰 구조물)으로 나누는 것을 검토할 만하다.
- **other_vehicle**: 대부분 트럭. 일부는 승합차·SUV처럼 car와 경계가 애매하다.
- **scooter**: 공유 전동킥보드, 어린이 킥보드, 전동 휠까지 포함. 개수가 너무 적다.
- **traffic_light**: 보행자·차량 신호등, 앞면·뒷면(불 꺼진 뒷면 포함)이 섞여 있다.

## 5. 실험

### 5.1 방식과 ID

- **A = Fusion**: FM feature를 YOLO 백본 P3/P4/P5에 더함(마지막 conv 0 초기화 → 시작 시점은 원래 YOLO와 동일). **추론할 때도 FM이 돈다** → 느림.
- **B = Distill**: FM은 학습할 때 teacher로만 쓰고, YOLO feature가 FM feature를 닮도록 cosine 손실 추가. **추론은 순수 YOLO11s-seg** → baseline과 같은 속도. `export_plain.py`로 보조 head를 떼어냄.
- **E0** = FM 없는 baseline. 번호는 인코더: 1 SigLIP2-B, 2a DINOv2-B, 2b DINOv3-B, 3a/3b 두 인코더 결합, 4 RADIOv2.5-B(비상업), 5 C-RADIOv3-B(상업 가능), 6 C-RADIOv4-SO400M(4억 파라미터).
- 모든 FM은 frozen. 공통: YOLO11s-seg COCO 사전학습 가중치, imgsz 640, copy_paste 0.3, AMP, seed 0.

### 5.2 1차 pilot (10클래스, 2026-10-01, A5000)

설정: train 996장(클래스마다 그 클래스가 든 **사진** 약 200장) / val 443장, 30 epoch, pilot val로 평가.

| ID | 방식 | 인코더 | mask mAP50-95 | E0 대비 | 지연 (ms, A5000 FP16 bs1) |
|---|---|---|---|---|---|
| A6 | fusion | C-RADIOv4-SO400M | **0.358** | +0.074 | 47.8 |
| A2a | fusion | DINOv2-B | 0.336 | +0.052 | 22.3 |
| A2b | fusion | DINOv3-B | 0.332 | +0.049 | 27.1 |
| A3a | fusion | DINOv2-B + SigLIP2-B | 0.332 | +0.048 | 34.0 |
| A5 | fusion | C-RADIOv3-B | 0.327 | +0.043 | 20.8 |
| A3b | fusion | DINOv3-B + SigLIP2-B | 0.326 | +0.042 | 46.9 |
| A4 | fusion | RADIOv2.5-B | 0.318 | +0.035 | 20.8 |
| A1 | fusion | SigLIP2-B | 0.313 | +0.030 | 28.6 |
| B5 | distill | C-RADIOv3-B | 0.286 | +0.003 | 13.1 |
| B4 | distill | RADIOv2.5-B | 0.285 | +0.001 | 10.7 |
| **E0** | baseline | – | **0.283** | 0 | **10.4** |
| B3a / B2a / B6 / B1 / B3b / B2b | distill | (각각) | 0.276 ~ 0.281 | −0.003 ~ −0.008 | 10.3 ~ 12.9 |

전체 표·클래스별 AP: [`insight_fm/README.md`](insight_fm/README.md), 원본: `insight_fm/results/pilot_summary.csv`, 모델·로그: `gdrive:sideguide/runs/<ID>_s0/`.

**되는 것**
- fusion은 7개 모두 확실히 오른다(+0.03 이상, seed 노이즈보다 큼). 많이 오른 클래스: other_vehicle(0.24 → 0.33~0.40), stairs, motorcycle, bus.
- 실용 후보: A2a(DINOv2), A5(C-RADIOv3). 둘 다 지연 약 2배. 둘의 차이 0.009는 노이즈 범위.

**안 되는 것**
- distill은 8개 모두 효과 없음(teacher를 SO400M으로 키운 B6도 마찬가지). 학습 데이터 996장·30 epoch·λ=1이라는 조건에서는 FM 지식이 전달되지 않았다. 최종 학습에서는 제외했다(데이터·epoch를 늘렸을 때 달라지는지는 확인하지 않음).
- 인코더 두 개 결합(A3a/A3b)은 하나보다 낫지 않고 느리다.
- scooter·traffic_light는 모든 모델에서 AP 0.1~0.19.

**한계**: seed 1개, val 443장 → ±0.01 이하 차이는 믿지 말 것. 지연은 GPU 기준이고, 노트북 CPU에서는 fusion이 훨씬 더 느려진다(⏳ 6장).

### 5.3 최종 학습: 5클래스 전용 모델 (진행 중, 2026-10-03~04)

요청: "COCO에 이미 있는 클래스 말고 우리가 말한 클래스 위주로", "100 epoch, 최대한 정확하게".

| 항목 | 설정 |
|---|---|
| 클래스 | scooter, stairs, obstacle, other_vehicle, traffic_light (나머지 5개 라벨은 제거) |
| train | scooter·stairs·traffic_light가 든 train 사진 **전부** = 8,120장 (obstacle 7,556장·other_vehicle 1,778장에도 자동 포함) |
| val | 클래스당 사진 약 100장 = 267장 (best epoch 선택용) |
| test | **전체 test 13,721장** (학습에 한 번도 안 쓴 데이터로 최종 점수) |
| 학습 | 100 epoch, patience 30, imgsz 640, seed 0 |
| 순서 | E0 → A5(C-RADIOv3 fusion). distill(B5)은 1차 pilot에서 효과가 없어 제외 |
| 명령 | [`insight_fm/tools/run_final5.sh`](insight_fm/tools/run_final5.sh) (설치 → 데이터 → 5클래스 빌드 → E0 → A5) |
| 결과 위치 | 서버 `~/repo/insight_fm/runs/final/`, Drive `gdrive:sideguide/runs_final5/` |

| ID | mask mAP50-95 (test) | mAP50 | scooter | stairs | obstacle | other_vehicle | traffic_light |
|---|---|---|---|---|---|---|---|
| E0 | ⏳ | | | | | | |
| A5 | ⏳ | | | | | | |

⏳ 결과가 나오면: 학습 곡선(과적합 여부), 혼동 행렬, 크기별·잘림별 recall(3장의 원인이 실제로 성능을 깎는지), stairs 오검출 사례를 추가한다.

## 6. 추론 속도 ⏳

`insight_fm/bench_speed.py`가 학습이 모두 끝난 뒤(GPU가 빈 상태) 자동으로 잰다(서버 `~/after_final.sh`). 결과는 `results/speed/speed.md`, Drive `gdrive:sideguide/runs_final5/results/speed/`.

측정 항목: GPU FP16/FP32 bs1 지연과 E0 대비 배수, GPU bs8 처리량, 실제 test 이미지 end-to-end(전처리+추론+NMS·마스크 후처리), A5에서 C-RADIO가 차지하는 시간·비율, CPU 4/8스레드(노트북 대용, 실제 노트북과 다를 수 있음), 파라미터·GFLOPs.

| 모델 | GPU FP16 (ms) | E0 대비 | end-to-end (ms / FPS) | CPU 4스레드 (ms) | FM 비중 |
|---|---|---|---|---|---|
| E0 | ⏳ | | | | – |
| A5 | ⏳ | | | | |

## 7. 재현 방법 (서버)

### 7.1 서버

- **Runyour AI(몬드리안)** GPU 대여. 접속: `ssh -i <발급받은 .pem> -p 22 ubuntu@machine.runyour.ai`. 인스턴스마다 .pem이 새로 나온다.
- 과금: **빌린 동안 분 단위**(A5000 약 900원/시간). GPU를 안 써도, SSH를 끊어도 나간다. **반환해야 멈추고, 반환하면 서버의 데이터는 모두 사라진다**(같은 머신을 다시 빌려도 복구 안 됨). 일시정지 기능은 없다. 크레딧이 떨어지면 자동 반환된다.
- 크레딧: 학교에서 20만 원 충전. 1차 인스턴스 약 15시간(약 1.4만 원) 사용 후 반환. 2차 인스턴스(2026-10-03 21:56~) 사용 중.
- **서버 이미지가 매번 다를 수 있다.** 1차는 conda(Python 3.12) 있음, 2차는 순수 Ubuntu 22.04(Python 3.10, pip 없음) → `python3 -m venv ~/venv` 후 설치. 둘 다 Docker 컨테이너이고 `sudo` 비밀번호 없음.
- GPU 선택 참고: L20(48GB) > A10×2 ≈ A5000(24GB) > V100(CUDA 13 미지원, 비추천).

### 7.2 처음부터 돌리기

```bash
# 이 PC(WSL)에서: rclone 설정을 서버로 (Drive 접근용, 한 번만)
ssh -i a5000.pem ubuntu@machine.runyour.ai "mkdir -p ~/.config/rclone"
scp -i a5000.pem ~/.config/rclone/rclone.conf ubuntu@machine.runyour.ai:~/.config/rclone/

# 서버에서: 반드시 tmux 안에서 (오래 걸리는 SSH 명령은 서버가 끊는다)
git clone -b detection https://github.com/sangjune2001/A-Proactive-Egocentric-Streaming-Video-Assistant-for-Blind-Navigation.git ~/repo
tmux new -s exp
cd ~/repo/insight_fm
read -s -p "HF token: " HF_TOKEN && export HF_TOKEN; echo     # DINOv3만 필요. history에 안 남음
bash setup.sh            # deps → data(약 1시간: zip 15개 받고 640px 추출, zip은 지움) → hf → prefetch → selftest → build
python run_all.py --phase pilot          # 1차 pilot 재현
# 5클래스 최종 학습
CLASSES=scooter,stairs,obstacle,other_vehicle,traffic_light YOLO_OUT=~/sg/yolo5 STEP=build bash setup.sh
SG_YOLO=~/sg/yolo5 python run_all.py --phase final --ids E0
SG_YOLO=~/sg/yolo5 python run_all.py --phase final --ids A5
python bench_speed.py --runs runs/final --images ~/sg/yolo5/full/images/test --out results/speed
```

- `RCLONE_REMOTE_RUNS=gdrive:...`를 export하면 epoch마다 결과가 Drive로 백업된다(반환 대비 필수).
- 끊겨도 같은 명령을 다시 치면 끝난 실험은 건너뛰고, 중단된 실험은 `last.pt`에서 이어서 학습한다.
- `touch results/HOLD_<phase>`로 해당 단계 직전에 멈출 수 있다.
- 데이터 분석 재생성: `python analyze_data.py --jsonl ~/sg/labels_all.jsonl --out analysis`, `python analyze_objects.py --jsonl ~/sg/labels_all.jsonl --images ~/sg/imgs --out analysis/objects`

### 7.3 파일 지도 (`insight_fm/`)

| 파일 | 역할 |
|---|---|
| `setup.sh` | 서버 세팅 전체. `STEP=<deps|data|hf|prefetch|selftest|build>`로 단계별 실행 |
| `extract_images.py` | Drive zip을 하나씩 받아 라벨된 프레임만 640px로 추출, zip 삭제(이어하기 가능) |
| `build_dataset.py` | jsonl → YOLO 폴더. 영상 단위 분할, pilot 샘플링(`--pilot-per-class`, `--pilot-all-classes`), 클래스 부분집합(`--classes`) |
| `encoders.py` | FM 인코더 로드(HF / torch hub), feature를 입력의 1/16 격자로 맞춤 |
| `fm_yolo.py` | fusion·distill 모델과 trainer (Ultralytics 확장) |
| `train_one.py` | 실험 1개 학습 → 평가 → 지연 측정 → `done.json` |
| `run_all.py` | 단계(sanity/pilot/tier2/full/final) 자동 실행, 결과표 |
| `export_plain.py` | distill 모델 → 순수 YOLO11s-seg `.pt` (배포용) |
| `bench_speed.py` | 추론 속도 종합 측정 |
| `analyze_data.py` | 개수·분할·크기 분석 (라벨만 사용) |
| `analyze_objects.py` | 실제 이미지에서 객체별 크기·잘림·선명도·밝기 + 예시 이미지 |
| `selftest.py` | 합성 데이터로 전체 흐름 검증 (CPU 가능) |

## 8. 그동안 겪은 문제와 해결 (같은 실수 방지용)

| 증상 | 원인 | 해결 |
|---|---|---|
| 이미지 추출이 0장 (`missing/bad 14826`) | 서버에 `libGL.so.1` 없음 → OpenCV import 실패를 코드가 조용히 삼킴 | `setup.sh`에서 `libgl1 libglib2.0-0` 설치, 추출 코드가 import 실패 시 즉시 멈추고 실패한 zip은 지우지 않게 수정 |
| S1.zip을 못 찾음 | 모든 zip을 `polygon/`에서 찾음 | jsonl의 `z` 경로(MyDrive 뒤)로 zip마다 실제 Drive 경로 사용 |
| obstacle/stairs 결과가 뒤바뀜 | 코드의 클래스 이름이 7=stairs, 8=obstacle로 잘못 적혀 있었음 | `seg10` 모델로 정답(7=obstacle, 8=stairs) 확인 후 수정. 1차 pilot E0는 수정 46초 전에 시작해 `done.json` 값을 사후 교정(`done.json.bak`에 원본) |
| HF 로그인 실패 | 폐기된 토큰 입력 | 토큰은 `read -s`로 입력. 로컬 스크립트로 유효성 검사 후 전송 |
| 긴 명령 중 `Connection closed by remote host` | 서버가 오래 걸리는 SSH 세션을 끊음 | 오래 걸리는 작업은 전부 tmux 안에서 |
| 터미널에 붙여넣은 명령이 깨짐 | 화면 폭에서 줄바꿈된 채로 복사됨 | 긴 명령은 스크립트 파일로 만들어 실행 |
| 밤사이 감시가 멈춤 | Windows Modern Standby(화면 꺼지면 절전) | 전원 연결 시 화면 끄기·절전 "안 함" |
| 2차 서버에서 `pip: command not found` | conda 없는 이미지 | venv 사용 (`insight_fm/tools/run_final5.sh` 참고) |

## 9. 결론: 무엇이 되고, 안 되고, 왜

| | 판단 | 근거 |
|---|---|---|
| ✅ FM fusion | 성능 확실히 오름 | 1차 pilot 7/7개 +0.03~+0.07 |
| ⚠️ FM fusion 실사용 | 속도가 관건 | GPU에서 약 2배, CPU에서는 더 느림(⏳ 실측) |
| ❌ FM distill | 1차에서 효과 없음 | 8/8개 ±0.008. 최종 학습에서는 제외 |
| ❌ traffic_light | 구조적으로 어려움 | 89%가 16px 미만. 입력 해상도를 올리거나(960~1280) 타일/크롭 추론이 필요 |
| ❌ stairs | 라벨이 틀림 | 원본 라벨의 상당수가 맨홀·보호판·연석. 재라벨링 필요 |
| ⚠️ scooter | 데이터 절대량 부족 | 사진 224장. 추가 수집 또는 외부 데이터 필요 |
| ⚠️ obstacle | 클래스가 너무 넓음 | 15종 혼합, 가는 기둥형 객체 다수 |
| ✅ 흐림·조명 | 문제 아님 | 흐린 객체 0.2%, 어두운 프레임 0.4% |

## 10. 다음에 할 일 (우선순위)

1. ⏳ 최종 결과(E0/A5) 확인, 5.3·6장 채우기.
2. **stairs 재라벨링**(372장, 몇 시간): 계단만 남기고, Polygon 영상 중 계단이 보이는 프레임도 추가 라벨. 이게 안 되면 stairs 수치는 의미가 없다.
3. **노트북(그램) CPU에서 E0 / A5 FPS 실측.** fusion을 쓸 수 있는지는 이 숫자로 결정된다.
4. traffic_light: imgsz 960 또는 1280으로 E0 재학습해 비교(작은 객체 개선 여부).
5. obstacle 세분화 검토(기둥형 / 낮은 장애물 / 큰 구조물).
6. 필요하면 traffic_sign(38,918개, 현재 미사용) 추가 검토.

## 11. 계정·자격 증명 (값은 적지 않음)

- **Hugging Face 토큰**: DINOv3(게이트 모델)에만 필요. 대화 중에 한 번 노출된 토큰은 폐기하고 새로 발급함.
- **GitHub**: 로컬 PC에 `git config --global credential.helper store`로 저장(평문 `~/.git-credentials`).
- **Google Drive**: 로컬 PC의 `~/.config/rclone/rclone.conf`(remote 이름 `gdrive:`)를 서버로 복사해서 사용.
- **서버 .pem**: 인스턴스마다 새로 발급. 로컬에서는 `~/.ssh/a5000.pem`(권한 600)으로 복사해 사용.
