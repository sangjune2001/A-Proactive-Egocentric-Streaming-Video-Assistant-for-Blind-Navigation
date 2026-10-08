# VLM 평가 데이터 — 새로 라벨링하지 않고 쓰는 방법

작성 2026-10-07 · 담당 박상준 · 관련: [`12`](12_VLM비교실험_결과_평가계획.md), [`13`](13_다음단계_파이프라인.md), [`10`](10_공개벤치마크_평가지표.md)

## 0. 결론

1. **하이브리드 구조에선 VLM이 대상 · 움직임 2칸만 맡는다** (방향 = bbox, 행동 = 규칙표) → VLM 평가에 필요한 정답도 **2칸**.
2. **VIABench 정답 문장은 정형화돼 있어** 2칸을 자동 추출할 수 있다 → 사람은 **표본 약 150개 검수 (약 1시간)** 만.
3. "계단만" 같은 부분집합은 **VIABench 과제 이름으로 바로 걸러진다.**
4. 보조로 **GuideDog Object** (대상 이름 + 방향 + bbox, 라벨링 0).
5. 신호등 바뀜: 별도 공개 영상 데이터셋은 못 찾았지만 **VIABench 안에 20개** (초록으로 13 · 빨강으로 7, §5) → 우리 영상 + 팀 촬영분과 합쳐 평가.

---

## 1. VIABench 정답 문장 분석 (`benchmarks/viabench_annotations/proactive_reminder.json`, 로컬 확인)

| 과제 | 정답 수 | 구간 길이 중앙값 | 문장 예 | 뽑을 수 있는 칸 |
|---|---|---|---|---|
| 장애물 경고 (Obstacle Alert) | 6,776 | 1.5 s | "Tree bed edge ahead on the right, please be cautious." / "Bus approaching from the right front, please be cautious." | 대상 ✓, 방향 ✓ (left 2,863 · right 2,926 · ahead/front 5,459), **다가옴 628** ✓ |
| 회피 지시 (Active Avoidance) | 1,275 | 2.1 s | "There is a row of electric scooters ahead. Please walk to the right." | 대상 ✓, 방향 ✓, 행동 ✓ |
| 계단 (Stairs Up/Down) | 637 | 1.9 s | "Step up ahead, please lift your feet carefully." / "Steps ahead." | 계단 ✓, **위/아래 명시 약 260** (up 190 · down 68) |
| 계단 (Staircase Up/Down) | 170 | 2.2 s | "There is an upward staircase ahead on the right." | 계단 ✓, 위/아래 일부 (up 16 · down 55) |
| 보행 신호등 (Pedestrian Traffic Light) | 257 | 1.9 s | "The pedestrian traffic light ahead is green." | 빨강 · 초록 ✓, **바뀜 20개** ("The red light ahead is turning green.") ✓ |
| 횡단보도 (Crosswalk) | 41 | 2.5 s | "You are at the right edge of the crosswalk. Please walk to the left." | 위치 · 행동 |
| 경사 (Slope Up/Down) | 107 | 2.0 s | "Uphill ahead" | 오르막 / 내리막 |

- 장애물 경고의 대상 명사 상위: pillar · pole · pedestrian · utility pole · stone pillar · tree · metal pole · row of electric scooters · row of parked cars · railing · wall · sign …
- 단순 정규식으로는 대상 추출 25 %뿐 → **LLM 추출이 필요** (아래 §2).

## 2. 최소 라벨링 방법

| 단계 | 방법 | 사람 작업 |
|---|---|---|
| ① 칸 추출 | 정답 문장 → 로컬 텍스트 LLM (Qwen2.5-7B-Instruct, vLLM)에 JSON 스키마 강제: {대상, 움직임, 방향}을 우리 어휘(`config.py`)에서 선택, 판단 불가 = `unknown` → 그 칸은 채점 제외. 약 8,000문장, A5000 약 20분 | 없음 |
| ② 대상 박스 | 정답 시각의 YOLO 트랙 중 **클래스가 맞는 것** 자동 선택. 0개 또는 여러 개면 박스 없이 입력 | 없음 |
| ③ 검수 | 과제별 층화 **약 150개** 사람 확인 → "자동 추출 정확도 N %" 보고 | 약 1시간 |

**주의:** VIABench 방향은 사람이 말한 표현 ("ahead on the right")이라 우리의 bbox 3등분 정의와 다르다 → 방향은 **보조 지표**로만.

## 3. 다른 공개 데이터셋

| 데이터셋 | 형태 | 라벨 | 우리 용도 | 한계 |
|---|---|---|---|---|
| **GuideDog** (ACL'26) | 1인칭 보행 **이미지** 22,084 (사람 검증 gold 2,106) | Object 설정: `answer`(대상) · `answer_direction`(시계 방향) · `answer_bbox` · `answer_depth` | 박스로 지정한 대상 이름 맞히기, **라벨링 0** | 이미지 → 움직임 없음. CC BY-NC 4.0, 승인 후 다운로드 (5.1 GB) |
| **WAD** (WalkVLM, ICCV'25) | 1인칭 보행 **영상 12k** (클립당 키프레임 10장) | 리마인더 6종 (장애물 · 교차로 · 길 막힘 · **다가오는 차량/사람** · 경로 이탈 · 표지), 대상 클래스, 시계 방향, 걸음 수 거리, 위험도. bbox = 검출기 + 사람 검증 | **움직이는 물체(S1)** 보강 — VIABench처럼 문장에서 추출 | 라이선스 표기 없음, 문장 형식 확인 필요 |
| **StairNet** | 가슴 카메라 이미지 51만 | 평지 / 평지→계단 / 계단 / 계단→평지 | 계단 존재 · 진입 시점 | 오르는 계단 위주로 보임 (내려가는 계단 포함 여부 확인 필요), 이미지 |
| **LYTNet** / **PTLD** | 보행 신호등 이미지 5,059 / 4,399 | 빨강 · 초록 · 카운트다운 / go · stop · off | 신호등 색 인식 | 이미지 → **바뀌는 순간 없음** |
| PEDESTRIAN (arXiv'25, CC BY 4.0) | 1인칭 보도 영상 340 | 장애물 29종 | 정지 장애물 대상 | 라벨 단위 (영상 / 프레임) 미확인 |

## 4. 추천 조합

| 평가할 칸 | 데이터 | 사람 작업 |
|---|---|---|
| 대상 + 움직임 (주 평가) | **VIABench 장애물 경고 + 회피 + 계단** (자동 추출) | 검수 150개 |
| 박스 지정 대상 인식 (보조) | **GuideDog Object** | 0 |
| 다가오는 물체 (S1) | VIABench "approaching" 628 + 필요 시 WAD | 0 (검수에 포함) |
| 신호등 바뀜 · 계단 위/아래 | VIABench 바뀜 20 + 계단 위/아래 343 + 종설 영상 + 팀 촬영분 | 팀 촬영분만 |

## 5. 자동 추출 결과 (규칙 기반, 10/7)

`code/viabench_extract.py` → `benchmarks/viabench_fields.jsonl` (정답 9,115개 = 장애물 경고 · 회피 · 계단 · 신호등)

| 과제 | n | 대상 추출 | 움직임 | 방향 | 제외 사유 |
|---|---|---|---|---|---|
| 장애물 경고 | 6,776 | 97 % | 56 % | 80 % | 매핑 실패 179 · 횡단보도 표현 모호 32 |
| 회피 지시 | 1,275 | 95 % | 60 % | 86 % | 매핑 실패 59 |
| 계단 (Stairs) | 637 | 100 % | 41 % | 97 % | — |
| 계단 (Staircase) | 170 | 100 % | 40 % | 97 % | — |
| 보행 신호등 | 257 | 74 % | 65 % | 41 % | 신호등 없음 66 |
| **전체** | **9,115** | **96 %** | **55 %** | **81 %** | 336 |

- **평가에 쓸 수 있는 정답:** 대상 8,779개 (영상 514편) · **대상 + 움직임 5,033개**
- 움직임 분포: 정지 3,861 · **다가옴 632** · 계단 위 215 / 아래 128 · 빨강 84 / 초록 65 · 가로지름 25 · **초록으로 13 / 빨강으로 7** · 멀어짐 3
- 움직임이 비는 이유는 대부분 **정답 문장에 움직임이 없어서** ("Pedestrian ahead on the left") — 추출 실패가 아니라 원 정답의 한계 → 해당 칸은 채점 제외
- 처리 규칙: 여러 물체가 나오면 모두 `target_ok` (하나만 맞혀도 정답) · 영어 한 단어가 우리 어휘 여럿에 걸치면 모두 허용 (pillar → pole/bollard, curb → stairs/other, 전동 이륜차 → scooter/motorcycle/bicycle) · "pedestrian crossing ahead"는 사람/횡단보도 모호 → 제외 · 지시문 ("please walk to the right")의 방향어는 대상 방향에서 제외
- **검수:** `benchmarks/viabench_fields_review150.csv` — 과제별 층화 150개, `ok_target` · `ok_motion` · `ok_direction` 열에 O/X, 틀리면 `fix`에 정답. 검수 결과로 "자동 추출 정확도"를 보고하고, 서버 대여 시 LLM 추출과 비교

## 출처

- GuideDog: https://huggingface.co/datasets/kjunh/GuideDog · https://jun297.github.io/GuideDog/ · https://arxiv.org/pdf/2503.12844
- WalkVLM / WAD: https://openaccess.thecvf.com/content/ICCV2025/html/Yuan_WalkVLM_Aid_Visually_Impaired_People_Walking_by_Vision_Language_Model_ICCV_2025_paper.html · https://arxiv.org/html/2412.20903 · https://walkvlm2024.github.io/
- StairNet: https://arxiv.org/pdf/2310.20666 · https://ieee-dataport.org/documents/stairnet-computer-vision-dataset-stair-recognition
- LYTNet: https://www.researchgate.net/publication/334644485 · Ampel-Pilot: https://github.com/patVlnta/Ampel-Pilot
- PEDESTRIAN: https://arxiv.org/abs/2512.19190
- VIABench: https://arxiv.org/pdf/2607.14660
