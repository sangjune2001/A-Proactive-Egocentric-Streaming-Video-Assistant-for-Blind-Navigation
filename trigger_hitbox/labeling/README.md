# 움직임 · 말하기 라벨링 (10/16 ~ )

히트박스 트리거 v5_29의 판정을 그대로 써서, 영상의 **프레임마다 · 물체(트랙)마다** 두 가지 라벨을 자동으로 만든다.
규칙으로 만든 초벌 라벨이라, 정답(GT)으로 쓰려면 사람이 검토해야 한다.

| 라벨 | 값 | 기준 (v5_29 변수) |
|---|---|---|
| **motion** | `approaching` 다가옴 · `receding` 멀어짐 · `static` 가만히 있음 · `crossing` 옆으로 지나감 | 트리거의 움직임 상태 그대로 (물체 자체 속도 = 바닥 거리 기울기 + 내 걸음, 0.2 s 다지기) |
| **speak** | `must` 말해야 됨 | 히트박스 판정이 주의·위험 (`lvB != safe`) — 트리거가 실제로 경고하는 단계 |
| | `may` 말해도 됨 | 걸러지지 않았고 내 진행 영역 근처(영역 안 · 영역에 걸침 · 3 s 안 진입 예측 · 영역 폭 +1 m / 길이 1.5배 안)지만 경고 단계는 아님 |
| | `must_not` 말하면 안 됨 | 차도 위 · 인도 끝선 너머 · 경로 밖(8°) · 멀어짐/위쪽 화살표 · 빨간불 대기 중 차 · 붐빔 속 사람 · 다가오지 않는 사람 · 지나가지 않는 버스 · 저신뢰인데 영역에 안 걸침 · 그 밖 먼 물체 |

## 파일

| 파일 | 내용 |
|---|---|
| [`make_label_pass2.py`](make_label_pass2.py) | `pass2_trigger_v5_29.py` → `pass2_label.py` 생성 (판정은 그대로, `labels_frames.jsonl` 출력만 추가) |
| [`JJY_보행영상_움직임말하기_라벨링_v5_29.ipynb`](JJY_보행영상_움직임말하기_라벨링_v5_29.ipynb) | Colab(T4) 노트북: 드라이브 `Video Dataset/` 영상 → pass1 전부 → pass2_label → `label/` |
| [`stream159.py`](stream159.py) · [`stream189.py`](stream189.py) | AI Hub 원본 zip을 디스크에 안 남기고 구글 드라이브로 바로 올리는 스크립트 (rclone rcat, 끊기면 이어받기) |

출력 `labels_frames.jsonl` 한 줄 = 한 프레임:
```json
{"f": 1, "t": 0.0, "surface": 1, "road": "", "walk_mps": 1.2,
 "objs": [{"tid": 3, "cls": "obstacle", "box": [0, 72.9, 204.7, 1018], "motion": "static",
           "speak": "must_not", "level": "safe", "dist_m": 44.37, "X": -13.36}]}
```
같은 폴더에 `events_runtime.json`(실제로 말하는 문장 단위 이벤트), `summary.json`, pass1 결과도 함께 저장한다.

## 데이터 (구글 드라이브, 공유 계정)

```
Video Dataset/
├─ train/1.mp4 … 20.mp4          우리 학습 영상 (카톡 1-7 · 8-14 · 15-20.zip)
├─ test/test_1.mp4 … test_3.mp4   테스트 영상 (190 cm 촬영자 → CAM_H 1.77)
└─ AIHub_159_1인칭보행영상/        BBOX, 평균 키(Average_stature · Average_staure_2) 실외(out)만 — 원천 26 + 라벨 26, 364.5 GB
label/                            ← 라벨 결과 (Video Dataset 경로 그대로: train/1.mp4 → label/train/1/)
JJY/                              ← Colab 노트북 + label_code/(코드·가중치)
```

- **AI Hub 159 원천은 동영상이 아니다.** 영상마다 30프레임(1 s) 간격 PNG(1920×1080) 30–50장 + 장면별 박스 JSON(트랙 ID 없음).
  광류·ByteTrack·걸음 속도가 필요한 v5_29 파이프라인은 그대로 못 돌려서, 159는 별도 방식(1 s 전후 박스 크기·깊이 비교)이 필요하다 — 미정.
- 업로드: 집 PC 업로드 0.26 MB/s라 불가 → AI Hub는 해외 IP 차단(도쿄 서버 실패) → **네이버 클라우드 서울 서버**에서 3줄 동시 ~30 MB/s, 약 3시간.

## 실행 (Colab)

1. 드라이브 `JJY/JJY_보행영상_움직임말하기_라벨링_v5_29.ipynb`를 Colab으로 열기 (계정: 데이터 드라이브)
2. 런타임 유형 **T4 GPU** → **모두 실행**. 끊기면 다시 모두 실행 — `label/.../done.json` 있는 영상은 건너뜀

로컬 확인 (test_1, 바닥 분할 없이): must 784 · may 337 · must_not 13,654 / static 13,331 · receding 840 · approaching 374 · crossing 230 (물체·프레임 수).

## 진행 기록

| 날짜 | 내용 |
|---|---|
| 10/15 | 공유 드라이브에 `Video Dataset/` 생성, 학습 20 + 테스트 3 업로드. 189(인도보행)로 잘못 시작했다가 159(1인칭 보행)로 변경 |
| 10/16 | 159 평균 키 실외 BBOX 52개(364.5 GB) 업로드 완료 (네이버 클라우드 서울). 라벨 규칙 · Colab 노트북 작성, 159 원천이 1 s 간격 PNG임을 확인 |
