# 트리거 → VLM → TTS: 코드 전체 공개 논문만으로 짠 비교 파이프라인

작성 2026-10-06 · 담당 박상준 · 관련: [`05_트리거_비교실험_설계.md`](05_트리거_비교실험_설계.md) (실험 ②를 이 문서가 대체), [`03_MVP_8주차_계획.md`](03_MVP_8주차_계획.md)

> 확인: 2026-10-06에 공식 GitHub README · Hugging Face 모델 페이지 · 학회 페이지에서 확인. 본문 수치를 직접 보지 않은 항목은 **(확인 필요)**.
> 지연 수치는 모두 **논문이 보고한 값**이다. 우리 A5000 수치는 09 문서의 실험으로 새로 잰다.
> 원칙: 각 비교군은 **원 논문 공식 코드·가중치를 그대로** 쓴다. 논문 부품을 섞어 만든 변형은 비교군에 넣지 않는다.

---

## 0. 한 장 요약

- **앞단은 모든 비교군에서 같다:** YOLO-seg (박주영 `.pt`) + ByteTrack + 규칙 트리거 → `event{t, track_id, cls, bbox, mask}`.
- **VLM 단계만 갈아 끼운다.** 비교군은 코드가 전부 공개된 2025–26 탑티어 논문 5편 + 기준선 1개 + end-to-end 참고군 1개.

| 비교군 | VLM 단계 | 논문 · 학회 | 무엇을 보려고 |
|---|---|---|---|
| **P0** | 트리거 때 최근 프레임으로 VLM 호출. **모델(S-M)·프레임 수(S-F)는 논문 근거로 후보를 정해 비교 실험으로 결정** ([`11`](11_P0_설계근거.md)) | (기준선) | 가장 단순한 구조의 성능 |
| **P1** | P0 + FastVID 토큰 가지치기 | FastVID · NeurIPS'25 | **같은 모델**에서 토큰을 줄이면 얼마나 빨라지고 얼마나 틀리나 |
| **P2** | TimeChat-Online-7B (DTD) | TimeChat-Online · ACM MM'25 | 시간 중복 제거 모델이 보행 영상에서도 통하나 |
| **P3** | StreamingVLM, 영상을 계속 넣어 KV 유지 → 트리거 때 질문만 | StreamingVLM · ICLR'26 | 트리거 전에 미리 계산해 두면 첫 응답이 얼마나 빨라지나 |
| **P4** | VideoRefer-VideoLLaMA3 (seg 마스크로 대상 지정) | VideoRefer · CVPR'25 | 대상 영역을 직접 받는 모델이 설명 정확도에서 이득인가 |
| P5 (참고) | MMDuet2 — 트리거 없이 모델이 말할 때를 결정 | MMDuet2 · ICLR'26 | 우리 분리형(트리거 + VLM)이 end-to-end보다 나은가 |

**MVP(8주차)는 P0 하나로 끝까지 연결한다** (비동기 경고 + TTS). P1–P5는 같은 인터페이스에 꽂아 8–9주차에 비교한다.

---

## 1. 코드 전체 공개 기준으로 거르기

**기준:** 공식 GitHub에 방법 재현에 필요한 코드가 전부 있고, 학습된 모델을 쓰는 방법이면 **가중치까지** 공개. 학습이 필요 없는 방법은 방법 코드 전부가 공개되면 통과.

### 1-1. 통과

| 논문 | 학회 | 공개 범위 (README 확인) | 본체 | 우리 쓰임 |
|---|---|---|---|---|
| **StreamingVLM** | ICLR'26 | 학습(SFT 1·2단계) · 추론 · 데이터 · 가중치 (`mit-han-lab/StmingVLM`, Qwen2.5-VL 계열 8B, BF16) | Qwen2.5-VL | **P3** |
| **TimeChat-Online** | ACM MM'25 | 학습 · 추론 · DTD 모듈 · 가중치 (`wyccccc/TimeChatOnline-7B`) | Qwen2.5-VL-7B | **P2** |
| **FastVID** | NeurIPS'25 | 학습 불필요, 모델별 구현 공개 (`fastvid_qwen25vl` 포함) | LLaVA-OV · LLaVA-Video · **Qwen2.5-VL** | **P1** |
| **VideoRefer Suite** | CVPR'25 | 코드 · 가중치 (VideoLLaMA3-2B/7B) · VideoRefer-700K · 추론 노트북 (단일·다중 객체) | VideoLLaMA3 | **P4** |
| **MMDuet2** | ICLR'26 | 학습(SFT · RL) · 추론 · 가중치 · 데이터 | Qwen2.5-VL-3B | **P5** |
| LiveCC | CVPR'25 | 학습 · 추론 · 가중치 · 데이터 | Qwen2-VL-7B | 쓰지 않음: 실시간 **해설** 모델이라 경고와 목적이 다름 |
| StreamForest | NeurIPS'25 Spotlight | 학습 5단계 · 추론 · 가중치(주행 특화 포함) · 데이터 | Qwen2-7B + SigLIP | 예비: 긴 기억이 핵심이라 우리에겐 우선순위 낮음 |
| StreamingTOM | CVPR'26 | 학습 불필요, 코드 공개 | **LLaVA-OV-7B만** | 예비: 본체가 달라 P0와 바로 비교 불가 (예비 실험) |
| DyCoke | CVPR'25 | 학습 불필요, 코드 공개 | **LLaVA-OV만** (Qwen2.5-VL은 외부 구현) | 예비: StreamingTOM과 같은 이유 |

### 1-2. 탈락

| 논문 | 학회 | 탈락 이유 |
|---|---|---|
| InfiniPot-V | NeurIPS'25 | README에 "연구용 재구현이며 **모든 구성 요소를 포함하지 않음**" 명시 |
| Dispider | CVPR'25 | 학습 코드 의도적 미포함 (추론·가중치만) |
| StreamMind | ICCV'25 | 가중치 공개 여부 불분명 |
| LION-FS | CVPR'25 | 가중치 공개 여부 불분명 |
| StreamBridge | NeurIPS'25 | 가중치 미공개 |
| WalkVLM | ICCV'25 | 원 버전은 보관 경로(WalkVLM-LR)에만 있고, 후속 WalkStream의 저지연 VLM은 "coming soon" — 구조 참고만 |
| CROP | EMNLP'25 | 코드 공개 확인 못 함. 아이디어(영역만 넣기)만 P0 입력 설계에 참고 |

> ACM MM은 CCF-A이지만 CVPR · ICCV · ECCV · NeurIPS · ICLR와 같은 급은 아니므로 발표 때 구분해서 적는다.

---

## 2. 파이프라인 구조

```
 ┌──────────────────────── 공통 앞단 (고정) ────────────────────────┐
 영상 ─▶ YOLO-seg + ByteTrack ─▶ 규칙 트리거 ─▶ event{t, id, cls, bbox, mask}
 └───────────────────────────────────────────────────────────────────┘
                     │                                   │
          (즉시)     ▼                                   ▼ (비동기)
      경고 템플릿 wav 재생                   ┌── VLM 단계: describe() ──┐
      "정면에 자전거"                        │  P0 (S-M 승자)           │
                                            │  P1 + FastVID             │
                                            │  P2 TimeChat-Online-7B    │
                                            │  P3 StreamingVLM (상시)   │
                                            │  P4 VideoRefer (마스크)   │
                                            └─────────────┬────────────┘
                                                          ▼
                                     JSON {target, direction, action}
                                                          ▼
                                   한국어 템플릿 조립 ─▶ TTS ─▶ 음성
```

### 2-1. 갈아 끼우는 경계 (한 함수)

```python
# 모든 비교군이 이 모양만 맞춘다
def describe(event: dict, frames: list[np.ndarray], masks: list[np.ndarray] | None) -> dict:
    """event: {t, track_id, cls, bbox}, frames: 트리거 시각까지 F1/F3/F8/F32 (docs/11 §2)
       → {"target": str, "direction": "left|front|right", "action": str, "latency": {...}}"""
```

- **P0–P2, P4**는 트리거 때 `describe()`가 한 번 불린다.
- **P3**는 영상 프레임을 계속 받아 KV를 유지하다가 `describe()`가 불리면 질문 텍스트만 붙여 디코딩한다 (StreamingVLM 공식 추론 루프 사용).
- **P5**는 `describe()`를 쓰지 않고 영상 전체를 받아 스스로 말한다. 트리거까지 포함한 전체 시스템과 비교한다.

### 2-2. 모든 비교군에 같게 맞추는 것

| 항목 | 고정값 | 이유 |
|---|---|---|
| 프레임 창 | S-F 결과 (F1 · F3 · F8 · F32 중 하나, docs/11 §2). P3만 영상 전체를 계속 받음 | 같은 정보량 |
| 대상 지정 | 이미지 모델(P0–P3): **프레임에 bbox를 그리고 좌표를 텍스트로 함께 줌**. P4: **seg 마스크** (모델 고유 방식) | 각 모델이 받을 수 있는 가장 자연스러운 방식 |
| 프롬프트 | 같은 영어 프롬프트, 출력은 **영어 JSON 3칸** | 한국어 생성 능력 차이를 빼고 "무엇을 봤나"만 비교 |
| 한국어 문장 | JSON → 한국어 템플릿으로 조립 (모든 비교군 같은 코드) | 문법 오류 0, TTS 입력 일정 |
| `max_tokens` | 48 | 디코딩 시간 상한 |
| 정밀도 | bf16, A5000 1장 | |

---

## 3. 비교군별 상세

### P0. 기준선 — 이벤트 호출 (모델 · 프레임은 [`11`](11_P0_설계근거.md)의 S-M · S-F로 결정)
- **동작:** 트리거 → 프레임 창(S-F) + 대상 bbox 표시 → vLLM 서버(OpenAI 호환 API, 로컬) → JSON 4칸. 코드 `code/p0/`.
- **코드:** Qwen 공식 가중치 + vLLM. 고정 시스템 프롬프트는 prefix caching으로 재사용.
- **MVP 구성이 이것이다.** S0 배관 점검은 가장 빠르고 vLLM 지원이 확실한 Qwen2.5-VL-7B로 하되, 채택 모델은 S-M 결과로 정한다.

### P1. FastVID (NeurIPS'25)
- **동작:** P0와 같은 입력·같은 모델에, 공식 `fastvid_qwen25vl` 구현으로 시각 토큰을 가지치기한다 (영상을 시간 순 구간으로 나누고 구간 안에서 밀도 기반으로 토큰을 남김).
- **보고값:** LLaVA-OV-7B에서 토큰 90.3% 제거, prefill 7.1배 빠름, 정확도 98.0% 유지.
- **볼 것:** 본체가 같아서 **가지치기 효과만 순수하게** 분리된다. bbox 박스를 그린 영역이 가지치기로 사라지지 않는지 사례 확인.
- **주의:** FastVID는 vLLM이 아닌 HF transformers 경로로 돈다 → P0도 HF 경로로 한 번 더 재서 비교한다.

### P2. TimeChat-Online (ACM MM'25)
- **동작:** 공식 7B 가중치 + DTD. 이웃 프레임의 같은 위치 패치가 비슷하면 버린다.
- **보고값:** 토큰 82.8% 제거, 정확도 98% 이상, 응답 1.76배 빠름. 학습 없이 Qwen2.5-VL에 DTD만 붙여도 VideoMME +5.7점.
- **볼 것:** 걷는 카메라는 배경이 계속 움직여서 **제거율이 논문(대부분 고정·완만한 영상)보다 낮을 수 있다.** 우리 영상에서의 실제 제거율이 핵심 결과.

### P3. StreamingVLM (ICLR'26)
- **동작:** 영상을 계속 넣으며 attention sink + 최근 시각 토큰 창 + 최근 텍스트 창으로 KV를 일정 크기로 유지. 트리거 때 질문만 넣고 디코딩.
- **보고값:** H100 1장에서 최대 8 FPS 실시간.
- **볼 것:** 트리거 → 첫 토큰(TTFT)이 P0보다 얼마나 줄어드는가 vs **상시 GPU 점유**(트리거가 없어도 계속 돎). A5000에서 2 fps 입력을 따라가는지 먼저 확인한다.

### P4. VideoRefer (CVPR'25)
- **동작:** 트리거 대상의 seg 마스크(우리 YOLO-seg 출력)를 영역 입력으로 주고 그 객체를 설명. 단일·다중 프레임 모드 모두 지원.
- **코드:** 공식 추론 노트북 (`videorefer_videollama3-infer-video` 등), VideoLLaMA3-7B / 2B.
- **볼 것:** 여러 객체가 섞인 장면에서 **대상을 헷갈리지 않는가** (P0의 박스 표시 방식 대비). 2B로 속도 이득도 본다.
- **확인 필요:** 영역 입력 형식(마스크 / 박스)을 노트북에서 확인. 논문은 마스크 기반.

### P5. MMDuet2 (ICLR'26) — 참고군
- **동작:** 트리거 없이 영상 전체를 넣고, 모델이 매 순간 응답 / "NO REPLY"를 결정.
- **볼 것:** 우리 시스템(트리거 + 최선의 VLM 단계)과 경고 시점 정확도(PDR · 오탐/분) · 문장 정확도 · GPU 사용량을 비교. 분리형이 연산량 면에서 이득이라는 주장의 근거가 된다 (교수님 "연산량" 지적에 대한 답).

---

## 4. 모델 · TTS

**VLM 본체:** 근거 없이 고르지 않는다. VIABench Table 4(우리와 같은 조건)에서 InternVL3.5-8B 43.9 > Qwen2.5-VL-7B 34.9이지만 지연은 15.5 s 대 4.6 s다. 그래서 후보 5개(InternVL3.5-8B · 4B, Qwen2.5-VL-7B · 3B, Qwen3-VL-8B)를 S-M으로 비교해 정한다 ([`11`](11_P0_설계근거.md)). P1–P3 효율화 공식 코드는 Qwen2.5-VL 기준이므로, P0 승자가 InternVL이면 "P0(InternVL) · P0(Qwen) · P1–P3(Qwen)"으로 보고한다.

**서빙:** vLLM OpenAI 호환 서버. 클라이언트는 이미지를 base64로 넣어 HTTP 요청만 보내므로, 이후 안경 → 서버 구조나 상용 API로 바꿔도 클라이언트 코드는 같다.
> 상용 API 참고: Gemini Live API는 영상을 1 FPS로 처리한다 (공식 문서). 위험 감지는 우리 트리거가 하고 API는 설명용으로만 쓴다.

**TTS**

| ID | 방법 | 한국어 | 실행 | 라이선스 | 근거 |
|---|---|---|---|---|---|
| **T1** | 경고 템플릿 **미리 합성한 wav** 재생 | — | 재생만 | — | 지연 ≈ 0. 클래스 × 방향 조합 수십 개 |
| **T2** | **Supertonic-3** (~99M) | 지원 | CPU (ONNX), 라즈베리파이 RTF 0.3 | 코드 MIT / 모델 OpenRAIL-M | 공식 README. 온디바이스 경량화(8–11주차)와 맞음 |
| T3 | CosyVoice 2 / Fun-CosyVoice 3 | 지원 | GPU, vLLM · TensorRT 가속 | Apache-2.0 | 스트리밍 첫 소리 150 ms (README), ICASSP'25 |

**MVP:** 경고 T1 + VLM 문장 T2. 부족하면 T3.

---

## 5. 비교 실험 설계

**평가 데이터 · 라벨 · 지표 · 실험 단계는 [`09_평가데이터_지표_실험설계.md`](09_평가데이터_지표_실험설계.md)로 옮겼다.** 요약:

- 경고마다 `대상 · 방향 · 움직임 · 행동` 네 칸 라벨 → VLM도 같은 네 칸 JSON으로 답함 → 사람 채점 없이 칸별 정답률을 스크립트로 계산
- 대표 지표 **적시 정답률(TCR)**: 네 칸이 모두 맞는 설명이 정답 구간이 끝나기 전에 소리로 나온 비율
- VLM 없는 템플릿 기준선 **PT**를 추가해 VLM의 이득을 숫자로 보임. VLM을 트리거 오탐 검증기로 쓰는 실험(S6) 포함
- 통계: 같은 경고에 대한 짝 비교(McNemar · Wilcoxon), 영상 단위 부트스트랩, Holm 보정

## 6. 일정 — 이번 주 (10/6–10/11)

날짜별 계획은 [`10_공개벤치마크_평가지표.md`](10_공개벤치마크_평가지표.md) §6에 있다. 요약: 10/7 P0 서버 · 10/8 채점 코드 + S0 · 10/9 모델 고르기(OVO-Bench RT만, API 키 없음) + S1 · 10/10 S2 · S-T · S3 · 10/11 S4 · S5 · 다음 주 S6 · S7. 10/15 통합 동결(MVP = P0).

---

## 부록. 2026-10-06 수업 피드백 · 팀 분담

- 교수님: 트리거 과도한 고도화 불필요 · MVP는 완벽하지 않아도 됨 · **전체 파이프라인 구상** · **다음 주 음성 필수** · 녹화 영상 시연 가능(실시간은 욕심내지 말 것) · 일반화 성능이 낮아도 파이프라인 먼저 · 알고리즘이 정교해질수록 연산량 증가 유의 · 접근 객체 경로 예측 질문 → 현재 트래킹 ID별 bbox 크기 증가로 판단.
- 분담: 박주영 = 킥보드·신호등 추가 학습 `.pt` (MVP는 기존 `.pt`) · 임태규 = 트리거 (최종 규칙 코드 추후 전달) · 박상준 = 현재 규칙 코드로 VLM + TTS 연결, 교체 가능한 인터페이스, 이 문서의 비교 실험.
- 촬영: 각자 2분 × 8–10 + 5분 × 2 (10/7–8). 어두운 시간·과밀 피함. 라벨 = `영상 / 구간 / 설명`. 경고하면 안 되는 상황은 라벨 없이 영상에만 담음.

---

## 출처

**비교군**
- StreamingVLM (ICLR'26): https://github.com/mit-han-lab/streaming-vlm · https://huggingface.co/mit-han-lab/StreamingVLM · https://arxiv.org/abs/2510.09608
- TimeChat-Online (ACM MM'25): https://github.com/yaolinli/TimeChat-Online · https://huggingface.co/wyccccc/TimeChatOnline-7B
- FastVID (NeurIPS'25): https://github.com/LunarShen/FastVID · https://arxiv.org/abs/2503.11187
- VideoRefer Suite (CVPR'25): https://github.com/DAMO-NLP-SG/VideoRefer
- MMDuet2 (ICLR'26): https://github.com/yellow-binary-tree/mmduet2

**예비 · 탈락**
- LiveCC (CVPR'25): https://github.com/showlab/livecc
- StreamForest (NeurIPS'25): https://github.com/MCG-NJU/StreamForest
- StreamingTOM (CVPR'26): https://github.com/YIGE24/StreamingTOM
- DyCoke (CVPR'25): https://github.com/KD-TAO/DyCoke
- InfiniPot-V (NeurIPS'25): https://github.com/aiha-lab/InfiniPot-V
- Dispider (CVPR'25): https://github.com/Mark12Ding/Dispider
- StreamMind (ICCV'25): https://github.com/xinding-sys/StreamMind
- LION-FS (CVPR'25): https://github.com/JiuTian-VL/LION-FS
- WalkVLM (ICCV'25): https://openaccess.thecvf.com/content/ICCV2025/html/Yuan_WalkVLM_Aid_Visually_Impaired_People_Walking_by_Vision_Language_Model_ICCV_2025_paper.html · https://github.com/xiaoyuan1996/walkvlm
- CROP (EMNLP'25): https://aclanthology.org/2025.emnlp-main.492.pdf

**모델 · 서빙 · TTS**
- Qwen3-VL 기술 보고서: https://arxiv.org/abs/2511.21631
- Supertonic: https://github.com/supertone-inc/supertonic
- CosyVoice: https://github.com/FunAudioLLM/CosyVoice
- Gemini Live API: https://ai.google.dev/gemini-api/docs/live-api/capabilities
