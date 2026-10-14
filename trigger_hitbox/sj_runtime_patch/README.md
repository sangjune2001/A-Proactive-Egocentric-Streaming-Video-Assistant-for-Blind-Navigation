# SJ 실행 시스템에 적용한 로컬 패치 (참고용)

`SJ` 브랜치(박상준)의 `code/runtime` 실행 시스템(트리거 → 재생 큐 → TTS)으로 히트박스 트리거의 음성을 냈다.
**SJ 브랜치는 수정·푸시하지 않았다.** 서버에 받은 사본에만 아래 패치를 적용해 실험했다 (기준 커밋 `ea78f8f`).

적용: `cd <SJ 사본> && git apply sj_runtime_v5_9.patch` 후 v5_10 문장 추가: `python after_v5_9/patch_names.py && python after_v5_9/patch_bollard.py` (스크립트 안 경로를 SJ 사본 위치로 바꿔서)

| 파일 | 바꾼 것 |
|---|---|
| `code/common.py` | 테스트 영상 `test_1~3`, 최종 10클래스 가중치 `F10` 등록 |
| `code/runtime/triggers.py` | replay 이벤트의 박스·움직임·행동·시계 방향·묶음·문장(`box, motion, action, clock, items, text`)을 넘김 |
| `code/runtime/dispatcher.py` | 시계 방향 한 문장("1시 방향 자동차가 지나가요. 멈추세요."), 여러 물체 묶음, 횡단보도·줄지은 장애물·주변 사람 안내 문장, 조각 미리 합성 목록 |
| `code/runtime/speech.py` | `synth_concat`: 미리 합성한 조각 wav를 이어 붙여 합성 시간 0 |
| `code/runtime/audio.py` | 재생 우선순위: 멈추세요 > 피하세요 > 위치 안내. 더 급한 경고가 오면 덜 급한 재생을 끊음 (검정 차 "멈추세요"가 앞 안내에 밀려 버려지던 문제) |
| `code/runtime/vlm_worker.py` | 신호 색 · 움직임 · 행동은 트리거 값을 따름 (VLM이 초록불을 '빨간불'이라 하던 문제) |
| `code/p0/ko_template.py` | 신호등 문장에서 방향 제거, 주의는 행동 안내 없음, (v5_10) '벽' 추가 |
| `after_v5_9/` | v5_10: 기둥·벽 조각과 "전방에 볼라드를 조심하세요" 문장 미리 합성 목록 |
