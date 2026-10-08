"""TTS 비교 문장 세트 → results/tts_bench/sentences.json  (docs/13 3단계)

  W : 즉시 경고 템플릿 전부 (클래스 × 방향 + 신호 2) — 실제로 미리 합성해 쓰는 문장 그대로
  D : VLM 설명 문장 (ko_template.sentence) 층화 추출 — 대상 · 움직임은 전부 한 번 이상, 방향 · 행동은 하이브리드 규칙대로
문장은 실행 시스템과 같은 코드로 만든다 (TTS가 실제로 받을 문장만 평가).
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent / "p0"), str(HERE.parent / "runtime")]
from common import OUT                                       # noqa: E402
from config import DIRECTIONS                                # noqa: E402
from dispatcher import all_warn_texts                        # noqa: E402
from ko_template import action_ok, sentence                  # noqa: E402

rng = random.Random(0)
S1 = ["person", "bicycle", "scooter", "motorcycle", "car", "bus", "other_vehicle"]
S2 = ["tree", "pole", "bollard", "sign", "stall", "parked_bicycle", "parked_scooter", "bench", "barrier", "other"]
combos = []
for t in S1:
    for m in ("approaching", "crossing", "receding"):
        combos.append(("S1", t, m))
for t in S2:
    combos.append(("S2", t, "static"))
for m in ("red", "green", "to_green", "to_red"):
    combos.append(("S3", "traffic_light", m))
for m in ("up", "down"):
    combos.append(("S4", "stairs", m))

# S1은 대상마다 움직임 하나씩 (움직임 3종이 고르게), 나머지는 전부
s1 = [c for c in combos if c[0] == "S1"]
pick = [s1[i * 3 + (i % 3)] for i in range(len(S1))] + [c for c in combos if c[0] != "S1"]
desc = []
for sc, t, m in pick:
    d = rng.choice(DIRECTIONS)
    out = dict(hazard=True, target=t, direction=d, motion=m, action=action_ok(sc, d, m)[0])
    desc.append(sentence(out))

items = [{"id": f"W{i:02d}", "kind": "warn", "text": x} for i, x in enumerate(all_warn_texts())]
items += [{"id": f"D{i:02d}", "kind": "desc", "text": x} for i, x in enumerate(dict.fromkeys(desc))]
od = OUT / "tts_bench"
od.mkdir(parents=True, exist_ok=True)
json.dump(items, open(od / "sentences.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"경고 {sum(i['kind'] == 'warn' for i in items)} · 설명 {sum(i['kind'] == 'desc' for i in items)} → {od / 'sentences.json'}")
for i in items[::6]:
    print(" ", i["id"], i["text"])
