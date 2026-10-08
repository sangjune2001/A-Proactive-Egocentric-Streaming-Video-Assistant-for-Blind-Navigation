"""VIABench 정답 문장 → 4칸(대상 · 움직임 · 방향) 규칙 기반 자동 추출 (docs/14 §2 ①).

  python viabench_extract.py            # → benchmarks/viabench_fields.jsonl, 요약 출력, 검수 표본 CSV

- 대상: 문장에 나온 물체를 config.TARGETS 어휘로 매핑. 여러 개면 target_ok에 모두 (하나만 맞혀도 정답, labels/v3와 같은 방식)
- 움직임: 다가옴 · 지나감 · 멀어짐 / 계단 위 · 아래 / 신호등 빨강 · 초록. 정지 물체 클래스는 static. 판단 불가 = None
- 방향: left / right / front. 양쪽 · 뒤 · 판단 불가 = None (VIABench 방향은 말로 한 표현 → 보조 지표)
- None인 칸은 채점에서 뺀다. 판단할 수 없는 문장(신호등 없음 등)은 skip 사유를 남긴다.
"""
from __future__ import annotations

import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "p0"))
from config import COARSE, TARGETS  # noqa: E402

BENCH = HERE.parent / "benchmarks"
SRC = BENCH / "viabench_annotations" / "proactive_reminder.json"
TASKS = ["Obstacle Alert", "Active Avoidance", "Stairs Up/Down", "Staircase Up/Down", "Pedestrian Traffic Light"]

# (정규식, 대상) — 위에서부터 먼저 맞는 것. 'row of / parked' 는 아래에서 parked_* 로 바꾼다
LEXICON = [
    (r"traffic lights?|signal lights?|crossing lights?", "traffic_light"),
    (r"stair(?:s|case|way)?|steps?\b|step up|step down|curbs?|kerbs?", "stairs"),
    (r"electric scooters?|e-scooters?|scooters?|mopeds?|electric (?:bikes?|bicycles?|vehicles?|cars?)|e-bikes?|battery cars?", "scooter"),
    (r"motorcycles?|motorbikes?|motor bikes?", "motorcycle"),
    (r"bicycles?|bikes?|cyclists?|shared bikes?", "bicycle"),
    (r"buses|bus\b", "bus"),
    (r"cars?\b|taxis?|suvs?|sedans?", "car"),
    (r"trucks?|vans?|vehicles?|tricycles?|carts?|trolleys?|forklifts?|tractors?", "other_vehicle"),
    (r"pedestrians?|people|persons?|person|man\b|men\b|woman|women|child(?:ren)?|kids?|crowds?|someone|passers?-?by|elderly|staff|workers?", "person"),
    (r"bollards?|stone pillars?|stone posts?|short pillars?|small pillars?|anti-?collision (?:posts?|pillars?|balls?)|stone balls?|spherical", "bollard"),
    (r"poles?|pillars?|posts?|columns?|street ?lights?|street ?lamps?|lamp ?posts?|lampposts?", "pole"),
    (r"trees?|tree trunks?|branches", "tree"),
    (r"signs?\b|signboards?|billboards?|placards?|board\b|boards\b", "sign"),
    (r"stalls?|vendors?|booths?|kiosks?|stands?\b", "stall"),
    (r"benches|bench\b|seating|seats?\b|stools?", "bench"),
    (r"railings?|fences?|barriers?|roadblocks?|guardrails?|handrails?|chains?|cones?|barricades?|isolation piles?|gates?", "barrier"),
    (r"walls?|flower ?pots?|planters?|flower ?beds?|tree (?:beds?|pits?|pools?)|greenbelts?|green belts?|boxes|box\b|trash (?:cans?|bins?)|garbage|bins?\b|"
     r"rocks?|stones?|potholes?|manholes?|puddles?|debris|bags?|chairs?|tables?|objects?|obstacles?|things?|items?|leaves|"
     r"construction|scaffold\w*|ladders?|pipes?|bricks?|sandbags?|goods|shelves|shelf|racks?|steps? ?stones?|water|"
     r"hydrants?|bush(?:es)?|shrubs?|hedges?|plants?|potted|greenery|grass|lawns?|doors?|doorways?|turnstiles?|backpacks?|"
     r"sticks?|baskets?|buckets?|dogs?|cats?|animals?|strollers?|prams?|wheelchairs?|suitcases?|luggage|umbrellas?|ropes?|"
     r"wires?|cables?|tents?|sheds?|awnings?|canopy|parasols?|mats?|barrels?|pallets?|planks?|cardboard|furniture|"
     r"machines?|equipment|mailbox\w*|lockers?|cabinets?|counters?|vending|ditch\w*|holes?|drains?|gutters?|lakes?|ponds?|"
     r"rivers?|sculptures?|statues?|pyramids?|ledges?|platforms?|signposts?", "other"),
]
AMBIG = [(r"\bpillars?\b|\bposts?\b", ["pole", "bollard"]), (r"\bcurbs?\b|\bkerbs?\b", ["stairs", "other"]),
         (r"railings?|handrails?", ["barrier", "other"]), (r"electric (?:bikes?|bicycles?|vehicles?)|e-bikes?|mopeds?",
                                                          ["scooter", "motorcycle", "bicycle"])]
MOVERS = {"person", "bicycle", "scooter", "motorcycle", "car", "bus", "other_vehicle"}
PARKED = re.compile(r"\bparked\b|\brows? of\b|\bline of\b|\bshared\b|\bstationary\b")


def targets_of(s: str) -> list[tuple[int, str]]:
    """문장 안 등장 위치 순서로 (위치, 대상). 같은 대상은 한 번."""
    found, taken = [], []
    for pat, t in LEXICON:
        for m in re.finditer(rf"\b(?:{pat})", s):
            if any(a <= m.start() < b for a, b in taken):
                continue
            taken.append((m.start(), m.end()))
            found.append((m.start(), t))
    seen, out = set(), []
    for pos, t in sorted(found):
        if t not in seen:
            seen.add(t)
            out.append((pos, t))
    return out


def motion_of(s: str, task: str, tgt: str | None) -> str | None:
    if task == "Pedestrian Traffic Light" or tgt == "traffic_light":
        if re.search(r"turn(?:s|ed|ing)? (?:to )?green|chang\w* to green|about to turn green", s):
            return "to_green"
        if re.search(r"turn(?:s|ed|ing)? (?:to )?red|chang\w* to red|about to turn red", s):
            return "to_red"
        r, g = "red" in s, "green" in s
        return "red" if r and not g else "green" if g and not r else None
    if tgt == "stairs":
        up = re.search(r"\bup\b|upward|upstairs|ascend|going up|step up", s)
        dn = re.search(r"\bdown\b|downward|downstairs|descend|going down|step down", s)
        return "up" if up and not dn else "down" if dn and not up else None
    if re.search(r"approach|coming (?:toward|towards|from|at)|oncoming|heading (?:toward|towards)|\btoward(?:s)? you", s):
        return "approaching"
    if re.search(r"cross(?:es|ing)? (?:in front|ahead|the path|your path)|passing (?:by|in front)|pass(?:es)? (?:by|in front)", s):
        return "crossing"
    if re.search(r"moving away|walking away|leaving|going away|riding away", s):
        return "receding"
    if tgt and (tgt not in MOVERS or PARKED.search(s)):
        return "static"
    return None


def direction_of(s: str) -> str | None:
    if re.search(r"both sides|left and right|right and left|on either side|\bbehind\b", s):
        return None
    l = re.search(r"\bleft\b", s)
    r = re.search(r"\bright\b", s)
    # 행동 지시(please walk to the right)의 방향어는 대상 방향이 아님 → 'please' 앞부분만 본다
    head = re.split(r"\bplease\b|\bkeep\b|\bwalk (?:on|to)\b|\bmove (?:to|toward)", s)[0]
    l, r = re.search(r"\bleft\b", head), re.search(r"\bright\b", head)
    if l and not r:
        return "left"
    if r and not l:
        return "right"
    if not l and not r and re.search(r"ahead|in front|front of|straight", head):
        return "front"
    return None


def extract(desc: str, task: str) -> dict:
    s = desc.lower().strip()
    if task == "Pedestrian Traffic Light" and re.search(r"\bno\b.*traffic light|there (?:is|are) no", s):
        return dict(skip="신호등 없음")
    ts = [t for _, t in targets_of(s)]
    if task in ("Stairs Up/Down", "Staircase Up/Down"):
        ts = ["stairs"] + [t for t in ts if t != "stairs"]
    if task == "Pedestrian Traffic Light":
        ts = ["traffic_light"]
    if not ts:
        return dict(skip="대상 매핑 실패")
    main = ts[0]
    # "pedestrian crossing ahead" = 사람이 가로지름 / 횡단보도 둘 다 가능 → 모호하면 제외
    if main in ("person", "traffic_light") and re.search(r"pedestrian crossing|zebra crossing|crosswalk", s) \
            and task != "Pedestrian Traffic Light":
        return dict(skip="횡단보도 표현 모호")
    if PARKED.search(s) and main in ("bicycle", "scooter"):
        main = f"parked_{main}"
        ts[0] = main
    ok = set(ts) | ({main.replace("parked_", "")} if main.startswith("parked_") else set())
    for pat, alts in AMBIG:                                  # 영어 표현 하나가 우리 어휘 여러 개에 걸치는 경우
        if re.search(pat, s):
            ok |= set(alts)
    ok = sorted(ok)
    return dict(target=main, target_ok=ok, target_coarse=COARSE.get(main, "obstacle"),
                motion=motion_of(s, task, main.replace("parked_", "") if main.startswith("parked_") else main),
                direction=direction_of(s), multi=len(ts) > 1)


def main():
    a = json.load(open(SRC, encoding="utf-8"))
    rows = []
    for vid, v in a.items():
        for x in v["annotations"]:
            if x["task"] not in TASKS:
                continue
            e = extract(x["description"], x["task"])
            rows.append(dict(video=vid, qid=x["qid"], task=x["task"], t_start=x["start_time"], t_end=x["end_time"],
                             text=x["description"], **e))
    assert all(r.get("target") in TARGETS + [None] for r in rows if "skip" not in r)

    out = BENCH / "viabench_fields.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"정답 {len(rows)}개 → {out.name}\n")
    print("| 과제 | n | 추출 | 대상 | 움직임 | 방향 | 여러 대상 | 제외 사유 |")
    print("|---|---|---|---|---|---|---|---|")
    for task in TASKS + ["전체"]:
        rr = [r for r in rows if task == "전체" or r["task"] == task]
        ok = [r for r in rr if "skip" not in r]
        p = lambda k: f"{sum(r.get(k) is not None for r in ok) / len(rr):.0%}"
        sk = Counter(r["skip"] for r in rr if "skip" in r)
        print(f"| {task} | {len(rr)} | {len(ok) / len(rr):.0%} | {p('target')} | {p('motion')} | {p('direction')} | "
              f"{sum(r['multi'] for r in ok) / len(rr):.0%} | {dict(sk)} |")
    ok = [r for r in rows if "skip" not in r]
    print("\n대상 분포:", Counter(r["target"] for r in ok).most_common())
    print("움직임 분포:", Counter(r["motion"] for r in ok).most_common())
    print("방향 분포:", Counter(r["direction"] for r in ok).most_common())

    # 매핑 실패 문장 예 (어휘 보강용)
    random.seed(0)
    fail = [r["text"] for r in rows if r.get("skip") == "대상 매핑 실패"]
    print(f"\n대상 매핑 실패 {len(fail)}개 예:")
    for t in random.sample(fail, min(15, len(fail))):
        print("  -", t)

    # 검수 표본: 과제별 층화 150개 (사람이 ok 열에 O/X, 틀리면 fix 열에 정답)
    per = {"Obstacle Alert": 70, "Active Avoidance": 30, "Stairs Up/Down": 20, "Staircase Up/Down": 10,
           "Pedestrian Traffic Light": 20}
    sample = []
    for task, k in per.items():
        rr = [r for r in rows if r["task"] == task]
        sample += random.sample(rr, min(k, len(rr)))
    with open(BENCH / "viabench_fields_review150.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["task", "video", "qid", "text", "target", "target_ok", "motion", "direction", "skip",
                    "ok_target(O/X)", "ok_motion(O/X)", "ok_direction(O/X)", "fix"])
        for r in sample:
            w.writerow([r["task"], r["video"], r["qid"], r["text"], r.get("target"), "|".join(r.get("target_ok", [])),
                        r.get("motion"), r.get("direction"), r.get("skip", ""), "", "", "", ""])
    print(f"\n검수 표본 {len(sample)}개 → viabench_fields_review150.csv")


if __name__ == "__main__":
    main()
