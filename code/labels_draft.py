"""VIABench 정답 → 우리 4개 시나리오 라벨 초안 (사람 검수 전).

kind
  warn     : 이 구간 안에 해당 시나리오 발화가 있어야 함 (PDR 분모)
  neutral  : 말해도 되고 안 해도 됨 (FP로 세지 않음) — 상태 안내, 우리 범위 밖, 애매한 옆 장애물
  suppress : 이 구간에 해당 시나리오 발화 = FP (5주차 '심화: 억제' 사례)

출력: labels/<clip>.json  (review=true 항목은 박상준·임태규가 영상 보고 확정)
"""
from __future__ import annotations

import json
import re

from common import CLIPS, DATA, ROOT

GT = json.load(open(DATA / "viabench_gt.json", encoding="utf-8"))
OUT = ROOT.parent / "labels" / "v2"
OUT.mkdir(parents=True, exist_ok=True)

MOVING = re.compile(r"cyclist|riding|moving|walking|passing|approach", re.I)
VEH = re.compile(r"bicycle|bike|scooter|tricycle|motor|vehicle|car\b|cyclist", re.I)
SIDE = re.compile(r"(on|to) (the|your) (left|right)( side)?(?! front)|side of the", re.I)
PERSON = re.compile(r"pedestrian|person|people", re.I)
TRANSITION = re.compile(r"chang|turn", re.I)


def map_one(a: dict) -> dict:
    task, d = a["task"], a["description"]
    base = {"t_start": a["start_time"], "t_end": a["end_time"], "src": "viabench",
            "viabench_task": task, "text": d}
    if task == "Pedestrian Traffic Light":
        dl = d.lower()
        tail = dl.split(" to ")[-1] if " to " in dl else dl
        state = "GO" if "green" in tail and not ("red" in tail and tail.index("red") > tail.index("green")) else (
            "STOP" if "red" in tail else None)
        if TRANSITION.search(d):
            return {**base, "scenario": "S3", "kind": "warn", "state": state, "review": False}
        return {**base, "scenario": "S3", "kind": "neutral", "state": state, "review": False}   # 상태 안내 (빨간불입니다)
    if task in ("Stairs Up/Down", "Staircase Up/Down"):
        return {**base, "scenario": "S4", "kind": "warn", "review": False}
    if task in ("Obstacle Alert", "Active Avoidance"):
        if PERSON.search(d) and not VEH.search(d):
            return {**base, "scenario": "-", "kind": "neutral", "review": False}  # 보행자: 4개 시나리오 밖
        if VEH.search(d) and MOVING.search(d):
            return {**base, "scenario": "S1", "kind": "warn", "review": True}
        if task == "Active Avoidance":                                            # "피해서 가라" = 경로 위
            return {**base, "scenario": "S2", "kind": "warn", "review": True}
        if SIDE.search(d):
            return {**base, "scenario": "S2", "kind": "neutral", "review": True}  # 옆 장애물: 억제 후보
        return {**base, "scenario": "S2", "kind": "warn", "review": True}
    return {**base, "scenario": "-", "kind": "neutral", "review": False}           # 방향 이탈·표지판·카메라 이상 등


# VIABench에 없는데 우리 시나리오 정의상 필요한 구간 (2026-10-05 박상준 영상 확인, 임태규 교차 검수 대상)
def _o(s, e, sc, kind, text):
    return {"t_start": s, "t_end": e, "scenario": sc, "kind": kind, "text": text, "review": True, "viabench_task": None}


OURS: dict[str, list[dict]] = {
    "c01_vehicle_cross": [_o(18.0, 24.0, "S1", "suppress", "횡단 중 먼 차로(10 m+)를 가로지르는 차·버스 — 충돌 코스 아님"),
                          _o(24.0, 30.5, "S1", "neutral", "왼쪽 4–6 m 전동 삼륜차 서행 — 애매")],
    "c03_offpath_obst": [_o(5.0, 56.0, "S2", "suppress", "넓은 광장·경로 옆 계단 난간·나무 — 경로 밖"),
                         _o(61.0, 70.0, "S2", "suppress", "화단 속 가로수 — 경로 밖")],
    "c04_parked_cars": [_o(0.0, 38.0, "S2", "suppress", "경로 옆 주차 차량·가로수·쓰레기통 줄 — 경로 밖"),
                        _o(0.0, 38.0, "S1", "suppress", "주차 차량 — 움직이지 않음")],
    "c06_wait_signal": [_o(40.0, 58.0, "S1", "suppress", "신호 대기 중(내가 멈춤) 앞 차로를 지나가는 차")],
}

# VIABench 매핑을 영상 확인 후 고친 것: (clip, start_time) → 덮어쓸 필드
OVERRIDE = {
    ("c10_front_obst_night", 20.59): {"scenario": "S2", "kind": "warn", "review": True,
                                     "note": "23–24 s 가로수가 진행 방향 정면 (VIABench는 '왼쪽'이라 표기)"},
    ("c03_offpath_obst", 56.77): {"kind": "neutral", "note": "방향 이탈(화단 쪽) — 4개 시나리오 밖"},
    # 2026-10-05 감사(labels/evidence/v2_audit/gt_*.png, 정답마다 ts−1 · ts · 중간 · te 4프레임 확인) — 5주차 정의상 경고 대상 아님
    ("c01_vehicle_cross", 5.61): {"kind": "neutral", "note": "표지판이 이미 왼쪽 0.5 m 옆을 지나는 중 — 옆 장애물"},
    ("c02_step", 1.11): {"kind": "neutral", "note": "에스컬레이터 오른쪽 난간 — 경로 안내이지 막힘 아님"},
    ("c02_step", 45.03): {"kind": "neutral", "note": "구간 내내 휴대폰이 화면을 가림 — 자전거가 보이지 않음"},
    ("c09_bike_on_tactile", 32.24): {"kind": "neutral", "note": "왼쪽 옆 가로수 — 옆 장애물"},
}

# 감사 결과 경고 정답의 등급: clear = 경로 위 막힘이 영상에서 분명 / ambiguous = 옆·안내 성격이 섞임
AMBIGUOUS = {("c01_vehicle_cross", 30.04): "왼쪽 군밤 노점 — 경로 왼쪽 끝",
             ("c02_step", 89.81): "벽 앞 꺾임 — 방향 안내 성격",
             ("c05_path_obst_recede", 10.76): "왼쪽 주차 이륜차 줄의 끝",
             ("c05_path_obst_recede", 15.8): "자전거 주차 구역 안내 — 왼쪽 줄",
             ("c09_bike_on_tactile", 33.54): "왼쪽 앞 주차 전동차"}


def main():
    for key, (fname, vid, _) in CLIPS.items():
        ev = []
        for a in GT[fname]["annotations"]:
            e = map_one(a)
            e.update(OVERRIDE.get((key, a["start_time"]), {}))
            if e["kind"] == "warn" and e["scenario"] in ("S1", "S2"):
                amb = AMBIGUOUS.get((key, a["start_time"]))
                e["tier"] = "ambiguous" if amb else "clear"
                if amb:
                    e["note"] = amb
            ev.append(e)
        ev += [{**e, "src": "ours"} for e in OURS.get(key, [])]
        ev.sort(key=lambda e: e["t_start"])
        for i, e in enumerate(ev, 1):
            e["id"] = i
        json.dump({"clip": key, "file": fname, "viabench_id": vid, "events": ev},
                  open(OUT / f"{key}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        n = {k: sum(e["kind"] == k for e in ev) for k in ("warn", "neutral", "suppress")}
        print(f"{key:<22} warn {n['warn']:>2}  neutral {n['neutral']:>2}  suppress {n['suppress']:>2}  "
              f"review {sum(e['review'] for e in ev)}")
        for e in ev:
            if e["kind"] != "neutral" or e["scenario"] != "-":
                print(f"   {e['t_start']:6.2f}-{e['t_end']:6.2f} {e['scenario']} {e['kind']:<8} {e['text'][:80]}")


if __name__ == "__main__":
    main()
