"""트리거 채점: 라벨 구간 vs 트리거 이벤트 → TP · FP · FN · Precision · Recall (+ F1 · 분당 오경보 · 경고 여유).

팀 기준 (strict):
  TP = 라벨 구간 안에서 그 사건에 맞는 트리거가 남 / FN = 안 남 / FP = 어느 라벨 구간에도 안 맞는 트리거
보완 (추가로 같이 보고):
  - 같은 구간 안 트리거 여러 개 = TP 1개 (나머지는 FP로 세지 않음)
  - 같은 물체(트랙)에서 연달아 난 FP는 1개로 묶음 (dedup)
  - 여유 판정 (tol): 구간 시작 2 s 전 ~ 끝 0.5 s 뒤까지 인정 — 미리 경고하는 방식(B)이 FP로 깎이지 않게.
    라벨이 사람이 대강 단 것이라 경계 오차 흡수 목적도 있음
  - 사건 맞춤: 이동체 라벨 ↔ 탈것 클래스, 장애물 라벨 ↔ obstacle/stairs, 신호 라벨 ↔ 같은 방향 신호 변경
usage: python eval_trigger.py LABELS.json SPLIT OUT_ROOT [--tol]   (OUT_ROOT/<video>/events_*.jsonl)
"""
import json
import sys
from pathlib import Path

VEH = {"bicycle", "scooter", "motorcycle", "car", "bus", "other_vehicle", "truck"}
OBS = {"obstacle", "stairs", "bench", "potted_plant", "chair", "tree"}
PRE_S, POST_S = 2.0, 0.5


def kind_of(cls):
    return "vehicle" if cls in VEH else "obstacle" if cls in OBS else "light" if cls == "traffic_light" else "other"


def load_events(vdir: Path, method: str):
    """→ [(t, kind, track, dir)]  method: B (히트박스 v5_1) | B0 (태규) | Bd (B 위험만)"""
    ev = []
    rd = lambda n: [json.loads(l) for l in (vdir / n).read_text(encoding="utf-8").splitlines() if l.strip()] \
        if (vdir / n).exists() else []
    if method in ("B", "Bd"):
        seen = set()
        for e in rd("events_B.jsonl"):
            if method == "Bd" and e["level"] != "danger":
                continue
            if e["track_id"] in seen:          # 한 물체는 처음 경고(주의 또는 위험) 한 번만 트리거로 셈
                continue
            seen.add(e["track_id"])
            ev.append((e["time_s"], kind_of(e["class"]), e["track_id"], None))
        for e in rd("events_floor.jsonl"):     # 바닥 장애물 (연석 턱 · 고정 장애물, v5_9)
            if method == "Bd" and e["level"] != "danger":
                continue
            ev.append((e["time_s"], "obstacle", ("F", e["frame"]), None))
        for e in rd("events_light.jsonl"):
            ev.append((e["time_s"], "light", ("L", e["track_id"]), "to_green" if e["event"].endswith("green") else "to_red"))
    else:
        for e in rd("events_B0.jsonl"):
            if e.get("trigger_type") == "light_change":          # event = light_red_to_green
                d = "to_green" if e["event"].endswith("green") else "to_red"
                ev.append((e["time_s"], "light", ("L", e.get("track_id")), d))
            else:
                ev.append((e["time_s"], kind_of(e["class"]), e.get("track_id"), None))
    return sorted(ev, key=lambda e: e[0])


def score(labels, events, dur, tol):
    pre, post = (PRE_S, POST_S) if tol else (0.0, 0.0)
    hit = [None] * len(labels)
    fp, fp_tracks = [], set()
    for t, k, tr, d in events:
        m = None
        for i, lab in enumerate(labels):
            s, e, lk = lab[0], lab[1], lab[2]
            if s - pre <= t <= e + post and lk == k and (lk != "light" or lab[3] == d):
                m = i
                break
        if m is None:
            if tr not in fp_tracks or tr is None:
                fp.append((t, k, tr))
                fp_tracks.add(tr)
        elif hit[m] is None:
            hit[m] = t
    tp = sum(h is not None for h in hit)
    lead = [labels[i][0] - h for i, h in enumerate(hit) if h is not None]
    return dict(TP=tp, FN=len(labels) - tp, FP=len(fp), fp_list=[(round(t, 1), k) for t, k, _ in fp],
                missed=[labels[i][:3] for i, h in enumerate(hit) if h is None], lead_s=lead, dur_min=dur / 60)


def summarize(rows):
    TP, FP, FN = (sum(r[k] for r in rows) for k in ("TP", "FP", "FN"))
    P = TP / (TP + FP) if TP + FP else 0.0
    R = TP / (TP + FN) if TP + FN else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    mins = sum(r["dur_min"] for r in rows)
    lead = [x for r in rows for x in r["lead_s"]]
    return dict(TP=TP, FP=FP, FN=FN, precision=round(P, 3), recall=round(R, 3), f1=round(F1, 3),
                fp_per_min=round(FP / mins, 2) if mins else None,
                lead_median_s=round(sorted(lead)[len(lead) // 2], 2) if lead else None)


def main(labels_json, split, out_root, tol=False):
    L = json.loads(Path(labels_json).read_text(encoding="utf-8"))[split]
    res = {}
    for method in ("B0", "B", "Bd"):
        rows = {}
        for vid, labs in L.items():
            vdir = Path(out_root) / vid
            if not (vdir / "summary.json").exists():
                continue
            sm = json.loads((vdir / "summary.json").read_text(encoding="utf-8"))
            dur = sm["frames"] / sm["fps"]
            rows[vid] = score(labs, load_events(vdir, method), dur, tol)
        res[method] = dict(total=summarize(list(rows.values())), per_video=rows)
    return res


if __name__ == "__main__":
    a = sys.argv[1:]
    out = {}
    for tol in (False, True):
        out["tol" if tol else "strict"] = main(a[0], a[1], a[2], tol)
    print(json.dumps({m: {k: v["total"] for k, v in out[m].items()} for m in out}, ensure_ascii=False, indent=1))
    Path(a[2], f"eval_{a[1]}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
