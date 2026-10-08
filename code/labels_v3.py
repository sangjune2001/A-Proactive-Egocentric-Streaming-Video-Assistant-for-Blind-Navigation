"""라벨 v3 — 경고마다 4칸(대상·방향·움직임·행동) + 대상 트랙 (docs/09 §3).

  python labels_v3.py sheet            # 경고마다 확인 시트 (t_start−1 s · t_start · 구간 중간, 모든 트랙 번호 표시)
  python labels_v3.py build            # labels/<clip>.json(v2) + MANUAL 표 → labels/v3/<clip>.json

사람이 정하는 칸: target_fine, motion, track_id (MANUAL). 나머지는 자동:
  scenario·target_coarse ← target_fine, direction ← 대상 bbox 중심 x (화면 1/3 기준), action_ok ← 규칙표.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "p0"))
from common import CLIPS, OUT, clip_path                       # noqa: E402
from config import COARSE                                     # noqa: E402
from frames import Tracks, Video                              # noqa: E402
from ko_template import action_ok, direction_of               # noqa: E402

LAB2 = HERE.parent / "labels" / "v2"
LAB3 = HERE.parent / "labels" / "v3"
SHEETS = HERE.parent / "labels" / "evidence" / "v3_sheets"

# ── 사람이 정한 칸 (clip, v2 id) ──────────────────────────────────────────────
# fine = 대표 세부 종류, ok = 정답으로 인정할 target 값들 (이름이 애매한 대상), track_id None = 검출기가 못 잡음 →
# direction은 사람이 정한 값. 근거: labels/evidence/v3_sheets/<clip>_<id>.jpg (t_start−1 s · t_start · 구간 중간)
# by = claude-draft: 2026-10-06 그림 확인 초안 → 팀원(임태규) 교차 확인 필요
PARKED_SCOOTER = ["parked_scooter", "scooter", "motorcycle"]
PARKED_BICYCLE = ["parked_bicycle", "bicycle"]
MANUAL: dict[tuple[str, int], dict] = {
    ("c01_vehicle_cross", 1):  dict(fine="sign", motion="static", track_id=1, note="정면 '此摆处摊' 입간판"),
    ("c01_vehicle_cross", 9):  dict(fine="stall", motion="static", track_id=586, note="왼쪽 군밤 노점 (트랙 586은 car로 오검출된 노점+사람 박스)"),
    ("c01_vehicle_cross", 10): dict(fine="stall", motion="static", track_id=None, direction="front", note="정면 과일 노점, t_start에 검출 없음"),
    ("c01_vehicle_cross", 11): dict(fine="stall", motion="static", track_id=None, direction="front", note="정면 핫도그 노점 줄, 검출 없음"),
    ("c02_step", 4):  dict(fine="stairs", motion="down", track_id=None, direction="front", note="정면 내려가는 계단 (小心台阶)"),
    ("c02_step", 5):  dict(fine="stairs", motion="up", track_id=None, direction="front", note="정면 올라가는 계단"),
    ("c02_step", 6):  dict(fine="parked_bicycle", ok=PARKED_BICYCLE + PARKED_SCOOTER, motion="static", track_id=None,
                           direction="front", note="정면 주차 자전거·전동스쿠터 무리, 검출 없음"),
    ("c02_step", 8):  dict(fine="other", ok=["other", "barrier"], motion="static", track_id=None, direction="front",
                           note="정면 벽 (점자블록 꺾임), tier ambiguous"),
    ("c02_step", 11): dict(fine="sign", ok=["sign", "barrier", "other"], motion="static", track_id=None, direction="right",
                           note="오른쪽 바로 앞 안내판 + 정면 검색대"),
    ("c03_offpath_obst", 1): dict(fine="stairs", motion="up", track_id=None, direction="front", note="정면 오르는 계단"),
    ("c05_path_obst_recede", 4): dict(fine="parked_scooter", ok=PARKED_SCOOTER, motion="static", track_id=705,
                                      note="경로 위 주차 전동스쿠터, tier ambiguous"),
    ("c05_path_obst_recede", 6): dict(fine="parked_bicycle", ok=PARKED_BICYCLE, motion="static", track_id=1252,
                                      note="왼쪽 앞 공유자전거 줄, tier ambiguous"),
    ("c06_wait_signal", 3):  dict(fine="traffic_light", motion="to_green", track_id=27,
                                  note="정면 보행 신호등 후보 27/37 중 27 (확인 필요)"),
    ("c06_wait_signal", 10): dict(fine="traffic_light", motion="to_green", track_id=555, note="손이 화면 대부분을 가림"),
    ("c06_wait_signal", 12): dict(fine="traffic_light", motion="to_red", track_id=555, note="같은 보행 신호등 555"),
    ("c07_light_rg", 2):  dict(fine="traffic_light", motion="to_green", track_id=62, note="왼쪽 아래 보행 신호등"),
    ("c07_light_rg", 7):  dict(fine="stairs", motion="up", track_id=None, direction="front", note="정면 연석 턱 (트랙 302는 안내견 오검출)"),
    ("c08_light_rgr_night", 2): dict(fine="traffic_light", motion="to_green", track_id=None, direction="front",
                                     note="야간, 먼 보행 신호등 검출 없음"),
    ("c08_light_rgr_night", 4): dict(fine="stairs", motion="up", track_id=None, direction="front",
                                     note="야간 정면 연석 턱 (트랙 292는 안내견 오검출)"),
    ("c09_bike_on_tactile", 2): dict(fine="parked_scooter", ok=PARKED_SCOOTER + PARKED_BICYCLE, motion="static",
                                     track_id=None, direction="front",
                                     note="점자블록 끝 노란 전동스쿠터 (정답 문장은 bicycle), 검출 없음 · 트랙 164는 가로수"),
    ("c09_bike_on_tactile", 6): dict(fine="parked_scooter", ok=PARKED_SCOOTER, motion="static", track_id=626,
                                     note="왼쪽 앞 주차 전동스쿠터, tier ambiguous"),
    ("c09_bike_on_tactile", 8): dict(fine="parked_scooter", ok=PARKED_SCOOTER, motion="static", track_id=723,
                                     note="정면 주차 전동스쿠터"),
    ("c10_front_obst_night", 3): dict(fine="tree", motion="static", track_id=None, direction="right",
                                      note="야간 가로수 줄. 화면상 보행로 오른쪽(정답 문장은 'left' → 회전 보정 확인 필요)"),
}


def warn_events(key: str) -> list[dict]:
    lab = json.load(open(LAB2 / f"{key}.json", encoding="utf-8"))
    return [e for e in lab["events"] if e.get("kind") == "warn"]


def _draw_tracks(im, tracks: Tracks, t: float, sx: float, sy: float):
    for tid, cls in tracks.visible(t, tol=0.07):
        b = tracks.bbox(tid, t, tol=0.07)
        if b is None:
            continue
        x1, y1, x2, y2 = int(b[0] * sx), int(b[1] * sy), int(b[2] * sx), int(b[3] * sy)
        col = (0, 0, 255) if cls != "person" else (200, 200, 200)
        cv2.rectangle(im, (x1, y1), (x2, y2), col, 2)
        cv2.putText(im, f"{tid}:{cls[:6]}", (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    return im


def sheet(keys):
    SHEETS.mkdir(parents=True, exist_ok=True)
    for key in keys:
        tr_path = OUT / "tracks" / "E0" / f"{key}.csv"
        tracks = Tracks(tr_path)
        video = Video(clip_path(key), CLIPS[key][2])
        for e in warn_events(key):
            ts = [max(0.0, e["t_start"] - 1.0), e["t_start"], (e["t_start"] + e["t_end"]) / 2]
            tiles = []
            for t in ts:
                im = video.at(t)
                sx, sy = im.shape[1] / tracks.W, im.shape[0] / tracks.H
                im = _draw_tracks(im.copy(), tracks, t, sx, sy)
                s = 480 / im.shape[0]
                im = cv2.resize(im, (round(im.shape[1] * s), 480))
                cv2.putText(im, f"t={t:.2f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                tiles.append(im)
            out = np.hstack(tiles)
            head = np.zeros((40, out.shape[1], 3), np.uint8)
            cv2.putText(head, f"{key} id={e['id']} {e['scenario']} | {e.get('text', '')[:110]}", (8, 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            (SHEETS / f"{key}_{e['id']:02d}.jpg").write_bytes(cv2.imencode(".jpg", np.vstack([head, out]))[1].tobytes())
        video.close()
        print(key, "ok", flush=True)


def scenario_of_fine(fine: str, motion: str) -> str:
    if fine == "traffic_light":
        return "S3"
    if fine == "stairs":
        return "S4"
    if fine in ("person", "bicycle", "scooter", "motorcycle", "car", "bus", "other_vehicle"):
        return "S1"
    return "S2"


def build():
    LAB3.mkdir(exist_ok=True)
    total = 0
    for key in CLIPS:
        if not (LAB2 / f"{key}.json").exists():
            continue
        tracks = Tracks(OUT / "tracks" / "E0" / f"{key}.csv")
        video = Video(clip_path(key), CLIPS[key][2])
        H, W = video.at(0).shape[:2]
        video.close()
        out = []
        for e in warn_events(key):
            m = MANUAL.get((key, e["id"]))
            if m is None:
                raise KeyError(f"MANUAL에 ({key!r}, {e['id']}) 없음")
            fine, motion, tid = m["fine"], m["motion"], m.get("track_id")
            box = tracks.bbox(tid, e["t_start"], tol=0.2) if tid is not None else None
            if box is not None:
                sx, sy = W / tracks.W, H / tracks.H
                box = [round(box[0] * sx, 1), round(box[1] * sy, 1), round(box[2] * sx, 1), round(box[3] * sy, 1)]
                direction = direction_of(box, W)
            else:
                direction = m["direction"]                 # 검출기가 못 잡은 대상: 사람이 정한 방향
            sc = scenario_of_fine(fine, motion)
            out.append(dict(
                id=e["id"], kind="warn", scenario=sc, tier=e.get("tier"), t_start=e["t_start"], t_end=e["t_end"],
                target_coarse=COARSE[fine], target_fine=fine, target_ok=m.get("ok", [fine]), motion=motion, direction=direction,
                direction_src="bbox" if box is not None else "manual",
                target_track={"track_id": tid, "bbox": box} if tid is not None else None,
                action_ok=action_ok(sc, direction, motion), text_ref=e.get("text"),
                annotator=m.get("by", "claude-draft"), note=m.get("note", ""), src=e.get("src")))
        json.dump({"clip": key, "version": 3, "W": W, "H": H, "events": out},
                  open(LAB3 / f"{key}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        total += len(out)
        print(f"{key:<22} {len(out)} warn", flush=True)
    print("total", total)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sheet", "build"])
    ap.add_argument("--clips", nargs="*", default=[k for k in CLIPS if (LAB2 / f"{k}.json").exists()])
    a = ap.parse_args()
    sheet(a.clips) if a.cmd == "sheet" else build()
