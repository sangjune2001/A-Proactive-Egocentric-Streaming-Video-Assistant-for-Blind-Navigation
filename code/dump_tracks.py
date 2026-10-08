"""인식 + 추적 덤프: 10편 → results/tracks/<model>/<clip>.csv (MVP 계약 형식) + 프레임별 처리 시간.

    python dump_tracks.py --model E0 --fps 15            # 로컬(CPU)도 가능, FPS 수치는 A5000에서 다시 잰다
    python dump_tracks.py --model A5 --fps 15 --device 0  # Mondrian A5000
"""
from __future__ import annotations

import argparse
import csv
import json
import time

import numpy as np
from ultralytics import YOLO

from common import CLASSES, CLIPS, OUT, WEIGHTS, clip_info, iter_frames

COLS = ["frame", "t_sec", "track_id", "cls", "conf", "x1", "y1", "x2", "y2",
        "foot_u", "foot_v", "edge_flag", "W", "H", "mask_area"]


def foot_point(poly: np.ndarray, box) -> tuple[float, float]:
    """마스크 최하단 접지점 (최하단 2 % 높이 띠의 x 평균). 마스크 없으면 bbox 하단 중앙."""
    if poly is None or len(poly) < 3:
        return (box[0] + box[2]) / 2, box[3]
    ys = poly[:, 1]
    vmax = ys.max()
    band = poly[ys >= vmax - 0.02 * max(1.0, box[3] - box[1])]
    return float(band[:, 0].mean()), float(vmax)


def run_clip(model_id: str, key: str, fps: float, device, imgsz: int) -> dict:
    model = YOLO(str(WEIGHTS[model_id]))          # 클립마다 새로 → 추적기 상태 초기화
    info = clip_info(key)
    out_dir = OUT / "tracks" / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, times = [], []
    for fidx, t, im in iter_frames(key, fps, max_side=1920):
        H, W = im.shape[:2]
        t0 = time.perf_counter()
        r = model.track(im, persist=True, tracker="bytetrack.yaml", imgsz=imgsz, conf=0.25,
                        device=device, verbose=False)[0]
        wall = (time.perf_counter() - t0) * 1000
        times.append({"t": round(t, 3), "wall_ms": round(wall, 2), **{k: round(v, 2) for k, v in r.speed.items()}})
        if r.boxes is None or r.boxes.id is None:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.cpu().numpy().astype(int)
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        polys = r.masks.xy if r.masks is not None else [None] * len(ids)
        for b, tid, c, p, poly in zip(xyxy, ids, cls, conf, polys):
            fu, fv = foot_point(poly, b)
            edge = int(b[0] <= 2 or b[1] <= 2 or b[2] >= W - 2 or b[3] >= H - 2)
            area = float(cv2_area(poly)) if poly is not None and len(poly) >= 3 else 0.0
            rows.append([fidx, round(t, 3), tid, CLASSES[c], round(float(p), 3),
                         *[round(float(v), 1) for v in b], round(fu, 1), round(fv, 1), edge, W, H, round(area, 1)])
    with open(out_dir / f"{key}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        w.writerows(rows)
    with open(out_dir / f"{key}_timing.json", "w", encoding="utf-8") as f:
        json.dump({"clip": key, "model": model_id, "fps_sampled": fps, "imgsz": imgsz, **info, "frames_proc": times}, f)
    wall = np.array([x["wall_ms"] for x in times])
    return {"clip": key, "frames": len(times), "rows": len(rows), "ms_median": float(np.median(wall)),
            "tracks": len({r[2] for r in rows})}


def cv2_area(poly):
    import cv2
    return cv2.contourArea(poly.astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="E0", choices=list(WEIGHTS))
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--clips", nargs="*", default=list(CLIPS))
    a = ap.parse_args()
    dev = int(a.device) if a.device.isdigit() else a.device
    for key in a.clips:
        s = run_clip(a.model, key, a.fps, dev, a.imgsz)
        print(json.dumps(s, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
