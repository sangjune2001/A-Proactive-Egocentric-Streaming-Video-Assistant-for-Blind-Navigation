"""AI Hub 159 (1인칭 시점 보행영상) 검증 세트로 인식 채점.

라벨: VIA 폴리곤 (이미지별 물체). 우리 10클래스로 바꿔 YOLO 검출과 맞춰 본다.
  - 클래스별 Recall · 전체 Precision (conf 0.5 = 태규 설정 / conf 0.2 = 히트박스용 저신뢰)
  - '내 앞길' 물체만 따로: 박스 아래 끝이 화면 아래 45 % 안 + 가로 가운데 60 % 안 (트리거가 실제로 신경 쓰는 물체)
usage: python eval_159.py WEIGHTS ROOT OUT_JSON   (ROOT 아래 *.json 라벨과 *.jpg 원본을 찾아 짝지음)
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np
from ultralytics import YOLO

MAP = {"car": "car", "motorcycle": "motorcycle", "bicycle": "bicycle", "person": "person",
       "tree": "obstacle", "chair": "obstacle", "pot": "obstacle", "bollard": "obstacle", "box": "obstacle",
       "bench": "obstacle", "table": "obstacle", "fire hydrant": "obstacle", "streetlight": "obstacle",
       "stretlight": "obstacle", "sign": "obstacle", "no parking sign": "obstacle", "no park sign": "obstacle",
       "no pack sign": "obstacle", "flower garden": "obstacle"}
VEH = {"car", "motorcycle", "bicycle", "scooter", "bus", "other_vehicle"}
IOU_T = 0.4


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def group(c):           # 탈것끼리는 서로 헷갈려도 트리거에는 같은 '이동체'
    return "vehicle" if c in VEH else c


def load_gt(root):
    imgs = {p.name: p for p in Path(root).rglob("*.jpg")}
    gt = {}
    for j in Path(root).rglob("*.json"):
        try:
            d = json.loads(j.read_text(encoding="utf-8"))
        except Exception:
            continue
        for v in d.values():
            if not isinstance(v, dict) or v.get("filename") not in imgs:
                continue
            objs = []
            for r in v["regions"].values():
                c = MAP.get((r["region_attributes"].get("object") or "").strip())
                sa = r["shape_attributes"]
                if c is None or "all_points_x" not in sa:
                    continue
                xs, ys = sa["all_points_x"], sa["all_points_y"]
                objs.append((c, (min(xs), min(ys), max(xs), max(ys))))
            gt[imgs[v["filename"]]] = objs
    return gt


def main(weights, root, out_json, max_imgs=3000):
    gt = load_gt(root)
    paths = sorted(gt)[:int(max_imgs)]
    print("images", len(paths), flush=True)
    m = YOLO(weights)
    names = m.names
    R = {}
    for conf in (0.5, 0.2):
        tp, n_gt = collections.Counter(), collections.Counter()
        tp_p, n_p = collections.Counter(), collections.Counter()   # 내 앞길
        n_pred, n_pred_ok = 0, 0
        for k in range(0, len(paths), 32):
            batch = paths[k:k + 32]
            res = m.predict([str(p) for p in batch], imgsz=640, conf=conf, device=0, verbose=False)
            for p, r in zip(batch, res):
                H, W = r.orig_shape
                preds = [(names[int(c)], b.tolist()) for c, b in zip(r.boxes.cls, r.boxes.xyxy)
                         if names[int(c)] not in ("traffic_light", "stairs")]
                used = set()
                for c, b in gt[p]:
                    front = b[3] > 0.55 * H and abs((b[0] + b[2]) / 2 - W / 2) < 0.3 * W
                    n_gt[c] += 1
                    n_p[c] += front
                    best = max(((iou(b, pb), i) for i, (pc, pb) in enumerate(preds)
                                if i not in used and group(pc) == group(c)), default=(0, None))
                    if best[0] >= IOU_T:
                        used.add(best[1])
                        tp[c] += 1
                        tp_p[c] += front
                n_pred += len(preds)
                n_pred_ok += len(used)
        R[f"conf{conf}"] = dict(
            recall={c: round(tp[c] / n_gt[c], 3) for c in n_gt}, n_gt=dict(n_gt),
            recall_front={c: round(tp_p[c] / n_p[c], 3) for c in n_p if n_p[c]}, n_front=dict(n_p),
            precision=round(n_pred_ok / n_pred, 3) if n_pred else None, n_pred=n_pred)
        print(conf, json.dumps(R[f"conf{conf}"], ensure_ascii=False), flush=True)
    Path(out_json).write_text(json.dumps(dict(images=len(paths), iou=IOU_T, **R), ensure_ascii=False, indent=1),
                              encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:4])
