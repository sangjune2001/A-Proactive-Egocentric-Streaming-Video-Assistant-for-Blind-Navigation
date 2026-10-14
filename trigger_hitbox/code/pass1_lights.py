"""신호등 전용 검출 + 상태 분류.

작은 보행신호 때문에 사용법 권장값(imgsz 1280, conf 0.25)으로 traffic_light만 검출하고,
신호 상태 분류기(yolo11n-cls, 96px: green/off/red/vehicle)로 판정한다.
신호는 천천히 바뀌므로 STRIDE 프레임마다 한 번만 본다 (CPU 시간 1/3).
usage: python pass1_lights.py DET_WEIGHTS CLS_WEIGHTS VIDEO OUT_DIR
→ OUT_DIR/lights.json
"""
import json
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

STRIDE = 3
import os
IMGSZ = int(os.environ.get("LIGHT_IMGSZ", 1280))   # v5_9: 작은 신호등용 1920
CONF = float(os.environ.get("LIGHT_CONF", 0.25))  # v5_9: 0.1
OUT_NAME = os.environ.get("LIGHT_OUT", "lights.json")
det_w, cls_w, video, out_dir = sys.argv[1:5]
det, clf = YOLO(det_w), YOLO(cls_w)
light_id = [k for k, v in det.names.items() if v == "traffic_light"][0]
cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS)
res_all, fi, t0 = [], 0, time.time()
while True:
    ok, frame = cap.read()
    if not ok:
        break
    fi += 1
    if (fi - 1) % STRIDE:
        continue
    r = det.track(frame, classes=[light_id], imgsz=IMGSZ, conf=CONF, persist=True, verbose=False,
                  tracker="bytetrack.yaml")[0]
    lights = []
    if r.boxes is not None:
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            w, h = x2 - x1, y2 - y1
            crop = frame[max(0, int(y1 - .3 * h)):int(y2 + .3 * h), max(0, int(x1 - .3 * w)):int(x2 + .3 * w)]
            if crop.size == 0:
                continue
            q = clf.predict(crop, imgsz=96, verbose=False)[0]
            lights.append(dict(box=[round(v, 1) for v in (x1, y1, x2, y2)], conf=round(float(b.conf[0]), 3),
                               tid=int(b.id[0]) if b.id is not None else None,
                               state=q.names[q.probs.top1], p=round(float(q.probs.top1conf), 3)))
    res_all.append(dict(f=fi, t=round((fi - 1) / fps, 4), lights=lights))
    if fi % 600 < STRIDE:
        print(f"{Path(video).name} {fi} frames {time.time() - t0:.0f}s", flush=True)
Path(out_dir).mkdir(parents=True, exist_ok=True)
(Path(out_dir) / OUT_NAME).write_text(json.dumps(dict(stride=STRIDE, fps=fps, frames=res_all)), encoding="utf-8")
n = sum(len(f["lights"]) for f in res_all)
print(f"done {Path(video).name} frames {fi} light dets {n} sec {time.time() - t0:.0f}", flush=True)
