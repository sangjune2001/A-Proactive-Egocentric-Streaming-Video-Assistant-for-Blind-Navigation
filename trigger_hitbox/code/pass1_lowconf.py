"""히트박스(B)용 저신뢰 검출: 태규 설정(conf 0.5)에서 빠지는 장애물(쌓인 상자·입간판·테이블 등)을 잡으려고
conf 0.2로 한 번 더 추적만 한다. B0(태규 규칙)·광류는 pass1_detect.py 결과를 그대로 쓴다.
usage: python pass1_lowconf.py WEIGHTS VIDEO OUT_DIR  →  OUT_DIR/dets_low.json
"""
import json
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

CONF = 0.2


def main(weights, video, out_dir):
    out = Path(out_dir)
    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    model = YOLO(weights)
    classes = [k for k, v in model.names.items() if v != "traffic_light"]
    frames, fi, t0 = [], 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1
        res = model.track(frame, classes=classes, conf=CONF, imgsz=640, tracker="bytetrack.yaml",
                          persist=True, verbose=False)[0]
        dets = []
        if res.boxes is not None:
            for b in res.boxes:
                dets.append(dict(box=[round(v, 1) for v in b.xyxy[0].tolist()], cls=int(b.cls[0]),
                                 conf=round(float(b.conf[0]), 3),
                                 tid=int(b.id[0]) if b.id is not None else None))
        frames.append(dict(f=fi, t=round((fi - 1) / fps, 4), dets=dets))
    cap.release()
    (out / "dets_low.json").write_text(json.dumps(dict(conf=CONF, frames=frames)), encoding="utf-8")
    print("done", Path(video).name, fi, "frames", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
