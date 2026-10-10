"""탐지(seg10 이어서 학습한 모델) + 신호 상태 분류 결합 추론.
traffic_light 박스를 잘라 분류 모델로 red/green/off/vehicle 판정. 출력은 bbox만 사용.
사용: python3 infer.py <탐지 best.pt> <분류 best.pt> <이미지/폴더/영상> [--save 결과폴더]"""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("det")
ap.add_argument("cls")
ap.add_argument("source")
ap.add_argument("--save", default="infer_out")
ap.add_argument("--imgsz", type=int, default=1280)
args = ap.parse_args()

det, cls = YOLO(args.det), YOLO(args.cls)
TL = [k for k, v in det.names.items() if v == "traffic_light"][0]
COL = {"red": (0, 0, 255), "green": (0, 255, 0), "off": (160, 160, 160), "vehicle": (255, 128, 0)}
out = Path(args.save)
out.mkdir(parents=True, exist_ok=True)
for i, r in enumerate(det.predict(args.source, imgsz=args.imgsz, stream=True, verbose=False)):
    img = r.orig_img.copy()
    for b, c, conf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
        x1, y1, x2, y2 = map(int, b)
        name = det.names[int(c)]
        if int(c) == TL:
            w, h = x2 - x1, y2 - y1
            crop = img[max(0, int(y1 - .3 * h)):int(y2 + .3 * h), max(0, int(x1 - .3 * w)):int(x2 + .3 * w)]
            if crop.size:
                p = cls.predict(crop, imgsz=96, verbose=False)[0]
                name = f"signal:{p.names[p.probs.top1]}"
        col = COL.get(name.split(":")[-1], (255, 255, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        cv2.putText(img, f"{name} {conf:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        print(i, name, round(conf, 2), [x1, y1, x2, y2])
    cv2.imwrite(str(out / f"{i:05d}.jpg"), img)
