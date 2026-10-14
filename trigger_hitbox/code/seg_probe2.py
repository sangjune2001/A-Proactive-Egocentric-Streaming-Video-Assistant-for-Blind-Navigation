"""인도/차도 분할 시험 2: Mask2Former (Mapillary Vistas) — 인도 · 차도 · 연석 · 횡단보도를 따로 배운 모델.
usage: python seg_probe2.py VIDEO OUT_JPG t1 t2 ...
"""
import sys
import time

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

MODEL = "facebook/mask2former-swin-large-mapillary-vistas-semantic"


def main(video, out_jpg, *times):
    proc = AutoImageProcessor.from_pretrained(MODEL)
    net = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL).cuda().eval()
    names = net.config.id2label
    want = {"Road": (255, 0, 160), "Sidewalk": (0, 220, 255), "Curb": (0, 120, 255),
            "Crosswalk - Plain": (255, 255, 255), "Lane Marking - Crosswalk": (255, 255, 255),
            "Pedestrian Area": (0, 255, 120), "Parking": (200, 0, 255), "Bike Lane": (255, 160, 0)}
    ids = {k: v for k, v in names.items() if v in want}
    print("classes", ids)
    cap = cv2.VideoCapture(video)
    tiles = []
    for ts in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(ts) * 1000)
        ok, im = cap.read()
        small = cv2.resize(im, (540, 960))
        x = proc(images=np.ascontiguousarray(small[:, :, ::-1]), return_tensors="pt").to("cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            o = net(**x)
        lab = proc.post_process_semantic_segmentation(o, target_sizes=[small.shape[:2]])[0].cpu().numpy()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        vis = small.copy()
        for k, nm in ids.items():
            m = lab == k
            vis[m] = (0.5 * np.array(want[nm]) + 0.5 * vis[m]).astype(np.uint8)
        cv2.putText(vis, f"{ts}s {ms:.0f}ms", (10, 40), 0, 1.1, (255, 255, 255), 3)
        tiles.append(vis)
    cv2.imwrite(out_jpg, cv2.resize(np.hstack(tiles), None, fx=0.5, fy=0.5))


if __name__ == "__main__":
    main(*sys.argv[1:])
