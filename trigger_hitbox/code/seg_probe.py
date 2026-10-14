"""인도/차도 분할 시험: SegFormer (Cityscapes) → road(차도) · sidewalk(인도) 색칠.
usage: python seg_probe.py VIDEO OUT_JPG t1 t2 ...
"""
import sys
import time

import cv2
import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

MODEL = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
COL = {0: (255, 0, 160), 1: (0, 220, 255), 9: (0, 160, 0)}      # road · sidewalk · terrain (BGR)


def main(video, out_jpg, *times):
    proc = SegformerImageProcessor.from_pretrained(MODEL)
    net = SegformerForSemanticSegmentation.from_pretrained(MODEL).cuda().eval().half()
    cap = cv2.VideoCapture(video)
    tiles = []
    for ts in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(ts) * 1000)
        ok, im = cap.read()
        small = cv2.resize(im, (540, 960))
        x = proc(images=np.ascontiguousarray(small[:, :, ::-1]), return_tensors="pt")["pixel_values"].cuda().half()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            lg = net(pixel_values=x).logits
        lab = torch.nn.functional.interpolate(lg.float(), size=small.shape[:2], mode="bilinear").argmax(1)[0].cpu().numpy()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        vis = small.copy()
        for k, c in COL.items():
            vis[lab == k] = (0.45 * np.array(c) + 0.55 * vis[lab == k]).astype(np.uint8)
        cv2.putText(vis, f"{ts}s {ms:.0f}ms", (10, 40), 0, 1.1, (255, 255, 255), 3)
        tiles.append(vis)
    cv2.imwrite(out_jpg, cv2.resize(np.hstack(tiles), None, fx=0.5, fy=0.5))


if __name__ == "__main__":
    main(*sys.argv[1:])
