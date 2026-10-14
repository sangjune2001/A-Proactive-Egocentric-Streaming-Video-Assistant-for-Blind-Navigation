"""바닥 종류 분할 (Mask2Former, Mapillary Vistas): 인도 · 차도 · 횡단보도를 STRIDE 프레임마다 저장.
트리거(pass2)가 '차도 위 차량은 무시' · '횡단보도 앞' 판단에 쓴다.
usage: python pass1_surface.py VIDEO OUT_DIR   →  OUT_DIR/surface.npz (frames, labels[N,h,w] uint8)
  라벨: 0 기타 · 1 인도 · 2 차도 · 3 횡단보도 · 4 연석 · 5 고정 장애물(기둥·화단·가로수·쓰레기통 등) · 6 벽·건물
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

MODEL = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
STRIDE = 5
OUT_W, OUT_H = 135, 240
GROUPS = {1: ("Sidewalk", "Pedestrian Area", "Curb Cut"),
          2: ("Road", "Parking", "Bike Lane", "Service Lane", "Rail Track"),
          3: ("Crosswalk - Plain", "Lane Marking - Crosswalk"),
          4: ("Curb",),                                         # 연석 = 턱 · 인도 경계 (v2)
          5: ("Pole", "Utility Pole", "Street Light", "Traffic Sign Frame", "Bench", "Bike Rack", "Fire Hydrant",
              "Junction Box", "Mailbox", "Phone Booth", "Trash Can", "Barrier", "Fence", "Guard Rail", "Pothole",
              "Vegetation", "Terrain"),                          # 바닥에 붙은 고정 장애물 · 화단 · 가로수
          6: ("Wall", "Building")}


def main(video, out_dir):
    proc = AutoImageProcessor.from_pretrained(MODEL)
    net = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL).cuda().eval().half()
    lut = np.zeros(len(net.config.id2label), np.uint8)
    for g, names in GROUPS.items():
        for k, v in net.config.id2label.items():
            if v in names:
                lut[k] = g
    cap = cv2.VideoCapture(video)
    frames, labs, fi = [], [], 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        fi += 1
        if (fi - 1) % STRIDE:
            continue
        small = cv2.resize(im, (540, 960))
        x = proc(images=np.ascontiguousarray(small[:, :, ::-1]), return_tensors="pt").to("cuda")
        x["pixel_values"] = x["pixel_values"].half()
        with torch.no_grad():
            o = net(**x)
        lab = proc.post_process_semantic_segmentation(o, target_sizes=[(OUT_H, OUT_W)])[0].cpu().numpy()
        frames.append(fi)
        labs.append(lut[lab])
    np.savez_compressed(Path(out_dir) / "surface2.npz", frames=np.array(frames), labels=np.stack(labs), stride=STRIDE)
    print("done", Path(video).name, len(frames), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
