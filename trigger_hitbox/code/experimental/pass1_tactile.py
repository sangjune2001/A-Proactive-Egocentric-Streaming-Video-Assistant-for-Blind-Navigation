"""1단계: 점자블록(노란색) 후보 픽셀 → 행별 가운데 점.
화면 아래 절반에서 노란색(HSV) 덩어리를 찾아, 행마다 가장 넓은 노란 구간의 가운데를 저장한다.
바닥 좌표 변환 · 인도 위 여부 · 직선 맞춤은 pass2에서 (지평선 · 걷는 방향이 프레임마다 다르므로).

usage: python pass1_tactile.py VIDEO OUT_DIR   →  OUT_DIR/tactile.json
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

STRIDE = 2           # 15 fps
SCALE = 0.25         # 1080 → 270 폭에서 찾음
ROW_STEP = 3         # 축소 영상에서 3 px마다 한 행
MIN_RUN = 3          # 축소 영상 기준 최소 노란 폭 (px)


def yellow_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (16, 90, 110), (36, 255, 255))      # 점자블록 노랑 (OpenCV H 0–180)
    m |= cv2.inRange(hsv, (15, 55, 70), (34, 255, 255))      # 색 바랜 점자블록 (황토색)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def main(video, out_dir):
    cap = cv2.VideoCapture(video)
    frames, fi = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        fi += 1
        if (fi - 1) % STRIDE:
            continue
        H, W = fr.shape[:2]
        small = cv2.resize(fr, (int(W * SCALE), int(H * SCALE)), interpolation=cv2.INTER_AREA)
        m = yellow_mask(small)
        h, w = m.shape
        pts = []
        for r in range(int(h * 0.45), h - 1, ROW_STEP):
            row = m[r] > 0
            if row.sum() < MIN_RUN:
                continue
            # 가장 긴 연속 노란 구간
            d = np.diff(np.r_[0, row.astype(np.int8), 0])
            st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
            k = int(np.argmax(en - st))
            if en[k] - st[k] < MIN_RUN:
                continue
            pts.append([round((r + 0.5) / SCALE, 1), round((st[k] + en[k]) / 2 / SCALE, 1), round((en[k] - st[k]) / SCALE, 1)])
        frames.append(dict(f=fi, pts=pts))
    Path(out_dir, "tactile.json").write_text(json.dumps(dict(stride=STRIDE, frames=frames)), encoding="utf-8")
    print("done", Path(video).name, "frames", fi, "with_yellow", sum(1 for x in frames if len(x["pts"]) >= 6))


if __name__ == "__main__":
    main(*sys.argv[1:3])
