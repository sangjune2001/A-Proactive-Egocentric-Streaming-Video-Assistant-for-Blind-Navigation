"""신호등 색을 상자 안 픽셀로 직접 판단 (v5_9, 사용자 제안).

분류기(signal_state_cls)는 기울어지거나 비스듬한 신호등을 '꺼짐'으로 자주 봄 (학습 8번: 277번 중 150번 꺼짐).
신호등 상자는 그대로 쓰고, 상자 안에서 '밝고 진한' 빨강 · 초록 픽셀 비율을 재서 확실한 쪽만 red / green.
둘 다 적으면 None (꺼짐 · 안 보임) → 확정 규칙(pass2 LightState)이 무시한다.
usage: python pass1_lightcolor.py VIDEO OUT_DIR   (OUT_DIR/lights.json → OUT_DIR/lights_hsv.json)
"""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

LIT_V, LIT_S = 140, 70          # 켜진 등: 밝고 채도 높음
MIN_FRAC, DOM = 0.015, 2.0      # 상자의 1.5 % 이상 + 다른 색의 2배 이상


def color_of(crop):
    if crop.size == 0:
        return None, 0.0, 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lit = (v >= LIT_V) & (s >= LIT_S)
    red = (lit & ((h <= 10) | (h >= 165))).mean()
    green = (lit & (h >= 40) & (h <= 95)).mean()           # LED 초록은 청록 쪽까지
    if red >= MIN_FRAC and red >= DOM * green:
        return "red", float(red), float(green)
    if green >= MIN_FRAC and green >= DOM * red:
        return "green", float(red), float(green)
    return None, float(red), float(green)


def main(video, out_dir):
    out = Path(out_dir)
    src = os.environ.get("LIGHT_IN", "lights.json")
    L = json.loads((out / src).read_text(encoding="utf-8"))
    want = {f["f"]: f for f in L["frames"] if f["lights"]}
    cap = cv2.VideoCapture(video)
    fi, n_red, n_green, n_none = 0, 0, 0, 0
    while want:
        ok, im = cap.read()
        if not ok:
            break
        fi += 1
        f = want.pop(fi, None)
        if f is None:
            continue
        H, W = im.shape[:2]
        for l in f["lights"]:
            x1, y1, x2, y2 = (int(round(v)) for v in l["box"])
            crop = im[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
            st, r, g = color_of(crop)
            l["cls_state"], l["cls_p"] = l["state"], l.get("p", 1.0)
            l["state"], l["p"] = (st or "off"), 1.0 if st else 0.0
            l["hsv"] = [round(r, 4), round(g, 4)]
            n_red += st == "red"
            n_green += st == "green"
            n_none += st is None
    (out / src.replace(".json", "_hsv.json")).write_text(json.dumps(L), encoding="utf-8")
    print("done", Path(video).name, "red", n_red, "green", n_green, "none", n_none, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
