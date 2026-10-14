"""클래스 없는 장애물 시험: 단안 깊이(Depth Anything V2 Small)로 '바닥보다 튀어나온 것'을 찾는다.

바닥이면 역깊이(disparity)가 화면 행 y에 대해 선형: 1/Z = (y − y_hor)/(f·h).
화면 아래 가운데(대부분 바닥)로 disparity = a·(y − y_hor) + b 를 맞추고,
같은 행의 바닥보다 확실히 가까운(disparity가 큰) 덩어리 = 바닥에서 솟은 물체 → 내 진행 영역 안이면 장애물.
usage: python depth_probe.py VIDEO OUT_DIR t1 t2 ...
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import pipeline

F_EQ, CAM_H = 26.0, 1.77


def main(video, out_dir, *times):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=0)
    cap = cv2.VideoCapture(video)
    for ts in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(ts) * 1000)
        ok, im = cap.read()
        H, W = im.shape[:2]
        f = F_EQ / 36 * max(W, H)
        y_hor = 0.47 * H
        from PIL import Image
        torch.cuda.synchronize()
        import time
        t0 = time.perf_counter()
        d = pipe(Image.fromarray(im[:, :, ::-1]))["predicted_depth"]
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        d = cv2.resize(d.squeeze().cpu().numpy().astype(np.float32), (W, H))
        # 바닥 맞추기: 화면 아래 30 % · 가운데 40 % 의 행별 중앙값
        ys = np.arange(int(0.70 * H), H - 5, 8)
        cols = slice(int(0.3 * W), int(0.7 * W))
        med = np.array([np.median(d[y, cols]) for y in ys])
        A = np.c_[ys - y_hor, np.ones_like(ys)]
        a, b = np.linalg.lstsq(A, med, rcond=None)[0]
        yy = np.arange(H)[:, None].astype(np.float32)
        ground = a * np.maximum(yy - y_hor, 1) + b
        # 바닥보다 25 % 이상 가까움 + 지평선 아래 + 영역 ±1.3 m 안 (발 행 기준 대략 열 범위)
        excess = (d - ground) / np.maximum(ground, 1e-3)
        mask = (excess > 0.25) & (yy > y_hor + 0.02 * H)
        Zrow = f * CAM_H / np.maximum(yy - y_hor, 1)
        half_px = 1.3 * f / Zrow                     # 행마다 영역 반폭(px)
        xx = np.arange(W)[None, :]
        mask &= np.abs(xx - W / 2) < half_px
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
        vis = im.copy()
        heat = cv2.applyColorMap(np.uint8(np.clip(excess, 0, 1) * 255), cv2.COLORMAP_JET)
        vis = cv2.addWeighted(vis, 0.6, heat, 0.4, 0)
        for k in range(1, n):
            x, y, w, h, area = st[k]
            if area < 0.004 * W * H:
                continue
            yb = y + h                                    # 덩어리 아래 끝 = 바닥 닿는 곳
            Z = f * CAM_H / max(yb - y_hor, 1)
            cv2.rectangle(vis, (x, y), (x + w, yb), (0, 0, 255), 6)
            cv2.putText(vis, f"obstacle? {Z:.1f}m", (x, max(40, y - 12)), 0, 1.4, (0, 0, 255), 4)
        cv2.putText(vis, f"t={ts}s depth {ms:.0f}ms", (20, 60), 0, 1.6, (255, 255, 255), 4)
        cv2.imwrite(str(out / f"depth_{ts}.jpg"), cv2.resize(vis, (W // 2, H // 2)))
        print(ts, f"{ms:.0f} ms", "blobs", n - 1)


if __name__ == "__main__":
    main(*sys.argv[1:])
