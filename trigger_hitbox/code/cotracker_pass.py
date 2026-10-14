"""CoTracker3(online)으로 물체 점 추적 → 박스 대신 점 움직임으로 경로를 구함.

YOLO 트랙이 처음 보일 때와 이후 1 s마다, 박스 아래쪽(발 근처)에 3x3 점을 찍어 CoTracker에 넘긴다.
usage: python cotracker_pass.py OUT_DIR START_S END_S
→ OUT_DIR/cotracker_{start}_{end}.npz  (tracks [T,N,2] 원본 좌표, vis [T,N], 점 메타)
"""
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

SEED_EVERY = 30            # 프레임 (1 s)마다 다시 점 찍기 (가려짐·이탈 대비)
GRID_X = (0.25, 0.5, 0.75)
GRID_Y = (0.6, 0.75, 0.9)  # 박스 아래쪽 = 발 근처
IN_W, IN_H = 288, 512      # CoTracker 입력 크기 (내부에서 384x512로 다시 맞춤)

out = Path(sys.argv[1])
t_start, t_end = float(sys.argv[2]), float(sys.argv[3])
data = json.loads((out / "dets.json").read_text(encoding="utf-8"))
meta, frames = data["meta"], data["frames"]
meta["video"] = os.environ.get("VIDEO", meta["video"])     # 서버에서는 영상 경로를 VIDEO로 지정
names = {int(k): v for k, v in meta["names"].items()}
fps, W, H = meta["fps"], meta["w"], meta["h"]
seg = [fr for fr in frames if t_start <= fr["t"] < t_end]
f0 = seg[0]["f"]
sx, sy = IN_W / W, IN_H / H

# --- 질의점: (구간 내 프레임 번호, x, y) + 메타 (tid, 발 위치까지 오프셋) ---
queries, qmeta, last_seed = [], [], {}
for k, fr in enumerate(seg):
    for d in fr["dets"]:
        tid = d["tid"]
        if tid is None or names[d["cls"]] == "traffic_light":
            continue
        if tid in last_seed and k - last_seed[tid] < SEED_EVERY:
            continue
        last_seed[tid] = k
        x1, y1, x2, y2 = d["box"]
        foot = ((x1 + x2) / 2, y2)
        for gx in GRID_X:
            for gy in GRID_Y:
                px, py = x1 + gx * (x2 - x1), y1 + gy * (y2 - y1)
                queries.append([k, px * sx, py * sy])
                qmeta.append([tid, k, foot[0] - px, foot[1] - py])
print(f"segment {t_start}-{t_end}s frames {len(seg)} tracks {len(last_seed)} points {len(queries)}", flush=True)

# --- 영상 읽기 ---
cap = cv2.VideoCapture(meta["video"])
cap.set(cv2.CAP_PROP_POS_FRAMES, f0 - 1)
imgs = []
for _ in seg:
    ok, fr = cap.read()
    if not ok:
        break
    imgs.append(cv2.cvtColor(cv2.resize(fr, (IN_W, IN_H)), cv2.COLOR_BGR2RGB))
cap.release()
video = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2)[None]   # uint8 유지 (메모리 1/4), 구간별로 float 변환
del imgs
T = video.shape[1]

torch.set_num_threads(max(1, torch.get_num_threads()))
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").eval().to(dev)
q = torch.tensor(queries, dtype=torch.float32)[None].to(dev)
print(f"device {dev}", flush=True)
t0 = time.time()
with torch.no_grad():
    model(video_chunk=video[:, :2].float().to(dev), is_first_step=True, queries=q, add_support_grid=True)
    step = model.step
    tracks = vis = None
    for ind in range(0, T - step, step):
        tracks, vis = model(video_chunk=video[:, ind: ind + step * 2].float().to(dev), add_support_grid=True)
        if ind // step % 5 == 0:
            el = time.time() - t0
            print(f"frame {ind + step * 2}/{T}  {el:.0f}s  ({el / (ind + step * 2):.2f}s/frame)", flush=True)
sec = time.time() - t0
tr = tracks[0, :, :len(queries)].cpu().numpy() / [sx, sy]
vi = vis[0, :, :len(queries)].cpu().numpy()
np.savez_compressed(out / f"cotracker_{int(t_start)}_{int(t_end)}.npz", tracks=tr.astype(np.float32), vis=vi,
                    qmeta=np.array(qmeta, np.float32), qframe=np.array([qq[0] for qq in queries]), f0=f0,
                    sec=sec)
print(f"done {T} frames {sec:.0f}s  {sec / T:.2f}s/frame", flush=True)
