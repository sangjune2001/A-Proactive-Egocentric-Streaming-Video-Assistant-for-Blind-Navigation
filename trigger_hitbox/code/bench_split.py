"""GPU/CPU 나눠 쓰기 시간 측정 (서버에서 실행).

1) YOLO를 GPU / CPU(쓰레드 수별)로 돌렸을 때 프레임당 시간
2) GPU가 VLM으로 바쁠 때(가짜 부하: 큰 행렬곱 반복) YOLO-GPU가 얼마나 느려지는지
3) 측정한 fps에서 태규 규칙(프레임 수 기준: 연속 10프레임 등)이 실제로 몇 초가 되는지

usage: python bench_split.py WEIGHTS VIDEO [n_frames=300]
"""
import json
import sys
import threading
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

weights, video = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 300
SETTINGS = dict(conf=0.5, imgsz=640, verbose=False)
STREAK_FRAMES, WINDOW_FRAMES = 10, 9          # 태규 FINAL_PARAMS

cap = cv2.VideoCapture(video)
frames = []
while len(frames) < N:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
cap.release()


def bench(device, threads=None, warm=10):
    if threads:
        torch.set_num_threads(threads)
    model = YOLO(weights)
    for f in frames[:warm]:
        model.track(f, device=device, persist=True, **SETTINGS)
    ts = []
    for f in frames:
        t0 = time.perf_counter()
        model.track(f, device=device, persist=True, **SETTINGS)
        if device != "cpu":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    ms = float(np.mean(ts))
    return dict(device=device, threads=threads or torch.get_num_threads(), ms_mean=round(ms, 1),
                ms_p95=round(float(np.percentile(ts, 95)), 1), fps=round(1000 / ms, 1),
                streak10_s=round(STREAK_FRAMES * ms / 1000, 2))


class FakeVLMLoad:
    """GPU를 VLM처럼 계속 바쁘게: 큰 fp16 행렬곱 반복."""

    def __init__(self):
        self.stop = False
        self.th = threading.Thread(target=self.run, daemon=True)

    def run(self):
        a = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
        while not self.stop:
            a @ a
            torch.cuda.synchronize()

    def __enter__(self):
        self.th.start()
        time.sleep(2)
        return self

    def __exit__(self, *e):
        self.stop = True
        self.th.join()


res = dict(frames=len(frames), cpu_threads_total=torch.get_num_threads(),
           gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, runs=[])
total = torch.get_num_threads()
if torch.cuda.is_available():
    res["runs"].append(dict(case="YOLO GPU 단독", **bench("cuda:0")))
    with FakeVLMLoad():
        res["runs"].append(dict(case="YOLO GPU + VLM 부하", **bench("cuda:0")))
for th in sorted({2, 4, 8, total} & set(range(1, total + 1))):
    res["runs"].append(dict(case=f"YOLO CPU {th}쓰레드", **bench("cpu", th)))
if torch.cuda.is_available():
    with FakeVLMLoad():
        res["runs"].append(dict(case="YOLO CPU(전체) + VLM 부하(GPU)", **bench("cpu", total)))

print(json.dumps(res, ensure_ascii=False, indent=2))
for r in res["runs"]:
    print(f"{r['case']:<32} {r['ms_mean']:7.1f} ms  p95 {r['ms_p95']:7.1f}  {r['fps']:5.1f} fps  "
          f"연속10프레임={r['streak10_s']} s")
