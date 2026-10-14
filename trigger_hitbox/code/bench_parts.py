"""파트별 시간 측정 (A5000 서버): GPU만 쓰는 구성 vs 무거운 것만 GPU(VLM) + 나머지 CPU 구성.

파트: 프레임 읽기 · YOLO 검출+추적 · 배경 광류(GMC) · 신호 분류 · 트리거 판단(B) · VLM 설명 · TTS 합성
각 파트를 '혼자' 잴 때와, 실제처럼 다른 파트가 동시에 돌 때(경합)를 같이 잰다.

usage (rt venv, vLLM :8000 · TTS :7788 이 떠 있어야 함):
  python bench_parts.py VIDEO DET_PT CLS_PT OUT_JSON
"""
import json
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, "/home/ubuntu/sj/code/p0")
sys.path.insert(0, "/home/ubuntu/sj/code")
sys.path.insert(0, str(Path(__file__).parent))

N_FR = 240           # YOLO 측정 프레임 수 (8 s @ 30 fps)
N_VLM = 8
N_TTS = 12
SENTS = ["정면 자전거", "왼쪽 장애물", "초록불로 바뀜", "정면에 자전거가 있어요. 오른쪽으로 피하세요.",
         "왼쪽에 킥보드가 다가오고 있어요. 멈추세요.", "오른쪽 신호등이 초록불이에요. 건너도 돼요."]


def stats(xs):
    xs = sorted(xs)
    return dict(n=len(xs), median_ms=round(1000 * statistics.median(xs), 1),
                p95_ms=round(1000 * xs[min(len(xs) - 1, int(0.95 * len(xs)))], 1),
                mean_ms=round(1000 * statistics.mean(xs), 1))


def load_frames(video, start_s=40.0, n=N_FR):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
    fr, dec = [], []
    for _ in range(n):
        t0 = time.perf_counter()
        ok, im = cap.read()
        dec.append(time.perf_counter() - t0)
        if not ok:
            break
        fr.append(im)
    return fr, stats(dec)


def bench_yolo(weights, frames, device, threads=None):
    if threads:
        torch.set_num_threads(threads)
    m = YOLO(weights)
    for im in frames[:5]:
        m.predict(im, imgsz=640, conf=0.5, device=device, verbose=False)
    ts = []
    for im in frames:
        t0 = time.perf_counter()
        m.track(im, imgsz=640, conf=0.5, device=device, persist=True, tracker="bytetrack.yaml", verbose=False)
        if device != "cpu":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return stats(ts)


def bench_cls(weights, frames, device):
    m = YOLO(weights)
    crops = [im[600:760, 280:340] for im in frames[:60]]        # 신호등 크기 정도 조각
    for c in crops[:5]:
        m.predict(c, imgsz=96, device=device, verbose=False)
    ts = []
    for c in crops:
        t0 = time.perf_counter()
        m.predict(c, imgsz=96, device=device, verbose=False)
        ts.append(time.perf_counter() - t0)
    return stats(ts)


def bench_gmc(frames):
    import pass1_detect as p1
    mo = p1.RecordingMotion()
    ts = []
    for im in frames:
        t0 = time.perf_counter()
        mo.update(im, [])
        ts.append(time.perf_counter() - t0)
    return stats(ts[1:])


def vlm_once(client, served, frames):
    from describe import describe
    imgs = [frames[i].copy() for i in range(len(frames) - 8 * 8, len(frames), 8)]     # 0.27 s 간격 8장
    box = (300.0, 900.0, 700.0, 1500.0)
    for im in imgs:
        cv2.rectangle(im, (300, 900), (700, 1500), (0, 0, 255), 6)
    times = [-(7 - k) * 0.27 for k in range(8)]
    r = describe(client, served, dict(cls="bicycle", scenario="S2"), imgs, times, [box] * 8, "draw",
                 (frames[0].shape[1], frames[0].shape[0]), prompt="h2")
    return r["lat"]["total_s"]


def tts_once(text):
    body = json.dumps({"model": "supertonic-3", "voice": "M1", "input": text, "response_format": "wav",
                       "lang": "ko"}).encode()
    req = urllib.request.Request("http://localhost:7788/v1/audio/speech", body, {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as r:
        r.read()
    return time.perf_counter() - t0


class Load(threading.Thread):
    """경합용 배경 부하: 실제 파이프라인처럼 YOLO를 계속 돌림."""

    def __init__(self, weights, frames, device, threads=None):
        super().__init__(daemon=True)
        self.w, self.fr, self.dev, self.th = weights, frames, device, threads
        self.stop = False

    def run(self):
        if self.th:
            torch.set_num_threads(self.th)
        m = YOLO(self.w)
        k = 0
        while not self.stop:
            m.predict(self.fr[k % len(self.fr)], imgsz=640, conf=0.5, device=self.dev, verbose=False)
            k += 1


def main(video, det_pt, cls_pt, out_json):
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")
    served = client.models.list().data[0].id
    frames, dec = load_frames(video)
    R = dict(gpu=torch.cuda.get_device_name(0), cpu_threads_total=torch.get_num_threads(), decode=dec)
    print("decode", dec, flush=True)
    R["gmc_cpu"] = bench_gmc(frames)
    print("gmc", R["gmc_cpu"], flush=True)
    R["yolo_gpu"] = bench_yolo(det_pt, frames, 0)
    print("yolo_gpu", R["yolo_gpu"], flush=True)
    for th in (4, 8, 16):
        R[f"yolo_cpu_{th}t"] = bench_yolo(det_pt, frames[:120], "cpu", th)
        print(f"yolo_cpu_{th}t", R[f"yolo_cpu_{th}t"], flush=True)
    torch.set_num_threads(R["cpu_threads_total"])
    R["cls_gpu"] = bench_cls(cls_pt, frames, 0)
    R["cls_cpu"] = bench_cls(cls_pt, frames, "cpu")
    print("cls", R["cls_gpu"], R["cls_cpu"], flush=True)

    vlm_once(client, served, frames)                 # 예열
    tts_once("예열")
    R["vlm_alone"] = stats([vlm_once(client, served, frames) for _ in range(N_VLM)])
    R["tts_alone"] = stats([tts_once(SENTS[k % len(SENTS)]) for k in range(N_TTS)])
    print("vlm/tts alone", R["vlm_alone"], R["tts_alone"], flush=True)
    for name, dev, th in (("gpu_only", 0, None), ("split_cpu8", "cpu", 8)):
        ld = Load(det_pt, frames, dev, th)
        ld.start()
        time.sleep(3)
        R[f"vlm_with_yolo_{name}"] = stats([vlm_once(client, served, frames) for _ in range(N_VLM)])
        R[f"tts_with_yolo_{name}"] = stats([tts_once(SENTS[k % len(SENTS)]) for k in range(N_TTS)])
        ld.stop = True
        ld.join()
        print(name, R[f"vlm_with_yolo_{name}"], R[f"tts_with_yolo_{name}"], flush=True)
    Path(out_json).write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", out_json)


if __name__ == "__main__":
    main(*sys.argv[1:5])
