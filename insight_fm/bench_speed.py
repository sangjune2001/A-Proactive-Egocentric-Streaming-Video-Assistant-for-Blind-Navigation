"""Inference speed of finished runs: GPU and CPU, model only and end to end, FM share for fusion models.

    python bench_speed.py --runs runs/final --images ~/sg/yolo5/full/images/test --out results/speed

Run it when the GPU is otherwise idle (no training), or the numbers are meaningless.
For every <runs>/<ID>_s<seed>/weights/best.pt it measures
    gpu_fp16_ms / gpu_fp32_ms   batch 1, 640x640, network forward only (FM included for fusion; the FM itself
                                always runs under fp16 autocast on CUDA, as in training)
    gpu_fp16_bs8_img_s          throughput at batch 8
    fm_only_fp16_ms             the foundation-model encoder alone (fusion only): how much of the time is the FM
    e2e_*_ms                    Ultralytics predict on real test images: preprocess / inference / postprocess (NMS +
                                masks), GPU fp16, batch 1
    cpu_t{N}_ms                 batch 1, fp32 on N CPU threads (a stand-in for a laptop CPU, not the laptop itself)
    params / gflops             parameters and FLOPs of one 640x640 forward, FM included for fusion
Distill models are also converted with export_plain.py and measured as the plain YOLO they deploy as.
Writes <out>/speed.csv and <out>/speed.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


def timed(fn, iters: int, warmup: int, cuda: bool) -> float:
    """median ms per call"""
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts)


def gflops(model, x) -> float | None:
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            model(x)
        return round(fc.get_total_flops() / 1e9, 1)
    except Exception:
        return None


def bench_one(rid: str, wt: Path, images: list[str], threads: list[int], dev: torch.device) -> dict:
    from ultralytics import YOLO

    from encoders import encode, get_encoder

    row: dict = {"id": rid, "weights": str(wt)}
    net = YOLO(str(wt)).model.eval().fuse()
    mode, fm = getattr(net, "fm_mode", "none"), getattr(net, "fm_name", "")
    row.update(mode=mode, encoder=fm or "-", params_yolo=sum(p.numel() for p in net.model.parameters()))
    row["params_fm"] = sum(p.numel() for p in get_encoder(fm, "cpu").parameters()) if mode == "fusion" else 0

    with torch.no_grad():
        # ---- GPU
        net = net.to(dev)
        x16 = torch.rand(1, 3, 640, 640, device=dev).half()
        x32 = x16.float()
        row["gflops"] = gflops(net.float(), x32)
        row["gpu_fp32_ms"] = round(timed(lambda: net(x32), 200, 30, True), 2)
        net.half()
        row["gpu_fp16_ms"] = round(timed(lambda: net(x16), 300, 50, True), 2)
        xb = torch.rand(8, 3, 640, 640, device=dev).half()
        row["gpu_fp16_bs8_img_s"] = round(8000 / timed(lambda: net(xb), 50, 10, True), 1)
        if mode == "fusion":
            row["fm_only_fp16_ms"] = round(timed(lambda: encode(fm, x16), 300, 50, True), 2)
            row["fm_share_pct"] = round(100 * row["fm_only_fp16_ms"] / row["gpu_fp16_ms"], 1)
        del xb
        torch.cuda.empty_cache()

        # ---- CPU (fp32)
        net = net.float().cpu()
        xc = torch.rand(1, 3, 640, 640)
        for t in threads:
            torch.set_num_threads(t)
            row[f"cpu_t{t}_ms"] = round(timed(lambda: net(xc), 20, 3, False), 1)
        if mode == "fusion":
            torch.set_num_threads(threads[0])
            row[f"cpu_t{threads[0]}_fm_only_ms"] = round(timed(lambda: encode(fm, xc), 20, 3, False), 1)
        torch.set_num_threads(max(threads))

    # ---- end to end on real images (GPU fp16, batch 1)
    m = YOLO(str(wt))
    pre, inf, post = [], [], []
    for i, p in enumerate(images):
        r = m.predict(p, imgsz=640, half=True, device=dev.index or 0, verbose=False)[0]
        if i >= 10:  # skip warm-up images
            pre.append(r.speed["preprocess"])
            inf.append(r.speed["inference"])
            post.append(r.speed["postprocess"])
    row["e2e_pre_ms"] = round(statistics.median(pre), 2)
    row["e2e_infer_ms"] = round(statistics.median(inf), 2)
    row["e2e_post_ms"] = round(statistics.median(post), 2)
    row["e2e_total_ms"] = round(row["e2e_pre_ms"] + row["e2e_infer_ms"] + row["e2e_post_ms"], 2)
    row["e2e_fps"] = round(1000 / row["e2e_total_ms"], 1)
    done = wt.parent.parent / "done.json"
    if done.exists():
        d = json.loads(done.read_text())
        row["mask_map50_95"], row["mask_map50"] = d.get("mask_map50_95"), d.get("mask_map50")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/final")
    ap.add_argument("--images", required=True, help="folder of real images for the end-to-end test")
    ap.add_argument("--n-images", type=int, default=210)
    ap.add_argument("--threads", default="4,8", help="CPU thread counts to test")
    ap.add_argument("--out", default="results/speed")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda:0")
    images = sorted(str(p) for p in Path(a.images).iterdir())[: a.n_images]
    threads = [int(t) for t in a.threads.split(",")]

    rows = []
    for wt in sorted(Path(a.runs).glob("*/weights/best.pt")):
        rid = wt.parent.parent.name.split("_s")[0]
        print(f"== {rid}", flush=True)
        rows.append(bench_one(rid, wt, images, threads, dev))
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
        if rows[-1]["mode"] == "distill":  # what actually ships: the plain YOLO without distill heads
            plain = out / f"{rid}_plain.pt"
            from export_plain import main as export
            export(str(wt), str(plain))
            rows.append(bench_one(f"{rid} (plain)", plain, images, threads, dev))
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "speed.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    base = next((r for r in rows if r["id"] == "E0"), None)
    gpu_name = torch.cuda.get_device_name(0)
    md = [f"# 추론 속도 ({gpu_name}, 640x640, batch 1)\n",
          "| 모델 | 방식 | 인코더 | 파라미터(YOLO+FM) | GFLOPs | GPU FP16 (ms) | E0 대비 | GPU FP32 (ms) | "
          "GPU bs8 (img/s) | 실제 이미지 end-to-end (ms / FPS) | " + " | ".join(f"CPU {t}스레드 (ms)" for t in threads) + " |",
          "|---|---|---|---|---|---|---|---|---|---|" + "---|" * len(threads)]
    for r in rows:
        rel = f"{r['gpu_fp16_ms'] / base['gpu_fp16_ms']:.2f}x" if base else "-"
        md.append(f"| {r['id']} | {r['mode']} | {r['encoder']} | {(r['params_yolo'] + r['params_fm']) / 1e6:.1f}M | "
                  f"{r['gflops']} | {r['gpu_fp16_ms']} | {rel} | {r['gpu_fp32_ms']} | {r['gpu_fp16_bs8_img_s']} | "
                  f"{r['e2e_total_ms']} / {r['e2e_fps']} | " + " | ".join(str(r[f'cpu_t{t}_ms']) for t in threads) + " |")
    fus = [r for r in rows if r["mode"] == "fusion"]
    if fus:
        md.append("\n**fusion 모델의 FM 비중** (GPU FP16)\n")
        for r in fus:
            md.append(f"- {r['id']}: 전체 {r['gpu_fp16_ms']}ms 중 FM {r['fm_only_fp16_ms']}ms ({r['fm_share_pct']}%), "
                      f"CPU {threads[0]}스레드에서는 FM만 {r[f'cpu_t{threads[0]}_fm_only_ms']}ms")
    md.append("\nend-to-end는 실제 test 이미지에서 전처리 + 추론 + 후처리(NMS, 마스크) 중앙값. "
              "CPU 수치는 서버 CPU의 스레드 수를 제한해 잰 값으로, 노트북 CPU의 실제 속도와는 다를 수 있음.")
    (out / "speed.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
