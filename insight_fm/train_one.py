"""Train + evaluate + benchmark ONE experiment. Resumable, idempotent.

    python train_one.py --id B3 --mode distill --encoder dinov3_b+siglip2_b \
        --data /data/sg/yolo/pilot/data.yaml --epochs 30 --seed 0 --eval-split val

Writes <project>/<id>_s<seed>/done.json when everything finished; re-running skips it.
If weights/last.pt exists but done.json does not, training is resumed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--mode", default="none", choices=["none", "fusion", "distill"])
    ap.add_argument("--encoder", default="")
    ap.add_argument("--lam", type=float, default=1.0, help="distill loss weight")
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", default="yolo11s-seg.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default=str(HERE / "runs"))
    ap.add_argument("--eval-split", default="val", choices=["val", "test"])
    return ap.parse_args()


def make_sync_callback(run_dir: Path):
    remote = os.environ.get("RCLONE_REMOTE_RUNS")  # e.g. gdrive:sideguide/runs
    state = {"p": None}

    def cb(trainer):
        if not remote:
            return
        if state["p"] is not None and state["p"].poll() is None:
            return  # previous upload still running
        state["p"] = subprocess.Popen(
            ["rclone", "copy", str(run_dir), f"{remote}/{run_dir.name}", "--exclude", "*.cache"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return cb


def count_params(module) -> int:
    return sum(p.numel() for p in module.parameters()) if module is not None else 0


def bench_latency(weights: Path, imgsz: int, device: str, iters: int = 100) -> dict:
    """Batch-1 FP16 latency on this GPU, FM included for fusion models."""
    import torch
    from ultralytics import YOLO

    from encoders import _REGISTRY, get_encoder

    if not torch.cuda.is_available():
        return {}
    dev = torch.device(f"cuda:{device}" if device.isdigit() else device)
    m = YOLO(str(weights)).model.to(dev).eval().fuse().half()
    x = torch.rand(1, 3, imgsz, imgsz, device=dev).half()
    with torch.no_grad():
        for _ in range(20):
            m(x)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(iters):
            m(x)
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t) / iters * 1000
    fm_params = 0
    if getattr(m, "fm_mode", "none") == "fusion":
        fm_params = count_params(get_encoder(m.fm_name, dev))
    return {
        "latency_ms_bs1_fp16": round(ms, 2),
        "fps_bs1_fp16": round(1000 / ms, 1),
        "params_yolo": count_params(m.model),
        "params_adapter_used_at_inference": count_params(getattr(m, "fm_adapt", None))
        if getattr(m, "fm_mode", "none") == "fusion" else 0,
        "params_fm_used_at_inference": fm_params,
    }


def main():
    a = parse()
    a.project = str(Path(a.project).resolve())  # Ultralytics nests relative projects under runs/<task>/
    os.environ["FM_MODE"], os.environ["FM_ENCODER"], os.environ["FM_LAMBDA"] = a.mode, a.encoder, str(a.lam)

    import torch
    from ultralytics import YOLO

    from fm_yolo import FMSegTrainer

    name = f"{a.id}_s{a.seed}"
    run_dir = Path(a.project) / name
    done = run_dir / "done.json"
    if done.exists():
        print(f"[skip] {name} already done")
        return
    last, best = run_dir / "weights" / "last.pt", run_dir / "weights" / "best.pt"

    t0 = time.time()
    finished = False
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        finished = ck.get("epoch", -1) == -1  # Ultralytics strips optimizer and sets epoch=-1 when a run ends
    if not finished:
        if last.exists():
            print(f"[resume] {name}")
            model = YOLO(str(last))
            model.add_callback("on_fit_epoch_end", make_sync_callback(run_dir))
            model.train(resume=True, trainer=FMSegTrainer)
        else:
            print(f"[train] {name}  mode={a.mode} encoder={a.encoder or '-'}")
            model = YOLO(a.weights)
            model.add_callback("on_fit_epoch_end", make_sync_callback(run_dir))
            model.train(
                trainer=FMSegTrainer, data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                seed=a.seed, patience=a.patience, workers=a.workers, fraction=a.fraction, device=a.device,
                project=a.project, name=name, exist_ok=True, copy_paste=0.3, amp=True, plots=True,
                deterministic=False,
            )
    train_hours = (time.time() - t0) / 3600

    # ------------------------------------------------------------ evaluation
    wt = best if best.exists() else last
    met = YOLO(str(wt)).val(data=a.data, split=a.eval_split, imgsz=a.imgsz, batch=16, device=a.device,
                            project=str(run_dir), name=f"eval_{a.eval_split}", exist_ok=True, plots=True)
    names = met.names
    per_class = {}
    for j, c in enumerate(met.seg.ap_class_index):
        per_class[names[int(c)]] = {"mask_ap50": round(float(met.seg.ap50[j]), 4),
                                    "mask_ap50_95": round(float(met.seg.ap[j]), 4)}
    res = {
        "id": a.id, "seed": a.seed, "mode": a.mode, "encoder": a.encoder, "lam": a.lam,
        "data": a.data, "eval_split": a.eval_split, "weights": str(wt),
        "mask_map50": round(float(met.seg.map50), 4), "mask_map50_95": round(float(met.seg.map), 4),
        "box_map50": round(float(met.box.map50), 4), "box_map50_95": round(float(met.box.map), 4),
        "per_class": per_class,
        "missing_classes_in_eval": [names[i] for i in range(len(names)) if i not in set(map(int, met.seg.ap_class_index))],
        "train_hours_this_session": round(train_hours, 2),
    }
    try:
        res.update(bench_latency(wt, a.imgsz, a.device))
    except Exception as e:  # never lose the metrics because of the benchmark
        res["bench_error"] = f"{type(e).__name__}: {e}"
    done.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in res.items() if k != "per_class"}, indent=2, ensure_ascii=False))
    make_sync_callback(run_dir)(None)


if __name__ == "__main__":
    main()
