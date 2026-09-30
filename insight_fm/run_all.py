"""Run the whole experiment plan unattended, one experiment after another on a single GPU.

    python run_all.py --phase sanity                 # ~20 min: 3 tiny runs to prove the pipeline works
    python run_all.py --phase pilot                  # 15 runs, 20% data, 30 epochs, scored on pilot val
    python run_all.py --phase tier2                  # C-RADIOv4-SO400M (fusion + distill), pilot setting
    python run_all.py --phase full                   # E0 + top-2 fusion + top-2 distill from pilot, 3 seeds, test set
    python run_all.py --phase full --ids E0,A5,B3    # or choose yourself
    python run_all.py --phase all                    # sanity -> pilot -> tier2 -> full
    python run_all.py --summary pilot                # just rebuild the summary table

Every run is idempotent: finished runs are skipped, interrupted runs resume from last.pt.
A failing run (e.g. a gated HF model without access) is logged and the loop continues.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("SG_YOLO", str(Path.home() / "sg" / "yolo"))).resolve()
RUNS = Path(os.environ.get("SG_RUNS", str(HERE / "runs"))).resolve()
RESULTS = HERE / "results"
EXTRA: list[str] = []

ENCODERS = [  # (suffix, encoder)
    ("1", "siglip2_b"),
    ("2a", "dinov2_b"),
    ("2b", "dinov3_b"),
    ("3a", "dinov2_b+siglip2_b"),
    ("3b", "dinov3_b+siglip2_b"),
    ("4", "radio_v2.5-b"),
    ("5", "c-radio_v3-b"),
]
EXPS = {"E0": ("none", "")}
EXPS.update({f"A{s}": ("fusion", e) for s, e in ENCODERS})
EXPS.update({f"B{s}": ("distill", e) for s, e in ENCODERS})
EXPS.update({"A6": ("fusion", "c-radio_v4-so400m"), "B6": ("distill", "c-radio_v4-so400m")})

PILOT_IDS = ["E0"] + [f"A{s}" for s, _ in ENCODERS] + [f"B{s}" for s, _ in ENCODERS]
TIER2_IDS = ["A6", "B6"]

# batch per run on a 24 GB RTX A5000 (FM frozen, AMP). Two ViT-B in fusion/teacher -> 12, SO400M -> 8.
def batch_for(eid: str) -> int:
    enc = EXPS[eid][1]
    if "so400m" in enc:
        return 8
    if "+" in enc:
        return 12
    return 16


def log(msg: str):
    RESULTS.mkdir(exist_ok=True)
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(RESULTS / "run_all.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(eid: str, data: Path, epochs: int, seed: int, project: Path, split: str, fraction: float = 1.0,
        extra: list[str] | None = None) -> bool:
    mode, enc = EXPS[eid]
    cmd = [sys.executable, str(HERE / "train_one.py"), "--id", eid, "--mode", mode, "--encoder", enc,
           "--data", str(data), "--epochs", str(epochs), "--seed", str(seed), "--batch", str(batch_for(eid)),
           "--project", str(project), "--eval-split", split, "--fraction", str(fraction)] + EXTRA + (extra or [])
    log(f"START {eid} s{seed} ({mode}, {enc or '-'}) -> {project.name}")
    t = time.time()
    rc = subprocess.call(cmd, cwd=HERE)
    log(f"{'DONE ' if rc == 0 else 'FAIL '} {eid} s{seed} rc={rc} {(time.time() - t) / 3600:.2f}h")
    return rc == 0


def load_done(project: Path) -> list[dict]:
    out = []
    for p in sorted(project.glob("*/done.json")):
        out.append(json.loads(p.read_text()))
    return out


def summarize(phase: str) -> list[dict]:
    project = RUNS / phase
    rows = load_done(project)
    if not rows:
        print(f"no finished runs in {project}")
        return []
    names = ["person", "bicycle", "scooter", "motorcycle", "car", "bus", "other_vehicle", "stairs", "obstacle",
             "traffic_light"]
    by_id: dict[str, list[dict]] = {}
    for r in rows:
        by_id.setdefault(r["id"], []).append(r)
    table = []
    for eid, rs in by_id.items():
        def ms(key, sub=None):
            vals = [(r["per_class"].get(sub, {}).get(key) if sub else r.get(key)) for r in rs]
            vals = [v for v in vals if v is not None]
            if not vals:
                return None, None
            return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)
        row = {"id": eid, "mode": rs[0]["mode"], "encoder": rs[0]["encoder"] or "-", "seeds": len(rs)}
        for k in ("mask_map50_95", "mask_map50", "box_map50_95"):
            m, s = ms(k)
            row[k], row[k + "_std"] = (round(m, 4), round(s, 4)) if m is not None else (None, None)
        for c in names:
            m, _ = ms("mask_ap50_95", c)
            row[f"{c}_ap"] = round(m, 4) if m is not None else None
        for k in ("latency_ms_bs1_fp16", "fps_bs1_fp16", "params_fm_used_at_inference"):
            row[k] = rs[0].get(k)
        table.append(row)
    base = next((t["mask_map50_95"] for t in table if t["id"] == "E0"), None)
    for t in table:
        t["delta_vs_E0"] = round(t["mask_map50_95"] - base, 4) if base is not None and t["mask_map50_95"] is not None else None
    table.sort(key=lambda t: -(t["mask_map50_95"] or 0))

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{phase}_summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    print(f"\n{phase} summary -> {path}")
    print(f"{'id':5s} {'mode':8s} {'encoder':22s} {'mAP50-95':>9s} {'±':>6s} {'Δ E0':>7s} {'scooter':>8s} {'ms':>6s}")
    for t in table:
        print(f"{t['id']:5s} {t['mode']:8s} {t['encoder']:22s} {t['mask_map50_95'] or 0:9.4f} "
              f"{t['mask_map50_95_std'] or 0:6.4f} {t['delta_vs_E0'] or 0:+7.4f} {t['scooter_ap'] or 0:8.4f} "
              f"{t['latency_ms_bs1_fp16'] or 0:6.1f}")
    return table


def pick_full_ids(k: int = 2) -> list[str]:
    table = summarize("pilot")
    fus = [t["id"] for t in table if t["mode"] == "fusion"][:k]
    dis = [t["id"] for t in table if t["mode"] == "distill"][:k]
    return ["E0"] + fus + dis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["sanity", "pilot", "tier2", "full", "all"])
    ap.add_argument("--ids", default="", help="comma list, overrides the default list of the phase")
    ap.add_argument("--seeds", default="0,1,2", help="seeds for the full phase")
    ap.add_argument("--pilot-epochs", type=int, default=30)
    ap.add_argument("--full-epochs", type=int, default=100)
    ap.add_argument("--summary", default="", help="only print/save the summary of a phase")
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()

    if a.summary:
        summarize(a.summary)
        return
    pilot_yaml, full_yaml = DATA_ROOT / "pilot" / "data.yaml", DATA_ROOT / "full" / "data.yaml"
    for y in (pilot_yaml, full_yaml):
        if not y.exists():
            sys.exit(f"missing {y}: run build_dataset.py first (or set SG_YOLO)")
    ids = [i for i in a.ids.split(",") if i]
    for i in ids:
        if i not in EXPS:
            sys.exit(f"unknown id {i}; known: {list(EXPS)}")

    global EXTRA
    EXTRA = ["--device", a.device, "--workers", str(a.workers), "--imgsz", str(a.imgsz)]
    phases = ["sanity", "pilot", "tier2", "full"] if a.phase == "all" else [a.phase]
    for ph in phases:
        log(f"===== phase {ph} =====")
        if ph == "sanity":
            for eid in ids or ["E0", "A2b", "B3b"]:
                ok = run(eid, pilot_yaml, 1, 0, RUNS / "sanity", "val", fraction=0.02)
                if not ok and a.phase == "all":
                    sys.exit("sanity failed: fix it before spending GPU hours (see results/run_all.log)")
        elif ph in ("pilot", "tier2"):
            for eid in ids or (PILOT_IDS if ph == "pilot" else TIER2_IDS):
                run(eid, pilot_yaml, a.pilot_epochs, 0, RUNS / "pilot", "val")
            summarize("pilot")
        elif ph == "full":
            chosen = ids or pick_full_ids()
            log(f"full phase ids: {chosen}")
            for seed in [int(s) for s in a.seeds.split(",")]:
                for eid in chosen:
                    run(eid, full_yaml, a.full_epochs, seed, RUNS / "full", "test")
            summarize("full")
    remote = os.environ.get("RCLONE_REMOTE_RUNS")
    if remote:
        subprocess.call(["rclone", "copy", str(RESULTS), f"{remote}/results"])


if __name__ == "__main__":
    main()
