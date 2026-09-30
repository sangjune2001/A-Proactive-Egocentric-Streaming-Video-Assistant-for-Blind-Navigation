"""CPU self-test on synthetic data: dataset build -> baseline / fusion / distill 1-epoch runs -> eval ->
resume -> plain export. Uses the 'dummy' encoder and random-init HF encoders, so no downloads beyond
yolo11s-seg.pt. Run it before touching the real data:  python selftest.py
"""

import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TMP = Path(os.environ.get("SELFTEST_DIR", HERE / "_selftest"))


def make_data():
    import cv2

    if TMP.exists():
        shutil.rmtree(TMP)
    img_dir = TMP / "imgs" / "shard_000"
    img_dir.mkdir(parents=True)
    rng = random.Random(0)
    recs = []
    for v in range(14):
        folder = f"Polygon_1_new/MP_SEL_{v:04d}"
        for f in range(6):
            key = f"{folder.replace('/', '_')}__frame_{f:05d}"
            im = np.full((256, 256, 3), 90, np.uint8)
            polys = []
            for _ in range(rng.randint(1, 4)):
                c = rng.choice(["person", "car", "scooter", "bollard", "traffic_light", "truck", "bicycle"])
                x, y, w, h = rng.randint(10, 150), rng.randint(10, 150), rng.randint(30, 90), rng.randint(30, 90)
                pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
                cv2.fillPoly(im, [np.array(pts)], (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)))
                polys.append({"label": c, "points": ";".join(f"{px},{py}" for px, py in pts)})
            cv2.imwrite(str(img_dir / f"{key}.jpg"), im)
            recs.append({"key": key, "width": 256, "height": 256, "polygons": polys})
    with open(TMP / "labels_all.jsonl", "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")


def sh(*args, env=None):
    print("\n$", " ".join(map(str, args)), flush=True)
    r = subprocess.run(list(map(str, args)), cwd=HERE, env={**os.environ, **(env or {})})
    if r.returncode != 0:
        sys.exit(f"SELFTEST FAILED at: {' '.join(map(str, args))}")


def main():
    make_data()
    py = sys.executable
    sh(py, "build_dataset.py", "--jsonl", TMP / "labels_all.jsonl", "--images", TMP / "imgs", "--inspect")
    sh(py, "build_dataset.py", "--jsonl", TMP / "labels_all.jsonl", "--images", TMP / "imgs",
       "--out", TMP / "yolo", "--pilot-frac", "0.5", "--split-tries", "5")

    env = {"FM_RANDOM_INIT": "1"}
    common = ["--data", TMP / "yolo" / "pilot" / "data.yaml", "--epochs", "1", "--imgsz", "128", "--batch", "4",
              "--workers", "0", "--device", "cpu", "--project", TMP / "runs", "--eval-split", "val"]
    for eid, mode, enc in [("E0", "none", ""), ("Afus", "fusion", "dummy"), ("Bdis", "distill", "dummy"),
                           ("Bcat", "distill", "dinov3_b+siglip2_b"), ("Afus2", "fusion", "dinov2_b")]:
        sh(py, "train_one.py", "--id", eid, "--mode", mode, "--encoder", enc, *common, env=env)
        res = json.loads((TMP / "runs" / f"{eid}_s0" / "done.json").read_text())
        print(f"   {eid}: mask mAP50-95={res['mask_map50_95']}")

    # idempotency: second call must skip
    sh(py, "train_one.py", "--id", "E0", "--mode", "none", *common, env=env)

    # resume: delete done.json and pretend training stopped mid-way (2-epoch run killed after epoch 1)
    sh(py, "-c", "import sys; sys.path.insert(0,'.'); import selftest_resume; selftest_resume.main()", env=env)

    # plain export of a distill model and validation of the plain file with stock Ultralytics
    plain = TMP / "deploy" / "Bdis_plain.pt"
    sh(py, "export_plain.py", TMP / "runs" / "Bdis_s0" / "weights" / "best.pt", plain)
    sh(py, "-c", f"from ultralytics import YOLO; m=YOLO(r'{plain}'); "
                 f"r=m.val(data=r'{TMP / 'yolo' / 'pilot' / 'data.yaml'}', imgsz=128, batch=4, device='cpu', "
                 f"workers=0, plots=False); print('plain val mAP50-95', r.seg.map); "
                 f"print(type(m.model).__name__)")
    print("\nSELFTEST PASSED")


if __name__ == "__main__":
    main()
