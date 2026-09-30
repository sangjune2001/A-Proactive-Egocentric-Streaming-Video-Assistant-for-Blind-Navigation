"""Part of selftest: start a 3-epoch distill run, kill it after epoch 1, then check train_one resumes."""

import os
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
TMP = Path(os.environ.get("SELFTEST_DIR", HERE / "_selftest"))


def main():
    args = [sys.executable, "train_one.py", "--id", "Bres", "--mode", "distill", "--encoder", "dummy",
            "--data", str(TMP / "yolo" / "pilot" / "data.yaml"), "--epochs", "3", "--imgsz", "128", "--batch", "4",
            "--workers", "0", "--device", "cpu", "--project", str(TMP / "runs")]
    last = TMP / "runs" / "Bres_s0" / "weights" / "last.pt"
    p = subprocess.Popen(args, cwd=HERE)
    t = time.time()
    while time.time() - t < 900:
        if last.exists():
            time.sleep(2)
            try:
                ep = torch.load(last, map_location="cpu", weights_only=False).get("epoch", -1)
            except Exception:
                continue
            if ep >= 0:
                p.kill()
                p.wait()
                print(f"killed after saving epoch {ep}", flush=True)
                break
        if p.poll() is not None:
            sys.exit("run ended before it could be interrupted")
        time.sleep(1)
    out = subprocess.run(args, cwd=HERE, capture_output=True, text=True)
    print(out.stdout[-1500:])
    if out.returncode != 0:
        print(out.stderr[-3000:])
        sys.exit("resume run failed")
    if "[resume]" not in out.stdout:
        sys.exit("resume path was not taken")
    if not (TMP / "runs" / "Bres_s0" / "done.json").exists():
        sys.exit("resume run did not finish")
    print("RESUME OK")


if __name__ == "__main__":
    main()
