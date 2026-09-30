"""Turn a trained distill (or baseline) checkpoint into a PLAIN YOLO11s-seg .pt for deployment.

The distill heads are training-only, so the result has exactly the YOLO11s-seg architecture and can be
exported to ONNX / OpenVINO for the laptop demo like any Ultralytics model:

    python export_plain.py runs/full/B3b_s0/weights/best.pt deploy/B3b_plain.pt
    yolo export model=deploy/B3b_plain.pt format=openvino imgsz=640

Fusion (Track A) models need the FM at inference and cannot be made plain.
"""

import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(src: str, dst: str):
    from ultralytics import __version__
    from ultralytics.nn.tasks import SegmentationModel

    ck = torch.load(src, map_location="cpu", weights_only=False)
    m = (ck.get("ema") or ck["model"]).float()
    mode = getattr(m, "fm_mode", "none")
    if mode == "fusion":
        sys.exit("fusion model: it needs the foundation model at inference, cannot export as plain YOLO")
    plain = SegmentationModel(deepcopy(m.yaml), ch=m.yaml.get("channels", 3), nc=len(m.names), verbose=False)
    sd = {k: v for k, v in m.state_dict().items() if not k.startswith("fm_adapt.")}
    missing, unexpected = plain.load_state_dict(sd, strict=False)
    if missing or unexpected:
        sys.exit(f"state_dict mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    plain.names = m.names
    plain.args = getattr(m, "args", {})
    plain.stride = m.stride
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"date": datetime.now().isoformat(), "version": __version__, "epoch": -1, "best_fitness": None,
                "model": plain.half(), "ema": None, "updates": None, "optimizer": None,
                "train_args": ck.get("train_args", {}), "train_metrics": ck.get("train_metrics"),
                "train_results": None}, dst)
    print(f"{src} ({mode}) -> {dst}  [{sum(p.numel() for p in plain.parameters()) / 1e6:.2f}M params]")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
