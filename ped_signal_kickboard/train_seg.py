"""기존 189 모델(yolo11n-seg, seg10 best.pt)을 그대로 이어서 학습. 클래스 10개 동일 → 헤드까지 가중치 전부 이어받음.
새 데이터의 bbox는 사각형 폴리곤 라벨(prepare_yolo.py)로 들어간다. 사용 시에는 결과의 boxes만 쓰면 됨.
끊기면 다시 실행 → last.pt에서 resume.
사용: python train_seg.py <seg10 best.pt> <data.yaml> <결과폴더> [--imgsz 1280] [--epochs 30]"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("weights")
ap.add_argument("data")
ap.add_argument("project")
ap.add_argument("--name", default="seg10_plus_tl_scooter")
ap.add_argument("--imgsz", type=int, default=1280)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--batch", type=float, default=-1)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--lr0", type=float, default=0.01)
ap.add_argument("--warmup", type=float, default=3.0)
ap.add_argument("--close-mosaic", type=int, default=10)
args = ap.parse_args()

last = Path(args.project) / args.name / "weights" / "last.pt"
if last.exists():
    YOLO(str(last)).train(resume=True)
else:
    YOLO(args.weights).train(
        data=args.data, imgsz=args.imgsz, epochs=args.epochs,
        batch=int(args.batch) if args.batch >= 1 else args.batch,
        patience=8, project=args.project, name=args.name, exist_ok=True, save_period=1, workers=args.workers,
        lr0=args.lr0, warmup_epochs=args.warmup, close_mosaic=args.close_mosaic,
    )
