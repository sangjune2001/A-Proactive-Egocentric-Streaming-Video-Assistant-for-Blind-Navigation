"""신호 상태 분류 모델(yolo11n-cls) 학습. 끊기면 다시 실행 → resume.
사용: python3 train_cls.py <cls 데이터 폴더> <결과폴더> [--epochs 30] [--imgsz 96]"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("data")
ap.add_argument("project")
ap.add_argument("--name", default="signal_state_cls")
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--imgsz", type=int, default=96)
args = ap.parse_args()
last = Path(args.project) / args.name / "weights" / "last.pt"
if last.exists():
    YOLO(str(last)).train(resume=True)
else:
    YOLO("yolo11n-cls.pt").train(data=args.data, imgsz=args.imgsz, epochs=args.epochs, batch=256, patience=8,
                                 project=args.project, name=args.name, exist_ok=True, workers=16,
                                 fliplr=0.0)  # 좌우반전 금지(신호 그림 방향 보존)
