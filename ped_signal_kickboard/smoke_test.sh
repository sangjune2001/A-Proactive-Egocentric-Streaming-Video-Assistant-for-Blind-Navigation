#!/usr/bin/env bash
# 본 학습 전에 작은 데이터(Roboflow 변환 결과 rftest)로 seg10 best.pt 이어 학습 1 epoch + 분류 1 epoch 시험
set -e
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
W=~/work; C=$W/code; T=$W/smoke
rm -rf $T; mkdir -p $T
rclone copy "gdrive:aihub189_yolo/runs/seg10/weights" $W/drive/aihub189_yolo/runs/seg10/weights --include best.pt
cat > $T/data.yaml <<EOF
path: $W/rftest
train: images/train
val: images/val
names:
  0: person
  1: bicycle
  2: scooter
  3: motorcycle
  4: car
  5: bus
  6: other_vehicle
  7: obstacle
  8: stairs
  9: traffic_light
EOF
python3 - <<EOF
from ultralytics import YOLO
m = YOLO("$W/drive/aihub189_yolo/runs/seg10/weights/best.pt")
print("names", m.names)
m.train(data="$T/data.yaml", imgsz=1280, epochs=1, batch=-1, project="$T", name="seg", exist_ok=True, workers=8, fraction=0.2)
EOF
echo SMOKE_SEG_OK
