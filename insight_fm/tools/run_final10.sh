#!/bin/bash
# 10-class final training on the same 8,120 pilot-train frames (all scooter/stairs/traffic_light frames),
# scored on the full test set: E0 100 epochs (patience 30) and A5 70 epochs (patience 15), trained at the same time.
# Inference speed is measured afterwards, one model at a time on the idle GPU.
set -eo pipefail
source ~/venv/bin/activate
cd ~/repo/insight_fm
[ -f ~/sg/yolo10/pilot/data.yaml ] || YOLO_OUT=$HOME/sg/yolo10 STEP=build bash setup.sh   # all 10 classes, same split
export SG_YOLO=$HOME/sg/yolo10 SG_RUNS=$HOME/repo/insight_fm/runs10
export RCLONE_REMOTE_RUNS=gdrive:sideguide/runs_final10
[ -f runs10/final/E0_s0/done.json ] || python run_all.py --phase final --final-epochs 100 --final-patience 30 --ids E0 > results/final10_E0.log 2>&1 &
[ -f runs10/final/A5_s0/done.json ] || pgrep -f "ids A5" >/dev/null || python run_all.py --phase final --final-epochs 70 --final-patience 15 --ids A5 > results/final10_A5.log 2>&1 &
wait
while pgrep -f "final-epochs 70 --final-patience 15 --ids A5" >/dev/null; do sleep 60; done   # A5 may run outside this script
python run_all.py --summary final
cp results/final_summary.csv results/final10_summary.csv
python eval_errors.py --weights runs10/final/E0_s0/weights/best.pt --data ~/sg/yolo10/full --out analysis/errors_E0
python eval_errors.py --weights runs10/final/A5_s0/weights/best.pt --data ~/sg/yolo10/full --out analysis/errors_A5
python bench_speed.py --runs runs10/final --images ~/sg/yolo10/full/images/test --out results/speed10
rclone copy results gdrive:sideguide/runs_final10/results
rclone copy analysis gdrive:sideguide/runs_final10/analysis
echo "== ALL FINAL10 RUNS DONE"
