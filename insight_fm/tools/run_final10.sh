#!/bin/bash
# 10-class final training on the same 8,120 pilot-train frames (all scooter/stairs/traffic_light frames),
# E0 then A5, 100 epochs, scored on the full test set. Waits for the 5-class run and its speed benchmark.
set -eo pipefail
source ~/venv/bin/activate
until grep -q SPEED_DONE ~/speed.log 2>/dev/null; do sleep 60; done
cd ~/repo/insight_fm
cp results/final_summary.csv results/final5_summary.csv 2>/dev/null || true   # keep the 5-class table
YOLO_OUT=$HOME/sg/yolo10 STEP=build bash setup.sh                              # all 10 classes, same split
export SG_YOLO=$HOME/sg/yolo10 SG_RUNS=$HOME/repo/insight_fm/runs10
export RCLONE_REMOTE_RUNS=gdrive:sideguide/runs_final10
python run_all.py --phase final --ids E0 2>&1 | tee -a results/final10.log
cp results/final_summary.csv results/final10_summary.csv
python eval_errors.py --weights runs10/final/E0_s0/weights/best.pt --data ~/sg/yolo10/full --out analysis/errors_E0_10cls
python run_all.py --phase final --ids A5 2>&1 | tee -a results/final10.log
cp results/final_summary.csv results/final10_summary.csv
rclone copy results gdrive:sideguide/runs_final10/results
echo "== ALL FINAL10 RUNS DONE"
