#!/bin/bash
# 5-class final training: E0 first, then C-RADIOv3 fusion (A5).
set -eo pipefail
# bare Ubuntu image: no conda/pip, so use a venv
if [ ! -x ~/venv/bin/python ]; then
  sudo apt-get update -y && sudo apt-get install -y python3-pip python3-venv python3-dev
  python3 -m venv ~/venv
fi
source ~/venv/bin/activate
pip install -q -U pip
cd ~/repo/insight_fm
export CLASSES=scooter,stairs,obstacle,other_vehicle,traffic_light
export YOLO_OUT=$HOME/sg/yolo5 SG_YOLO=$HOME/sg/yolo5
export RCLONE_REMOTE_RUNS=gdrive:sideguide/runs_final5
mkdir -p results
STEP=deps bash setup.sh
STEP=data bash setup.sh
STEP=selftest bash setup.sh
STEP=build bash setup.sh
python encoders.py --prefetch c-radio_v3-b
python run_all.py --phase final --ids E0 2>&1 | tee -a results/final.log
python run_all.py --phase final --ids A5 2>&1 | tee -a results/final.log
echo "== ALL FINAL RUNS DONE"
