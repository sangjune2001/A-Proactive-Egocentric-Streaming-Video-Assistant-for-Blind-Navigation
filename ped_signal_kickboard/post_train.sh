#!/usr/bin/env bash
# server_run.sh(학습)가 끝나면 자동 평가 → RESULTS.md + 이미지 → 드라이브(aihub_traffic_code/docs) 업로드
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
W=~/work; C=$W/code
while pgrep -f server_run.sh > /dev/null; do sleep 120; done
mkdir -p $W/docs
python3 $C/eval_all.py $W/ds $W/runs/seg10_plus_tl_scooter/weights/best.pt \
  $W/drive/aihub189_yolo/runs/seg10/weights/best.pt $W/runs/signal_state_cls/weights/best.pt $W/docs
python3 $C/show_results.py $W/runs/seg10_plus_tl_scooter/results.csv 30 > $W/docs/epochs.txt
rclone copy $W/docs "gdrive:aihub_traffic_code/docs" --transfers 4
rclone copy $W/runs "gdrive:aihub_ped_kick_yolo/runs_server" --transfers 4
echo POST_DONE
