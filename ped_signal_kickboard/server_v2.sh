#!/usr/bin/env bash
# v2: v1 best.pt에서 이어서 짧게 추가 학습. 189 ×10, 188 1만, 614 5천으로 비율 조정(부분 라벨로 떨어진 기존 클래스 회복)
# 끝나면 seg10 vs v2 평가 → docs_v2/RESULTS.md, 드라이브 업로드
set -e
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
W=~/work; C=$W/code; DS=$W/ds
python3 $C/make_list.py $DS "$W/drive/aihub_신호등_킥보드/71579_신호등신호정보/Validation" $DS/data_v2.yaml \
  --repeat 189_=10 --cap 188_=10000 --cap 614_=5000 --hot 5
python3 $C/train_seg.py $W/runs/seg10_plus_tl_scooter/weights/best.pt $DS/data_v2.yaml $W/runs \
  --name seg10_v2_balanced --epochs 15 --lr0 0.002 --warmup 1 --close-mosaic 5 --workers 16
mkdir -p $W/docs_v2
python3 $C/eval_all.py $DS $W/runs/seg10_v2_balanced/weights/best.pt \
  $W/drive/aihub189_yolo/runs/seg10/weights/best.pt $W/runs/signal_state_cls/weights/best.pt $W/docs_v2
python3 $C/show_results.py $W/runs/seg10_v2_balanced/results.csv 15 > $W/docs_v2/epochs.txt
rclone copy $W/runs/seg10_v2_balanced "gdrive:aihub_ped_kick_yolo/runs_server/seg10_v2_balanced" --transfers 4
rclone copy $W/docs_v2 "gdrive:aihub_traffic_code/docs_v2" --transfers 4
echo V2_DONE
