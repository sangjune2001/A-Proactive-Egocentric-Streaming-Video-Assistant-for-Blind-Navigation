#!/usr/bin/env bash
# 런유어AI(RTX A5000)에서: 드라이브 데이터 받기 -> 데이터셋 만들기 -> seg10 best.pt(yolo11n-seg) 그대로 이어서 학습 -> 결과 드라이브로
# 다시 실행하면 이어서 진행(이미 받은 파일/데이터셋은 건너뛰고, 학습은 last.pt에서 resume)
set -euo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
W=~/work; D=$W/drive; DS=$W/ds; C=$W/code
mkdir -p $D $DS

# 1) 드라이브에서 받기
rclone copy "gdrive:aihub_신호등_킥보드" "$D/aihub_신호등_킥보드" --transfers 8 --checkers 16
rclone copy "gdrive:aihub189_yolo" "$D/aihub189_yolo" --include "state.json" --include "labels/**" --include "runs/seg10/weights/best.pt" --transfers 16
rclone copy "gdrive:sideguide/polygon" "$D/sideguide/polygon" --include "P1.zip"

# 2) 데이터셋: 기존 10클래스 그대로. 새 데이터는 traffic_light / scooter 만(원래 라벨, 사각형 폴리곤), 기존 189(폴리곤) 포함
[ -f $DS/data.yaml ] || python3 $C/prepare_yolo.py "$D/aihub_신호등_킥보드" $DS --max-188 30000 --max-188-val 3000 --max-614 20000
python3 $C/build_189.py $D $DS
python3 -m pip install -q pyyaml
python3 $C/rf_to_yolo.py $DS $W/rf_cache   # 샘플 확인된 한국 보행신호등 5종 + 킥보드(Roboflow)
sed -i "s#^path: .*#path: $DS#" $DS/data.yaml
# 신호가 바뀌는 장면(71579 보행신호 red<->green 변화 클립) 프레임을 5배로 강조
python3 $C/oversample.py $DS "$D/aihub_신호등_킥보드/71579_신호등신호정보/Validation/clips_pedestrian_signal.csv" --factor 5

# 2-1) 신호 상태 분류 모델(red/green/off/vehicle): crop 만들고 학습 — 탐지 학습과 동시에 백그라운드로 (GPU 소량 사용)
( python3 $C/build_cls.py "$D/aihub_신호등_킥보드" $W/rf_cache $W/cls_ds && \
  python3 $C/train_cls.py $W/cls_ds $W/runs --epochs 30 ) > $W/cls.log 2>&1 &
CLS_PID=$!

# 3) 학습: seg10 best.pt(yolo11n-seg) 그대로 이어서
python3 $C/train_seg.py "$D/aihub189_yolo/runs/seg10/weights/best.pt" $DS/data.yaml $W/runs --imgsz 1280 --epochs 30 --workers 16

wait $CLS_PID || true

# 4) 결과를 드라이브로 (탐지 runs/seg10_plus_tl_scooter, 분류 runs/signal_state_cls, 결합 추론은 infer.py)
rclone copy $W/runs "gdrive:aihub_ped_kick_yolo/runs_server" --transfers 4
echo DONE
