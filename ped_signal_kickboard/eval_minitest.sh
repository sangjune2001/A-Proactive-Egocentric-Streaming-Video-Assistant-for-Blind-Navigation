#!/usr/bin/env bash
# eval_all.py 사전 시험: 각 부분집합 8장짜리 미니 데이터셋 + 현재 last.pt
cd ~/work
rm -rf mini minidocs
mkdir -p mini/images/val mini/labels/val mini/images/train
for p in 189_ 188_ 614_ rf_; do
  ls ds/images/val | grep "^$p" | head -8 | while read f; do
    ln -s ~/work/ds/images/val/$f mini/images/val/$f
    ln -s ~/work/ds/labels/val/${f%.jpg}.txt mini/labels/val/${f%.jpg}.txt
  done
done
ls ds/images/train | grep '^71579_' | head -2 | while read f; do ln -s ~/work/ds/images/train/$f mini/images/train/$f; done
ln -sfn ~/work/cls_ds ~/work/cls_ds_link
# eval_all 은 <ds>/../cls_ds 를 쓰므로 mini 옆에도 연결
python3 code/eval_all.py ~/work/mini ~/work/runs/seg10_plus_tl_scooter/weights/last.pt \
  ~/work/drive/aihub189_yolo/runs/seg10/weights/best.pt ~/work/runs/signal_state_cls/weights/best.pt ~/work/minidocs
echo EVALTEST_EXIT $?
