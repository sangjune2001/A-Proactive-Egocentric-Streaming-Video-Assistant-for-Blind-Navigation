# 1단계(검출·신호등·바닥 분할)를 테스트 영상 3개에 한 번 돌림.  bash run_pass1.sh  (code/ 에서)
. ~/fast/venv/rt/bin/activate; export HF_HOME=~/fast/hf
for v in test_3 test_1 test_2; do
  O=../out/$v; mkdir -p $O
  python pass1_detect.py final_10class_best.pt ../data/test/$v.mp4 $O 2>&1 | grep -E "^done|Error|Trace"
  python pass1_lowconf.py final_10class_best.pt ../data/test/$v.mp4 $O 2>&1 | grep -E "^done|Error|Trace"
  python pass1_lights.py final_10class_best.pt signal_state_cls_best.pt ../data/test/$v.mp4 $O 2>&1 | grep -E "^done|Error|Trace"
  python pass1_lightcolor.py ../data/test/$v.mp4 $O 2>&1 | grep -E "^done|Error|Trace"
  python pass1_surface.py ../data/test/$v.mp4 $O 2>&1 | grep -E "^done|Error|Trace"
  touch $O/PASS1_DONE; echo "P1DONE $v"
done
