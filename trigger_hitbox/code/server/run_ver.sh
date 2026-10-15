# 버전 하나를 영상 하나에: 판정 → 히트박스 영상 → SJ 실행 시스템(음성) → 최종 좌우 비교 영상
# usage: bash run_ver.sh test_3 v5_29 versions/pass2_trigger_v5_29.py [sapi|server]
#   sapi   = Heami (노트북에서 heami_pieces.ps1로 미리 합성해 캐시에 넣어둔 조각만 씀)
#   server = Supertonic-3 (supertonic serve --model supertonic-3 --port 7788)
. ~/fast/venv/rt/bin/activate
v=$1; VER=$2; PYF=$3; TTS=${4:-sapi}; TAG=${VER//_/}
O=../out_$VER/$v; mkdir -p $O
until [ -f ../out/$v/PASS1_DONE ]; do sleep 20; done
for f in dets.json dets_low.json flows.npz lights.json lights_hsv.json events_B0.jsonl surface2.npz; do ln -sf $PWD/../out/$v/$f $O/$f; done
FF=$(python -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
cp notes_$VER.txt $O/ 2>/dev/null || cp ../notes/notes_$VER.txt $O/
CAM_H=1.77 PANELS=B VERSION=$VER VIDEO=$PWD/../data/test/$v.mp4 python $PYF $O | tail -1 && $FF -y -loglevel error -i $O/hitbox_B_$VER.mp4 -c:v libx264 -crf 26 -preset fast -pix_fmt yuv420p $O/${v}_히트박스_${VER}_수정본.mp4 &
DUMP=1 CAM_H=1.77 PANELS=B VIDEO=$PWD/../data/test/$v.mp4 python $PYF $O > /dev/null
wait
mkdir -p ~/sj/results/events/trigB_$TAG && cp $O/events_runtime.json ~/sj/results/events/trigB_$TAG/$v.json
if [ "$TTS" = sapi ]; then TA="--tts sapi"; else TA="--tts server --tts-url http://localhost:7788/v1 --tts-model supertonic-3 --tts-voice M1"; fi
cd ~/sj/code/runtime && python run_demo.py --clip $v --perception yolo --tracks F10 --device 0 --trigger replay:trigB_$TAG --vlm none $TA --run $TAG > ~/work/e2e_${TAG}_$v.log 2>&1
cd ~/work/trigger_proto/code && python final_render.py $O ~/sj/results/runtime/$TAG/$v $O/hitbox_B.mp4 $O/${v}_최종_음성_비교_${VER}.mp4
echo "VERDONE $VER $v"
