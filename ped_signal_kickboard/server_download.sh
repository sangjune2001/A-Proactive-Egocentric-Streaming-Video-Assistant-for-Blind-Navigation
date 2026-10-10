#!/usr/bin/env bash
# 런유어AI 서버에서 남은 AI Hub 다운로드 실행 (PC보다 ~6배 빠름: 단일 20MB/s)
# 188 Validation 1줄 + 188 Training 2줄(claims로 묶음 나눠 받음) + 업로더(staging -> 드라이브)
cd ~/work/code
export PATH=$HOME/bin:$PATH
start() {  # $1=tag $2=jobs
  AIHUB_TAG=$1 AIHUB_JOBS=$2 nohup python3 -u -X utf8 pipeline.py > stdout_$1.log 2> stderr_$1.log &
}
start sv  188:Validation
start st1 188:Training
sleep 5
start st2 188:Training
nohup bash -c 'while true; do rclone move ~/work/code/staging "gdrive:aihub_신호등_킥보드" --exclude "*.tmp" \
  --transfers 4 --min-age 10s --delete-empty-src-dirs --retries 5 --log-level ERROR --log-file ~/work/code/uploader.log; \
  sleep 60; done' > /dev/null 2>&1 &
echo started
