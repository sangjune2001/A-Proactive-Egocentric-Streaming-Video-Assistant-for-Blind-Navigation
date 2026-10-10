#!/usr/bin/env bash
# 서버 다운로드(pipeline.py)가 모두 끝나고 staging 업로드까지 비면 학습(server_run.sh) 시작
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
while pgrep -f "pipeline.py" > /dev/null; do sleep 60; done
while [ "$(find ~/work/code/staging -name '*.zip' 2>/dev/null | wc -l)" != "0" ]; do sleep 60; done
sleep 90   # 마지막 업로드 마무리
bash ~/work/code/server_run.sh
