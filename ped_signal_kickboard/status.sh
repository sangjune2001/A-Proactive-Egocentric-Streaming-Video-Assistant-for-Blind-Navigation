#!/usr/bin/env bash
# 서버 다운로드/학습 상태 요약
cd ~/work/code
date '+%m-%d %H:%M'
for t in sv st1 st2; do
  [ -f pipeline_$t.log ] || continue
  echo "== $t"; grep -E "GB 받음" pipeline_$t.log | tail -n 1; grep -E "원천 .* 완료|전체 완료|실패" pipeline_$t.log | tail -n 2
  tail -n 1 stderr_$t.log 2>/dev/null
done
echo "== 실행 중: $(pgrep -f pipeline.py | wc -l) pipeline, staging $(find staging -name '*.zip' 2>/dev/null | wc -l) zip 대기"
[ -f ~/work/run.log ] && { echo "== train"; tail -n 3 ~/work/run.log; }
