#!/usr/bin/env bash
# 로컬 디스크(반납하면 사라짐)의 결과 · 로그 · 코드를 NFS storage(246 GB, 남음)로 복사.
#   bash code/server/sync_to_storage.sh [storage 경로]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
STORE=${1:-/home/ubuntu/runyourai/vlmcall}
DST="$STORE/backup_$(date +%m%d)"
mkdir -p "$DST"
# NFS는 작은 파일이 느리므로 tar 하나로
tar cf "$DST/results.tar" -C "$ROOT" results
tar cf "$DST/logs.tar" -C /home/ubuntu/work logs
tar cf "$DST/code.tar" -C "$ROOT" code docs labels
ls -lh "$DST"
df -h "$STORE" | tail -1
