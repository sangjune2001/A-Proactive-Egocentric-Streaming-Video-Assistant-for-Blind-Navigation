"""PC 작업을 멈춘 뒤 실행: 미완성 shard(.zip.tmp)에 들어 있던 파일은 saved.txt에서 빼고(다시 받도록) tmp를 지운다.
saved.txt 경로 표기를 '/'로 통일한다."""
import struct
from pathlib import Path

import sys

BASE = Path(__file__).resolve().parent
# 인자로 TAG를 주면(예: python migrate.py 188v 71579) 그 작업들의 tmp/saved 만 정리 → 다른 작업은 계속 돌려도 됨
TAGS = sys.argv[1:]
lost = set()
for tmp in (BASE / "staging").rglob("*.zip.tmp"):
    if TAGS and not any(tmp.name.startswith(f"shard_{t}_") for t in TAGS):
        continue
    ds_split = "/".join(tmp.relative_to(BASE / "staging").parts[:2])
    data = tmp.read_bytes()
    pos = 0
    while True:
        i = data.find(b"PK\x03\x04", pos)
        if i < 0 or i + 30 > len(data):
            break
        nlen, xlen = struct.unpack("<HH", data[i + 26:i + 30])
        name = data[i + 30:i + 30 + nlen].decode("utf-8", "replace")
        lost.add(f"{ds_split}/{name}")
        pos = i + 30 + nlen + xlen
    tmp.unlink()
    print("removed", tmp.name)

for saved_path in ([BASE / f"saved_{t}.txt" for t in TAGS] if TAGS else BASE.glob("saved*.txt")):
    if not saved_path.exists():
        continue
    lines = [l.replace("\\", "/") for l in saved_path.read_text(encoding="utf-8").split("\n") if l.strip()]
    keep = [l for l in dict.fromkeys(lines) if l not in lost]
    saved_path.write_text("\n".join(keep) + "\n", encoding="utf-8")
    print(f"{saved_path.name}: {len(lines)} -> {len(keep)} (미완성 shard에서 {len(lost)}개 중 제외)")
