"""SJ 사본 음성: 움직임 '천천히 가요 · 멈추고 있어요 · 서 있어요' 추가 (v5_20)."""
from pathlib import Path
p = Path("/home/ubuntu/sj/code/runtime/dispatcher.py")
s = p.read_text()
old = '"receding": "멀어져요"}'
if '"slow"' not in s:
    assert old in s
    s = s.replace(old, '"receding": "멀어져요", "slow": "천천히 가요", "stopping": "멈추고 있어요", "standing": "서 있어요"}', 1)
    p.write_text(s)
print("patched")
