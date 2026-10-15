"""SJ 실행 시스템 사본(서버 ~/sj)에 v5_12 음성 조각 추가 — SJ 브랜치에는 푸시하지 않음.
'경로상에' 위치, '경로를 유지하세요.' 행동, '긴 기둥/짧은 기둥' 이름, 새 안내 문장 미리 합성 목록."""
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/sj")
p = root / "code/p0/ko_template.py"
s = p.read_text()
if '"pole_long"' not in s:
    key = '"obstacle": "장애물"'
    assert key in s
    s = s.replace(key, key + ', "pole_long": "긴 기둥", "pole_short": "짧은 기둥", "pole": "기둥"', 1)
    p.write_text(s)

p = root / "code/runtime/dispatcher.py"
s = p.read_text()
old = 'CLOCK_ACTION = {"stop": "멈추세요.", "veer_left": "왼쪽으로 피하세요.", "veer_right": "오른쪽으로 피하세요."}'
if old in s:
    s = s.replace(old, old[:-1] + ', "keep": "경로를 유지하세요."}', 1)
assert '"keep": "경로를 유지하세요."' in s
old = '''out.append("정면" if it["clock"] == 12 else f"{it['clock']}시 방향")'''
if old in s:
    s = s.replace(old, '''out.append("정면" if it["clock"] == 12 else "경로상에" if it["clock"] == 0 else f"{it['clock']}시 방향")''', 1)
assert '"경로상에" if it["clock"] == 0' in s
old = '''clocks = [f"{h}시 방향" for h in (9, 10, 11, 1, 2, 3)] + ["정면"]'''
if old in s:
    s = s.replace(old, old[:-1] + ', "경로상에"]', 1)
for a in ('for c in CLASSES + ["bollard", "pole", "wall"]', 'for c in CLASSES + ["bollard"]'):
    s = s.replace(a, 'for c in CLASSES + ["pole_long", "pole_short", "pole", "wall"]')
assert s.count('"pole_long", "pole_short"') >= 2
extra = ["횡단보도 앞이에요.", "횡단보도 앞이에요. 빨간불이에요.", "횡단보도 앞이에요. 초록불이에요.",
         "오른쪽에 기둥이 이어져 있어요. 주의하세요.", "왼쪽에 기둥이 이어져 있어요. 주의하세요."]
if "V512_TEXTS" not in s:
    s = s.replace("def clock_pieces(ev)", "V512_TEXTS = %r\n\n\ndef clock_pieces(ev)" % extra, 1)
    s = s.replace("+ list(CLOCK_ACTION.values()) + XWALK_TEXTS", "+ list(CLOCK_ACTION.values()) + XWALK_TEXTS + V512_TEXTS", 1)
assert "XWALK_TEXTS + V512_TEXTS" in s
p.write_text(s)
print("patched")
