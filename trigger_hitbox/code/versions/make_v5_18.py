"""10/15 사용자: 경로 추적형 경고. v5_16 → v5_18, v5_17 → v5_19.
처음 본 위치로 한 번만 말하던 문제 (test_1: 1시 → 경로를 틀며 11시 → 정면이 된 볼라드).
- 모든 물체는 멀리서부터 계속 추적하고 매 프레임 '지금 내 걷는 방향' 기준으로 위치(정면·경로상·시계)를 다시 계산
- 옆(경로 밖)에 멈춰 있는 물체의 주의는 말하지 않고 추적만 (위험 = 바짝 붙으면 '경로를 유지하세요'는 그대로)
- 정면·경로상으로 들어오는 순간 경고. 이미 옆으로 알린 물체라도 경로 안으로 들어오면 다시 알림 (한 번)
usage: python make_v5_18.py SRC DST VERSION_NAME"""
import sys
from pathlib import Path

src, dst, ver = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
s = src.read_text(encoding="utf-8")
old = '''                if tag == "B" and fired and not (lv == "danger" and "danger" not in fired):
                    continue
                if lv != "safe" and lv not in fired:
                    fired.add(lv)
'''
assert s.count(old) == 1
s = s.replace(old, '''                if tag == "B":
                    # 경로 추적 ({v}): 지금 내 걷는 방향 기준 위치를 매 프레임 다시 봄
                    in_path_now = gp is not None and place_of((X, Z), heading, half_m)[0] in (0, 12)
                    if lv == "warn" and state == "static" and not in_path_now:
                        continue                     # 옆에 멈춰 있는 물체: 말하지 않고 추적만
                    re_path = in_path_now and lv != "safe" and bool(fired) and "path" not in fired
                    if fired and not (lv == "danger" and "danger" not in fired) and not re_path:
                        continue
                    if re_path:
                        fired.discard(lv)            # 옆 → 경로 안으로 들어옴: 다시 알림
                    if in_path_now and lv != "safe":
                        fired.add("path")
                if lv != "safe" and lv not in fired:
                    fired.add(lv)
'''.replace("{v}", ver))
s = s.replace('"""v5_1', f'"""{ver} (10/15): 아래 버전 + 경로 추적형 경고 (옆 정지 물체는 추적만, 경로 안으로 들어오면 알림).\nv5_1', 1)
dst.write_text(s, encoding="utf-8")
print("ok", ver)
