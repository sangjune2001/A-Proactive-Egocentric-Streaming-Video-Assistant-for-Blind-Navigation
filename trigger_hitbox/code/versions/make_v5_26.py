"""10/15 사용자: v5_20부터 주차된 자전거·오토바이·차가 '다가와요'로 나오던 문제. v5_22 → v5_26, v5_23 → v5_27.
원인: 물체 자체 속도(거리 기울기 + 내 걸음)가 조금만 음수여도 '다가옴'으로 올리던 줄 (내 걸음 추정이 2 m/s대로 크면 더 심함).
수정: 이 승격은 사람만 · 화면에서도 배경보다 커질 때(g_rel > 0.05)만 · 기준도 사람 기준의 0.7배 이상.
탈것은 v5_14와 같은 다가옴 기준(확대율 + 바닥 속도 + 0.2 s 유지)만 씀.
usage: python make_v5_26.py SRC DST VERSION_NAME"""
import sys
from pathlib import Path

src, dst, ver = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
s = src.read_text(encoding="utf-8")
old = '''                        elif vm_ < -0.5:
                            state = "approaching"        # 자체 속도로 나를 향해 옴 (걸어서 다가오는 사람)'''
assert s.count(old) == 1
s = s.replace(old, f'''                        elif cls == "person" and vm_ < -max(0.5, 0.7 * thr2_) and g_rel > 0.05:
                            state = "approaching"        # 걸어서 다가오는 사람만 ({ver}: 탈것은 기존 기준만 — 주차 차 '다가옴' 오판)''')
s = s.replace('"""v5_2', f'"""{ver} (10/15): 아래 버전 + 자체 속도 다가옴 승격은 사람만 (주차된 탈것 다가옴 오판 수정).\nv5_2', 1)
dst.write_text(s, encoding="utf-8")
print("ok", ver)
