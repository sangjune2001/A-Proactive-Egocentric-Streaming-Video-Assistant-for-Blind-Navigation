"""v5_28 → v5_29 (10/15 사용자: test_2 9.4 s 차량 '다가와요' · test_3 47.6 s 자전거 '지나가요'는 틀림, v5_15가 맞음).
원인: 1 s 유지 규칙으로 경고가 늦게 나가면서, 그 순간 잠깐 튄 움직임(카메라 돌림 · 새로 잡힌 트랙의 속도 튐)이 문장에 들어감.
수정: 사람 말고는 움직임 판정(정지 · 다가옴 · 지나감 · 멀어짐)이 바뀌어도 0.2 s 넘게 계속돼야 인정 (0.5 s는 진짜 지나가는 차까지 빠짐).
usage: python make_v5_29.py SRC DST VERSION_NAME"""
import sys
from pathlib import Path

src, dst, ver = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
s = src.read_text(encoding="utf-8")
old = '''                # 히트박스 길이용 속도'''
assert s.count(old) == 1
s = s.replace(old, f'''                # 움직임 판정 다지기 ({ver}): 사람 말고는 바뀐 판정이 0.2 s 넘게 계속돼야 인정
                if cls != "person":
                    if getattr(tr, "st_ok", None) is None:
                        tr.st_ok, tr.st_new, tr.st_t0 = state, state, t
                    if state != tr.st_new:
                        tr.st_new, tr.st_t0 = state, t
                    if state != tr.st_ok and t - tr.st_t0 >= 0.2:
                        tr.st_ok = state
                    state = tr.st_ok
''' + old, 1)
s = s.replace('"""v5_2', f'"""{ver} (10/15): v5_28 + 사람 말고는 움직임 판정이 0.2 s 계속돼야 인정.\nv5_2', 1)
dst.write_text(s, encoding="utf-8")
print("ok", ver)
