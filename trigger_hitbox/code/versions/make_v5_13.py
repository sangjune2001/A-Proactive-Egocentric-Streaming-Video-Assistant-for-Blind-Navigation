"""v5_12 → v5_13: 사람 경고 규칙만 바꿈 (10/15 사용자 요청).
- 사람은 내 길 방향 ±10° 안에서 아주 빠르게(물체 자체 속도 2 m/s 이상) 다가올 때만 경고
- 사람이 많으면(가까이 3명 이상, 또는 2 s 안 사람 경고 3개 이상) 사람 경고는 빼고 물체·장애물 위주
usage: python make_v5_13.py pass2_trigger.py(v5_12) pass2_trigger_v5_13.py"""
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
s = src.read_text(encoding="utf-8")


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), old[:80])
    s = s.replace(old, new)


rep("PERSON_BLOCK_Z = 4.0 ", '''PERSON_DEG = 10.0          # 사람: 내 길 방향 ±10° 안만 (v5_13)
PERSON_FAST = 2.0          # 사람: 물체 자체 속도로 2 m/s 넘게 다가올 때만 = 뛰어옴 (v5_13)
CROWD_N, CROWD_Z = 3, 8.0  # 앞 8 m · 옆 ±4 m 안 사람 3명 이상 = 붐빔 → 사람 경고 안 함 (v5_13)
CROWD_EV_N, CROWD_EV_S = 3, 2.0   # 사람 경고가 2 s 안 3개 이상 겹치면 그 사람 경고들은 뺌 (v5_13)
PERSON_BLOCK_Z = 4.0 ''')

# 사람 판단: v5_12의 막힘·다가옴·가로지름 규칙 → 정면 ±10° + 빠르게 다가옴, 붐비면 안 함
start = s.index("                person_appr = cls == \"person\" and (")
end = s.index("                if person_block:")
s = s[:start] + '''                # 사람 (v5_13, 사용자): 우르르 지나가는 사람들은 알아서 피해 가므로 경고 안 함.
                # 내 길 방향 ±10° 안에서 2 m/s 넘게(뛰어서) 다가오는 사람만 경고. 붐비면 그것도 안 함
                p_ang = abs(math.degrees(math.atan2(X, Z)) - heading)
                person_block = False
                person_appr = (cls == "person" and not crowd_now and 0 < Z < APPR_MAX_Z and p_ang <= PERSON_DEG
                               and vZw < -PERSON_FAST)
''' + s[end:]

# 붐빔 판단: 이번 프레임 검출 중 가까운 사람 수 (바닥 위치는 발 기준 간이 계산)
rep('''        for d in kept_:
            cls = names[d["cls"]]''', '''        n_people = 0
        for d in kept_:
            if names[d["cls"]] == "person":
                gp_ = ground_pos(d["box"], "person", f_px, head_x, y_hor, H)
                if gp_ is not None and gp_[1] < CROWD_Z and abs(gp_[0]) < 4.0:
                    n_people += 1
        crowd_now = n_people >= CROWD_N
        for d in kept_:
            cls = names[d["cls"]]''')

# 2 s 안 사람 경고 3개 이상 → 그 사람 경고들 제거 (물체·장애물 경고는 그대로)
rep('''    rt = group_speech(rt)
''', '''    pe_ = sorted(r["t"] for r in rt if r.get("cls") == "person")
    busy_ = {t_ for t_ in pe_ if sum(abs(u_ - t_) <= CROWD_EV_S for u_ in pe_) >= CROWD_EV_N}
    rt = [r for r in rt if not (r.get("cls") == "person" and r["t"] in busy_)]     # (v5_13)
    rt = group_speech(rt)
''')

s = s.replace('"""v5_12 (10/15):', '"""v5_13 (10/15): v5_12 + 사람 경고 규칙(정면 ±10° 빠르게 다가옴만 · 붐비면 사람 경고 안 함).\nv5_12 (10/15):', 1)
dst.write_text(s, encoding="utf-8")
print("ok")
