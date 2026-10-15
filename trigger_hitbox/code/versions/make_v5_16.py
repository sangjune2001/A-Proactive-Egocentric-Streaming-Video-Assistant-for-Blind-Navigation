"""10/15 요청 4개: v5_14 → v5_16, v5_15 → v5_17.
1) 지나가요 = 무조건 '멈추세요' (주의·위험 모두)
2) 정면·경로상 판단은 걷는 방향(카메라)과 길 방향 둘 중 하나라도 걸리면 경로상 — 길 방향 오판으로
   정면 볼라드에 '경로를 유지하세요'가 나가던 문제 (test_1 3.3 s, 길 방향 -9° 오판)
3) 정지 장애물은 1 s 이상 (그동안 60 % 이상 프레임에서) 잡혀야 경고 — 순간 잡힌 기둥 (test_3)
4) 줄지은 가로수·기둥: 첫 경고 때 같은 쪽 ±1 m 줄에 정지 물체가 2개 이상 보이면 바로
   '오른쪽에 기둥이 이어져 있어요. 주의하세요.' 한 번, 그 줄의 나머지는 말 안 함 (행동이 '경로 유지'여도 묶음)
usage: python make_v5_16.py SRC DST VERSION_NAME"""
import sys
from pathlib import Path

src, dst, ver = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
s = src.read_text(encoding="utf-8")


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), old[:90])
    s = s.replace(old, new)


rep("ROW_GAP_S, ROW_DX_M = 8.0, 0.7 ", "STATIC_MIN_AGE = 1.0      # 정지 장애물은 1 s 이상 잡혀야 경고 (순간 잡힌 기둥 제외, {v})\nROW_GAP_S, ROW_DX_M = 8.0, 1.0 ".replace("{v}", ver))

# 2) 위치: 카메라 방향(0°)과 길 방향 둘 다 봄
start, end = s.index("def place_of("), s.index("def choose_action(")
s = s[:start] + '''def place_of(g, heading, half_m=0.0):
    """위치 ({v}): 걷는 방향(카메라 0°)과 길 방향(heading) 둘 다 봄 — 길 방향을 잘못 잡아도 정면 물체를 옆으로 보내지 않게.
    둘 중 하나라도 ±4° 안 = 정면(12) / 하나라도 물체 가장자리가 내 몸 폭(+0.3 m) 안 = 경로상(0) /
    나머지 = 길 방향 기준 시계 방향. lat = 옆 거리 (+ 오른쪽, 경로상이면 그렇게 판단한 기준의 값)."""
    if g is None:
        return 12, 0.0
    best = None
    for hd in (0.0, heading):
        hr = math.radians(hd)
        lat = g[0] * math.cos(hr) - g[1] * math.sin(hr)
        a = math.degrees(math.atan2(g[0], g[1])) - hd
        if abs(a) <= FRONT_DEG:
            return 12, lat
        if max(abs(lat) - half_m, 0.0) < BODY_R + 0.3 and best is None:
            best = (0, lat)
    if best is not None:
        return best
    k = 1 if abs(a) < 45 else 2 if abs(a) < 75 else 3
    return (k if a > 0 else 12 - k), lat


'''.replace("{v}", ver) + s[end:]

# 1) 지나가요 = 무조건 멈추세요 (위험만 보던 첫 줄보다 먼저)
rep('''    if e["level"] != "danger":
        return "none"
    me = next((d for d in draws if d["tid"] == e["track_id"] and d["g"] is not None), None)
    if me is None:
        return "none"
    if e["state"] == "crossing" and (e["class"] in VEHICLES or e["class"] == "person"):
        return "none"
''', '''    if e["state"] == "crossing" and (e["class"] in VEHICLES or e["class"] == "person"):
        return "stop"                                # 지나가요 = 무조건 멈추세요 ({v}, 사용자)
    if e["level"] != "danger":
        return "none"
    me = next((d for d in draws if d["tid"] == e["track_id"] and d["g"] is not None), None)
    if me is None:
        return "none"
'''.replace("{v}", ver))

# 3) 정지 장애물 1 s 유지
rep('''                elif (age < MIN_AGE_S or not gate''', '''                elif (age < MIN_AGE_S or (cls in OBSTACLES and (age < STATIC_MIN_AGE or tr.n_obs < 0.6 * age * fps))
                      or not gate''')

# 4) 줄지은 장애물: 첫 경고 때 같은 쪽 줄에 정지 물체가 2개 이상 보이는지 (그 프레임 그림 정보로, 실시간 가능)
rep('''        rt.append(dict(t=e["time_s"], cls=say_, scenario=sc, track_id=e["track_id"], level=e["level"], box=box,''',
    '''        row_n, lat_ = 0, None
        if me and me["g"] is not None and clock in (1, 2, 3, 9, 10, 11) and e["state"] == "static":
            hr_ = math.radians(pf_.get("heading", 0.0))
            lat_ = me["g"][0] * math.cos(hr_) - me["g"][1] * math.sin(hr_)
            for o_ in pf_["draws"]:
                if o_["g"] is None or o_["cls"] not in OBSTACLES or o_["state"] != "static" or not 0 < o_["g"][1] < 20:
                    continue
                lo_ = o_["g"][0] * math.cos(hr_) - o_["g"][1] * math.sin(hr_)
                if lo_ * lat_ > 0 and abs(lo_ - lat_) < ROW_DX_M:
                    row_n += 1
        rt.append(dict(row_n=row_n, lat=lat_, t=e["time_s"], cls=say_, scenario=sc, track_id=e["track_id"], level=e["level"], box=box,''')

start, end = s.index("def merge_rows(rt):"), s.index("def calm_veer(rt):")
s = s[:start] + '''def merge_rows(rt):
    """줄지어 선 가로수·기둥 ({v}, 사용자): 같은 쪽(1–3시 / 9–11시) 정지 장애물 줄.
    첫 경고 때 그 줄(옆 거리 ±1 m)에 정지 물체가 2개 이상 보이거나, 8 s 안에 같은 줄 물체가 또 나오면
    '오른쪽에 기둥이 이어져 있어요. 주의하세요.' 한 번만 말하고, 줄이 끊길 때(8 s)까지 그 줄은 말하지 않음.
    행동이 '경로를 유지하세요'인 것도 묶음 (정면·경로상 '피하세요/멈추세요'는 그대로)."""
    out, rows = [], {}
    for r in sorted(rt, key=lambda r: r["t"]):
        side = ("right" if r.get("clock") in (1, 2, 3) else "left" if r.get("clock") in (9, 10, 11) else None)
        if (side is None or r["scenario"] != "S2" or r.get("motion") != "static"
                or r.get("action", "none") not in ("none", "keep") or r.get("lat") is None):
            out.append(r)
            continue
        row = rows.get(side)
        if row and r["t"] - row["t"] < ROW_GAP_S and abs(r["lat"] - row["lat"]) < ROW_DX_M:
            row.update(t=r["t"], lat=0.5 * (row["lat"] + r["lat"]), n=row["n"] + 1)
            if not row["said"]:
                row["said"] = True
                out.append(row_msg(r, side))
            continue                                    # 줄의 나머지는 말하지 않음
        many = r.get("row_n", 0) >= 2
        rows[side] = dict(t=r["t"], lat=r["lat"], n=1, said=many)
        out.append(row_msg(r, side) if many else r)
    return out


def row_msg(r, side):
    what = "기둥이" if str(r.get("cls", "")).startswith("pole") else "장애물이"
    return dict({k: v for k, v in r.items() if k not in ("clock", "items", "motion", "action")},
                cls="row", scenario="S5", track_id=None, urgent=True,
                text=f"{'오른쪽' if side == 'right' else '왼쪽'}에 {what} 이어져 있어요. 주의하세요.")


'''.replace("{v}", ver) + s[end:]

s = s.replace('"""v5_1', f'"""{ver} (10/15): 아래 버전 + 지나가요=멈추세요 · 정면 판단 두 방향 · 정지 장애물 1 s 유지 · 줄지은 기둥 한 번에.\nv5_1', 1)
dst.write_text(s, encoding="utf-8")
print("ok", ver)
