"""v5_15 기준 (사용자: 경고 생성은 v5_15가 가장 정확) + 판정 안 바꾸는 요청만 → v5_28.
(아래는 v5_16 설명 중 1·3·4만 적용, 2(위치 두 방향)는 뺌)
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
rep('''                elif (age < MIN_AGE_S or not gate''', '''                elif (age < MIN_AGE_S or (cls != "person" and state != "crossing"
                                          and (age < STATIC_MIN_AGE or tr.n_obs < 0.6 * age * fps))
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
    return dict({k: v for k, v in r.items() if k not in ("clock", "items", "motion", "action")},
                cls="row", scenario="S5", track_id=None, urgent=True,
                text=f"{'오른쪽' if side == 'right' else '왼쪽'}에 기둥이 연속해 있어요.")     # 사용자 문구


'''.replace("{v}", ver) + s[end:]


# 버스: 옆으로 지나갈 때만 (사용자)
rep('''                if cls == "person" and not (person_block or person_appr):
                    rawB = "safe"''', '''                if cls == "person" and not (person_block or person_appr):
                    rawB = "safe"
                elif cls == "bus" and state != "crossing":
                    rawB = "safe"                # 버스는 지나갈 때만 ({v}, 사용자)'''.replace("{v}", ver))

# 사람만: 말할 때 움직임 (판정 state는 그대로) — 자체 속도 = 거리 기울기 + 내 걸음, 0.5 s 평균
rep('''                lk_ = tr.lock_v
                big_near_ = y2 >= H - 5 or (x2 - x1) > 0.5 * W''', '''                tr.say_m = None                  # 사람 말하기용 움직임 ({v}, 판정에는 안 씀)
                if cls == "person":
                    tr.vh = getattr(tr, "vh", None) or deque(maxlen=60)
                    tr.vh.append((t, vZw))
                    vm_ = float(np.mean([v_ for t_, v_ in tr.vh if t - t_ <= 0.5]))
                    ago_ = [v_ for t_, v_ in tr.vh if 0.8 <= t - t_ <= 1.3]
                    v_prev = float(np.mean(ago_)) if ago_ else None
                    if 0.4 < vm_ < walk - 0.2:
                        tr.say_m = "stopping" if v_prev is not None and v_prev - vm_ > 0.5 else "slow"
                    elif v_prev is not None and v_prev > 0.9 and vm_ <= 0.4:
                        tr.say_m = "stopping"
                    elif abs(vm_) <= 0.3 and (v_prev is None or abs(v_prev) <= 0.3):
                        tr.say_m = "standing"
                lk_ = tr.lock_v
                big_near_ = y2 >= H - 5 or (x2 - x1) > 0.5 * W'''.replace("{v}", ver))
rep('''                              gpoly=None if gp is None else dang_poly, half_m=None if gp is None else half_m,''',
    '''                              gpoly=None if gp is None else dang_poly, half_m=None if gp is None else half_m,
                              say_m=getattr(tr, "say_m", None),''')
rep('''        rt.append(dict(row_n=row_n, lat=lat_, t=e["time_s"], cls=say_, scenario=sc, track_id=e["track_id"], level=e["level"], box=box,
                       motion=e["state"],''', '''        mo_ = e["state"]
        if say_ == "person" and mo_ not in ("approaching", "crossing") and me and me.get("say_m"):
            mo_ = me["say_m"]                # 사람만: 천천히 가요 · 멈추고 있어요 · 서 있어요
        rt.append(dict(row_n=row_n, lat=lat_, t=e["time_s"], cls=say_, scenario=sc, track_id=e["track_id"], level=e["level"], box=box,
                       motion=mo_,''')

s = s.replace('"""v5_1', f'"""{ver} (10/15): v5_15 + 지나가요=멈추세요 · 정지 장애물 1 s 유지 · 줄지은 기둥 한 번에 · 버스는 지나갈 때만 · 사람 움직임 말하기.\nv5_1', 1)
dst.write_text(s, encoding="utf-8")
print("ok", ver)
