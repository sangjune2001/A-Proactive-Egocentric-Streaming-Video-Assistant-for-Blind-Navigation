"""v5_7 복원본 → v5_12: 사용자가 고른 v5_8~v5_11 항목 + 10/15 요청만 더함."""
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
s = src.read_text(encoding="utf-8")


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), old[:80])
    s = s.replace(old, new)


# ---------- 11-1: 판정 대기만 짧게 (0.5 s 안쪽) ----------
rep("MIN_AGE_S = 0.4            # 트랙이 생긴 지 0.4 s 미만이면 판정 안 함 (v5: 0.5 → 0.4, 경고를 더 일찍)",
    "MIN_AGE_S = 0.25           # 트랙이 생긴 지 0.25 s 미만이면 판정 안 함 (v5_12: 0.4 → 0.25)")
rep("ON_HOLD_S, OFF_HOLD_S = 0.2, 0.6   # 히스테리시스 (초 단위, v5: 켜짐 0.3 → 0.2)",
    "ON_HOLD_S, OFF_HOLD_S = 0.1, 0.6   # 히스테리시스 (초 단위, v5_12: 켜짐 0.2 → 0.1)")
rep("APPR_HOLD_S = 0.3          # 다가옴 확정까지 (v5_7: 0.5 → 0.3 s, 다가오는 물체 경고를 더 빨리)",
    "APPR_HOLD_S = 0.2          # 다가옴 확정까지 (v5_12: 0.3 → 0.2 s)")
rep("GROUP_S = 1.0             # 급하지 않은 경고는 1 s 동안 모아서 한 문장으로",
    "GROUP_S = 0.5             # 급하지 않은 경고는 0.5 s 동안 모아서 한 문장으로 (v5_12: 1 → 0.5 s)")
rep("HEADON_X = 0.9           # 물체 중심이 내 정면 ±0.9 m 안 = 부딪힐 각 → 이때만 '피하세요/멈추세요'\n",
    "HEADON_X = 0.9           # 물체 중심이 내 정면 ±0.9 m 안 = 부딪힐 각 → 이때만 '피하세요/멈추세요'\n"
    "DUP_IOU = 0.8             # 박스가 80 % 이상 겹치면 같은 물체 (v5_12 ← v5_9)\n"
    "FRONT_DEG = 4.0           # 길 방향 ±4° 안만 '정면' (v5_12 ← v5_8)\n"
    "CLOSE_Z = 2.5             # 이보다 가까운 경로 위 위험 물체는 '멈추세요' (v5_12 ← v5_8)\n"
    "ROW_GAP_S, ROW_DX_M = 8.0, 0.7   # 줄지은 장애물: 8 s 안 · 옆 거리 ±0.7 m (v5_12 ← v5_8)\n"
    "POLE_ASPECT, POLE_W_M = 3.0, 0.5   # 세로/가로 3배 넘고 폭 0.5 m 이하인 정지 장애물 = 기둥 (볼라드 포함, v5_12)\n"
    "POLE_LONG_M = 1.5         # 기둥 높이 1.5 m 이상(또는 화면 위로 잘림) = 긴 기둥, 아니면 짧은 기둥 (v5_12)\n")

# ---------- 공용 함수: IoU · 위치(정면/경로상/시계) · 행동 · 줄지은 장애물 ----------
old_start = s.index("def choose_action(e, draws, road):")
old_end = s.index("def group_speech(rt):")
s = s[:old_start] + '''def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def place_of(g, heading, half_m=0.0):
    """위치 (v5_12): 길 방향 기준.
    ±4° 안 = 정면(12) / 그 밖이라도 물체 가장자리가 내 몸 폭(+0.3 m) 길 위 = 경로상(0) /
    나머지 = 시계 방향 (4–45° 1·11시, 45–75° 2·10시, 그 밖 3·9시).
    lat = 길 중심선에서 옆 거리 (+ 오른쪽)."""
    if g is None:
        return 12, 0.0
    hr = math.radians(heading)
    lat = g[0] * math.cos(hr) - g[1] * math.sin(hr)
    a = math.degrees(math.atan2(g[0], g[1])) - heading
    if abs(a) <= FRONT_DEG:
        return 12, lat
    if max(abs(lat) - half_m, 0.0) < BODY_R + 0.3:
        return 0, lat
    k = 1 if abs(a) < 45 else 2 if abs(a) < 75 else 3
    return (k if a > 0 else 12 - k), lat


def choose_action(e, draws, road, heading=0.0):
    """행동 안내 (v5_12): 주의 = 안내 없음 / 위험일 때만.
    - 옆으로 지나가는 차·사람 = '지나가요'만 (행동 없음)
    - 11·1시 등 옆(경로 밖): 정지 물체면 '경로를 유지하세요', 움직이는 물체는 위치만
    - 계단 · 경로 위 2.5 m 안으로 바짝 = 멈추세요
    - 경로상 = 물체 반대쪽으로 피하세요 (11시 쪽 물체 → 오른쪽, 1시 쪽 → 왼쪽)
    - 정면 = 덜 막힌 쪽으로 피하세요 (같으면 물체 반대쪽)"""
    if e["level"] != "danger":
        return "none"
    me = next((d for d in draws if d["tid"] == e["track_id"] and d["g"] is not None), None)
    if me is None:
        return "none"
    if e["state"] == "crossing" and (e["class"] in VEHICLES or e["class"] == "person"):
        return "none"
    clock, lat = place_of(me["g"], heading, me.get("half_m") or 0.0)
    if clock not in (0, 12):
        return "keep" if e["state"] not in ("approaching", "crossing") else "none"
    if e["class"] == "stairs" or me["g"][1] < CLOSE_Z:
        return "stop"
    away = "left" if lat > 0 else "right"            # 물체 반대쪽
    if clock == 0:
        return "veer_" + away
    hr = math.radians(heading)

    def to_path(g):
        return (g[0] * math.cos(hr) - g[1] * math.sin(hr), g[0] * math.sin(hr) + g[1] * math.cos(hr))
    X0, Z0 = to_path(me["g"])
    others = [to_path(d["g"]) for d in draws if d["g"] is not None and d["tid"] != e["track_id"]]

    def blocked(side):               # 피해 갈 쪽 1.8 m 폭 · 앞뒤 3 m 안 물체까지 거리 (없으면 inf)
        lo, hi = (X0 - 2.0, X0 - 0.2) if side == "left" else (X0 + 0.2, X0 + 2.0)
        ds = [math.hypot(X - X0, Z - Z0) for X, Z in others if lo < X < hi and abs(Z - Z0) < 3.0]
        return min(ds) if ds else math.inf

    order = [away, "right" if away == "left" else "left"]
    return "veer_" + max(order, key=lambda sd: (blocked(sd), sd == away))


def merge_rows(rt):
    """줄지어 선 가로수·기둥 (v5_8 → v5_12): 같은 쪽(1–3시 / 9–11시)에 옆 거리가 비슷한(±0.7 m) 정지 장애물이
    8 s 안에 또 나오면, 두 번째는 '오른쪽에 기둥이 이어져 있어요. 주의하세요.' 한 번만, 그 뒤 같은 줄은 말하지 않음."""
    out, rows = [], {}
    for r in sorted(rt, key=lambda r: r["t"]):
        side = ("right" if r.get("clock") in (1, 2, 3) else "left" if r.get("clock") in (9, 10, 11) else None)
        if (side is None or r["scenario"] != "S2" or r.get("motion") != "static" or r.get("action", "none") != "none"
                or r.get("X") is None):
            out.append(r)
            continue
        row = rows.get(side)
        if row and r["t"] - row["t"] < ROW_GAP_S and abs(r["X"] - row["X"]) < ROW_DX_M:
            row.update(t=r["t"], X=0.5 * (row["X"] + r["X"]), n=row["n"] + 1)
            if row["n"] == 2:
                what = "기둥이" if str(r.get("cls", "")).startswith("pole") else "장애물이"
                out.append(dict({k: v for k, v in r.items() if k not in ("clock", "items", "motion", "action")},
                                cls="row", scenario="S5", track_id=None, urgent=True,
                                text=f"{'오른쪽' if side == 'right' else '왼쪽'}에 {what} 이어져 있어요. 주의하세요."))
            continue                                    # 세 번째부터는 말하지 않음
        rows[side] = dict(t=r["t"], X=r["X"], n=1)
        out.append(r)
    return out


def calm_veer(rt):
    """반대쪽 '피하세요'가 1.5 s 안에 또 나오면 뒤엣것은 위치만 (오른쪽으로 → 바로 왼쪽으로 헷갈림 방지, v5_12)."""
    last = None
    for r in sorted(rt, key=lambda r: r["t"]):
        a = r.get("action", "none")
        if a in ("veer_left", "veer_right"):
            if last and r["t"] - last[0] < 1.5 and a != last[1]:
                r["action"] = "none"
            else:
                last = (r["t"], a)
    return rt


''' + s[old_end:]
rep('''    out, pend = [], []

    def flush():''', '''    out, pend = [], []
    rt = calm_veer(merge_rows(rt))

    def flush():''')

# ---------- 9-8: 신호등 = 상자 안 픽셀 색 (lights_hsv.json 있으면) ----------
rep('''    lj = out / "lights.json"
''', '''    lj = out / os.environ.get("LIGHT_FILE", "lights_hsv.json")      # v5_12 ← v5_9: 기울어진 신호등도 상자 안 색으로
    if not lj.exists() or os.environ.get("LIGHT_HSV", "1") == "0":
        lj = out / "lights.json"
''')

# ---------- 9-4: 연석·화단까지 나눈 바닥 분할(surface2) → 인도 끝선 ----------
rep('''    sf = out / "surface.npz"
''', '''    sf = out / "surface2.npz" if (out / "surface2.npz").exists() else out / "surface.npz"   # v5_12 ← v5_9
''')
rep('''    walk_edge = [None, None]   # 내가 걷는 인도의 왼쪽·오른쪽 끝 X (m)
    heading = 0.0''', '''    walk_edge = [None, None]   # 내가 걷는 인도의 왼쪽·오른쪽 끝 X (m)
    crowd_hist, crowd_said = deque(maxlen=90), -1e9   # 주변 사람 걷기 시작 감지 (v5_12 ← v5_10 요청)
    last_walk_side = -1e9      # 마지막으로 인도 위에 있던 시각
    road_t0 = 0.0              # 발밑이 차도가 된 시각
    turning = False
    heading = 0.0''')
rep('''                user_surf = int(np.bincount(foot_reg.ravel(), minlength=4).argmax())
''', '''                user_surf = int(np.bincount(foot_reg.ravel(), minlength=7).argmax())
                if user_surf in (4, 5, 6):                   # 연석·화단 가장자리에 서 있어도 인도로 봄
                    user_surf = 1
''')

s = s.replace("minlength=4)", "minlength=7)")   # surface2 라벨 0–6

# ---------- 8-2: 횡단보도 없는 곳에서 차도 건너기 ----------
rep('''            xs_ = ("cross" if on_xw and walk >= XWALK_WALK else
                   "wait" if walk < XWALK_WALK and road == "crosswalk" else None)
''', '''            xs_ = ("cross" if on_xw and walk >= XWALK_WALK else
                   "wait" if walk < XWALK_WALK and road == "crosswalk" else None)
        elif (user_surf == 2 and walk >= XWALK_WALK and t - road_t0 >= 1.5
              and road_t0 - last_walk_side < 3.0):    # 발밑이 1.5 s 넘게 계속 차도 + 그 직전엔 인도
            xs_ = "road"            # 횡단보도 없는 곳에서 인도 → 차도로 내려섬 (v5_12 ← v5_8)
''')
rep('''            text_ = "횡단보도 앞이에요." + sig_ if xs_ == "wait" else "횡단보도를 건너고 있어요. 차를 조심하세요."
''', '''            text_ = {"wait": "횡단보도 앞이에요." + sig_,
                     "cross": "횡단보도를 건너고 있어요. 차를 조심하세요.",
                     "road": "차도를 건너고 있어요. 차를 조심하세요."}[xs_]
''')
rep('''        elif road != "crosswalk":
            xw_state = None
''', '''        elif road != "crosswalk":
            xw_state = None
        if user_surf == 1:
            last_walk_side = t
        if user_surf != 2:
            road_t0 = t
''')

# ---------- 9-7: 같은 물체 중복 박스 제거 + 트랙 이어 붙이기 ----------
rep('''        for d in fr["dets"]:
            cls = names[d["cls"]]''', '''        # 같은 물체를 여러 번 잡은 박스 (v5_12 ← v5_9): IoU 0.8 이상 겹치면 신뢰도 높은 하나만
        kept_ = []
        for d in sorted(fr["dets"], key=lambda d: -d.get("conf", 1.0)):
            if all(box_iou(d["box"], k["box"]) < DUP_IOU for k in kept_):
                kept_.append(d)
        for d in kept_:
            cls = names[d["cls"]]''')
rep('''            if tr is None or t - tr.last_t > 0.5:
                tr = tracks[tid] = Track(tid, cls, t)
            gap = t - tr.last_t
            tr.last_t = t
''', '''            if tr is None or t - tr.last_t > 0.5:
                tr = tracks[tid] = Track(tid, cls, t)
                # 트랙 이어 붙이기: 1 s 안에 사라진 트랙의 마지막 발 위치 근처 또는 직전 박스와 80 % 이상 겹치는 새 ID
                # = 같은 물체 — 나이 · 경고 기록 · 바닥 칼만 등을 넘겨 받음 (같은 물체 경고 반복 방지)
                best_, bd_ = None, 0.12 * W
                for o_ in tracks.values():
                    if o_ is tr or not o_.hist or not (0 < t - o_.last_t < 1.0) or getattr(o_, "merged", False):
                        continue
                    dd_ = math.hypot(o_.hist[-1][1] - (x1 + x2) / 2, o_.hist[-1][2] - y2)
                    if getattr(o_, "last_box", None) is not None and box_iou(o_.last_box, d["box"]) >= DUP_IOU:
                        dd_ = 0.0
                    if dd_ < bd_:
                        best_, bd_ = o_, dd_
                if best_ is not None:
                    best_.merged = True
                    for k_ in ("t0", "firedB", "hB", "gkf", "zh", "lock_v", "lock_t", "votes", "n_obs", "max_conf",
                               "zoneB_t", "appr_t0"):
                        if hasattr(best_, k_):
                            setattr(tr, k_, getattr(best_, k_))
                    tr.hist = deque(best_.hist, maxlen=45)
            gap = t - tr.last_t
            tr.last_t = t
            tr.last_box = d["box"]
''')

# ---------- 9-5 · 9-2: 다가옴(달려오는 사람) · 옆으로 지나감(검정 차) ----------
rep('''                    appr_now = state == "approaching" and vZw < -(own_v + (0.6 if cls == "person" else 0.0))
''', '''                    thr_ = 0.5 + 0.05 * Z if cls == "person" else own_v
                    # 화면 확대율이 약해도 바닥 속도로 기준의 2배 넘게 다가오면 다가옴 (45.7 s 달려오는 사람, v5_12 ← v5_9)
                    # 화면 아래 끝에 닿거나 폭이 화면 절반 넘는 큰 물체: 발 위치 거리가 튀어서 바닥 속도를 안 믿음
                    big_near = y2 >= H - 5 or (x2 - x1) > 0.5 * W
                    appr_now = ((cls == "person" and state == "approaching" and g_rel > G_REL_MOVING) if big_near else
                                vZw < -thr_ and (state == "approaching" or g_rel > 0.05
                                                 or (cls == "person" and vZw < -2 * thr_)))
''')
rep('''                             "crossing" if abs(vX) > own_v and state == "crossing" else "static")
                # 히트박스 길이용 속도''', '''                             "crossing" if abs(vX) > own_v and state == "crossing" else "static")
                lk_ = tr.lock_v
                big_near_ = y2 >= H - 5 or (x2 - x1) > 0.5 * W
                # 화면 옆 끝에 잘린 채 들어오는 물체 (검정 차, v5_12 ← v5_9): 안쪽 끝이 0.4 s 넘게 화면 폭의 30 %/s 넘게
                # 옆으로 움직이면 '옆에서 들어와 지나감' — 잘린 상자라 바닥 거리·확대율은 믿지 않음
                bh_ = getattr(tr, "bh", None) or deque(maxlen=30)
                tr.bh = bh_
                bh_.append((t, x1, x2))
                old_ = [b for b in bh_ if t - b[0] >= 0.4]
                edge_l, edge_r = x1 <= 0.1 * W, x2 >= 0.9 * W
                if old_ and not turning and (edge_l != edge_r) and (cls in VEHICLES or cls == "person"):
                    b0_ = old_[-1]
                    inner_v = ((x2 - b0_[2]) if edge_l else (b0_[1] - x1)) / max(t - b0_[0], 1e-3)
                    if inner_v > 0.3 * W and abs(vX) > own_v:
                        state = "crossing"
                # 화살표가 옆(수평)을 향하면 다가옴보다 '지나감'이 우선 — 내 길과 수직으로 움직임 (v5_12 ← v5_9)
                if (not turning and (cls in VEHICLES or cls == "person") and lk_ is not None and age >= 0.3
                        and abs(lk_[0]) > 0.25 * (x2 - x1) and abs(lk_[1]) < 0.6 * abs(lk_[0]) and Z < 8.0
                        and (abs(vX) > own_v or big_near_)):
                    state = "crossing"
                # 히트박스 길이용 속도''')

# 예측 진입을 사람 판단보다 먼저 계산 (다가오는·가로지르는 사람도 미리 경고, 9-5)
pred = '''                    # 예측 진입 (v5_7): 움직이는 물체의 0.25 s 간격 3 s 미래 위치 (상대속도 = 칼만 바닥 속도)
                    enter_w = enter_d = False
                    if state in ("approaching", "crossing"):
                        for k_ in range(1, 13):
                            tf = 0.25 * k_
                            Xf, Zf = X + vX * tf, Z + vZ * tf
                            latf = max(abs(Xf) - half_m, 0.0)
                            if 0 < Zf < zone_z and latf < zone_w:
                                enter_w = True
                                if tf <= DANGER_REACH_S and latf < BODY_R + 0.5 and Zf < 6.0:
                                    enter_d = True
                                    break
'''
rep(pred, "")
pred_out = "\n".join(l[4:] if l.strip() else l for l in pred.split("\n"))
rep('''                person_appr = cls == "person" and state == "approaching" and in_zone
''', pred_out + '''                person_appr = cls == "person" and (       # v5_12 ← v5_9: 달려오는 사람 · 가로질러 들어오는 사람 일찍
                    (state == "approaching" and (in_zone or enter_w or (Z < 8.0 and lat < zone_w + 1.5)))
                    or (state == "crossing" and enter_w)
                    or (in_body and 0 < Z < 8.0 and t_reach < DANGER_REACH_S))
''')

# ---------- 오늘: 주의 뒤 위험이 되면 다시 알림 ----------
rep('''                if tag == "B" and fired and not (lv == "danger" and "danger" not in fired and gp is not None and in_body):
''', '''                # v5_12 (사용자): 주의로 한 번 알린 물체라도 위험으로 바뀌면 다시 알림 (위치와 상관없이 한 번)
                if tag == "B" and fired and not (lv == "danger" and "danger" not in fired):
''')

# draws에 물체 반폭 · 행동에 길 방향
rep('''                              gpoly=None if gp is None else dang_poly,
''', '''                              gpoly=None if gp is None else dang_poly, half_m=None if gp is None else half_m,
''')

# ---------- 10-1: 주변 사람 걷기 시작 → 신호 확인 ----------
rep('''        for e in evB[nB_before:]:           # 이번 프레임에 새로 난 경고의 행동 안내 (v5_3)
            e["action"] = choose_action(e, draws, road)
''', '''        # 주변 사람 신호 (v5_12 ← v5_10 요청): 횡단보도 앞 대기 중 앞쪽 사람 2명 이상이 멈춰 있다 걷기 시작
        # → 내가 기다리는 신호가 바뀌었을 가능성 (사거리에서 다른 신호를 잡았을 때 보조). 10 s 간격
        n_go = sum(1 for d_ in draws if d_["cls"] == "person" and d_["g"] is not None and d_["g"][1] < 12.0
                   and abs(d_["g"][0]) < 4.0 and d_["state"] in ("receding", "crossing"))
        crowd_hist.append((t, n_go))
        if (xw_state == "wait" and n_go >= 2 and t - crowd_said > 10.0
                and any(n_ == 0 for t_, n_ in crowd_hist if t - t_ <= 2.0)):
            evX.append(dict(frame=fi, time_s=round(t, 3), kind="crowd", text="주변 사람들이 건너기 시작했어요. 신호를 확인하세요."))
            crowd_said = t
        for e in evB[nB_before:]:           # 이번 프레임에 새로 난 경고의 행동 안내
            e["action"] = choose_action(e, draws, road, heading)
''')

# ---------- 말할 이름 · 위치 (10-3 + 오늘) ----------
rep('''        clock = 12
        if me and me["g"] is not None:      # 시계 방향: 정면 12시, 오른쪽 1–3시, 왼쪽 11–9시 (30°마다)
            k = int(np.clip(round(math.degrees(math.atan2(me["g"][0], me["g"][1])) / 30), -3, 3))
            clock = 12 + k if k <= 0 else k
        rt.append(dict(t=e["time_s"], cls=e.get("say", e["class"]), scenario=sc, track_id=e["track_id"], level=e["level"], box=box,
                       motion=e["state"], action=e.get("action", "none"), clock=clock, dist=e["dist_m"]))''',
    '''        pf_ = per_frame_draw[e["frame"] - 1]
        # 위치 (v5_12): 정면(길 방향 ±4°) 12 / 경로상 0 / 그 밖 시계 방향 — 길 방향 기준
        clock, _ = place_of(me["g"] if me else None, pf_.get("heading", 0.0), (me.get("half_m") or 0.0) if me else 0.0)
        say_ = e.get("say", e["class"])
        # 이름 (v5_12): 멈춰 있는 탈것은 '장애물'. 가늘고 긴 정지 장애물은 '기둥' (볼라드도 기둥) — 높이로 긴/짧은
        if e["state"] not in ("approaching", "crossing") and say_ in VEHICLES:
            say_ = "obstacle"
        Zb_ = me["g"][1] if me and me["g"] is not None else None
        if (say_ == "obstacle" and box is not None and Zb_ is not None
                and (box[3] - box[1]) > POLE_ASPECT * (box[2] - box[0]) and (box[2] - box[0]) * Zb_ / f_px <= POLE_W_M):
            h_m = (box[3] - box[1]) * Zb_ / f_px
            say_ = "pole_long" if (box[1] <= 5 or h_m >= POLE_LONG_M) else "pole_short"
        e["say"] = say_
        if say_ in ("obstacle", "stairs", "pole_long", "pole_short"):
            e["state"] = "static"
        rt.append(dict(t=e["time_s"], cls=say_, scenario=sc, track_id=e["track_id"], level=e["level"], box=box,
                       motion=e["state"], action=e.get("action", "none"), clock=clock, dist=e["dist_m"],
                       X=None if not me or me["g"] is None else round(me["g"][0], 2)))''')

# ---------- 그림: 인도 끝선을 영상 위에도 (연두선, 9-4) ----------
rep('''            floor_fill(img, band, (255, 200, 0), 0.16)          # 형광펜처럼 연하게
''', '''            floor_fill(img, band, (255, 200, 0), 0.16)          # 형광펜처럼 연하게
            if tag == "B":                                          # 인도 끝선 (소실점 방향 직선, v5_12 ← v5_9)
                for ex_ in (pf.get("edge") or (None, None)):
                    if ex_ is not None:
                        ln = project(np.array([[ex_, z_] for z_ in (1.0, zz)], np.float32), f_px, hx, yh)
                        cv2.line(img, tuple(map(int, ln[0])), tuple(map(int, ln[1])), (120, 255, 120), 4, cv2.LINE_AA)
''')

s = s.replace('"""2단계:', '"""v5_12 (10/15): v5_7 + 사용자가 고른 v5_8~v5_11 항목 + 10/15 요청.\n\n2단계:', 1)
dst.write_text(s, encoding="utf-8")
print("ok", len(s))
