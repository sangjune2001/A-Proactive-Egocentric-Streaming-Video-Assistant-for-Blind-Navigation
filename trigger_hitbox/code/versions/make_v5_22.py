"""10/15 사용자: 내 경로 = 카메라 시야가 아니라 도로 형태 (살짝 옆으로 피하는 정도). v5_20 → v5_22, v5_21 → v5_23.
카메라가 좌우로 크게 흔들려 경로·위치(정면/11·1시)가 같이 흔들리던 문제.
- 길 방향: 카메라가 돈 만큼(배경이 옆으로 움직인 양) 매 프레임 바로 반영(자이로처럼) + 바닥 분할 도로 방향으로 천천히 보정
- 분할 도로 방향이 갑자기 20° 넘게 튀면 1 s 계속될 때만 믿음 (카메라 뒤집힘 · 순간 오판), ±35° 제한
- 진행 영역 · 몸 폭 · 도달 시간 · 예측 진입을 모두 길 방향 좌표로 계산, 영역 그림도 길 방향으로
- 위치(정면·경로상·시계)는 길 방향 기준. 2 s 넘게 도로를 못 잡으면 카메라 방향도 함께 봄 (안전 쪽)
usage: python make_v5_22.py SRC DST VERSION_NAME"""
import sys
from pathlib import Path

src, dst, ver = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
s = src.read_text(encoding="utf-8")


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), old[:90])
    s = s.replace(old, new)


V = lambda x: x.replace("{v}", ver)

# place_of: 길 방향만 (불확실하면 카메라도)
rep('''def place_of(g, heading, half_m=0.0):''', '''def place_of(g, heading, half_m=0.0, both=True):''')
rep('''    for hd in (0.0, heading):
        hr = math.radians(hd)
        lat = g[0] * math.cos(hr) - g[1] * math.sin(hr)''', '''    for hd in ((0.0, heading) if both else (heading,)):
        hr = math.radians(hd)
        lat = g[0] * math.cos(hr) - g[1] * math.sin(hr)''')
rep('''def choose_action(e, draws, road, heading=0.0):''', '''def choose_action(e, draws, road, heading=0.0, both=True):''')
rep('''    clock, lat = place_of(me["g"], heading, me.get("half_m") or 0.0)''',
    '''    clock, lat = place_of(me["g"], heading, me.get("half_m") or 0.0, both)''')

# 상태 변수
rep('''    heading = 0.0            # 길 방향 (도, 걷는 방향 기준 +오른쪽)''', V('''    heading = 0.0            # 길 방향 (도, 걷는 방향 기준 +오른쪽)
    head_x_prev, hd_seen_t, hd_out_t0 = None, -1e9, None   # 길 방향 추적 ({v})
    leaving, lv_t0 = False, None                             # 도로 벗어남'''))

# 카메라가 돈 만큼 길 방향 예측
rep('''        y_hor_prev = y_hor
''', V('''        y_hor_prev = y_hor
        # 길 방향 예측 ({v}): 카메라가 옆으로 돈 만큼(지평선 배경 이동) 길도 화면에서 같이 돎 → 바로 반영
        if head_x_prev is not None:
            dx_bg = 0.0
            if cam_ok:
                c2_ = np.array([head_x_prev, y_hor])
                dx_bg = float((m[:, :2] @ c2_ + m[:, 2] - c2_)[0])
            heading = float(np.clip(heading + math.degrees(math.atan((dx_bg - (head_x - head_x_prev)) / f_px)), -35, 35))
        head_x_prev = head_x
        if t - hd_seen_t > 2.0:
            heading *= 0.995                 # 도로를 오래 못 잡으면 천천히 정면으로
        # 도로 벗어남 ({v}): 걷는 방향이 도로와 20° 넘게 2 s 계속 어긋나면 도로를 떠나는 중 → 걷는 방향을 따름
        if not leaving:
            lv_t0 = (lv_t0 if lv_t0 is not None else t) if abs(heading) > 20 else None
            if lv_t0 is not None and t - lv_t0 >= 2.0:
                leaving, lv_t0 = True, None
        if leaving:
            heading *= 0.9
'''))

# 분할 보정: 천천히 + 튀는 값 거름
rep('''                    heading = 0.8 * heading + 0.2 * float(np.clip(math.degrees(math.atan(b_)), -30, 30))''', V('''                    meas_ = float(np.clip(math.degrees(math.atan(b_)), -35, 35))
                    if leaving:                      # 도로를 벗어나는 중: 걷는 방향을 따름, 도로와 다시 맞으면 복귀
                        if abs(meas_) < 10:
                            leaving, heading, hd_seen_t = False, meas_, t
                    elif abs(meas_ - heading) > 20:  # 갑자기 튐: 1 s 계속될 때만 믿음 (뒤집힘 · 꺾인 길)
                        hd_out_t0 = t if hd_out_t0 is None else hd_out_t0
                        if t - hd_out_t0 >= 1.0:
                            heading, hd_out_t0, hd_seen_t = meas_, None, t
                    else:
                        heading, hd_out_t0, hd_seen_t = 0.85 * heading + 0.15 * meas_, None, t'''))

# 영역 판정을 길 방향 좌표로
rep('''                lat = max(abs(X) - half_m, 0.0)
                in_zone = lat < zone_w + ZONE_EDGE_M and 0 < Z < zone_z''', V('''                hd_ok = t - hd_seen_t <= 2.0
                hrp_ = math.radians(heading)          # 길 방향 좌표 ({v}): 옆 = lat_c, 앞 = along
                lat_c = X * math.cos(hrp_) - Z * math.sin(hrp_)
                along = X * math.sin(hrp_) + Z * math.cos(hrp_)
                lat = max(abs(lat_c) - half_m, 0.0)
                if not hd_ok:                          # 도로를 못 잡으면 카메라 기준도 봄 (안전 쪽)
                    lat = min(lat, max(abs(X) - half_m, 0.0))
                in_zone = lat < zone_w + ZONE_EDGE_M and 0 < along < zone_z'''))
rep('''                hx_, hz_ = dang_poly[:, 0], dang_poly[:, 1]''', '''                hx_ = dang_poly[:, 0] * math.cos(hrp_) - dang_poly[:, 1] * math.sin(hrp_)
                hz_ = dang_poly[:, 0] * math.sin(hrp_) + dang_poly[:, 1] * math.cos(hrp_)''')
rep('''                t_reach = Z / closing if closing > 0.2 else math.inf''', '''                t_reach = max(along, 0.0) / closing if closing > 0.2 else math.inf''')
rep('''                        latf = max(abs(Xf) - half_m, 0.0)''', '''                        latf = max(abs(Xf * math.cos(hrp_) - Zf * math.sin(hrp_)) - half_m, 0.0)
                        Zf = Xf * math.sin(hrp_) + Zf * math.cos(hrp_)''')

# 위치 판단 호출
rep('''                    in_path_now = gp is not None and place_of((X, Z), heading, half_m)[0] in (0, 12)''',
    '''                    in_path_now = gp is not None and place_of((X, Z), heading, half_m, not hd_ok)[0] in (0, 12)''')
rep('''            e["action"] = choose_action(e, draws, road, heading)''',
    '''            e["action"] = choose_action(e, draws, road, heading, t - hd_seen_t > 2.0)''')
rep('''surf=user_surf, heading=heading,''', '''surf=user_surf, heading=heading, hd_ok=t - hd_seen_t <= 2.0, leaving=leaving,''')
rep('''        clock, _ = place_of(me["g"] if me else None, pf_.get("heading", 0.0), (me.get("half_m") or 0.0) if me else 0.0)''',
    '''        clock, _ = place_of(me["g"] if me else None, pf_.get("heading", 0.0), (me.get("half_m") or 0.0) if me else 0.0,
                            not pf_.get("hd_ok", False))''')

# 그림: 진행 영역을 길 방향으로
rep('''                edge = np.array([[zw, z] for z in np.linspace(1.0, zz, 30)], np.float32)
                band = np.vstack([project(edge * [-1, 1], f_px, hx, yh), project(edge, f_px, hx, yh)[::-1]])''',
    '''                hrr_ = math.radians(pf.get("heading", 0.0))      # 길 방향으로 돌린 진행 영역
                al_ = np.linspace(1.0, zz, 30)
                side_ = lambda la: np.c_[la * math.cos(hrr_) + al_ * math.sin(hrr_), -la * math.sin(hrr_) + al_ * math.cos(hrr_)].astype(np.float32)
                band = np.vstack([project(side_(-zw), f_px, hx, yh), project(side_(zw), f_px, hx, yh)[::-1]])''')

s = s.replace('"""v5_2', f'"""{ver} (10/15): 아래 버전 + 내 경로 = 도로 방향 (카메라 흔들림 보정 · 영역·판정 길 좌표).\nv5_2', 1)
dst.write_text(s, encoding="utf-8")
print("ok", ver)
