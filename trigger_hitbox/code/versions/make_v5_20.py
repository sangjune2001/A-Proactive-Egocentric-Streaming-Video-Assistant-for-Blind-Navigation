"""10/15 사용자: 상대속도로 움직임 구분. v5_18 → v5_20, v5_19 → v5_21.
앞에서 천천히 가는 사람에게 내가 가까워져 박스가 커지는 걸 '다가와요'라고 하던 문제.
물체 자체 속도 = 바닥 거리 변화 기울기 + 내 걸음 (최근 0.5 s 평균으로 흔들림 줄임)으로
  다가옴(나를 향해 옴) · 천천히 감(같은 방향, 나보다 느림 → 내가 따라잡음) · 멈추는 중(1 s 전보다 확 느려짐) ·
  멈춰 있음(내가 다가감) · 같이/더 빨리 감(위협 아님)을 나눔.
- 다가옴은 자체 속도가 0.5 s 평균으로도 기준 넘게 나를 향할 때만
- 음성: 사람이 천천히 가요 / 멈추고 있어요 / 서 있어요 (다가와요는 진짜 다가올 때만)
- 옆 경로 밖 '추적만' 규칙을 다가옴·지나감이 아닌 모든 상태에 적용
usage: python make_v5_20.py SRC DST VERSION_NAME"""
import sys
from pathlib import Path

src, dst, ver = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
s = src.read_text(encoding="utf-8")


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), old[:90])
    s = s.replace(old, new)


rep('''                             "crossing" if abs(vX) > own_v and state == "crossing" else "static")
                lk_ = tr.lock_v''', '''                             "crossing" if abs(vX) > own_v and state == "crossing" else "static")
                # 상대속도로 움직임 구분 ({v}, 사용자): 물체 자체 속도 vZw (+ 나에게서 멀어지는 쪽) 최근 0.5 s 평균
                tr.vh = getattr(tr, "vh", None) or deque(maxlen=60)
                tr.vh.append((t, vZw))
                if cls not in OBSTACLES and age >= 0.8 and Z <= APPR_MAX_Z and state != "crossing":
                    vm_ = float(np.mean([v_ for t_, v_ in tr.vh if t - t_ <= 0.5]))
                    ago_ = [v_ for t_, v_ in tr.vh if 0.8 <= t - t_ <= 1.3]
                    v_prev = float(np.mean(ago_)) if ago_ else None
                    thr2_ = 0.5 + 0.05 * Z if cls == "person" else own_v
                    if state == "approaching" and vm_ > -0.7 * thr2_:
                        state = "static"                 # 박스만 커짐 (앞사람이 느려 내가 다가감) → 다가옴 아님
                    if state != "approaching":
                        if vm_ > walk - 0.2 and vm_ > 0.4:
                            state = "receding"           # 나와 같이 또는 더 빨리 감 → 위협 아님
                        elif vm_ > 0.4:
                            state = ("stopping" if v_prev is not None and v_prev - vm_ > 0.5 else "slow")
                        elif vm_ < -0.5:
                            state = "approaching"        # 자체 속도로 나를 향해 옴 (걸어서 다가오는 사람)
                        elif v_prev is not None and v_prev > 0.9:
                            state = "stopping"
                        elif abs(vm_) <= 0.3 and (v_prev is None or abs(v_prev) <= 0.3):
                            state = "still"              # 정말 멈춰 있음 → 내가 다가감 ('서 있어요')
                        else:
                            state = "static"
                lk_ = tr.lock_v'''.replace("{v}", ver))

rep('''                if person_block:
                    state, going_up = "static", False''', '''                if person_block:
                    state = state if state in ("slow", "stopping", "still") else "static"
                    going_up = False''')

rep('''                    if lv == "warn" and state == "static" and not in_path_now:''',
    '''                    if lv == "warn" and state not in ("approaching", "crossing") and not in_path_now:''')

# 버스 (사용자): 인도·골목으로 들어올 일이 거의 없음 → 옆으로 지나갈 때만 알림, 그 밖엔 경고 대상 아님
rep('''                if cls == "person" and not (person_block or person_appr):
                    rawB = "safe"''', '''                if cls == "person" and not (person_block or person_appr):
                    rawB = "safe"
                elif cls == "bus" and state != "crossing":
                    rawB = "safe"                # 버스는 지나갈 때만 ({v}, 사용자)'''.replace("{v}", ver))

# 음성 움직임: 사람이 멈춰 있으면 '서 있어요'
rep('''        rt.append(dict(row_n=row_n, lat=lat_, t=e["time_s"], cls=say_, scenario=sc, track_id=e["track_id"], level=e["level"], box=box,
                       motion=e["state"],''', '''        mo_ = ("standing" if (say_ == "person" and e["state"] == "still") else
               "static" if e["state"] == "still" else e["state"])
        rt.append(dict(row_n=row_n, lat=lat_, t=e["time_s"], cls=say_, scenario=sc, track_id=e["track_id"], level=e["level"], box=box,
                       motion=mo_,''')

s = s.replace('"""v5_1', f'"""{ver} (10/15): 아래 버전 + 상대속도로 다가옴·천천히 감·멈추는 중·멈춰 있음 구분.\nv5_1', 1)
dst.write_text(s, encoding="utf-8")
print("ok", ver)
