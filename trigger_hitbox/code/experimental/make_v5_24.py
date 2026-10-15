"""10/15 사용자: 점자블록(노란색)을 따라가는 것을 우선. v5_22 → v5_24, v5_23 → v5_25.
- pass1_tactile.py가 찾은 노란 점 → 바닥 좌표 → 인도 위(차도 노란선 제외) · 폭 0.15–0.8 m(노란 차 제외)만
- 6행 이상 · 앞뒤 2 m 이상 · 직선에서 0.35 m 안으로 맞으면 점자블록 → 그 방향을 내 경로로 (도로 방향보다 우선)
- 점자블록이 보이는 동안은 바닥 분할 도로 방향 보정 안 함, 도로 벗어남 판단도 해제
- 영상에 점자블록 선(자홍) 표시
usage: python make_v5_24.py SRC DST VERSION_NAME"""
import sys
from pathlib import Path

src, dst, ver = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
s = src.read_text(encoding="utf-8")
V = lambda x: x.replace("{v}", ver)


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), old[:90])
    s = s.replace(old, new)


rep('''    leaving, lv_t0 = False, None                             # 도로 벗어남''', V('''    leaving, lv_t0 = False, None                             # 도로 벗어남
    tac_by_f, tac_stride, tac_t = {}, 2, -1e9                # 점자블록 ({v})
    tj_ = out / "tactile.json"
    if tj_.exists() and os.environ.get("TACTILE", "1") != "0":
        td_ = json.loads(tj_.read_text(encoding="utf-8"))
        tac_stride, tac_by_f = td_["stride"], {x["f"]: x["pts"] for x in td_["frames"]}'''))

rep('''                if len(gz) >= 6 and np.ptp(gz) > 2.0:
                    b_ = float(np.polyfit(gz, gx, 1)[0])''', '''                if len(gz) >= 6 and np.ptp(gz) > 2.0 and t - tac_t > 1.0:     # 점자블록이 보이면 그쪽 우선
                    b_ = float(np.polyfit(gz, gx, 1)[0])''')

rep('''        # 길 유형: 횡단보도(정면 보행 신호 · 바닥 줄무늬) > 차도 인접(2 s 안 차량) > 인도''', V('''        # 점자블록 ({v}, 사용자): 노란 점 → 바닥 좌표 직선. 보이면 그 방향이 내 경로 (도로 방향보다 우선)
        tac_now = None
        pts_ = tac_by_f.get(fi - ((fi - 1) % tac_stride)) if tac_by_f else None
        if pts_:
            gxs_, gzs_, wms_ = [], [], []
            for yy_, xx_, ww_ in pts_:
                if yy_ - y_hor < 40:
                    continue
                Zt_ = f_px * CAM_H / (yy_ - y_hor)
                if not 1.2 < Zt_ < 12.0 or not 0.15 <= ww_ * Zt_ / f_px <= 0.8:
                    continue
                if surf is not None and surf[min(int(yy_ / H * sh_), sh_ - 1), min(int(xx_ / W * sw_), sw_ - 1)] == 2:
                    continue                        # 차도 위 노란 선
                gxs_.append((xx_ - head_x) * Zt_ / f_px)
                gzs_.append(Zt_)
                wms_.append(ww_ * Zt_ / f_px)
            zs_ = np.sort(np.asarray(gzs_)) if gzs_ else np.zeros(0)
            wm_ = np.asarray(wms_)
            # 바닥에 끊김 없이 이어짐(점 사이 1.5 m 안) · 가까이(4 m 안)까지 옴 · 폭이 일정 → 볼라드 띠 · 간판 제외
            if (len(gzs_) >= 6 and np.ptp(gzs_) >= 2.0 and zs_[0] < 4.0 and float(np.max(np.diff(zs_))) < 1.5
                    and float(wm_.std() / max(wm_.mean(), 1e-3)) < 0.5):
                bt_, at_ = np.polyfit(gzs_, gxs_, 1)
                if (float(np.std(np.asarray(gxs_) - (at_ + bt_ * np.asarray(gzs_)))) < 0.35
                        and abs(math.degrees(math.atan(bt_))) < 40):
                    tac_now = (float(at_), float(bt_))
        if tac_now is not None:
            heading = 0.7 * heading + 0.3 * float(np.clip(math.degrees(math.atan(tac_now[1])), -35, 35))
            hd_seen_t = tac_t = t
            leaving, lv_t0 = False, None
        # 길 유형: 횡단보도(정면 보행 신호 · 바닥 줄무늬) > 차도 인접(2 s 안 차량) > 인도'''))

rep('''hd_ok=t - hd_seen_t <= 2.0, leaving=leaving,''', '''hd_ok=t - hd_seen_t <= 2.0, leaving=leaving, tactile=tac_now,''')

rep('''            if tag == "B":                                          # 인도 끝선 (소실점 방향 직선, v5_12 ← v5_9)''',
    V('''            if tag == "B" and pf.get("tactile") is not None:       # 점자블록 선 (자홍, {v})
                at_, bt_ = pf["tactile"]
                ln_ = project(np.array([[at_ + bt_ * z_, z_] for z_ in (1.2, min(zz, 12.0))], np.float32), f_px, hx, yh)
                cv2.line(img, tuple(map(int, ln_[0])), tuple(map(int, ln_[1])), (255, 0, 255), 6, cv2.LINE_AA)
            if tag == "B":                                          # 인도 끝선 (소실점 방향 직선, v5_12 ← v5_9)'''))

s = s.replace('"""v5_2', f'"""{ver} (10/15): 아래 버전 + 점자블록(노란색) 방향을 내 경로로 우선.\nv5_2', 1)
dst.write_text(s, encoding="utf-8")
print("ok", ver)
