"""YOLO 박스 경로(v3) vs CoTracker 점 경로 비교: 흔들림 수치 + 나란히 영상.

usage: python cotracker_compare.py OUT_DIR START_S END_S   (cotracker_pass.py 결과가 있어야 함)
"""
import json
import math
import sys
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pass2_trigger as p2  # noqa: E402

out = Path(sys.argv[1])
t_start, t_end = int(float(sys.argv[2])), int(float(sys.argv[3]))
ct = np.load(out / f"cotracker_{t_start}_{t_end}.npz")
TR, VI, QM, f0 = ct["tracks"], ct["vis"], ct["qmeta"], int(ct["f0"])
meta, pfd = p2.run(out, make_video=False)
flows = np.load(out / "flows.npz")
fps, W, H = meta["fps"], meta["w"], meta["h"]
f_px = p2.F_EQ_MM / 36.0 * max(W, H)
T = TR.shape[0]
q_tid, q_seed = QM[:, 0].astype(int), QM[:, 1].astype(int)


class PathEst:
    """경로 기록(카메라 보정) + 1 s 직선 맞춤 속도, 고정(lock) 있음/없음 둘 다."""

    def __init__(self):
        self.hist = deque(maxlen=45)
        self.lock_v, self.lock_t = None, None

    def warp(self, m):
        if self.hist:
            a = np.asarray(self.hist, float)
            a[:, 1:] = a[:, 1:] @ m[:, :2].T + m[:, 2]
            self.hist = deque(map(tuple, a), maxlen=45)

    def add(self, t, x, y):
        self.hist.append((t, x, y))
        raw = p2.lock_fit([p for p in self.hist if t - p[0] <= p2.LOCK_WIN_S])
        if raw is not None and (self.lock_t is None or t - self.lock_t >= p2.LOCK_WIN_S):
            if self.lock_v is None:
                self.lock_v = raw
            else:
                so, sn = math.hypot(*self.lock_v), math.hypot(*raw)
                ang = abs(math.degrees(math.atan2(self.lock_v[0] * raw[1] - self.lock_v[1] * raw[0],
                                                  self.lock_v[0] * raw[0] + self.lock_v[1] * raw[1])))
                if ang > p2.LOCK_ANGLE_DEG or abs(sn - so) > p2.LOCK_SPEED_RATIO * max(so, 1.0):
                    self.lock_v = raw
            self.lock_t = t
        return raw, self.lock_v


est = {m: defaultdict(PathEst) for m in ("yolo", "ct")}
ct_pos = {}                      # CoTracker 발 위치 (점 움직임 중앙값을 누적)
angles = {m: defaultdict(list) for m in ("yolo_raw", "yolo_lock", "ct_raw", "ct_lock")}
accel = {"yolo": [], "ct": []}   # 발 위치 2차 차분 (px) = 프레임 간 떨림
prev_foot = {"yolo": {}, "ct": {}}
frames_draw = []
for k in range(T):
    pf = pfd[f0 - 1 + k]
    t, fi = pf["t"], f0 + k
    pairs = flows[f"f{fi}"] if f"f{fi}" in flows else None
    m, _ = p2.camera_motion(pairs)
    if m is not None:
        for d_ in est.values():
            for e in d_.values():
                e.warp(m)
    rows = []
    for d in pf["draws"]:
        tid, (x1, y1, x2, y2) = d["tid"], d["box"]
        foot_y = ((x1 + x2) / 2, y2)
        # CoTracker 발 위치: 이 트랙 점들 중 직전·현재 프레임 둘 다 보이는 점의 이동 중앙값만큼 옮김
        sel = (q_tid == tid) & (q_seed <= k - 1) & VI[k] & (VI[k - 1] if k else False)
        if tid not in ct_pos:
            ct_pos[tid] = foot_y
        elif sel.sum() >= 3:
            dmv = np.median(TR[k, sel] - TR[k - 1, sel], axis=0)
            ct_pos[tid] = (ct_pos[tid][0] + dmv[0], ct_pos[tid][1] + dmv[1])
        else:          # 점을 다 잃으면 YOLO 발 위치로 다시 맞춤
            ct_pos[tid] = foot_y
        foot_c = ct_pos[tid]
        res = {}
        for name, foot in (("yolo", foot_y), ("ct", foot_c)):
            raw, lock = est[name][tid].add(t, *foot)
            res[name] = (raw, lock)
            pv = prev_foot[name].get(tid)
            if pv is not None and pv[0] == k - 1 and pv[3] == k - 2:
                accel[name].append(math.hypot(foot[0] - 2 * pv[1][0] + pv[2][0], foot[1] - 2 * pv[1][1] + pv[2][1]))
            prev_foot[name][tid] = (k, foot, pv[1] if pv else foot, pv[0] if pv else -9)
            still = p2.STILL_FRAC * (x2 - x1)
            for tag, v in (("raw", raw), ("lock", lock)):
                if v is not None and math.hypot(*v) >= still:
                    angles[f"{name}_{tag}"][tid].append((k, math.atan2(v[1], v[0])))
        pts = TR[k, (q_tid == tid) & (q_seed <= k) & (q_seed > k - 30) & VI[k]]
        rows.append(dict(d=d, foot_y=foot_y, foot_c=foot_c, res=res, pts=pts,
                         hist_y=[(p[1], p[2]) for p in est["yolo"][tid].hist],
                         hist_c=[(p[1], p[2]) for p in est["ct"][tid].hist]))
    frames_draw.append((pf, rows))


def jitter(series):
    """연속 프레임 사이 화살표 각도 변화 평균 (°/프레임), 1초 안에 90° 넘게 뒤집힌 횟수."""
    dif, flips = [], 0
    for s in series.values():
        for (ka, a), (kb, b) in zip(s, s[1:]):
            if kb == ka + 1:
                dif.append(abs(math.degrees(math.atan2(math.sin(b - a), math.cos(b - a)))))
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                if s[j][0] - s[i][0] > 30:
                    break
                if abs(math.atan2(math.sin(s[j][1] - s[i][1]), math.cos(s[j][1] - s[i][1]))) > math.pi / 2:
                    flips += 1
                    break
    return (round(float(np.mean(dif)), 2) if dif else None), flips, sum(len(s) for s in series.values())


stats = {k: dict(zip(("deg_per_frame", "flips_90deg_in_1s", "arrow_frames"), jitter(v))) for k, v in angles.items()}
stats["foot_shake_px"] = {k: round(float(np.median(v)), 2) for k, v in accel.items()}
stats["cotracker_sec_per_frame"] = round(float(ct["sec"]) / T, 2)
stats["points"] = int(len(QM))
(out / f"cotracker_stats_{t_start}_{t_end}.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
print(json.dumps(stats, indent=2))

# ---------------- 영상 ----------------
s_ = stats
notes = out / f"notes_ct_{t_start}_{t_end}.txt"
notes.write_text(f"""# CoTracker 경로 비교 ({t_start}~{t_end}초)
# 왼쪽: YOLO 박스 발 위치 + 방향 고정 (v3)
# 오른쪽: CoTracker 점 추적 (방향 고정 없음)
- 물체마다 박스 아래쪽에 점 9개, 1초마다 다시 찍음
- 발 위치 = 점들 이동량 중앙값을 누적
- 작은 점 = CoTracker가 따라가는 점
# 수치 (작을수록 안정)
- 발 위치 프레임 간 떨림 (px)
   YOLO {s_['foot_shake_px']['yolo']} / CoTracker {s_['foot_shake_px']['ct']}
- 화살표 각도 변화 (°/프레임)
   YOLO {s_['yolo_raw']['deg_per_frame']} → 고정 {s_['yolo_lock']['deg_per_frame']}
   CoTracker {s_['ct_raw']['deg_per_frame']} → 고정 {s_['ct_lock']['deg_per_frame']}
- 1초 안에 화살표가 90° 넘게 뒤집힘 (회)
   YOLO {s_['yolo_raw']['flips_90deg_in_1s']} → 고정 {s_['yolo_lock']['flips_90deg_in_1s']}
   CoTracker {s_['ct_raw']['flips_90deg_in_1s']} → 고정 {s_['ct_lock']['flips_90deg_in_1s']}
# 속도 (이 노트북 CPU)
- CoTracker {s_['cotracker_sec_per_frame']} s/프레임 (점 {s_['points']}개)
- YOLO 0.14 s/프레임, 트리거 0.002 s/프레임
""", encoding="utf-8")
panel = p2.notes_panel(notes, int(W * 0.5), int(H * 0.5))
cap = cv2.VideoCapture(meta["video"])
cap.set(cv2.CAP_PROP_POS_FRAMES, f0 - 1)
sc = 0.5
vw = cv2.VideoWriter(str(out / f"compare_cotracker_{t_start}_{t_end}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                     (int(W * sc) * 2 + panel.shape[1], int(H * sc)))


def arrow(img, a, v, col):
    if v is None or math.hypot(*v) < 5:
        return
    b = (int(a[0] + v[0]), int(a[1] + v[1]))
    a = (int(a[0]), int(a[1]))
    cv2.arrowedLine(img, a, b, (255, 255, 255), 16, cv2.LINE_AA, tipLength=0.35)
    cv2.arrowedLine(img, a, b, col, 9, cv2.LINE_AA, tipLength=0.35)


for pf, rows in frames_draw:
    ok, frame = cap.read()
    if not ok:
        break
    panels = []
    for side in ("yolo", "ct"):
        img = frame.copy()
        hx, yh, zz = pf["head_x"], pf["y_hor"], pf["zone_z"]
        edge = np.array([[p2._ZW, z] for z in np.linspace(1.0, zz, 30)], np.float32)
        band = np.vstack([p2.project(edge * [-1, 1], f_px, hx, yh), p2.project(edge, f_px, hx, yh)[::-1]])
        p2.floor_fill(img, band, (255, 200, 0), 0.16)
        p2.dashed_poly(img, band, (255, 200, 0), 5)
        for r in rows:
            d = r["d"]
            if not d["inB"] and d["lvB"] == "safe":
                continue
            tc = p2.track_color(d["tid"])
            x1, y1, x2, y2 = map(int, d["box"])
            cv2.rectangle(img, (x1, y1), (x2, y2), p2.COL[d["lvB"]], 4)
            if side == "yolo":
                if len(r["hist_y"]) > 1:
                    p2.draw_trail(img, r["hist_y"], tc)
                arrow(img, r["foot_y"], r["res"]["yolo"][1] if d["lock"] is not None else None, tc)
            else:
                for p in r["pts"]:
                    cv2.circle(img, (int(p[0]), int(p[1])), 7, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(img, (int(p[0]), int(p[1])), 5, tc, -1, cv2.LINE_AA)
                if len(r["hist_c"]) > 1:
                    p2.draw_trail(img, r["hist_c"], tc)
                v = r["res"]["ct"][0]
                if v is not None and math.hypot(*v) >= p2.STILL_FRAC * (x2 - x1):
                    arrow(img, r["foot_c"], v, tc)
        title = "YOLO box + lock (v3)" if side == "yolo" else "CoTracker points (no lock)"
        for k_, s in enumerate((title, f"t={pf['t']:.1f}s")):
            cv2.putText(img, s, (20, 60 + 50 * k_), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 7, cv2.LINE_AA)
            cv2.putText(img, s, (20, 60 + 50 * k_), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(cv2.resize(img, (int(W * sc), int(H * sc))))
    vw.write(np.hstack(panels + [panel]))
vw.release()
cap.release()
print("video done")
