# -*- coding: utf-8 -*-
"""v3 기하 트리거 (박상준, 2026-10-03 ~ 10-05) — 한 파일 복원본.

"언제 말할지"를 학습 없이 미터 단위 기하로 정한다.
  영상 15 fps
   → [이미 있음] E0 YOLO11s-seg + ByteTrack      results/tracks/E0/<clip>.csv   (code/dump_tracks.py)
   → [이미 있음] 배경 움직임                      results/motion/<clip>.csv      (code/motion.py)
   → ① MoGe-2 ViT-S 깊이 (2 Hz)                    results/depth/<clip>.npz       python geom_trigger_v3.py depth
   → ② 시각 오도메트리 (깊이 + LK + PnP)          results/vo/<clip>.csv          python geom_trigger_v3.py vo
   → ③ 진행방향 기준 미터 좌표 (X, Z)             results/geom/E0/<clip>.csv     python geom_trigger_v3.py geom
   → ④ S1·S2 칼만 [x,z,vx,vz] + 64표본 충돌확률 + 정지물 4.5 m 거리 규칙
     ⑤ S3 신호등 crop → HSV(+TinyTLC 선택) → 안정 상태 전이
     ⑥ S4 지면 모델 외삽 + stairs 검출
     ⑦ 스케줄러 → 템플릿 문장                    results/events/v3_E0/<clip>.json  python geom_trigger_v3.py run
   → 채점 (labels/v2, code/evaluate.py)                                              python geom_trigger_v3.py eval

  python geom_trigger_v3.py all            # ①–⑦ + 채점 전부
  python geom_trigger_v3.py run --no-s4    # S4 끄고 (S4 기하 단서는 오탐이 많았음)

설명 문서: 같은 폴더의 v3_기하트리거_설명.md
원래 파일 → 이 파일 절:  depth_moge.py §1 · vo.py §2 · geom.py §3 · trigger/trigger_v2.py §4 ·
                         trigger/run_trigger.py §5 · run_ours.py §6 · signal_s3.py §7 · step_s4.py §8
복원: 10/5 삭제된 코드를 Claude Code 세션 기록의 Write/Edit/패치 이력을 순서대로 재생해 복원.
      로직은 바꾸지 않았고 import·경로만 한 파일에 맞게 정리했다.
      바뀐 점 1개 — TinyTLC 가중치(tlc_go_stop.pt)는 복원 불가 → S3는 HSV 단독이 기본, --tlc 로 가중치 주면 원래대로.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))        # common.py (클립 목록 · iter_frames · results 경로)
from common import CLIPS, OUT, clip_info, iter_frames  # noqa: E402


# =====================================================================================================
# §1  깊이 — MoGe-2 (metric point map + 화각 동시 추정) → results/depth/<clip>.npz
#   DAv2-Metric-Outdoor는 주행 장면 학습이라 1인칭 보행 영상에서 발밑 지팡이를 3.9 m로 냄 → MoGe-2로 교체.
#   로컬 CPU 장당 ~8–14 s → 2 Hz만 저장, 그 사이는 looming(§3)과 칼만 예측(§4)으로 메움.
#   저장: t, depth(1/4 해상도 float16, m), K = fx/fy/cx/cy(원본 픽셀)
# =====================================================================================================
def moge_load(device="cpu"):
    from moge.model.v2 import MoGeModel
    return MoGeModel.from_pretrained("Ruicheng/moge-2-vits-normal").to(device).eval()


def moge_infer(m, im_bgr, device="cpu", num_tokens=900):
    import torch
    with torch.no_grad():
        x = torch.from_numpy(np.ascontiguousarray(im_bgr[:, :, ::-1]) / 255.0).float().permute(2, 0, 1).to(device)
        o = m.infer(x, resolution_level=0, num_tokens=num_tokens, use_fp16=device != "cpu")
    H, W = im_bgr.shape[:2]
    K = o["intrinsics"].float().cpu().numpy()          # 정규화 intrinsics (0–1)
    dep = o["depth"].float().cpu().numpy()
    return dep, np.array([K[0, 0] * W, K[1, 1] * H, K[0, 2] * W, K[1, 2] * H], np.float32)


def run_depth(clips, hz=2.0, device="cpu", threads=4, tokens=900):
    import torch
    torch.set_num_threads(threads)
    m = moge_load(device)
    od = OUT / "depth"
    od.mkdir(parents=True, exist_ok=True)
    for key in clips:
        ts, deps, Ks, ms = [], [], [], []
        for fidx, t, im in iter_frames(key, hz, max_side=1920):
            t0 = time.perf_counter()
            dep, K = moge_infer(m, im, device, tokens)
            ms.append((time.perf_counter() - t0) * 1000)
            ts.append(t)
            deps.append(np.nan_to_num(dep[::4, ::4], nan=0.0, posinf=0.0).astype(np.float16))
            Ks.append(K)
        np.savez_compressed(od / f"{key}.npz", t=np.array(ts, np.float32), depth=np.stack(deps),
                            K=np.stack(Ks), stride=4, hz=hz)
        print(f"[depth] {key}: {len(ts)} frames, median {np.median(ms):.0f} ms", flush=True)


# =====================================================================================================
# §2  내 움직임 — metric depth 기반 시각 오도메트리 → results/vo/<clip>.csv
#   깊이 앵커 프레임 k에서
#     ① 배경 특징점 (검출 박스 · 화면 하단 20 %[손·안내견·지팡이] 제외) → MoGe 깊이로 3D (카메라 좌표)
#     ② 15 fps 프레임마다 LK 광류로 추적 (정·역 검사)
#     ③ 다음 앵커(≈0.5 s 뒤)에서 solvePnPRansac(3D, 2D, K) → 카메라 이동 R, t (깊이가 미터라 t도 미터)
#   출력: 구간마다 내 속도 |t_xz|/Δt, 진행 방향 atan2(tx, tz), yaw 변화, 인라이어 수
#   (초판 '정지물이 다가오는 속도'는 차도·광장처럼 정지물 트랙이 없으면 0 → c01 횡단에서 0 m/s 나와 교체)
# =====================================================================================================
VO_SCALE_W = 640


def run_vo_clip(model_id, key, fps=15.0):
    d = np.load(OUT / "depth" / f"{key}.npz")
    td, dep, Ks, stride = d["t"], d["depth"].astype(np.float32), d["K"], int(d["stride"])
    boxes = defaultdict(list)
    for r in csv.DictReader(open(OUT / "tracks" / model_id / f"{key}.csv", encoding="utf-8")):
        boxes[int(r["frame"])].append(tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2")))
    ai = 0
    state = None              # dict(t0, P3 (N,3), pts (N,1,2) 축소 좌표, gray)
    rows = []
    for fidx, t, im in iter_frames(key, fps, max_side=1920):
        H, W = im.shape[:2]
        s = VO_SCALE_W / max(H, W)
        gray = cv2.cvtColor(cv2.resize(im, (round(W * s), round(H * s))), cv2.COLOR_BGR2GRAY)
        if state is not None and len(state["pts"]):
            p1, st, _ = cv2.calcOpticalFlowPyrLK(state["gray"], gray, state["pts"], None, winSize=(21, 21), maxLevel=3)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, state["gray"], p1, None, winSize=(21, 21), maxLevel=3)
            ok = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm((back - state["pts"]).reshape(-1, 2), axis=1) < 1.5)
            state["pts"], state["P3"] = p1[ok], state["P3"][ok]
            state["gray"] = gray
        is_anchor = ai < len(td) and abs(td[ai] - t) <= 0.07
        if not is_anchor:
            continue
        K = Ks[ai]
        Km = np.array([[K[0] * s, 0, K[2] * s], [0, K[1] * s, K[3] * s], [0, 0, 1]], np.float64)
        # ③ 이전 앵커 → 지금: PnP
        if state is not None and len(state["pts"]) >= 12:
            okp, rvec, tvec, inl = cv2.solvePnPRansac(state["P3"].astype(np.float64), state["pts"].reshape(-1, 2).astype(np.float64),
                                                      Km, None, reprojectionError=2.0, iterationsCount=200,
                                                      flags=cv2.SOLVEPNP_EPNP)
            dt = t - state["t0"]
            if okp and inl is not None and len(inl) >= 10 and dt > 0:
                R, _ = cv2.Rodrigues(rvec)
                c = (-R.T @ tvec).ravel()            # 새 카메라 중심 (이전 카메라 좌표)
                v = math.hypot(c[0], c[2]) / dt
                head = math.degrees(math.atan2(c[0], c[2])) if math.hypot(c[0], c[2]) > 0.05 else float("nan")
                yaw = math.degrees(math.atan2(R[0, 2], R[2, 2]))
                rows.append([round(state["t0"], 3), round(t, 3), round(v, 3), round(c[0], 3), round(c[1], 3), round(c[2], 3),
                             None if math.isnan(head) else round(head, 2), round(yaw, 2), int(len(inl)), len(state["pts"])])
            else:
                rows.append([round(state["t0"], 3), round(t, 3), None, None, None, None, None, None, 0, len(state["pts"])])
        # ① 새 앵커: 배경 특징점 + 3D
        mask = np.full(gray.shape, 255, np.uint8)
        mask[int(gray.shape[0] * 0.8):, :] = 0
        for x1, y1, x2, y2 in boxes.get(fidx, []):
            mask[max(0, int(y1 * s) - 4):int(y2 * s) + 4, max(0, int(x1 * s) - 4):int(x2 * s) + 4] = 0
        p = cv2.goodFeaturesToTrack(gray, maxCorners=400, qualityLevel=0.01, minDistance=8, mask=mask, blockSize=7)
        P3, pts = np.zeros((0, 3)), np.zeros((0, 1, 2), np.float32)
        if p is not None:
            dm = dep[ai]
            uv = p.reshape(-1, 2) / s                                   # 원본(1920 축소) 좌표
            du = np.clip((uv[:, 0] / stride).astype(int), 0, dm.shape[1] - 1)
            dv = np.clip((uv[:, 1] / stride).astype(int), 0, dm.shape[0] - 1)
            z = dm[dv, du]
            good = (z > 0.3) & (z < 25)
            uv, z = uv[good], z[good]
            X = (uv[:, 0] - K[2]) / K[0] * z
            Y = (uv[:, 1] - K[3]) / K[1] * z
            P3 = np.stack([X, Y, z], 1)
            pts = p[good].astype(np.float32)
        state = dict(t0=t, P3=P3, pts=pts, gray=gray)
        ai += 1
    od = OUT / "vo"
    od.mkdir(parents=True, exist_ok=True)
    with open(od / f"{key}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t0", "t1", "speed", "cx", "cy", "cz", "heading_deg", "yaw_deg", "inliers", "tracked"])
        w.writerows(rows)
    sp = [r[2] for r in rows if r[2] is not None]
    print(f"[vo] {key}: {len(rows)} segs, ok {len(sp)}, speed median {np.median(sp) if sp else 0:.2f} m/s", flush=True)


# =====================================================================================================
# §3  미터 좌표 — tracks + 깊이(2 Hz) + VO → results/geom/<model>/<clip>.csv  (§4 입력 계약)
#   거리 Z
#     · 깊이 프레임: 접지점(마스크 맨 아래 점) 바로 아래 지면 깊이 중앙값
#       (기둥·볼라드처럼 가는 물체는 박스 안 깊이가 대부분 배경 → c09 기둥 9 m ↔ 2.7 m 진동 → 접지점 방식)
#       접지점이 화면 하단에 잘렸으면 박스 안쪽(가운데 50 % 폭, 아래 40 %) 30 백분위
#     · 사이 프레임: looming으로 이어 붙임  Z(t) = Z(t_d)·s(t_d)/s(t),  s = √(마스크 면적)  (최대 0.6 s)
#       크기 변화율은 깊이 스케일 오차와 무관 (Lee 1976 τ). 화면에 잘린 박스는 이어 붙이지 않음
#   내 속도 = VO 최근 3구간 중앙값 (0–2.5 m/s),  진행 방향 h = 최근 4구간 이동 벡터 합 방향 (±30°)
#   좌표를 h만큼 돌려 Z = 내가 걷는 방향 (고개를 돌려도 '경로'는 걷는 방향 기준)
# =====================================================================================================
def load_depth(key):
    d = np.load(OUT / "depth" / f"{key}.npz")
    return d["t"], d["depth"].astype(np.float32), d["K"], int(d["stride"])


def box_depth(dmap, stride, r):
    H, W = dmap.shape
    fu, fv = float(r["foot_u"]) / stride, float(r["foot_v"]) / stride
    bh = (float(r["y2"]) - float(r["y1"])) / stride
    if fv < H - 3:
        ua, ub = int(fu - 2), int(fu + 3)
        va, vb = int(fv + 1), int(min(H, fv + 1 + max(3, 0.04 * bh)))
        patch = dmap[max(0, va):vb, max(0, ua):min(W, ub)]
        patch = patch[patch > 0.05]
        if patch.size >= 4:
            return float(np.median(patch))
    x1, y1, x2, y2 = (float(r[k]) / stride for k in ("x1", "y1", "x2", "y2"))
    w, h = x2 - x1, y2 - y1
    xa, xb = int(x1 + 0.25 * w), int(math.ceil(x2 - 0.25 * w))
    ya, yb = int(y2 - 0.4 * h), int(math.ceil(y2))
    xa, xb, ya, yb = max(0, xa), min(W, max(xb, xa + 1)), max(0, ya), min(H, max(yb, ya + 1))
    patch = dmap[ya:yb, xa:xb]
    patch = patch[patch > 0.05]
    return float(np.percentile(patch, 30)) if patch.size else None


def size_of(r):
    a = float(r.get("mask_area") or 0)
    if a <= 0:
        a = (float(r["x2"]) - float(r["x1"])) * (float(r["y2"]) - float(r["y1"]))
    return math.sqrt(max(a, 1.0))


def run_geom_clip(model_id, key, max_prop_s=0.6):
    rows = list(csv.DictReader(open(OUT / "tracks" / model_id / f"{key}.csv", encoding="utf-8")))
    mot = {int(r["frame"]): r for r in csv.DictReader(open(OUT / "motion" / f"{key}.csv", encoding="utf-8"))}
    td, dep, Ks, stride = load_depth(key)
    by_f = defaultdict(list)
    for r in rows:
        by_f[int(r["frame"])].append(r)
    frames = sorted(set(by_f) | set(mot))
    t_of = {f: float((mot.get(f) or by_f[f][0])["t_sec"]) for f in frames}

    anchor_frame = {}                 # 각 깊이 시각에 가장 가까운 트랙 프레임 (±1/15 s)
    for i, t in enumerate(td):
        f = min(frames, key=lambda f: abs(t_of[f] - t)) if frames else None
        if f is not None and abs(t_of[f] - t) <= 0.07:
            anchor_frame[f] = i

    fx_med = float(np.median(Ks[:, 0])) if len(Ks) else 1000.0
    cx_med = float(np.median(Ks[:, 2])) if len(Ks) else 0.0
    vo = list(csv.DictReader(open(OUT / "vo" / f"{key}.csv", encoding="utf-8")))
    vo_t = np.array([float(r["t1"]) for r in vo]) if vo else np.zeros(0)

    def ego_at(t):
        """t 이전 최근 3구간 → (속도, 진행방향 deg, 신뢰)"""
        if not len(vo_t):
            return 0.0, 0.0, 0
        i = int(np.searchsorted(vo_t, t + 1e-6))
        seg = [vo[j] for j in range(max(0, i - 3), i) if vo[j]["speed"]]
        if not seg:
            return 0.0, 0.0, 0
        v = float(np.clip(np.median([float(r["speed"]) for r in seg]), 0, 2.5))
        # 진행 방향 = 최근 2 s(4구간) 이동 벡터 합의 방향 — 이동 거리 가중이라 느릴 때 흔들림이 작다
        seg4 = [vo[j] for j in range(max(0, i - 4), i) if vo[j]["speed"]]
        cx_, cz_ = sum(float(r["cx"]) for r in seg4), sum(float(r["cz"]) for r in seg4)
        h = float(np.clip(math.degrees(math.atan2(cx_, cz_)), -30, 30)) if math.hypot(cx_, cz_) > 0.5 and cz_ > 0 else 0.0
        okk = int(min(int(r["inliers"]) for r in seg) >= 20)
        return v, h, okk

    anchors = {}                      # tid -> (t, Z, size)
    out = []
    for f in frames:
        t = t_of[f]
        v_ego, head, heading_ok = ego_at(t)
        delta = math.radians(head)
        cd, sd = math.cos(delta), math.sin(delta)
        di = anchor_frame.get(f)
        if di is not None:
            fx, fy, cx, cy = Ks[di]
            for r in by_f.get(f, []):
                z = box_depth(dep[di], stride, r)
                if z is not None:
                    anchors[int(r["track_id"])] = (t, z, size_of(r))
        else:
            fx, cx = fx_med, cx_med

        for r in by_f.get(f, []):
            tid = int(r["track_id"])
            an = anchors.get(tid)
            if an is None:
                continue
            ta, za, sa = an
            if t - ta > max_prop_s:
                continue
            if abs(t - ta) < 1e-6:
                z = za
            elif int(r["edge_flag"]):
                continue
            else:
                z = za * sa / size_of(r)
            u = float(r["foot_u"])
            xc = (u - cx) / fx * z
            X, Z = xc * cd - z * sd, xc * sd + z * cd       # 진행 방향 h만큼 회전
            out.append([f, round(t, 3), tid, round(X, 3), 0.0, round(Z, 3), round(math.hypot(X, Z), 3),
                        0.0, round(v_ego, 3), round(v_ego, 3), heading_ok, round(head, 2),
                        int(di is not None)])
    od = OUT / "geom" / model_id
    od.mkdir(parents=True, exist_ok=True)
    with open(od / f"{key}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "t_sec", "track_id", "X", "Y", "Z", "dist_m", "ego_vx", "ego_vz", "ego_speed",
                    "heading_ok", "heading_deg", "depth_anchor"])
        w.writerows(out)
    vs = [r[8] for r in out]
    print(f"[geom] {key}: rows {len(out)} / tracks {len(rows)}  ego v median {np.median(vs) if vs else 0:.2f} "
          f"max {max(vs) if vs else 0:.2f}", flush=True)


# =====================================================================================================
# §4  S1·S2 칼만 트리거 (원래 trigger/trigger_v2.py, 10/3)
#   입력 (프레임마다)  dets = [{track_id, cls, X, Z, x1..y2, W, H}],  ego = {vx, vz, heading_ok}
#   ① 대상별 칼만 필터  상태 [x, z, vox, voz]  (위치 = 나 기준, 속도 = 대상 자신의 월드 속도)
#                       예측 p ← p + (v_obj − v_ego)·dt   → 내가 멈추면 정지물의 상대속도도 즉시 0
#   ② 충돌 확률        P(t_contact ≤ T_warn), t_contact = 상대 직선운동이 '내 경로 원(반경 W)'에 처음 들어가는 시각
#                       칼만 공분산에서 고정 표본 64개 → 거리 오차가 크면 자동으로 보수적
#   ③ 판정 + 이유      접근 / 측면 진입 / 따라잡음 / 정적-경로내 / 갑작 등장 / 근접   (침묵 사유도 코드로 남김)
#   ④ 히스테리시스     p ≥ p_on 이 hold_on 유지 → 켬,  p < p_off 가 hold_off 유지 → 끔
#   ⑤ 스케줄러         음성은 직렬 채널. 수준 상승(주의→위험)에만 재발화, 위험은 선점, 동시 후보는 한 문장으로
# =====================================================================================================
GROUP_OF = {
    "person": "person",
    "bicycle": "pm", "scooter": "pm", "kickboard": "pm", "motorcycle": "pm",
    "car": "vehicle", "bus": "vehicle", "truck": "vehicle", "other_vehicle": "vehicle",
    "obstacle": "obstacle", "bollard": "obstacle", "pole": "obstacle", "barricade": "obstacle",
    "tree_trunk": "obstacle", "bench": "obstacle", "kiosk": "obstacle",
    "stairs": "stairs",
}
NAME_KO = {"person": "사람", "bicycle": "자전거", "scooter": "킥보드", "kickboard": "킥보드",
           "motorcycle": "오토바이", "car": "차", "bus": "버스", "truck": "트럭", "other_vehicle": "트럭",
           "obstacle": "장애물", "bollard": "볼라드", "pole": "기둥", "barricade": "바리케이드",
           "tree_trunk": "나무", "bench": "벤치", "kiosk": "키오스크", "stairs": "계단", "traffic_light": "신호등"}

# 그룹별 물리 파라미터
#   radius  : 대상 반폭 (m)               → 경로 반경 W = ego_half_width + radius + margin
#   q       : 칼만 가속 노이즈 (m/s²)     → 얼마나 갑자기 방향을 바꿀 수 있나
#   v_prior : 처음 봤을 때 속도 사전 σ    → 정지물은 작게, 이동체는 크게
#   t_warn  : 주의 리드타임 (s) = TTS 길이 + 반응 + 회피 여유,  t_danger: 위험(선점) 리드타임
#   prio    : 동시 후보 정렬 우선순위
# v3 조정 (run_ours.py):
#   · radius — 거리를 마스크 '접지점'(대상의 가장 가까운 모서리)에서 재므로 반폭을 다시 더하지 않게 줄임
#              (v2는 중심 좌표 가정: vehicle 1.0 / pm 0.4 / obstacle 0.25)
#   · obstacle t_warn 3→4 s, t_danger 1.5→2 s — 지팡이 보행 0.6–1.0 m/s에서 2.4–4 m (10편으로 정한 값, 과적합 주의)
CLASS_CFG = {
    "person":   dict(radius=0.30, q=1.0, v_prior=1.0, t_warn=3.0, t_danger=1.5, prio=1),
    "pm":       dict(radius=0.20, q=2.0, v_prior=2.5, t_warn=4.0, t_danger=2.0, prio=3),
    "vehicle":  dict(radius=0.30, q=2.0, v_prior=3.0, t_warn=4.0, t_danger=2.0, prio=3),
    "obstacle": dict(radius=0.10, q=0.2, v_prior=0.3, t_warn=4.0, t_danger=2.0, prio=2),
    "stairs":   dict(radius=0.80, q=0.2, v_prior=0.3, t_warn=4.0, t_danger=2.0, prio=2),
}
LEVEL_RANK = {None: 0, "caution": 1, "danger": 2}


@dataclass
class TriggerConfig:
    ego_half_width: float = 0.30   # 어깨 반폭 (m)
    margin: float = 0.20           # 스치는 여유 (m)
    p_on: float = 0.50             # 켜는 충돌확률
    p_off: float = 0.25            # 끄는 충돌확률 (히스테리시스)
    p_on_noheading: float = 0.70   # 진행방향 신뢰 낮으면 더 확실해야 켬
    hold_on_sec: float = 0.15      # 디바운스 (초 단위 — 1차의 프레임 단위 결함 수정)
    hold_off_sec: float = 0.50
    min_age_sec: float = 0.25      # 트랙이 이만큼 살아야 판정
    near_m: float = 1.2            # 경로 안 근접은 TTC와 무관하게 경고
    sudden_age_sec: float = 0.6    # 등장 직후 이 시간 안에
    sudden_dist_m: float = 3.5     #   이 거리 안이면 "갑작 등장"
    self_dist_m: float = 0.6       # 착용자 팔·지팡이 (EgoBlind에서 확인된 1순위 오탐)
    static_speed: float = 0.4      # 월드 속도 이 이하 = 정지물
    depth_rel: float = 0.12        # 측정 노이즈 σ_Z = depth_rel·Z + 0.05  (v2 0.06 → MoGe-2 실측 흔들림이 커서 0.12)
    heading_sigma_deg: float = 2.0 # 진행방향 추정 오차
    ego_v_sigma: float = 0.10      # 내 속도 추정 오차 (m/s)
    nis_reset: float = 30.0        # 혁신이 이보다 크면 ID 스위치로 보고 필터 재시작
    stitch_sec: float = 0.6        # 이 시간 안에 사라진 트랙과
    stitch_m: float = 0.8          #   예측 위치가 (0.8 m + 거리의 10 %) 안이면 같은 대상으로 재연결
    n_samples: int = 64
    # 스케줄러
    min_gap_sec: float = 1.0       # 발화 사이 최소 침묵 (위험은 예외)
    max_caution_per_10s: int = 3
    pending_ttl_sec: float = 1.0   # 이 시간 안에 못 말한 주의 후보는 버림 (낡은 정보)
    sec_per_char: float = 0.11     # TTS 길이 추정 (한국어 음절당)
    upgrade_margin_sec: float = 0.8
    class_cfg: dict = field(default_factory=lambda: copy.deepcopy(CLASS_CFG))

    def W(self, group):
        return self.ego_half_width + self.class_cfg[group]["radius"] + self.margin


def contact_times(p, v, W):
    """상대 직선운동 p + v·t 가 반경 W 원에 처음 들어가는 시각. 벡터화. 못 들어가면 inf."""
    a = (v * v).sum(-1)
    b = 2.0 * (p * v).sum(-1)
    c = (p * p).sum(-1) - W * W
    t = np.full(p.shape[:-1], np.inf)
    inside = c <= 0
    t[inside] = 0.0
    disc = b * b - 4 * a * c
    ok = (~inside) & (a > 1e-8) & (disc >= 0) & (b < 0)
    t[ok] = (-b[ok] - np.sqrt(disc[ok])) / (2 * a[ok])
    return t


def cpa(p, v):
    """TCPA / DCPA (최근접 시각·거리). 해양 충돌회피(ARPA)의 표준 정의."""
    vv = float(v @ v)
    if vv < 1e-6:
        return math.inf, float(np.linalg.norm(p))
    tcpa = -float(p @ v) / vv
    return tcpa, float(np.linalg.norm(p + v * max(tcpa, 0.0)))


def clock_of(x, z):
    """진행방향 기준 방위 → 시계 방향 (정면 12시)."""
    deg = math.degrees(math.atan2(x, max(z, 1e-6)))
    h = int(round(deg / 30.0)) % 12
    return 12 if h == 0 else h


class KF:
    """상태 [x, z, vox, voz]. 위치 = 나 기준 상대, 속도 = 대상의 월드 속도 (진행방향 좌표)."""

    H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])

    def __init__(self, z, R, v_prior, q):
        self.x = np.array([z[0], z[1], 0.0, 0.0])      # 사전: 정지물
        self.P = np.diag([R[0, 0], R[1, 1], v_prior ** 2, v_prior ** 2])
        self.q = q

    def predict(self, dt, v_ego):
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        self.x = F @ self.x
        self.x[0] -= v_ego[0] * dt
        self.x[1] -= v_ego[1] * dt
        G = np.array([[0.5 * dt * dt, 0], [0, 0.5 * dt * dt], [dt, 0], [0, dt]])
        self.P = F @ self.P @ F.T + (self.q ** 2) * (G @ G.T)

    def update(self, z, R):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        nis = float(y @ np.linalg.solve(S, y))
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)
        return nis


class Track:
    def __init__(self, tid, name, t):
        self.tid = tid
        self.name = name
        self.group = GROUP_OF.get(name)
        self.kf = None
        self.t_first = t
        self.t_last = t
        self.first_dist = None
        self.on = False
        self.level = None              # 현재 켜진 수준
        self.announced = 0             # 이미 말한 최고 수준 (LEVEL_RANK)
        self.above_since = None
        self.below_since = None
        self.self_votes = 0
        self.n_obs = 0
        self.info = {}


class KalmanTrigger:
    def __init__(self, cfg=None, seed=0):
        self.cfg = cfg or TriggerConfig()
        self.eps = np.random.default_rng(seed).standard_normal((self.cfg.n_samples, 4))
        self.tracks = {}
        self.events = []               # 실제로 말한 것
        self.pending = {}              # tid -> 후보 (아직 말 못 한 것)
        self.busy_until = -1e9
        self.busy_level = None
        self.last_end = -1e9
        self.caution_times = []
        self.frame_log = []            # (t, [info...])  — §6 거리 규칙이 다시 읽음

    def _R(self, X, Z, heading_ok):
        """측정 노이즈: 거리 비례 깊이 오차 + 진행방향 오차가 만드는 측면 오차."""
        c = self.cfg
        sz = c.depth_rel * abs(Z) + 0.05
        sh = math.radians(c.heading_sigma_deg) * (1.0 if heading_ok else 3.0)
        sx = c.depth_rel * abs(X) + sh * abs(Z) + 0.05
        return np.diag([sx * sx, sz * sz])

    def _stitch(self, tid, name, t, X, Z, v_ego, seen):
        """3D 재연결: 혼잡한 장면에서 추적 ID가 바뀌어도 같은 대상이면 상태를 이어받는다."""
        g = GROUP_OF.get(name)
        best, best_d = None, None
        for otid, o in self.tracks.items():
            if otid in seen or o.kf is None or o.group != g:
                continue
            dt = t - o.t_last
            if not 0 < dt <= self.cfg.stitch_sec:
                continue
            px = o.kf.x[0] + (o.kf.x[2] - v_ego[0]) * dt
            pz = o.kf.x[1] + (o.kf.x[3] - v_ego[1]) * dt
            dist = math.hypot(px - X, pz - Z)
            gate = self.cfg.stitch_m + 0.1 * math.hypot(X, Z)
            if dist < gate and (best_d is None or dist < best_d):
                best, best_d = otid, dist
        if best is None:
            return None
        o = self.tracks.pop(best)
        self.pending.pop(best, None)
        o.tid = tid
        o.name = name
        return o

    def _risk(self, tr, v_ego):
        """대상 하나의 위험: 칼만 분포에서 64표본 → 접촉 시각 분포 → P(접촉 ≤ T_warn / T_danger)."""
        c = self.cfg
        cc = c.class_cfg[tr.group]
        W = c.W(tr.group)
        P = tr.kf.P.copy()
        P[2, 2] += c.ego_v_sigma ** 2
        P[3, 3] += c.ego_v_sigma ** 2
        L = np.linalg.cholesky(P + 1e-9 * np.eye(4))
        s = tr.kf.x + self.eps @ L.T
        p, v_rel = s[:, :2], s[:, 2:] - v_ego
        tc = contact_times(p, v_rel, W)
        p_warn = float((tc <= cc["t_warn"]).mean())
        p_danger = float((tc <= cc["t_danger"]).mean())
        t_med = float(np.median(tc))

        pm = tr.kf.x[:2]
        vw = tr.kf.x[2:]
        vr = vw - v_ego
        tcpa, dcpa = cpa(pm, vr)
        return dict(W=W, p_warn=p_warn, p_danger=p_danger, t_contact=t_med,
                    tcpa=tcpa, dcpa=dcpa, x=float(pm[0]), z=float(pm[1]),
                    dist=float(np.hypot(*pm)), v_world=vw.copy(), v_rel=vr.copy(),
                    speed_world=float(np.hypot(*vw)), closing=float(-(pm @ vr) / max(np.hypot(*pm), 1e-6)))

    def _reason(self, tr, r, age):
        c = self.cfg
        if age < c.sudden_age_sec and tr.first_dist is not None and tr.first_dist < c.sudden_dist_m:
            return "sudden_appear"
        if r["speed_world"] < c.static_speed:
            return "static_in_path"
        vx, vz = r["v_world"]
        sp = r["speed_world"]
        if vz < -0.5 * sp:
            return "approach"          # 마주 옴
        if abs(vx) > 0.5 * sp:
            return "crossing"          # 옆에서 경로로 들어옴
        return "catch_up"              # 같은 방향인데 내가 더 빠름

    def _silent_code(self, r):
        if r["z"] < 0.2:
            return "behind"
        if np.hypot(*r["v_rel"]) < 0.3:
            return "following"
        if r["tcpa"] < 0:
            return "receding"
        if r["dcpa"] > r["W"]:
            return "out_of_path"
        return "far_future"

    def step(self, t, dets, ego):
        c = self.cfg
        v_ego = np.array([float(ego.get("vx", 0.0)), float(ego.get("vz", 0.0))])
        heading_ok = bool(ego.get("heading_ok", True))
        p_on = c.p_on if heading_ok else c.p_on_noheading
        seen = set()
        infos = []

        for d in dets:
            name = d["cls"]
            tid = d["track_id"]
            tr = self.tracks.get(tid)
            if tr is None or tr.name != name and GROUP_OF.get(name) != tr.group:
                tr = self._stitch(tid, name, t, float(d["X"]), float(d["Z"]), v_ego, seen) or Track(tid, name, t)
                self.tracks[tid] = tr
            if tr.group is None:                    # 신호등 등 — 이 트리거의 대상 아님
                continue
            seen.add(tid)
            X, Z = float(d["X"]), float(d["Z"])
            R = self._R(X, Z, heading_ok)
            cc = c.class_cfg[tr.group]
            if tr.kf is None:
                tr.kf = KF(np.array([X, Z]), R, cc["v_prior"], cc["q"])
                tr.first_dist = float(np.hypot(X, Z))
            else:
                tr.kf.predict(max(t - tr.t_last, 1e-3), v_ego)
                if tr.kf.update(np.array([X, Z]), R) > c.nis_reset:
                    tr.kf = KF(np.array([X, Z]), R, cc["v_prior"], cc["q"])
            tr.t_last = t
            tr.n_obs += 1

            # 착용자 신체: 아주 가깝고 화면 하단에 붙어 있음
            H = float(d.get("H", 0) or 0)
            bottom = H > 0 and float(d.get("y2", 0)) >= 0.96 * H
            if tr.group == "person" and bottom and np.hypot(X, Z) < c.self_dist_m:
                tr.self_votes += 1
            is_self = tr.self_votes >= max(3, 0.5 * tr.n_obs)

            r = self._risk(tr, v_ego)
            age = t - tr.t_first
            r.update(track_id=tid, name=name, group=tr.group, age=age, is_self=is_self)

            # --- 판정
            cand_level, code = None, None
            if is_self:
                code = "self_body"
            elif age < c.min_age_sec:
                code = "too_young"
            elif r["z"] < 0.2:
                code = "behind"
            else:
                near = r["dist"] < c.near_m and abs(r["x"]) < r["W"]
                if r["p_danger"] >= p_on or near:
                    cand_level = "danger"
                elif r["p_warn"] >= p_on:
                    cand_level = "caution"
                else:
                    code = self._silent_code(r)

            # --- 히스테리시스 (켜기 p_on, 끄기 p_off)
            if cand_level:
                tr.below_since = None
                tr.above_since = tr.above_since if tr.above_since is not None else t
                if not tr.on and t - tr.above_since >= c.hold_on_sec:
                    tr.on = True
                if tr.on:
                    tr.level = cand_level if LEVEL_RANK[cand_level] >= LEVEL_RANK[tr.level] else tr.level
            else:
                tr.above_since = None
                if tr.on and r["p_warn"] < c.p_off:
                    tr.below_since = tr.below_since if tr.below_since is not None else t
                    if t - tr.below_since >= c.hold_off_sec:
                        tr.on, tr.level = False, None
                        tr.announced = 0          # 위험이 해소되면 다음 접근 때 다시 말할 수 있다

            r["reason"] = self._reason(tr, r, age) if tr.on else None
            if tr.on and r["dist"] < c.near_m and r["reason"] != "sudden_appear":
                r["reason"] = r["reason"] if r["p_warn"] >= p_on else "near"
            r["code"] = code
            r["on"], r["level"] = tr.on, tr.level
            tr.info = r
            infos.append(r)

            if tr.on and LEVEL_RANK[tr.level] > tr.announced:
                self.pending[tid] = dict(t=t, level=tr.level, info=r)

        for tid in [k for k, v in self.tracks.items() if t - v.t_last > 1.5]:      # 사라진 트랙 정리
            del self.tracks[tid]
            self.pending.pop(tid, None)

        self.frame_log.append((t, infos))
        return self._schedule(t)

    def _schedule(self, t):
        """음성은 한 번에 하나."""
        c = self.cfg
        for tid in list(self.pending):                # 낡은 후보 / 꺼진 후보 제거
            pnd = self.pending[tid]
            tr = self.tracks.get(tid)
            stale = pnd["level"] == "caution" and t - pnd["t"] > c.pending_ttl_sec
            if tr is None or not tr.on or stale or LEVEL_RANK[tr.level] <= tr.announced:
                del self.pending[tid]
        if not self.pending:
            return []

        def key(p):
            i = p["info"]
            return (LEVEL_RANK[p["level"]], c.class_cfg[i["group"]]["prio"], -i["t_contact"])

        best = max(self.pending.values(), key=key)
        # 처음 말하는 대상이 이미 위험 리드타임 근처면 '주의'를 건너뛰고 바로 '위험'으로 (이중 발화 방지)
        bi = best["info"]
        t_up = c.class_cfg[bi["group"]]["t_danger"] + c.upgrade_margin_sec
        if best["level"] == "caution" and bi["t_contact"] <= t_up:
            best = dict(best, level="danger")
        danger = best["level"] == "danger"
        busy = t < self.busy_until
        if busy and not (danger and self.busy_level != "danger"):
            return []
        if not danger:
            if t - self.last_end < c.min_gap_sec:
                return []
            self.caution_times = [x for x in self.caution_times if t - x < 10.0]
            if len(self.caution_times) >= c.max_caution_per_10s:
                return []

        # 같은 그룹·이유의 동시 후보는 한 문장으로
        group = [p for p in self.pending.values()
                 if p["info"]["group"] == bi["group"] and p["info"]["reason"] == bi["reason"]]
        tids = [p["info"]["track_id"] for p in group]
        text = sentence(bi, best["level"], n=len(group))
        dur = c.sec_per_char * len(text.replace(" ", "")) + 0.3
        ev = dict(t=round(t, 3), track_id=bi["track_id"], track_ids=tids, cls=bi["name"],
                  group=bi["group"], reason=bi["reason"], level=best["level"],
                  dist_m=round(bi["dist"], 2), clock=clock_of(bi["x"], bi["z"]),
                  t_contact_s=round(min(bi["t_contact"], 99.0), 2),
                  tcpa_s=round(min(bi["tcpa"], 99.0), 2) if np.isfinite(bi["tcpa"]) else None,
                  dcpa_m=round(bi["dcpa"], 2), p_coll=round(bi["p_warn"], 2),
                  text=text, duration_s=round(dur, 2), preempt=bool(busy))
        self.events.append(ev)
        self.busy_until = t + dur
        self.busy_level = best["level"]
        self.last_end = t + dur
        if not danger:
            self.caution_times.append(t)
        for tid in tids:
            tr = self.tracks.get(tid)
            if tr is not None:
                tr.announced = max(tr.announced, LEVEL_RANK[best["level"]])
            self.pending.pop(tid, None)
        return [ev]


def sentence(info, level, n=1):
    """구조화된 판정 → 짧은 한국어 한 문장. VLM 없이도 즉시 나가야 하는 문장."""
    name = NAME_KO.get(info["name"], "물체")
    unit = "명" if info["group"] == "person" else "개"
    who = name if n == 1 else "%s %d%s" % (name, n, unit)
    clk = "%d시" % clock_of(info["x"], info["z"])
    d = info["dist"]
    dist = "가까이" if d < 1.5 else "%d미터" % int(round(d))
    reason = info.get("reason")
    if level == "danger":
        return "멈춤, %s %s" % (clk, who)
    if reason == "sudden_appear":
        return "%s 가까이 %s 나타남" % (clk, who)
    if reason == "static_in_path":
        return "%s %s %s" % (clk, dist, who)
    if reason == "crossing":
        side = "왼쪽에서" if info["v_world"][0] > 0 else "오른쪽에서"
        return "%s %s 다가옴" % (side, who)
    if reason == "catch_up":
        return "%s %s 앞 %s" % (clk, dist, who)
    return "%s %s %s 접근" % (clk, dist, who)


# =====================================================================================================
# §5  tracks.csv + geom.csv → 프레임별 입력 → 칼만 트리거 실행 (원래 trigger/run_trigger.py)
# =====================================================================================================
def join_frames(tracks, geom):
    """(frame, track_id)로 묶어 프레임별 dets와 ego를 만든다. 깊이가 없는 검출은 쓸 수 없다."""
    g = {(int(r["frame"]), int(r["track_id"])): r for r in geom}
    frames, ego, times = defaultdict(list), {}, {}
    for r in geom:
        fi = int(r["frame"])
        ego[fi] = dict(vx=float(r["ego_vx"]), vz=float(r["ego_vz"]),
                       heading_ok=bool(int(float(r.get("heading_ok", 1)))))
        times[fi] = float(r["t_sec"])
    for r in tracks:
        fi, tid = int(r["frame"]), int(r["track_id"])
        times.setdefault(fi, float(r["t_sec"]))
        gr = g.get((fi, tid))
        if gr is None:
            continue
        frames[fi].append(dict(track_id=tid, cls=r["cls"], X=float(gr["X"]), Z=float(gr["Z"]),
                               x1=float(r["x1"]), y1=float(r["y1"]), x2=float(r["x2"]),
                               y2=float(r["y2"]), W=float(r["W"]), H=float(r["H"]),
                               conf=float(r.get("conf", 1.0))))
    return frames, ego, times


def run_kalman(tracks, geom, cfg=None):
    frames, ego, times = join_frames(tracks, geom)
    trig = KalmanTrigger(cfg or TriggerConfig())
    last_ego = dict(vx=0.0, vz=0.0, heading_ok=True)
    for fi in sorted(times):
        last_ego = ego.get(fi) or last_ego
        trig.step(times[fi], frames.get(fi, []), last_ego)
    return trig


# =====================================================================================================
# §6  v3 어댑터 — 입력 필터 + S2 거리 규칙 (원래 run_ours.py)
# =====================================================================================================
EXCLUDE = {"person"}      # 5주차 시나리오 4개 = 이동수단·장애물·신호·단차. 보행자는 범위 밖

# 실제 높이(m) 허용 범위 — 미터 깊이가 있으니 '물리적으로 말이 되는가'를 검사할 수 있다.
# 상한은 넉넉히 (가림막 단 전동 스쿠터 ~2.2 m + 깊이 오차) — 자막 위 3 cm '자전거' 같은 것만 거른다
H_RANGE = {"bicycle": (0.3, 3.5), "scooter": (0.25, 3.0), "motorcycle": (0.4, 4.0), "car": (0.5, 5.0),
           "bus": (1.2, 7.0), "other_vehicle": (0.4, 6.0), "obstacle": (0.12, 10.0), "stairs": (0.05, 10.0)}
MAX_AREA = {"obstacle": 0.85}       # 탈것 클래스는 0.45 (계단·에스컬레이터가 '트럭'으로 잡힘)


def companion_ids(rows, near_frac=0.5, min_obs=8):
    """화면 하단에 붙어 다니는 트랙 = 안내견·지팡이·손·휴대폰 (나와 같이 움직임) → 무시."""
    st = defaultdict(lambda: [0, 0])
    for r in rows:
        H = float(r["H"])
        bottom = float(r["y2"]) >= 0.95 * H
        big = (float(r["y2"]) - float(r["y1"])) >= 0.25 * H
        s = st[int(r["track_id"])]
        s[0] += 1
        s[1] += int(bottom and big)
    return {tid for tid, (n, b) in st.items() if n >= min_obs and b / n >= near_frac}


def plausible(r, g, fy):
    """실제 크기 게이트: 박스 높이(px)/fy × 거리 = 실제 높이(m). 자막 위 '자전거', 화면을 덮는 '트럭' 제거."""
    W, H = float(r["W"]), float(r["H"])
    bw, bh = float(r["x2"]) - float(r["x1"]), float(r["y2"]) - float(r["y1"])
    if bw * bh / (W * H) > MAX_AREA.get(r["cls"], 0.45):
        return False
    lo, hi = H_RANGE.get(r["cls"], (0.1, 10.0))
    h_m = bh / fy * float(g["Z"])
    if int(r["edge_flag"]):              # 화면에 잘린 박스는 실제보다 작게 보임 → 하한만 완화
        lo = lo * 0.3
    return lo <= h_m <= hi


S2_DIST_M = 4.5      # 경로 통로 안 정지물: 이 거리 안에 들어오면 속도와 무관하게 한 번 알림
S2_HOLD_S = 0.5


def s2_distance_events(trig, already):
    """칼만은 '몇 초 뒤 닿나'만 본다 → 느리게 걷거나 멈추면 3–5 m 앞 장애물에 침묵.
    VIABench 정답(사람 안내자)은 '앞에 ○○ 있음'을 3–6 m에서 말한다 → 거리 기준을 보조로 추가."""
    since, said, out, last_t = {}, set(already), [], -1e9
    for t, infos in trig.frame_log:
        for i in infos:
            tid = i["track_id"]
            # 장애물 클래스는 원래 정지물 — 내 속도 추정 오차로 '움직인다'고 나와도 정지로 본다
            static = i["group"] == "obstacle" or i["speed_world"] < 0.6
            inside = (i["group"] in ("obstacle", "pm", "vehicle") and static and not i["is_self"]
                      and abs(i["x"]) <= i["W"] and 0.3 < i["z"] <= S2_DIST_M and i["age"] >= 0.25)
            if not inside:
                since.pop(tid, None)
                continue
            since.setdefault(tid, t)
            if tid not in said and t - since[tid] >= S2_HOLD_S and t - last_t >= 1.0:
                said.add(tid)
                last_t = t
                info = dict(i, reason="static_in_path")
                out.append(dict(t=round(t, 3), track_id=tid, track_ids=[tid], cls=i["name"], group=i["group"],
                                reason="static_in_path", level="caution", dist_m=round(i["dist"], 2),
                                clock=clock_of(i["x"], i["z"]), text=sentence(info, "caution"),
                                scenario="S2", kind="warn", vlm="enrich", rule="distance"))
    return out


def run_s12(model_id, key, cfg=None):
    tracks = list(csv.DictReader(open(OUT / "tracks" / model_id / f"{key}.csv", encoding="utf-8")))
    geom = list(csv.DictReader(open(OUT / "geom" / model_id / f"{key}.csv", encoding="utf-8")))
    comp = companion_ids(tracks)
    tracks = [r for r in tracks if r["cls"] not in EXCLUDE and r["cls"] != "traffic_light"
              and int(r["track_id"]) not in comp]
    fy = float(np.median(np.load(OUT / "depth" / f"{key}.npz")["K"][:, 1]))
    trow = {(r["frame"], r["track_id"]): r for r in tracks}
    geom = [g for g in geom if (g["frame"], g["track_id"]) in trow and plausible(trow[(g["frame"], g["track_id"])], g, fy)]
    trig = run_kalman(tracks, geom, cfg)
    ev = []
    for e in trig.events:
        sc = "S2" if e["group"] == "obstacle" or e["reason"] == "static_in_path" else "S1"
        ev.append(dict(e, scenario=sc, kind="warn", vlm="enrich", rule="ttc"))
    ev += s2_distance_events(trig, {e["track_id"] for e in ev})
    ev.sort(key=lambda e: e["t"])
    return ev, len(comp)


# =====================================================================================================
# §7  S3 보행 신호 변화 — 기하가 아니라 색 '상태 전이' (원래 signal_s3.py)
#   ① traffic_light 트랙 crop  ② 색 = HSV 픽셀 비율(임태규 값) [× TinyTLC p(GO), 가중치 있을 때]
#   ③ 트랙별 0.6 s 창 60 % 이상 같은 판정 → 안정 상태
#   ④ 신호 하나에 '고정' (1 s 이상 산 트랙 중 크기 × 중앙 × 확신 점수 최고) → 여러 신호 사이 색 뒤집힘 방지
#   ⑤ 고정 신호의 안정 상태 STOP↔GO 전이 = warn, 처음 고정 / 갈아탈 때 = info
#   안전 우선: GO는 확률 0.9 이상만 (빨강을 초록이라 하는 것이 치명 오류)
# =====================================================================================================
class TLC:
    """선택: TinyTLC(STOP/GO, ImVisible PTL 학습) — 가중치 tlc_go_stop.pt + tlc_model.py가 있어야 함."""

    def __init__(self, path):
        import torch
        sys.path.insert(0, str(Path(path).resolve().parent))
        from tlc_model import TinyTLC
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.m = TinyTLC(ck["ncls"]).eval()
        self.m.load_state_dict(ck["state"])
        self.mean = torch.tensor(ck["mean"], dtype=torch.float32)
        self.std = torch.tensor(ck["std"], dtype=torch.float32)
        self.crop = ck["crop"]
        self.go_idx = ck["names"].index("GO")

    def p_go(self, crops_bgr):
        import torch
        if not crops_bgr:
            return np.zeros(0)
        X = np.stack([cv2.resize(c, (self.crop, self.crop), interpolation=cv2.INTER_AREA) for c in crops_bgr])
        t = torch.from_numpy(X[..., ::-1].copy()).float() / 255.0
        t = ((t - self.mean) / self.std).permute(0, 3, 1, 2)
        with torch.no_grad():
            return torch.softmax(self.m(t), 1)[:, self.go_idx].numpy()


def hsv_scores(crop_bgr):
    """빨강·초록 램프 픽셀 비율 (임태규 노트북 기준값)."""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    red = (((h < 10) | (h > 170)) & (s > 70) & (v > 100)).mean()
    green = ((h > 35) & (h < 90) & (s > 50) & (v > 80)).mean()
    return float(red), float(green)


def frame_state(p_go, red, green, go_th=0.9, stop_th=0.3):
    """TLC 없으면 p_go = 0.5 (중립) → HSV 규칙만으로 판정된다."""
    if red > 0.02 and red > 1.5 * green:
        return "STOP"
    if p_go >= go_th and not (red > green):
        return "GO"
    if green > 0.02 and green > 1.5 * red and p_go >= 0.5:
        return "GO"
    if p_go <= stop_th:
        return "STOP"
    return "UNK"


class SignalTrigger:
    def __init__(self, tlc_path=None, window_s=0.6, agree=0.6, min_age_s=1.0, pad=0.15):
        self.tlc = TLC(tlc_path) if tlc_path else None
        self.window_s, self.agree, self.min_age_s, self.pad = window_s, agree, min_age_s, pad
        self.hist = defaultdict(deque)        # tid -> deque[(t, state)]
        self.first = {}
        self.last_seen = {}
        self.stable = None                    # 전역 안정 상태 (고정된 신호 기준)
        self.stable_tid = None
        self.prev_track_state = None
        self.events = []
        self.log = []                         # (t, tid, p_go, red, green, state)

    def _crop(self, im, r):
        H, W = im.shape[:2]
        x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
        pw, ph = (x2 - x1) * self.pad, (y2 - y1) * self.pad
        x1, y1 = int(max(0, x1 - pw)), int(max(0, y1 - ph))
        x2, y2 = int(min(W, x2 + pw)), int(min(H, y2 + ph))
        if x2 - x1 < 3 or y2 - y1 < 3:
            return None
        return im[y1:y2, x1:x2]

    def _track_stable(self, tid):
        q = self.hist[tid]
        states = [x for _, x in q if x != "UNK"]
        if len(states) < 3:
            return None, 0
        top = max(set(states), key=states.count)
        return (top if states.count(top) / len(q) >= self.agree else None), len(states)

    def step(self, t, im, lights):
        crops, rows = [], []
        for r in lights:
            c = self._crop(im, r)
            if c is not None:
                crops.append(c)
                rows.append(r)
        pg = self.tlc.p_go(crops) if self.tlc else np.full(len(crops), 0.5)
        H, W = im.shape[:2]
        cand = []
        for r, c, p in zip(rows, crops, pg):
            tid = int(r["track_id"])
            red, green = hsv_scores(c)
            st = frame_state(float(p), red, green)
            self.first.setdefault(tid, t)
            self.last_seen[tid] = t
            q = self.hist[tid]
            q.append((t, st))
            while q and t - q[0][0] > self.window_s:
                q.popleft()
            self.log.append((round(t, 3), tid, round(float(p), 3), round(red, 4), round(green, 4), st))
            if t - self.first[tid] < self.min_age_s:
                continue
            stab, n_conf = self._track_stable(tid)
            if stab is None:
                continue
            area = (r["x2"] - r["x1"]) * (r["y2"] - r["y1"]) / (W * H)
            cx = (r["x1"] + r["x2"]) / 2 / W
            cand.append((n_conf * np.sqrt(area) * (1.0 - abs(cx - 0.5)), tid, stab))
        out = []
        lost = self.stable_tid is None or t - self.last_seen.get(self.stable_tid, -1e9) > 1.0
        if lost and cand:
            _, tid, stab = max(cand)
            prev = self.stable
            self.stable_tid, self.stable = tid, stab
            self.prev_track_state = stab
            if prev != stab:
                out.append(self._ev(t, tid, "info", prev, stab))
        elif not lost:
            stab, _ = self._track_stable(self.stable_tid)
            if stab is not None and stab != self.prev_track_state:
                prev = self.prev_track_state
                self.prev_track_state = stab
                self.stable = stab
                out.append(self._ev(t, self.stable_tid, "warn", prev, stab))
        self.events += out
        return out

    def _ev(self, t, tid, kind, prev, cur):
        if kind == "warn":
            text = "초록불로 바뀜, 건너도 됩니다" if cur == "GO" else "빨간불로 바뀜, 멈추세요"
        else:
            text = "초록불입니다" if cur == "GO" else "빨간불입니다"
        return dict(t=round(t, 3), scenario="S3", kind=kind, track_id=tid, prev=prev, state=cur,
                    text=text, level="caution", vlm_verify=kind == "warn" or prev is None)


def run_s3(model_id, key, tlc_path=None):
    by = defaultdict(list)
    for r in csv.DictReader(open(OUT / "tracks" / model_id / f"{key}.csv", encoding="utf-8")):
        if r["cls"] == "traffic_light":
            by[int(r["frame"])].append({k: (float(v) if k in ("x1", "y1", "x2", "y2", "conf") else v) for k, v in r.items()})
    if not by:
        return [], []
    st = SignalTrigger(tlc_path)
    for fidx, t, im in iter_frames(key, 15.0, max_side=1920):
        if fidx in by:
            st.step(t, im, by[fidx])
    return st.events, st.log


# =====================================================================================================
# §8  S4 단차·계단 (원래 step_s4.py) — stairs 클래스(AP 0.21, 라벨 2/3가 맨홀·연석)만 믿지 않고 지면 모델로도 판정
#   깊이 프레임마다 통로(가운데 24 % 폭) 세로 깊이 프로파일
#   평평한 지면에서는 1/깊이가 행 v의 1차식 (핀홀) → 근거리 바닥 행(58–80 %)에서 1/d = a·v + b 맞춤
#   → 위쪽(먼 쪽) 행(22–58 %)으로 연장해 예측 깊이와 실제 비교
#     drop : 실제가 예측보다 1.5배 이상 멀고 행 사이 깊이가 25 % 이상 뜀 → 내려가는 계단·꺼짐
#     rise : 실제가 예측의 0.75배 미만 (예측 1.2–4.5 m) → 올라가는 계단·턱 (또는 벽 → VLM이 가름)
#   후보 = (score > 0.35 가 1 s 지속) 또는 stairs 트랙 conf ≥ 0.4 가 통로 안 → 발화, 같은 단차 6 s 재발화 금지
# =====================================================================================================
def frame_score(dep, K, stride, boxes=(), fit=(0.58, 0.80), look=(0.22, 0.58)):
    H, W = dep.shape
    u0, u1 = int(W * 0.38), int(W * 0.62)
    m = np.ones((H, W), bool)
    for x1, y1, x2, y2 in boxes:
        m[max(0, int(y1 / stride)):int(y2 / stride) + 1, max(0, int(x1 / stride)):int(x2 / stride) + 1] = False
    prof = np.full(H, np.nan)
    for v in range(int(H * look[0]), int(H * fit[1])):
        r = dep[v, u0:u1][m[v, u0:u1]]
        r = r[r > 0.2]
        if r.size >= 0.3 * (u1 - u0):
            prof[v] = np.median(r)
    out = dict(step=0.0, sign=0, kind="none", fit_ok=False)
    fv = np.arange(int(H * fit[0]), int(H * fit[1]))
    fv = fv[np.isfinite(prof[fv]) & (prof[fv] > 0.5) & (prof[fv] < 4.0)]
    if len(fv) < 8:
        return out
    A = np.stack([fv, np.ones_like(fv)], 1).astype(float)
    coef, *_ = np.linalg.lstsq(A, 1.0 / prof[fv], rcond=None)
    if coef[0] <= 0:                               # 위로 갈수록 멀어져야 지면
        return out
    out["fit_ok"] = True
    lv = np.arange(int(H * look[0]), int(H * fit[0]))
    pred_inv = coef[0] * lv + coef[1]
    ok = np.isfinite(prof[lv]) & (pred_inv > 1 / 6.0)   # 예측 6 m 이내 (지평선 아래)
    if ok.sum() < 6:
        return out
    lv, pred, act = lv[ok], 1.0 / pred_inv[ok], prof[lv][ok]
    ratio = act / pred
    near = pred < 4.5
    rise = (ratio < 0.75) & near & (pred > 1.2)
    jumps = act[:-1] / np.maximum(act[1:], 1e-3)          # 위 행 / 아래 행
    drop = (ratio > 1.5) & near
    drop_score = float(drop.mean()) if np.nanmax(jumps, initial=0) > 1.25 else 0.0
    rise_score = float(rise.mean())
    if drop_score >= rise_score:
        out.update(step=drop_score, sign=-1, kind="down")
    else:
        out.update(step=rise_score, sign=1, kind="up")
    return out


def run_s4(model_id, key, hold_s=1.0, refractory_s=6.0, step_m=0.35):
    d = np.load(OUT / "depth" / f"{key}.npz")
    ts, deps, Ks, stride = d["t"], d["depth"].astype(np.float32), d["K"], int(d["stride"])
    stairs, bx = defaultdict(list), defaultdict(list)
    for r in csv.DictReader(open(OUT / "tracks" / model_id / f"{key}.csv", encoding="utf-8")):
        if r["cls"] != "stairs":
            bx[round(float(r["t_sec"]) * 2) / 2].append(tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2")))
        if r["cls"] == "stairs" and float(r["conf"]) >= 0.4:
            cx = (float(r["x1"]) + float(r["x2"])) / 2 / float(r["W"])
            if 0.2 < cx < 0.8 and float(r["y2"]) > 0.5 * float(r["H"]):
                stairs[round(float(r["t_sec"]) * 2) / 2].append(r)
    events = []
    cand_since, last_ev = None, -1e9
    for t, dep, K in zip(ts, deps, Ks):
        s = frame_score(dep, K, stride, boxes=bx.get(round(float(t) * 2) / 2, ()))
        geo = s["step"] > step_m
        det = len(stairs.get(round(float(t) * 2) / 2, [])) > 0
        if geo or det:
            cand_since = cand_since if cand_since is not None else float(t)
            if (float(t) - cand_since >= hold_s - 1e-6 or det) and float(t) - last_ev > refractory_s:
                kind = "up" if s["sign"] > 0 else "down"
                txt = "앞에 계단·턱, 올라감" if kind == "up" else "앞에 계단·턱, 내려감"
                events.append(dict(t=round(float(t), 3), scenario="S4", kind="warn", text=txt, vlm="verify",
                                   cue="det" if det else kind, score=round(s["step"], 3)))
                last_ev = float(t)
        else:
            cand_since = None
    return events


# =====================================================================================================
# §9  실행 · 채점
# =====================================================================================================
def run_events(model_id, clips, use_s4=True, tlc_path=None, tag="v3"):
    od = OUT / "events" / f"{tag}_{model_id}"
    od.mkdir(parents=True, exist_ok=True)
    for key in clips:
        ev12, ncomp = run_s12(model_id, key)
        ev3, log3 = run_s3(model_id, key, tlc_path)
        ev4 = run_s4(model_id, key) if use_s4 else []
        ev = sorted(ev12 + ev3 + ev4, key=lambda e: e["t"])
        json.dump({"clip": key, "system": tag, "model": model_id, "dur": clip_info(key)["dur"],
                   "events": ev, "companion_tracks": ncomp, "s3_log": log3},
                  open(od / f"{key}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
        print(f"[run] {key:<22} {len(ev):>3} events  "
              + "  ".join(f"{e['t']:.1f}s {e['scenario']} {e.get('text', '')}" for e in ev[:8]), flush=True)


def run_eval(model_id, tags=("v3", "b0")):
    from evaluate import eval_system
    for tag in tags:
        d = OUT / "events" / f"{tag}_{model_id}"
        if not d.exists():
            continue
        evs, durs = {}, {}
        for key in CLIPS:
            j = json.load(open(d / f"{key}.json", encoding="utf-8"))
            evs[key] = [e for e in j["events"] if e.get("kind", "warn") == "warn"]
            durs[key] = j["dur"]
        _, s = eval_system(evs, durs)
        by = "  ".join(f"{k} {v['hit_lenient']}/{v['warn']} fp{v['fp']}" for k, v in s["by_scenario"].items())
        print(f"[eval] {tag:<4} PDR(2s 앞 허용) {s['PDR_lenient']:.2f}  정밀도 {s['precision']:.2f}  "
              f"오탐/분 {s['fp_per_min']:.2f}  정탐 {s['tp']} / 오탐 {s['fp']}  |  {by}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="v3 기하 트리거")
    ap.add_argument("stage", choices=["depth", "vo", "geom", "run", "eval", "all"])
    ap.add_argument("--model", default="E0")
    ap.add_argument("--clips", nargs="*", default=list(CLIPS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--hz", type=float, default=2.0, help="깊이 추정 빈도 (CPU 2 Hz, GPU면 15)")
    ap.add_argument("--no-s4", action="store_true")
    ap.add_argument("--tlc", default=None, help="tlc_go_stop.pt 경로 (없으면 S3 HSV 단독)")
    a = ap.parse_args()
    st = a.stage
    if st in ("depth", "all"):
        run_depth(a.clips, a.hz, a.device)
    if st in ("vo", "all"):
        for k in a.clips:
            run_vo_clip(a.model, k)
    if st in ("geom", "all"):
        for k in a.clips:
            run_geom_clip(a.model, k)
    if st in ("run", "all"):
        run_events(a.model, a.clips, use_s4=not a.no_s4, tlc_path=a.tlc)
    if st in ("eval", "all"):
        run_eval(a.model)


if __name__ == "__main__":
    main()
