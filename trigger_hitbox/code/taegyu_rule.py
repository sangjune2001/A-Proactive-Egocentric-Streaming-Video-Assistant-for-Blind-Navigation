# 임태규 taegyu 브랜치 rule_trigger_upload.ipynb 셀 4~7 원문 그대로 (B0 기준선)
from collections import defaultdict, deque
from types import MappingProxyType

import cv2
import numpy as np

# Test Recall 46.67%를 기록한 기존 규칙의 확정값입니다.
# MappingProxyType은 의도하지 않은 항목 변경을 막습니다.
FINAL_PARAMS = MappingProxyType({
    "GROWTH_THRESHOLD": 1.10,
    "CENTER_SPEED_MAX": 0.08,
    "TRIGGER_STREAK": 10,
    "RAPID_GROWTH_THRESHOLD": 2.00,
    "RAPID_CENTER_SPEED_MIN": 0.10,
    "RAPID_TRIGGER_STREAK": 3,
    "APPROACH_WINDOW": 9,
    "APPROACH_AVG": 3,
    "MAX_OBSERVATION_GAP": 0.30,
})

VEHICLE_NAMES = (
    "bicycle", "car", "motorcycle", "bus", "truck", "other_vehicle", "scooter",
)
OBSTACLE_NAMES = (
    "bench", "potted_plant", "chair", "obstacle", "stairs", "tree",
)

# YOLO / BYTETrack 연결부에 전달할 기존 검출 설정
DETECTOR_SETTINGS = MappingProxyType({
    "conf": 0.5,
    "imgsz": 640,
    "tracker": "bytetrack.yaml",
    "persist": True,
    "verbose": False,
})

# 기존 카메라 움직임 보정(GMC) 설정
GMC_IMAGE_SIZE = 640
GMC_MIN_POINTS = 20
GMC_MIN_INLIER_RATIO = 0.6
GMC_MAX_FB_ERROR = 1.5

def classify_light_color(frame, box):
    x1, y1, x2, y2 = map(int, box)
    height, width = frame.shape[:2]

    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    red_mask = ((h < 10) | (h > 170)) & (s > 70) & (v > 100)
    green_mask = (h > 35) & (h < 90) & (s > 50) & (v > 80)

    red_score = red_mask.sum()
    green_score = green_mask.sum()
    min_pixels = max(3, int(roi.shape[0] * roi.shape[1] * 0.005))

    if max(red_score, green_score) < min_pixels:
        return "unknown"
    if red_score > green_score * 1.2:
        return "red"
    if green_score > red_score * 1.2:
        return "green"
    return "unknown"


class BackgroundMotion:
    def __init__(self):
        self.prev_gray = None
        self.prev_mask = None

    def update(self, frame, boxes):
        h, w = frame.shape[:2]
        scale = min(1.0, GMC_IMAGE_SIZE / max(h, w))
        sw = max(2, round(w * scale))
        sh = max(2, round(h * scale))

        gray = cv2.cvtColor(
            cv2.resize(frame, (sw, sh)), cv2.COLOR_BGR2GRAY
        )
        mask = np.full((sh, sw), 255, dtype=np.uint8)
        sx, sy = sw / w, sh / h

        for box in boxes:
            x1, y1, x2, y2 = map(float, box)
            margin = 5
            left = max(0, int(np.floor(x1 * sx)) - margin)
            top = max(0, int(np.floor(y1 * sy)) - margin)
            right = min(sw, int(np.ceil(x2 * sx)) + margin)
            bottom = min(sh, int(np.ceil(y2 * sy)) + margin)
            if right > left and bottom > top:
                mask[top:bottom, left:right] = 0

        prev_gray, prev_mask = self.prev_gray, self.prev_mask
        self.prev_gray, self.prev_mask = gray, mask

        info = dict(
            camera_ok=False,
            camera_reason="INIT",
            camera_points=0,
            camera_inliers=0,
            camera_inlier_ratio=0.0,
        )

        if prev_gray is None:
            return None, info
        if prev_gray.shape != gray.shape:
            info["camera_reason"] = "SIZE_CHANGED"
            return None, info

        points0 = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=500,
            qualityLevel=0.01,
            minDistance=10,
            mask=prev_mask,
            blockSize=7,
        )

        if points0 is None or len(points0) < GMC_MIN_POINTS:
            info["camera_reason"] = "FEW_FEATURES"
            return None, info

        points1, status1, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, points0, None,
            winSize=(21, 21), maxLevel=3,
        )

        if points1 is None or status1 is None:
            info["camera_reason"] = "FLOW_FAILED"
            return None, info

        valid = (
            status1.ravel().astype(bool)
            & np.isfinite(points1.reshape(-1, 2)).all(axis=1)
        )
        p0 = points0.reshape(-1, 2)[valid]
        p1 = points1.reshape(-1, 2)[valid]

        if len(p0) < GMC_MIN_POINTS:
            info["camera_reason"] = "FEW_TRACKS"
            return None, info

        back, status2, _ = cv2.calcOpticalFlowPyrLK(
            gray, prev_gray, p1.reshape(-1, 1, 2), None,
            winSize=(21, 21), maxLevel=3,
        )

        if back is None or status2 is None:
            info["camera_reason"] = "BACK_FLOW_FAILED"
            return None, info

        fb_error = np.linalg.norm(back.reshape(-1, 2) - p0, axis=1)
        valid = (
            status2.ravel().astype(bool)
            & np.isfinite(fb_error)
            & (fb_error <= GMC_MAX_FB_ERROR)
            & (p1[:, 0] >= 0) & (p1[:, 0] < sw)
            & (p1[:, 1] >= 0) & (p1[:, 1] < sh)
        )
        p0, p1 = p0[valid], p1[valid]

        if len(p1):
            indices = np.floor(p1).astype(int)
            background = mask[indices[:, 1], indices[:, 0]] > 0
            p0, p1 = p0[background], p1[background]

        info["camera_points"] = len(p0)

        if len(p0) < GMC_MIN_POINTS:
            info["camera_reason"] = "FEW_BACKGROUND_TRACKS"
            return None, info

        matrix, inliers = cv2.estimateAffinePartial2D(
            p0, p1,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.0,
            maxIters=2000,
            confidence=0.99,
            refineIters=10,
        )

        if matrix is None or inliers is None:
            info["camera_reason"] = "RANSAC_FAILED"
            return None, info

        inlier_mask = inliers.ravel().astype(bool)
        count = int(inlier_mask.sum())
        ratio = count / len(p0)

        info["camera_inliers"] = count
        info["camera_inlier_ratio"] = ratio

        if (
            not np.isfinite(matrix).all()
            or count < GMC_MIN_POINTS
            or ratio < GMC_MIN_INLIER_RATIO
        ):
            info["camera_reason"] = "LOW_CONFIDENCE"
            return None, info

        spread = np.ptp(p0[inlier_mask], axis=0)
        if spread[0] < sw * 0.2 or spread[1] < sh * 0.2:
            info["camera_reason"] = "LOCAL_FEATURES"
            return None, info

        estimated_scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
        if not 0.9 <= estimated_scale <= 1.1:
            info["camera_reason"] = "EXTREME_SCALE"
            return None, info

        small_transform = np.eye(3)
        small_transform[:2] = matrix
        resize_transform = np.diag([sx, sy, 1.0])
        full_transform = (
            np.linalg.inv(resize_transform)
            @ small_transform
            @ resize_transform
        )

        info["camera_ok"] = True
        info["camera_reason"] = "OK"
        return full_transform[:2], info

class RuleBasedTrigger:
    """고정 규칙. BGR 프레임과 추적 결과를 받아 이번 프레임의 이벤트만 반환합니다."""

    def __init__(self, class_names, fps):
        names = dict(enumerate(class_names)) if isinstance(class_names, list) else class_names
        self.names = {
            int(k): str(v).strip().lower().replace(" ", "_")
            for k, v in names.items()
        }
        name_to_id = {v: k for k, v in self.names.items()}
        if "traffic_light" not in name_to_id:
            raise ValueError("traffic_light 클래스가 필요합니다.")
        self.vehicle_classes = {name_to_id[n] for n in VEHICLE_NAMES if n in name_to_id}
        self.obstacle_classes = {name_to_id[n] for n in OBSTACLE_NAMES if n in name_to_id}
        self.traffic_light_class = name_to_id["traffic_light"]
        self.target_classes = (
            [name_to_id[n] for n in VEHICLE_NAMES if n in name_to_id]
            + [name_to_id[n] for n in OBSTACLE_NAMES if n in name_to_id]
            + [self.traffic_light_class]
        )
        self.params = FINAL_PARAMS
        self.reset(fps=fps)

    def reset(self, *, fps=None):
        """새 영상/스트림 또는 tracker ID 초기화 시 호출합니다. Tracker도 별도로 초기화하세요."""
        if fps is not None:
            fps = float(fps)
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError("fps는 양수여야 합니다.")
            self.fps = fps
        self.motion = BackgroundMotion()
        self.history = defaultdict(lambda: deque(maxlen=self.params["APPROACH_WINDOW"]))
        self.last_seen = {}
        self.normal_streak = {}
        self.rapid_streak = {}
        self.approach_fired = set()
        self.colors = defaultdict(lambda: deque(maxlen=5))
        self.stable_colors = {}
        self.light_fired = set()
        self._last_frame = 0
        self._last_time = None

    def update(self, frame_bgr, detections, *, frame_index=None, time_s=None):
        """detections: box(xyxy 픽셀), class_id, track_id를 가진 dict의 iterable.

        빈 프레임도 detections=[]로 호출합니다. time_s는 영상/캡처 시각(초)입니다.
        frame_index는 1부터 시작하는 원본 프레임 번호이며 반드시 증가해야 합니다.
        """
        fi = self._last_frame + 1 if frame_index is None else frame_index
        if not isinstance(fi, (int, np.integer)) or fi <= self._last_frame:
            raise ValueError("frame_index는 이전보다 큰 양의 정수여야 합니다.")
        fi = int(fi)
        t = (fi - 1) / self.fps if time_s is None else float(time_s)
        if not np.isfinite(t) or t < 0 or (self._last_time is not None and t <= self._last_time):
            raise ValueError("time_s는 0 이상의 증가하는 영상 시각이어야 합니다.")
        if (not isinstance(frame_bgr, np.ndarray) or frame_bgr.dtype != np.uint8
                or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3
                or min(frame_bgr.shape[:2]) < 2):
            raise ValueError("frame_bgr는 uint8 H×W×3 BGR 이미지여야 합니다.")
        prepared = []
        for order, detection in enumerate(detections):
            cid = int(detection["class_id"])
            if cid not in self.target_classes:
                continue
            box = np.asarray(detection["box"], dtype=float)
            if (box.shape != (4,) or not np.isfinite(box).all()
                    or box[2] < box[0] or box[3] < box[1]):
                raise ValueError("box는 유한한 [x1, y1, x2, y2] 픽셀 좌표여야 합니다.")
            tid = detection.get("track_id")
            prepared.append(dict(
                box=box.tolist(), class_id=cid,
                **{"class": self.names[cid]},
                track_id=None if tid is None else int(tid),
                order=int(detection.get("order", order)),
                color=(classify_light_color(frame_bgr, box)
                       if cid == self.traffic_light_class else None),
            ))
        matrix, camera = self.motion.update(frame_bgr, [d["box"] for d in prepared])
        height, width = frame_bgr.shape[:2]
        events = self._step(dict(
            frame=fi, time_s=t, width=width, height=height,
            matrix=matrix, camera=camera, detections=prepared,
        ))
        self._last_frame, self._last_time = fi, t
        return events

    def update_yolo(self, frame_bgr, result, *, frame_index=None, time_s=None):
        """Ultralytics model.track(...)[0] 결과를 바로 연결하는 어댑터."""
        detections = []
        if result.boxes is not None:
            for order, box in enumerate(result.boxes):
                detections.append(dict(
                    box=box.xyxy[0].cpu().tolist(),
                    class_id=int(box.cls[0]),
                    track_id=int(box.id[0]) if box.id is not None else None,
                    order=order,
                ))
        return self.update(
            frame_bgr, detections, frame_index=frame_index, time_s=time_s
        )

    def _step(self, observation):
        """동일한 GMC/HSV 관측에 원래 규칙을 순차 적용합니다."""
        p = self.params
        window, avg, gap = p["APPROACH_WINDOW"], p["APPROACH_AVG"], p["MAX_OBSERVATION_GAP"]
        f = observation
        fi, t, camera = f["frame"], f["time_s"], f["camera"]
        if not camera["camera_ok"]:
            self.history.clear()
        else:
            m = np.asarray(f["matrix"], dtype=float)
            for key in list(self.history):
                hist = self.history[key]
                if not hist or (fi - hist[-1]["frame"]) / self.fps > gap:
                    del self.history[key]
                    continue
                for old in hist:
                    old["corrected"] = m[:, :2] @ old["corrected"] + m[:, 2]

        events = []
        for detection in f["detections"]:
            tid, cid = detection["track_id"], detection["class_id"]
            if tid is None:
                continue
            base = dict(
                frame=fi, time_s=t, track_id=tid, class_id=cid,
                **{"class": detection["class"]}, order=detection["order"],
            )
            if cid == self.traffic_light_class:
                key = (cid, tid)
                self.colors[key].append(detection["color"])
                recent = list(self.colors[key])[-3:]
                now = (
                    recent[0] if len(recent) == 3 and len(set(recent)) == 1
                    and recent[0] in ("red", "green") else None
                )
                previous = self.stable_colors.get(key)
                event_key = (cid, tid, previous, now)
                if (now is not None and previous in ("red", "green")
                        and now != previous and event_key not in self.light_fired):
                    self.light_fired.add(event_key)
                    events.append(dict(
                        base, event=f"light_{previous}_to_{now}",
                        trigger_type="light_change",
                        message=f"light ID={tid} {previous}->{now}",
                    ))
                if now is not None:
                    self.stable_colors[key] = now
                continue

            if cid not in self.vehicle_classes and cid not in self.obstacle_classes:
                continue
            x1, y1, x2, y2 = detection["box"]
            center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            missing = max(0, fi - self.last_seen[tid] - 1) if tid in self.last_seen else 0
            self.last_seen[tid] = fi
            hist = self.history[tid]
            hist.append(dict(
                frame=fi, time=t, area=max(0, x2 - x1) * max(0, y2 - y1),
                raw=center.copy(), corrected=center.copy(),
            ))
            growth = speed = None
            if len(hist) >= window:
                seq = list(hist)
                past, recent = seq[:avg], seq[-avg:]
                growth = float(
                    np.mean([o["area"] for o in recent])
                    / max(np.mean([o["area"] for o in past]), 1.0)
                )
                dt = max(
                    np.mean([o["time"] for o in recent])
                    - np.mean([o["time"] for o in past]), 1e-6,
                )
                delta = (
                    np.mean([o["corrected"] for o in recent], axis=0)
                    - np.mean([o["corrected"] for o in past], axis=0)
                )
                speed = float(np.linalg.norm(delta / np.array([f["width"], f["height"]])) / dt)
            valid = camera["camera_ok"] and len(hist) >= window
            normal = valid and growth >= p["GROWTH_THRESHOLD"] and speed <= p["CENTER_SPEED_MAX"]
            rapid = (valid and cid in self.vehicle_classes
                     and growth >= p["RAPID_GROWTH_THRESHOLD"]
                     and speed >= p["RAPID_CENTER_SPEED_MIN"])
            ns = (0 if missing else self.normal_streak.get(tid, 0)) + 1 if normal else 0
            rs = (0 if missing else self.rapid_streak.get(tid, 0)) + 1 if rapid else 0
            self.normal_streak[tid], self.rapid_streak[tid] = ns, rs
            if tid not in self.approach_fired and (ns >= p["TRIGGER_STREAK"] or rs >= p["RAPID_TRIGGER_STREAK"]):
                self.approach_fired.add(tid)
                kind = "rapid_entry" if rs >= p["RAPID_TRIGGER_STREAK"] else "approach"
                events.append(dict(
                    base, event="approach", trigger_type=kind,
                    message=f"{detection['class']} ID={tid} {kind} g={growth:.3f} v={speed:.4f}",
                ))
        events.sort(key=lambda event: (event["frame"], event["order"]))
        return events
