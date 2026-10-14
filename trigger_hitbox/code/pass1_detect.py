"""1단계: 영상마다 YOLO 검출·ByteTrack 추적 + 태규 규칙(B0) 실행 + 배경 광류 저장.

무거운 계산(YOLO)은 여기서 한 번만 하고, 2단계(새 트리거 A/B)는 저장된 결과로 돈다.
usage: python pass1_detect.py WEIGHTS VIDEO OUT_DIR
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
import taegyu_rule as tr  # noqa: E402


class RecordingMotion(tr.BackgroundMotion):
    """태규 GMC와 같은 계산을 하면서, 배경 특징점 쌍(p0→p1, 원본 좌표)도 남긴다."""

    def update(self, frame, boxes):
        h, w = frame.shape[:2]
        scale = min(1.0, tr.GMC_IMAGE_SIZE / max(h, w))
        sw, sh = max(2, round(w * scale)), max(2, round(h * scale))
        gray = cv2.cvtColor(cv2.resize(frame, (sw, sh)), cv2.COLOR_BGR2GRAY)
        mask = np.full((sh, sw), 255, np.uint8)
        sx, sy = sw / w, sh / h
        for box in boxes:
            x1, y1, x2, y2 = map(float, box)
            mask[max(0, int(y1 * sy) - 5):min(sh, int(np.ceil(y2 * sy)) + 5),
                 max(0, int(x1 * sx) - 5):min(sw, int(np.ceil(x2 * sx)) + 5)] = 0
        self.last_pairs = None
        if self.prev_gray is not None and self.prev_gray.shape == gray.shape:
            p0 = cv2.goodFeaturesToTrack(self.prev_gray, 500, 0.01, 10,
                                         mask=self.prev_mask, blockSize=7)
            if p0 is not None and len(p0) >= 8:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None,
                                                     winSize=(21, 21), maxLevel=3)
                if p1 is not None:
                    back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev_gray, p1, None,
                                                            winSize=(21, 21), maxLevel=3)
                    a, b = p0.reshape(-1, 2), p1.reshape(-1, 2)
                    fb = np.linalg.norm(back.reshape(-1, 2) - a, axis=1)
                    ok = st.ravel().astype(bool) & st2.ravel().astype(bool) & (fb <= 1.5)
                    self.last_pairs = np.hstack([a[ok] / [sx, sy], b[ok] / [sx, sy]]).astype(np.float32)
        return super().update(frame, boxes)


def main(weights, video, out_dir, max_frames=0):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap_w, cap_h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    model = YOLO(weights)
    trig = tr.RuleBasedTrigger(model.names, fps)
    trig.motion = RecordingMotion()
    # 태규 규칙과 같은 대상 클래스 + 사람(시각화·참고용)도 함께 추적
    classes = sorted(set(trig.target_classes) | {k for k, v in model.names.items() if v == "person"})
    frames, flows, b0_events = [], {}, []
    fi, t0 = 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok or (int(max_frames) and fi >= int(max_frames)):
            break
        fi += 1
        t = (fi - 1) / fps
        res = model.track(frame, classes=classes, **dict(tr.DETECTOR_SETTINGS))[0]
        dets = []
        if res.boxes is not None:
            for b in res.boxes:
                dets.append(dict(box=[round(v, 1) for v in b.xyxy[0].tolist()], cls=int(b.cls[0]),
                                 conf=round(float(b.conf[0]), 3),
                                 tid=int(b.id[0]) if b.id is not None else None))
        # B0는 태규 코드 그대로 (사람은 그 안에서 걸러짐)
        evs = trig.update_yolo(frame, res, frame_index=fi, time_s=t)
        b0_events += evs
        m = trig.motion
        frames.append(dict(f=fi, t=round(t, 4), dets=dets))
        if m.last_pairs is not None:
            flows[f"f{fi}"] = m.last_pairs
        if fi % 300 == 0:
            print(f"{Path(video).name} {fi} frames {time.time() - t0:.0f}s", flush=True)
    cap.release()
    meta = dict(video=str(video), fps=fps, n=fi, w=int(cap_w), h=int(cap_h), names=model.names, sec=round(time.time() - t0, 1))
    (out / "dets.json").write_text(json.dumps(dict(meta=meta, frames=frames)), encoding="utf-8")
    np.savez_compressed(out / "flows.npz", **flows)
    (out / "events_B0.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in b0_events), encoding="utf-8")
    print("done", meta, "B0 events", len(b0_events), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:5])
