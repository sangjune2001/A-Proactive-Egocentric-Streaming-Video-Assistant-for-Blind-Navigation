"""시연 영상: run_demo 결과(dets · log · timeline) → demo.mp4 (원본 영상 + 박스 + 트리거 표시 + 자막 + 실제 재생된 소리).

소리는 timeline의 실제 재생 구간 그대로 (끊긴 설명은 끊긴 지점까지) 섞는다 → 영상에서 들리는 것 = 시스템이 낸 것.
    python render.py <clip> <run 폴더>       # run_demo.py --render 가 자동으로 부름
"""
from __future__ import annotations

import json
import subprocess
import sys
import wave
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import core  # noqa: F401
from common import OUT, clip_info, iter_frames

SR = 22050
FONT = "C:/Windows/Fonts/malgunbd.ttf"


def read_wav(path):
    with wave.open(str(path)) as w:
        sr, n, ch, sw = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
        x = np.frombuffer(w.readframes(n), dtype={1: np.int8, 2: np.int16, 4: np.int32}[sw]).astype(np.float32)
    x = x.reshape(-1, ch).mean(1) / float(2 ** (8 * sw - 1))
    if sr != SR:
        from scipy.signal import resample_poly
        g = np.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g)
    return x


def mix_audio(timeline, dur):
    out = np.zeros(int((dur + 1.0) * SR), np.float32)
    for it in timeline:
        if "t_start" not in it:
            continue
        x = read_wav(it["wav"])
        a = int(it["t_start"] * SR)
        n = min(len(x), int((it["t_end"] - it["t_start"]) * SR), len(out) - a)
        if n > 0:
            out[a:a + n] += x[:n]
    return np.clip(out, -1, 1)


def write_wav(path, x):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((x * 32767).astype(np.int16).tobytes())


def render(key, od: Path, fps=15.0, long_side=1280):
    od = Path(od)
    timeline = json.load(open(od / "timeline.json", encoding="utf-8"))
    rows = [json.loads(x) for x in open(od / "log.jsonl", encoding="utf-8")]
    events = [r["ev"] for r in rows if r["stage"] == "event"]
    dets = {}
    for line in open(od / "dets.jsonl", encoding="utf-8"):
        r = json.loads(line)
        dets[r["f"]] = r["d"]
    dur = clip_info(key)["dur"]
    write_wav(od / "audio.wav", mix_audio(timeline, dur))

    font = ImageFont.truetype(FONT, 30)
    small = ImageFont.truetype(FONT, 20)
    played = [it for it in timeline if "t_start" in it]
    ff = None
    import imageio_ffmpeg
    for fidx, t, im in iter_frames(key, fps, max_side=1920):
        s = min(1.0, long_side / max(im.shape[:2]))
        hot = {e["track_id"] for e in events if 0 <= t - e["t"] <= 1.5}
        for tid, cls, x1, y1, x2, y2 in dets.get(fidx, []):
            c = (0, 0, 255) if tid in hot else (0, 200, 0)
            cv2.rectangle(im, (x1, y1), (x2, y2), c, 6 if tid in hot else 2)
        im = cv2.resize(im, (round(im.shape[1] * s) // 2 * 2, round(im.shape[0] * s) // 2 * 2))
        if ff is None:
            H, W = im.shape[:2]
            ff = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                                   "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{fps:g}", "-i", "-",
                                   "-i", str(od / "audio.wav"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                   "-crf", "23", "-c:a", "aac", "-b:a", "128k", str(od / "demo.mp4")],
                                  stdin=subprocess.PIPE)
        pil = Image.fromarray(im[:, :, ::-1])
        d = ImageDraw.Draw(pil)
        d.rectangle([0, 0, W, 34], fill=(0, 0, 0))
        d.text((10, 4), f"{t:5.1f}s   빨간 박스 = 트리거", font=small, fill=(255, 255, 255))
        now = [it for it in played if it["t_start"] <= t < it["t_end"]]
        for k, it in enumerate(now[:1]):
            tag = "경고" if it["kind"] == "warn" else "설명"
            col = (255, 80, 80) if it["kind"] == "warn" else (120, 200, 255)
            txt = f"[{tag}] {it['text']}"
            f = font
            bw = d.textlength(txt, font=f)
            if bw > W - 30:                                  # 긴 설명은 화면 폭에 맞춰 글자를 줄임
                f = ImageFont.truetype(FONT, max(14, int(30 * (W - 30) / bw)))
                bw = d.textlength(txt, font=f)
            y = H - 60
            d.rectangle([W / 2 - bw / 2 - 12, y - 6, W / 2 + bw / 2 + 12, y + 42], fill=(0, 0, 0))
            d.text((W / 2 - bw / 2, y), txt, font=f, fill=col)
        ff.stdin.write(np.ascontiguousarray(np.asarray(pil)[:, :, ::-1]).tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"→ {od / 'demo.mp4'}", flush=True)


if __name__ == "__main__":
    render(sys.argv[1], Path(sys.argv[2]) if len(sys.argv) > 2 else OUT / "runtime")
