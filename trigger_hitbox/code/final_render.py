"""최종 비교 영상: 왼쪽 = 사용자에게 보이는/들리는 것만 (위험 물체 박스 · 신호 · 음성 자막),
오른쪽 = 히트박스 등 판단 과정 전체 (pass2_trigger.py 의 B 화면). 소리 = SJ 실행 시스템이 실제로 재생한 것.

usage: python final_render.py OUT_DIR RUN_DIR DEBUG_MP4 OUT_MP4
  OUT_DIR   : pass2_trigger.py 결과 폴더 (pfd.pkl — DUMP=1 로 실행)
  RUN_DIR   : SJ code/runtime run_demo.py 결과 폴더 (timeline.json · log.jsonl)
  DEBUG_MP4 : PANELS=B 로 만든 히트박스 영상 (오른쪽)
"""
import json
import pickle
import subprocess
import sys
import wave
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SR = 22050
FONTS = ["/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf", "C:/Windows/Fonts/malgunbd.ttf"]
FONT = next(f for f in FONTS if Path(f).exists())
KR = {"person": "사람", "bicycle": "자전거", "scooter": "킥보드", "motorcycle": "오토바이", "car": "자동차",
      "bus": "버스", "other_vehicle": "차량", "obstacle": "장애물", "stairs": "계단", "traffic_light": "신호등"}
LV = {"warn": ("주의", (255, 200, 0)), "danger": ("위험", (255, 40, 40))}          # RGB
LIGHT = {"red": ("빨간불", (255, 40, 40)), "green": ("초록불", (40, 220, 40))}


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


def mix_audio(played, dur, path):
    out = np.zeros(int((dur + 1.0) * SR), np.float32)
    for it in played:
        x = read_wav(it["wav"])
        a = int(it["t_start"] * SR)
        n = min(len(x), int((it["t_end"] - it["t_start"]) * SR), len(out) - a)
        if n > 0:
            out[a:a + n] += x[:n]
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(out, -1, 1) * 32767).astype(np.int16).tobytes())


def main(out_dir, run_dir, debug_mp4, out_mp4):
    out_dir, run_dir = Path(out_dir), Path(run_dir)
    d = pickle.load(open(out_dir / "pfd.pkl", "rb"))
    meta, pfd = d["meta"], d["pfd"]
    timeline = json.load(open(run_dir / "timeline.json", encoding="utf-8"))
    played = [it for it in timeline if "t_start" in it]
    dur = meta["n"] / meta["fps"]
    wav = run_dir / "final_audio.wav"
    mix_audio(played, dur, wav)

    cap, dbg = cv2.VideoCapture(meta["video"]), cv2.VideoCapture(str(debug_mp4))
    W, H = meta["w"], meta["h"]
    sc = 0.5
    w2, h2 = int(W * sc) // 2 * 2, int(H * sc) // 2 * 2
    big = ImageFont.truetype(FONT, 34)
    mid = ImageFont.truetype(FONT, 26)
    small = ImageFont.truetype(FONT, 20)
    import imageio_ffmpeg
    ff = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w2 * 2}x{h2}", "-r", f"{meta['fps']:g}",
                           "-i", "-", "-i", str(wav), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24",
                           "-preset", "fast", "-c:a", "aac", "-b:a", "128k", "-shortest", str(out_mp4)],
                          stdin=subprocess.PIPE)
    for pf in pfd:
        ok, im = cap.read()
        ok2, right = dbg.read()
        if not ok:
            break
        t = pf["t"]
        left = cv2.resize(im, (w2, h2))
        pil = Image.fromarray(left[:, :, ::-1])
        dr = ImageDraw.Draw(pil)
        # 위험·주의 물체만 박스 + 한글 이름
        for o in pf["draws"]:
            if o["lvB"] == "safe":
                continue
            name, col = LV[o["lvB"]]
            x1, y1, x2, y2 = (v * sc for v in o["box"])
            dr.rectangle([x1, y1, x2, y2], outline=col, width=6 if o["lvB"] == "danger" else 4)
            lab = f"{name} {KR.get(o['cls'], o['cls'])}" + (f" {o['dist']:.0f}m" if o.get("dist") else "")
            tw = dr.textlength(lab, font=mid)
            dr.rectangle([x1, max(0, y1 - 34), x1 + tw + 12, max(34, y1)], fill=col)
            dr.text((x1 + 6, max(0, y1 - 32)), lab, font=mid, fill=(0, 0, 0))
        # 신호등: 확정된 색만
        for L in pf["lights"]:
            if not L.get("stable"):
                continue
            name, col = LIGHT[L["stable"]]
            x1, y1, x2, y2 = (v * sc for v in L["box"])
            dr.rectangle([x1 - 4, y1 - 4, x2 + 4, y2 + 4], outline=col, width=4)
            dr.text((x1 - 4, y2 + 6), name, font=small, fill=col, stroke_width=2, stroke_fill=(0, 0, 0))
        # 지금 나오는 음성 (경고 = 빨강, 설명 = 하늘색)
        now = [it for it in played if it["t_start"] <= t < it["t_end"]]
        dr.rectangle([0, 0, w2, 40], fill=(0, 0, 0))
        dr.text((12, 7), f"{t:5.1f}s   사용자 화면 (박스 = 주의·위험 물체)", font=small, fill=(255, 255, 255))
        for it in now[:1]:
            tag, col = ("경고", (255, 90, 90)) if it["kind"] == "warn" else ("설명", (120, 200, 255))
            txt = f"음성 [{tag}] {it['text']}"
            f = big
            while dr.textlength(txt, font=f) > w2 - 30 and f.size > 16:
                f = ImageFont.truetype(FONT, f.size - 2)
            bw = dr.textlength(txt, font=f)
            y = h2 - 90
            dr.rectangle([w2 / 2 - bw / 2 - 14, y - 8, w2 / 2 + bw / 2 + 14, y + f.size + 14], fill=(0, 0, 0))
            dr.text((w2 / 2 - bw / 2, y), txt, font=f, fill=col)
        left = np.asarray(pil)[:, :, ::-1]
        right = cv2.resize(right, (w2, h2)) if ok2 else np.zeros_like(left)
        ff.stdin.write(np.ascontiguousarray(np.hstack([left, right])).tobytes())
    ff.stdin.close()
    ff.wait()
    print("->", out_mp4)


if __name__ == "__main__":
    main(*sys.argv[1:5])
