"""71579 클립 시각화: 위는 프레임(보행신호등 박스), 아래는 보행신호등 확대.
사용: python show_clip.py <라벨 tar> <이미지 루트> <출력폴더> [클립 수]
라벨 tar에서 보행신호 상태가 바뀌는 클립 중, 이미지가 로컬에 있는 것을 골라 그린다."""
import io
import json
import sys
import tarfile
import zipfile
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

lab_tar, img_root, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
n_clips = int(sys.argv[4]) if len(sys.argv) > 4 else 2

with tarfile.open(lab_tar) as t:
    parts = sorted((m for m in t.getmembers() if m.isfile()), key=lambda m: m.name)
    z = zipfile.ZipFile(io.BytesIO(b"".join(t.extractfile(m).read() for m in parts)))
clips = defaultdict(list)
for i in z.infolist():
    if i.filename.endswith(".json"):
        d = json.loads(z.read(i))
        d.setdefault("objects", [])
        d["file_name"] = d.get("file_name") or d.get("image_name") or i.filename.rsplit("/", 1)[-1][:-5] + ".jpg"
        clips[i.filename.split("/")[0]].append(d)

COL = {"red": (255, 40, 40), "green": (40, 220, 80), "etc": (255, 200, 0)}
font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 22)
small = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 16)


def peds(d):
    return [o for o in d["objects"] if o["class_name"] == "pedestrian_signal"]


def score(frames):
    seq = [tuple(sorted(o["attribute"]["signal"] for o in peds(d))) for d in frames]
    seq = [s for s in seq if s]
    return sum(1 for a, b in zip(seq, seq[1:]) if a != b)


local = {p.name: p for p in img_root.rglob("*.jpg")}
cands = []
for c, frames in clips.items():
    frames.sort(key=lambda d: d["file_name"])
    if all(d["file_name"] in local for d in frames) and any(peds(d) for d in frames):
        cands.append((score(frames), c, frames))
cands.sort(key=lambda x: -x[0])
print(f"로컬에 이미지가 있는 보행신호 클립 {len(cands)}개")

W = 640
for sc, clip, frames in cands[:n_clips]:
    tiles, crops = [], []
    for d in frames:
        img = Image.open(local[d["file_name"]]).convert("RGB")
        draw = ImageDraw.Draw(img)
        states = []
        for o in d["objects"]:
            (x1, y1), (x2, y2) = o["data"]
            sig = str(o["attribute"].get("signal"))
            if o["class_name"] == "pedestrian_signal":
                draw.rectangle([x1, y1, x2, y2], outline=COL.get(sig, (255, 0, 255)), width=5)
                states.append(sig)
                pad = max(x2 - x1, y2 - y1) * 0.6
                crops.append((img.crop((x1 - pad, y1 - pad, x2 + pad, y2 + pad)).resize((200, 200)), sig, d["file_name"]))
            else:
                draw.rectangle([x1, y1, x2, y2], outline=(120, 120, 255), width=2)
        t = img.resize((W, int(img.height * W / img.width)))
        ImageDraw.Draw(t).text((8, 6), f"{d['file_name'][-10:-4]}  보행: {', '.join(states) or '-'}",
                               fill=(255, 255, 0), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
        tiles.append(t)
    th, cols = tiles[0].height, len(tiles)
    per_row = (W * cols) // 213
    rows = (len(crops) + per_row - 1) // per_row
    sheet = Image.new("RGB", (W * cols, th + rows * 230 + 10), (25, 25, 25))
    for i, t in enumerate(tiles):
        sheet.paste(t, (i * W, 0))
    for i, (c, sig, fn) in enumerate(crops):
        x, y = (i % per_row) * 213, th + 10 + (i // per_row) * 230
        sheet.paste(c, (x, y))
        ImageDraw.Draw(sheet).text((x + 4, y + 202), f"{fn[-10:-4]} {sig}", fill=COL.get(sig, (255, 0, 255)), font=small)
    out = out_dir / f"{clip}.jpg"
    sheet.save(out, quality=85)
    print(out, sheet.size, "변화", sc)
