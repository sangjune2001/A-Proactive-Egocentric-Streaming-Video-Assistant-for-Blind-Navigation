"""YOLO 데이터셋 이미지 몇 장에 박스를 그려 한 장으로 합친다. 사용: python show_yolo.py <ds폴더> <출력.jpg> [장수]"""
import random
import sys
from pathlib import Path
from PIL import Image, ImageDraw

ds, out = Path(sys.argv[1]), Path(sys.argv[2])
n = int(sys.argv[3]) if len(sys.argv) > 3 else 6
COL = [(255, 40, 40), (40, 220, 80), (200, 200, 200), (255, 160, 0)]
NAMES = ["red", "green", "off", "kick"]
imgs = sorted((ds / "images").rglob("*.jpg"))
random.seed(0)
tiles = []
for p in random.sample(imgs, min(n, len(imgs))):
    im = Image.open(p).convert("RGB")
    d = ImageDraw.Draw(im)
    lab = ds / "labels" / p.parent.name / (p.stem + ".txt")
    for line in lab.read_text().splitlines():
        c, x, y, w, h = line.split()
        c = int(c)
        W, H = im.size
        x, y, w, h = float(x) * W, float(y) * H, float(w) * W, float(h) * H
        d.rectangle([x - w / 2, y - h / 2, x + w / 2, y + h / 2], outline=COL[c], width=4)
        d.text((x - w / 2, y - h / 2 - 12), NAMES[c], fill=COL[c])
    im.thumbnail((800, 800))
    tiles.append(im)
W = 800
sheet = Image.new("RGB", (W * 2, sum(t.height for t in tiles[::2]) + 10), (20, 20, 20))
y = 0
for i in range(0, len(tiles), 2):
    for j, t in enumerate(tiles[i:i + 2]):
        sheet.paste(t, (j * W, y))
    y += max(t.height for t in tiles[i:i + 2])
sheet.save(out, quality=85)
print(out)
