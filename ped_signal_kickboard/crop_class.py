"""YOLO 데이터셋에서 특정 클래스 박스만 잘라 격자로 모은다. 사용: python crop_class.py <ds> <클래스번호> <출력.jpg> [개수]"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw

ds, cls, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
n = int(sys.argv[4]) if len(sys.argv) > 4 else 40
crops = []
for lab in sorted((ds / "labels").rglob("*.txt")):
    img = None
    for line in lab.read_text().splitlines():
        c, x, y, w, h = line.split()
        if c != cls:
            continue
        if img is None:
            img = Image.open(ds / "images" / lab.parent.name / (lab.stem + ".jpg")).convert("RGB")
        W, H = img.size
        x, y, w, h = float(x) * W, float(y) * H, float(w) * W, float(h) * H
        pad = max(w, h) * 0.5
        crops.append(img.crop((x - w / 2 - pad, y - h / 2 - pad, x + w / 2 + pad, y + h / 2 + pad)).resize((120, 120)))
        if len(crops) >= n:
            break
    if len(crops) >= n:
        break
cols = 10
sheet = Image.new("RGB", (cols * 124, ((len(crops) + cols - 1) // cols) * 124), (20, 20, 20))
for i, c in enumerate(crops):
    sheet.paste(c, ((i % cols) * 124, (i // cols) * 124))
sheet.save(out, quality=90)
print(out, len(crops))
