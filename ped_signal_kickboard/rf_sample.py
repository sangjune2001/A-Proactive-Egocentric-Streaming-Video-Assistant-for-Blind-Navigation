"""Roboflow 데이터셋을 yolov8 형식으로 받아(캐시) 샘플 12장에 박스+클래스명을 그린 확인용 이미지를 만든다.
사용: python rf_sample.py <출력폴더> ws/proj [ws/proj ...]"""
import io
import random
import sys
import time
import zipfile
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

KEY = (Path.home() / ".roboflow_key").read_text().strip()
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
try:
    font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 18)
except OSError:
    font = ImageFont.load_default()


def export_zip(slug):
    dst = out / (slug.replace("/", "__") + ".zip")
    if dst.exists():
        return dst
    info = requests.get(f"https://api.roboflow.com/{slug}", params={"api_key": KEY}, timeout=60).json()
    ver = sorted(info["versions"], key=lambda v: v.get("images", 0))[0]["id"].rsplit("/", 1)[-1]  # 증강 없는 작은 버전
    for _ in range(60):
        r = requests.get(f"https://api.roboflow.com/{slug}/{ver}/yolov8", params={"api_key": KEY}, timeout=60).json()
        link = (r.get("export") or {}).get("link")
        if link:
            dst.write_bytes(requests.get(link, timeout=600).content)
            return dst
        time.sleep(5)
    raise RuntimeError(f"export 실패: {r}")


for slug in sys.argv[2:]:
    try:
        zp = export_zip(slug)
    except Exception as e:
        print(slug, "실패", e)
        continue
    z = zipfile.ZipFile(zp)
    names = None
    for n in z.namelist():
        if n.endswith("data.yaml"):
            import yaml
            names = yaml.safe_load(z.read(n))["names"]
    imgs = [n for n in z.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png")) and "/images/" in n]
    random.seed(1)
    tiles = []
    for n in random.sample(imgs, min(12, len(imgs))):
        im = Image.open(io.BytesIO(z.read(n))).convert("RGB")
        d = ImageDraw.Draw(im)
        lab = n.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        if lab in z.namelist():
            for line in z.read(lab).decode().splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                c, xs, ys = int(p[0]), [float(v) for v in p[1::2]], [float(v) for v in p[2::2]]
                W, H = im.size
                if len(p) == 5:
                    x, y, w, h = map(float, p[1:5])
                    box = [(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H]
                else:
                    box = [min(xs) * W, min(ys) * H, max(xs) * W, max(ys) * H]
                d.rectangle(box, outline=(255, 0, 255), width=3)
                d.text((box[0], max(0, box[1] - 20)), str(names[c] if names else c), fill=(255, 255, 0), font=font,
                       stroke_width=2, stroke_fill=(0, 0, 0))
        im.thumbnail((480, 480))
        tiles.append(im)
    sheet = Image.new("RGB", (480 * 4, 480 * ((len(tiles) + 3) // 4)), (20, 20, 20))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % 4) * 480, (i // 4) * 480))
    sp = out / (slug.replace("/", "__") + ".jpg")
    sheet.save(sp, quality=80)
    print(slug, len(imgs), "장", names, "->", sp.name, flush=True)
