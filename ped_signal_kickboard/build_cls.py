"""신호 상태 분류용 crop 데이터셋: <out>/{train,val}/{red,green,off,vehicle}/*.jpg
- 188: traffic_light type==pedestrian -> attribute red on=red, green on=green, 그 외=off / 그 외 type -> vehicle
- 71579: pedestrian_signal red/green/etc(off) / vehicular_signal -> vehicle (invisible/unusual 제외)
- Roboflow 한국 보행신호 5종: green*/red* -> green/red (rf_cache zip)
- 71579 신호 변화 클립(csv)의 crop은 train에 3배로 넣어 강조
사용: python3 build_cls.py <shard루트(aihub_신호등_킥보드)> <rf_cache> <out> [--cap 40000]"""
import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("root", type=Path)
ap.add_argument("rf_cache", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--cap", type=int, default=40000, help="train 클래스별 최대 crop 수")
args = ap.parse_args()
CLASSES = ["red", "green", "off", "vehicle"]
cnt = Counter()


def h(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def stem(n):
    b = n.rsplit("/", 1)[-1]
    for e in (".json", ".jpg", ".png", ".jpeg"):
        if b.lower().endswith(e):
            return b[: -len(e)]
    return b


def load_json(raw):
    for enc in ("utf-8-sig", "cp949"):
        try:
            return json.loads(raw.decode(enc))
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass


def save(img, box, cls, sp, name, copies=1):
    x1, y1, x2, y2 = box
    w, hh = x2 - x1, y2 - y1
    if w < 5 or hh < 8 or (sp == "train" and cnt[(sp, cls)] >= args.cap):
        return
    px, py = w * 0.3, hh * 0.3
    c = img.crop((max(0, x1 - px), max(0, y1 - py), min(img.width, x2 + px), min(img.height, y2 + py)))
    c.thumbnail((96, 96))
    d = args.out / sp / cls
    d.mkdir(parents=True, exist_ok=True)
    for k in range(copies if sp == "train" else 1):
        c.save(d / f"{name}_{k}.jpg", quality=92)
        cnt[(sp, cls)] += 1


# 71579 변화 클립 목록
hot = set()
for zp in (args.root / "71579_신호등신호정보" / "Validation").glob("*.zip"):
    with zipfile.ZipFile(zp) as z:
        for n in z.namelist():
            if n.endswith("clips_pedestrian_signal.csv"):
                for r in csv.DictReader(io.StringIO(z.read(n).decode("utf-8-sig"))):
                    if int(r["ped_signal_changes"]) > 0:
                        hot.add(r["clip"].split("_")[-1])

# AI Hub 188 / 71579
for dsname in ("188_신호등표지판", "71579_신호등신호정보"):
    dsdir = args.root / dsname
    ann = {}
    for zp in sorted(dsdir.rglob("*.zip")):
        with zipfile.ZipFile(zp) as z:
            for n in z.namelist():
                if n.startswith("labels/") and n.endswith(".json"):
                    ann[stem(n)] = (load_json(z.read(n)), zp.parent.name)
    for jp in dsdir.rglob("*.json"):
        ann.setdefault(stem(jp.name), (load_json(jp.read_bytes()), "Validation"))
    for zp in sorted(dsdir.rglob("*.zip")):
        with zipfile.ZipFile(zp) as z:
            for n in z.namelist():
                if not n.startswith("images/"):
                    continue
                s = stem(n)
                d, split_dir = ann.get(s, (None, None))
                if not d:
                    continue
                boxes = []
                if dsname.startswith("188"):
                    for a in d.get("annotation", []):
                        if a.get("class") != "traffic_light":
                            continue
                        if a.get("type") == "pedestrian":
                            at = (a.get("attribute") or [{}])[0]
                            cls = "red" if at.get("red") == "on" else "green" if at.get("green") == "on" else "off"
                        else:
                            cls = "vehicle"
                        boxes.append((cls, a["box"]))
                    sp = "val" if split_dir == "Validation" else "train"
                    copies = 1
                else:
                    for o in d.get("objects", []):
                        (x1, y1), (x2, y2) = o["data"]
                        if o["class_name"] == "pedestrian_signal":
                            sig = str(o.get("attribute", {}).get("signal"))
                            boxes.append(({"red": "red", "green": "green"}.get(sig, "off"), [x1, y1, x2, y2]))
                        elif o["class_name"] == "vehicular_signal":
                            boxes.append(("vehicle", [x1, y1, x2, y2]))
                    sp = "val" if h(s) % 10 == 0 else "train"
                    copies = 3 if s.split("_")[2] in hot else 1
                if not boxes:
                    continue
                img = Image.open(io.BytesIO(z.read(n))).convert("RGB")
                for i, (cls, b) in enumerate(boxes):
                    save(img, b, cls, sp, f"{dsname[:5]}_{s}_{i}", copies)
        print(dsname, zp.name, dict(cnt), flush=True)

# Roboflow 한국 보행신호 (yolov8 export zip, bbox 또는 polygon 라벨)
import yaml

for zp in sorted(args.rf_cache.glob("*_full.zip")):
    if "scooter" in zp.name:
        continue
    z = zipfile.ZipFile(zp)
    names = yaml.safe_load(z.read(next(n for n in z.namelist() if n.endswith("data.yaml"))))["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    cmap = {i: ("green" if n.lower().startswith("green") or n.lower().startswith("greenlight") else
                "red" if n.lower().startswith("red") else None) for i, n in enumerate(names)}
    for n in z.namelist():
        if "/images/" not in n:
            continue
        lab = n.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        if lab not in z.namelist():
            continue
        rows = [l.split() for l in z.read(lab).decode().splitlines() if len(l.split()) >= 5]
        rows = [r for r in rows if cmap.get(int(r[0]))]
        if not rows:
            continue
        img = Image.open(io.BytesIO(z.read(n))).convert("RGB")
        W, H = img.size
        s = Path(n).stem
        sp = "val" if h(s.split(".rf.")[0]) % 10 == 0 else "train"  # 증강본 누수 방지
        for i, r in enumerate(rows):
            v = list(map(float, r[1:]))
            if len(v) == 4:
                box = [(v[0] - v[2] / 2) * W, (v[1] - v[3] / 2) * H, (v[0] + v[2] / 2) * W, (v[1] + v[3] / 2) * H]
            else:
                box = [min(v[0::2]) * W, min(v[1::2]) * H, max(v[0::2]) * W, max(v[1::2]) * H]
            save(img, box, cmap[int(r[0])], sp, f"rf_{zp.stem[:15]}_{s[:40]}_{i}")
    print(zp.name, dict(cnt), flush=True)
print("완료", dict(cnt))
