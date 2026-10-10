"""확인된 Roboflow 데이터셋(한국 보행신호등, 킥보드)을 받아 기존 10클래스 seg 데이터셋에 추가.
보행신호 클래스 -> traffic_light(9), 킥보드 -> scooter(2). 그 외 클래스(crosswalk, button 등)는 버림.
bbox는 사각형 폴리곤으로 저장(seg10 그대로 학습). 해시 10% -> val. 파일명 접두어 rf_<프로젝트>_
사용: python3 rf_to_yolo.py <ds 폴더> <다운로드 캐시 폴더>"""
import hashlib
import io
import sys
import time
import zipfile
from pathlib import Path

import requests
import yaml
from PIL import Image

KEY = (Path.home() / ".roboflow_key").read_text().strip()
TL, SCOOTER = 9, 2
# 샘플 이미지를 직접 보고 한국 보행신호등으로 확인한 것만 (2026-10-09)
DATASETS = {
    "crosswalk-traffic-light/robot-hsuip": {"greenlight": TL, "redlight": TL},          # 부산 등, 보행자 시점
    "chanyoung/pedestrian-light-crosswalk": {"greenlight": TL, "redlight": TL},         # 여의대로 등
    "cap-8nhra/crosswalk-pedestrian-light": {"greenlight": TL, "redlight": TL},         # 서울 초교 앞 등
    "usrg2/pedestrian-signal-p6xjj": {"Green": TL, "Red": TL},                           # 청운대 앞
    "s-workspace-ddokc/pedestrian-signal": {"green_pedestrian_signal": TL, "red_pedestrian_signal": TL},  # 영남대 등
    "kdigital/electric-scooter-cd7hw": {"electric scooter": SCOOTER},                   # 킥보드
}
ds, cache = Path(sys.argv[1]), Path(sys.argv[2])
cache.mkdir(parents=True, exist_ok=True)


def export(slug):
    dst = cache / (slug.replace("/", "__") + "_full.zip")
    if dst.exists():
        return dst
    info = requests.get(f"https://api.roboflow.com/{slug}", params={"api_key": KEY}, timeout=60).json()
    ver = max(info["versions"], key=lambda v: v.get("images", 0))["id"].rsplit("/", 1)[-1]
    for _ in range(120):
        r = requests.get(f"https://api.roboflow.com/{slug}/{ver}/yolov8", params={"api_key": KEY}, timeout=60).json()
        link = (r.get("export") or {}).get("link")
        if link:
            with requests.get(link, stream=True, timeout=600) as resp:
                with open(dst, "wb") as f:
                    for c in resp.iter_content(1 << 20):
                        f.write(c)
            return dst
        time.sleep(5)
    raise RuntimeError(r)


for slug, cmap in DATASETS.items():
    z = zipfile.ZipFile(export(slug))
    names = yaml.safe_load(z.read(next(n for n in z.namelist() if n.endswith("data.yaml"))))["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    idx = {i: cmap[n] for i, n in enumerate(names) if n in cmap}
    tag = "rf_" + slug.split("/")[1].replace("-", "")[:20]
    n_img = n_box = 0
    for n in z.namelist():
        if "/images/" not in n or not n.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        lab = n.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        if lab not in z.namelist():
            continue
        lines = []
        for line in z.read(lab).decode().splitlines():
            p = line.split()
            if len(p) < 5 or int(p[0]) not in idx:
                continue
            v = list(map(float, p[1:]))
            if len(v) == 4:
                x, y, w, h = v
                x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
            else:
                x1, x2, y1, y2 = min(v[0::2]), max(v[0::2]), min(v[1::2]), max(v[1::2])
            x1, y1, x2, y2 = [min(1, max(0, t)) for t in (x1, y1, x2, y2)]
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue
            lines.append(f"{idx[int(p[0])]} {x1:.6f} {y1:.6f} {x2:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x1:.6f} {y2:.6f}")
        if not lines:
            continue
        stem = Path(n).stem
        orig = stem.split(".rf.")[0]  # 같은 원본의 증강본은 같은 split으로 (train/val 누수 방지)
        sp = "val" if int(hashlib.md5(orig.encode()).hexdigest()[:8], 16) % 10 == 0 else "train"
        out_img = ds / "images" / sp / f"{tag}_{stem[:60]}.jpg"
        if out_img.exists():
            continue
        out_img.parent.mkdir(parents=True, exist_ok=True)
        (ds / "labels" / sp).mkdir(parents=True, exist_ok=True)
        im = Image.open(io.BytesIO(z.read(n))).convert("RGB")
        if max(im.size) > 1280:
            im.thumbnail((1280, 1280))
        im.save(out_img, quality=90)
        (ds / "labels" / sp / f"{out_img.stem}.txt").write_text("\n".join(lines))
        n_img += 1
        n_box += len(lines)
    print(slug, "이미지", n_img, "박스", n_box, flush=True)
