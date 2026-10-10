"""드라이브의 shard zip(aihub_신호등_킥보드) -> YOLO detect 데이터셋(/content/ds 등 로컬 디스크).
클래스: 기존 189 모델의 10개 그대로 (아래 NAMES). 새 데이터는 원래 라벨 중 해당 클래스만 사용, 재라벨링 없음.
- 188 : traffic_light(차량/보행 모두) -> traffic_light. Validation -> val, Training -> train
- 71579: 신호등 4종(vehicular/pedestrian/unusual/invisible) -> traffic_light, train
- 614 : annotations.PM[] bbox [x,y,w,h] -> scooter. video_id 해시 10% -> val (같은 영상이 train/val에 섞이지 않게)
사용: python prepare_yolo.py <shard루트> <출력폴더> [--max-188 N] [--max-614 N] [--max-side 1280]
"""
import argparse
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image

# 기존 189 모델(yolo11n-seg, aihub189_yolo/runs/seg10)의 10개 클래스를 순서 그대로 유지 (새 클래스 추가 안 함)
NAMES = ["person", "bicycle", "scooter", "motorcycle", "car", "bus", "other_vehicle", "obstacle", "stairs",
         "traffic_light"]
SCOOTER, TRAFFIC_LIGHT = 2, 9
# 새 데이터는 원래 라벨 중 기존 클래스에 해당하는 것만 넣는다(재라벨링 없음):
#   188 traffic_light(차량/보행 전부), 71579 신호등 4종 -> traffic_light / 614 PM(킥보드) -> scooter


def h(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def stem(name):
    base = name.rsplit("/", 1)[-1]
    for ext in (".png.json", ".jpg.json", ".json", ".jpg", ".png", ".jpeg"):
        if base.lower().endswith(ext):
            return base[: -len(ext)]
    return base


def load_json(raw):
    for enc in ("utf-8-sig", "cp949"):
        try:
            return json.loads(raw.decode(enc))
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass


def parse(ds, d):
    """-> (w, h, [(cls, x1, y1, x2, y2)], split_hint)"""
    boxes = []
    if ds.startswith("188"):
        w, hgt = d["image"]["imsize"]
        for a in d.get("annotation", []):
            if a.get("class") == "traffic_light":
                x1, y1, x2, y2 = a["box"]
                boxes.append((TRAFFIC_LIGHT, x1, y1, x2, y2))
        return w, hgt, boxes, None
    if ds.startswith("71579"):
        w, hgt = d.get("image_size", [1920, 1080])
        for o in d.get("objects", []):
            if o.get("class_name") in ("vehicular_signal", "pedestrian_signal", "unusual_signal", "invisible_signal"):
                (x1, y1), (x2, y2) = o["data"]
                boxes.append((TRAFFIC_LIGHT, x1, y1, x2, y2))
        return w, hgt, boxes, None
    if ds.startswith("614"):
        w, hgt = d["description"]["imageWidth"], d["description"]["imageHeight"]
        for pm in (d.get("annotations") or {}).get("PM", []):
            x, y, bw, bh = pm["points"]
            boxes.append((SCOOTER, x, y, x + bw, y + bh))
        vid = d.get("info", {}).get("video_id", "")
        return w, hgt, boxes, ("val" if h(vid) % 10 == 0 else "train")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--max-188", type=int, default=30000, help="188 train 최대 이미지 수")
    ap.add_argument("--max-188-val", type=int, default=3000)
    ap.add_argument("--max-614", type=int, default=20000)
    ap.add_argument("--max-side", type=int, default=1280)
    args = ap.parse_args()
    names = NAMES

    # 1) 라벨 전부 읽기 (shard 안 labels/ + 낱개 json)
    ann = {}  # (dsdir, stem) -> (w,h,boxes,split)
    present = set()  # 실제로 이미지가 shard에 있는 것
    for dsdir in sorted(p for p in args.root.iterdir() if p.is_dir()):
        ds = dsdir.name
        for zp in sorted(dsdir.rglob("*.zip")):
            split_dir = zp.parent.name
            with zipfile.ZipFile(zp) as z:
                for n in z.namelist():
                    if n.startswith("images/"):
                        present.add((ds, stem(n)))
                    elif n.startswith("labels/") and n.endswith(".json"):
                        d = load_json(z.read(n))
                        r = d and parse(ds, d)
                        if r and r[2]:
                            ann[(ds, stem(n))] = (*r[:3], r[3] or ("val" if split_dir == "Validation" and ds.startswith("188") else "train"))
        for jp in dsdir.rglob("*.json"):
            d = load_json(jp.read_bytes())
            r = d and parse(ds, d)
            if r and r[2]:
                ann[(ds, stem(jp.name))] = (*r[:3], "train")
        print(ds, "라벨 누적", len(ann), flush=True)

    # 2) 샘플 수 제한 (해시로 고정 샘플)
    caps = {("188", "train"): args.max_188, ("188", "val"): args.max_188_val,
            ("614", "train"): args.max_614, ("614", "val"): max(1, args.max_614 // 9)}
    by = {}
    print("라벨", len(ann), "/ 이미지 있음", len(present))
    for k, v in ann.items():
        if k not in present:
            continue
        by.setdefault((k[0][:3] if not k[0].startswith("715") else "71579", v[3]), []).append(k)
    chosen = set()
    for grp, keys in by.items():
        keys.sort(key=lambda k: h(k[1]))
        chosen.update(keys[: caps.get(grp, len(keys))])
        print("선택", grp, min(len(keys), caps.get(grp, len(keys))), "/", len(keys))

    # 3) 이미지 꺼내서 저장 + YOLO txt
    for sp in ("train", "val"):
        (args.out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / sp).mkdir(parents=True, exist_ok=True)
    cnt = Counter()
    for dsdir in sorted(p for p in args.root.iterdir() if p.is_dir()):
        ds = dsdir.name
        for zp in sorted(dsdir.rglob("*.zip")):
            with zipfile.ZipFile(zp) as z:
                for n in z.namelist():
                    if not n.startswith("images/") or not n.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue
                    key = (ds, stem(n))
                    if key not in chosen:
                        continue
                    w, hgt, boxes, sp = ann[key]
                    name = f"{ds[:5]}_{key[1]}"
                    if (args.out / "labels" / sp / f"{name}.txt").exists():
                        continue
                    img = Image.open(io.BytesIO(z.read(n))).convert("RGB")
                    w, hgt = img.size if (w, hgt) != img.size else (w, hgt)
                    if max(img.size) > args.max_side:
                        img.thumbnail((args.max_side, args.max_side))
                    img.save(args.out / "images" / sp / f"{name}.jpg", quality=90)
                    lines = []
                    for c, x1, y1, x2, y2 in boxes:
                        x1, x2 = sorted((max(0, x1), min(w, x2)))
                        y1, y2 = sorted((max(0, y1), min(hgt, y2)))
                        if x2 - x1 < 1 or y2 - y1 < 1:
                            continue
                        # yolo11n-seg 그대로 학습하므로 bbox를 사각형 폴리곤으로 저장
                        # (seg 데이터에 bbox 줄이 섞이면 Ultralytics가 폴리곤을 전부 버림)
                        a, b, cc, dd = x1 / w, y1 / hgt, x2 / w, y2 / hgt
                        lines.append(f"{c} {a:.6f} {b:.6f} {cc:.6f} {b:.6f} {cc:.6f} {dd:.6f} {a:.6f} {dd:.6f}")
                        cnt[(sp, NAMES[c])] += 1
                    (args.out / "labels" / sp / f"{name}.txt").write_text("\n".join(lines))
                    cnt[(sp, "images")] += 1
            print(zp.name, dict(cnt), flush=True)

    (args.out / "data.yaml").write_text(
        f"path: {args.out}\ntrain: images/train\nval: images/val\nnames:\n" +
        "".join(f"  {i}: {n}\n" for i, n in enumerate(names)), encoding="utf-8")
    print("완료", dict(cnt))


if __name__ == "__main__":
    main()
