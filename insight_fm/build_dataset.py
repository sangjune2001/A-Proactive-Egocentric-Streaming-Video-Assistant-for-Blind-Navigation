"""labels_all.jsonl + extracted 640px image shards  ->  YOLO-seg folders with VIDEO-level splits.

    # 1) look first: prints the jsonl fields, a parsed example and the image index
    python build_dataset.py --jsonl /data/sg/labels_all.jsonl --images /data/sg/imgs --inspect
    # 2) build
    python build_dataset.py --jsonl /data/sg/labels_all.jsonl --images /data/sg/imgs --out /data/sg/yolo

Output
    <out>/full/{images,labels}/{train,val,test}   70 / 15 / 15 by video folder
    <out>/full/data.yaml
    <out>/pilot/...  train = every full-train frame containing a --pilot-all-classes class, then up to
                     --pilot-per-class frames per remaining class (rarest class first),
                     val = up to --pilot-val-per-class frames per class from full val (test is not used)
    <out>/stats.json   instance counts per class per split
Images are symlinked, not copied.

jsonl record formats understood (the key is `{folder with / replaced}__{stem}`):
    {"k": ..., "z": zip, "m": member, "l": ["cls x1 y1 ...", ...]}          labels_all.jsonl from Colab
    {"key": ..., "lines": ["cls x1 y1 x2 y2 ...", ...]}                        already YOLO-seg
    {"key": ..., "polys"|"polygons"|"objects"|"annotations": [
          {"cls"|"class"|"label"|"category": int or raw name, "points"|"pts"|"polygon"|"segmentation": [...]}
          or [cls, points] ], "w"/"width": W, "h"/"height": H}                 points normalised or pixels
Raw CVAT label names are mapped with MAP below (29 -> 10 classes).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

NAMES = ["person", "bicycle", "scooter", "motorcycle", "car", "bus",
         "other_vehicle", "obstacle", "stairs", "traffic_light"]
MAP = {"person": 0, "bicycle": 1, "scooter": 2, "motorcycle": 3, "car": 4, "bus": 5,
       "truck": 6, "carrier": 6, "stroller": 6, "wheelchair": 6, "traffic_light": 9,
       **{k: 7 for k in ["bollard", "pole", "tree_trunk", "potted_plant", "barricade", "fire_hydrant", "kiosk",
                         "bench", "chair", "table", "power_controller", "traffic_light_controller",
                         "parking_meter", "stop", "movable_signage"]},
       **{n: i for i, n in enumerate(NAMES)}}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

KEY_F = ("k", "key", "id", "name", "image", "file", "stem")
LINES_F = ("l", "lines", "labels", "yolo")
POLY_F = ("polys", "polygons", "objects", "annotations", "anns", "shapes")
CLS_F = ("cls", "class", "class_id", "label", "category", "name")
PTS_F = ("points", "pts", "polygon", "poly", "segmentation", "xy")
W_F, H_F = ("w", "width", "orig_w", "img_w"), ("h", "height", "orig_h", "img_h")


def first(d: dict, fields, default=None):
    for f in fields:
        if f in d and d[f] is not None:
            return d[f]
    return default


def norm_key(k: str) -> str:
    k = str(k).replace("\\", "/")
    stem = Path(k).name
    stem = stem.rsplit(".", 1)[0] if Path(stem).suffix.lower() in IMG_EXT else stem
    return stem


def video_of(key: str) -> str:
    return key.split("__")[0] if "__" in key else str(Path(key).parent)


def to_cls(c) -> int | None:
    if isinstance(c, (int, float)) and not isinstance(c, bool):
        c = int(c)
        return c if 0 <= c < len(NAMES) else None
    return MAP.get(str(c).strip())


def flat_points(p) -> list[float]:
    if isinstance(p, str):  # CVAT "x1,y1;x2,y2"
        return [float(v) for pair in p.split(";") for v in pair.split(",")]
    if p and isinstance(p[0], (list, tuple)):
        if p[0] and isinstance(p[0][0], (list, tuple)):  # COCO-style [[x,y,...]] or [[[x,y],...]]
            p = p[0]
        if p and isinstance(p[0], (list, tuple)):
            return [float(v) for xy in p for v in xy]
    return [float(v) for v in p]


def parse_record(r: dict) -> tuple[str, list[str]]:
    key = first(r, KEY_F)
    if key is None:
        raise ValueError(f"no key field (looked for {KEY_F}) in {list(r)}")
    key = norm_key(key)

    lines = first(r, LINES_F)
    if isinstance(lines, str):
        lines = [ln for ln in lines.splitlines() if ln.strip()]
    if lines is not None and (not lines or isinstance(lines[0], str)):
        out = []
        for ln in lines:
            parts = ln.split()
            c = to_cls(int(float(parts[0]))) if parts[0].lstrip("-").replace(".", "", 1).isdigit() else to_cls(parts[0])
            if c is not None and len(parts) >= 7:
                out.append(" ".join([str(c)] + parts[1:]))
        return key, out

    polys = first(r, POLY_F, lines)
    if polys is None:
        raise ValueError(f"no label field (looked for {LINES_F + POLY_F}) in {list(r)}")
    W, H = first(r, W_F), first(r, H_F)
    out = []
    for o in polys:
        if isinstance(o, dict):
            c, pts = to_cls(first(o, CLS_F)), first(o, PTS_F)
        else:
            c, pts = to_cls(o[0]), o[1] if len(o) == 2 else o[1:]
        if c is None or pts is None:
            continue
        xy = flat_points(pts)
        if len(xy) < 6:
            continue
        if max(xy) > 1.5:  # pixel coordinates
            if not (W and H):
                raise ValueError(f"pixel coordinates but no width/height fields ({W_F}/{H_F}) in record {key}")
            xy = [v / (W if i % 2 == 0 else H) for i, v in enumerate(xy)]
        xy = [min(max(v, 0.0), 1.0) for v in xy]
        out.append(f"{c} " + " ".join(f"{v:.6f}" for v in xy))
    return key, out


def index_images(root: Path) -> dict[str, Path]:
    idx = {}
    for dp, _, fs in os.walk(root, followlinks=True):
        for f in fs:
            p = Path(dp) / f
            if p.suffix.lower() in IMG_EXT:
                idx[p.stem] = p
    return idx


def split_videos(by_video: dict, seed: int):
    vids = sorted(by_video)
    rng = random.Random(seed)
    rng.shuffle(vids)
    n = len(vids)
    a, b = int(n * 0.70), int(n * 0.85)
    return {**{v: "train" for v in vids[:a]}, **{v: "val" for v in vids[a:b]}, **{v: "test" for v in vids[b:]}}


def class_counts(by_video, assign, split):
    c = Counter()
    for v, recs in by_video.items():
        if assign[v] == split:
            for _, lines, _ in recs:
                c.update(int(ln.split()[0]) for ln in lines)
    return c


def choose_split(by_video, tries: int, seed0: int):
    """Try several seeds; keep the one whose worst-covered class in val/test is best covered."""
    present = set()
    for recs in by_video.values():
        for _, lines, _ in recs:
            present.update(int(ln.split()[0]) for ln in lines)
    best = None
    for s in range(seed0, seed0 + tries):
        asg = split_videos(by_video, s)
        cv, ct = class_counts(by_video, asg, "val"), class_counts(by_video, asg, "test")
        score = min(min(cv[c], ct[c]) for c in present) if present else 0
        if best is None or score > best[0]:
            best = (score, s, asg)
    return best


def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def per_class_sample(items, per_class: int, rng: random.Random, take_all=frozenset()):
    """Pick frames so every class appears in about `per_class` frames (fewer if the class has fewer).
    Every frame containing a class in `take_all` is taken first. Then rarest class first; a frame picked
    for one class also counts for every other class it contains."""
    items = list(items)
    rng.shuffle(items)
    classes_of = [{int(ln.split()[0]) for ln in t[1]} for t in items]
    freq = Counter(c for cs in classes_of for c in cs)
    have, picked, chosen = Counter(), set(), []
    for i, cs in enumerate(classes_of):
        if cs & take_all:
            picked.add(i)
            chosen.append(items[i])
            have.update(cs)
    for c in sorted(freq, key=freq.get):
        for i, cs in enumerate(classes_of):
            if have[c] >= per_class:
                break
            if c in cs and i not in picked:
                picked.add(i)
                chosen.append(items[i])
                have.update(cs)
    return chosen


def write_yaml(root: Path, splits):
    txt = f"path: {root}\n" + "".join(f"{s}: images/{s}\n" for s in splits)
    txt += f"nc: {len(NAMES)}\nnames: {NAMES}\n"
    (root / "data.yaml").write_text(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--images", required=True, help="folder with the extracted shard images (searched recursively)")
    ap.add_argument("--out", default=str(Path.home() / "sg" / "yolo"))
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-tries", type=int, default=50)
    ap.add_argument("--pilot-per-class", type=int, default=200, help="pilot train frames per class")
    ap.add_argument("--pilot-all-classes", default="scooter,stairs,traffic_light",
                    help="comma list of classes whose train frames all go into the pilot ('' for none)")
    ap.add_argument("--pilot-val-per-class", type=int, default=100, help="pilot val frames per class")
    a = ap.parse_args()

    print("indexing images ...", flush=True)
    idx = index_images(Path(a.images))
    print(f"  {len(idx)} images found under {a.images}")

    if a.inspect:
        with open(a.jsonl, encoding="utf-8") as f:
            for i, line in enumerate(f):
                r = json.loads(line)
                print(f"\n--- record {i} fields: {list(r)}")
                print(json.dumps(r, ensure_ascii=False)[:600])
                try:
                    k, lines = parse_record(r)
                    print(f"parsed key={k}  image_found={k in idx}  n_objects={len(lines)}")
                    print("  " + "\n  ".join(ln[:90] for ln in lines[:3]))
                except Exception as e:
                    print(f"PARSE ERROR: {e}")
                if i == 2:
                    break
        print("\nimage file examples:", [str(p) for p in list(idx.values())[:3]])
        return

    by_video = defaultdict(list)
    miss, bad, n = 0, 0, 0
    with open(a.jsonl, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n += 1
            try:
                key, lines = parse_record(json.loads(line))
            except Exception as e:
                bad += 1
                if bad <= 3:
                    print(f"  parse error: {e}")
                continue
            img = idx.get(key)
            if img is None:
                miss += 1
                continue
            by_video[video_of(key)].append((key, lines, img))
    kept = sum(len(v) for v in by_video.values())
    print(f"records {n} | kept {kept} | no image {miss} | parse errors {bad} | videos {len(by_video)}")
    if kept == 0:
        sys.exit("nothing kept: run with --inspect and check the key/image naming")

    score, seed, assign = choose_split(by_video, a.split_tries, a.seed)
    print(f"video split seed {seed} (worst class count in val/test = {score})")

    out = Path(a.out)
    full = out / "full"
    stats = {}
    for s in ("train", "val", "test"):
        stats[s] = {NAMES[k]: v for k, v in sorted(class_counts(by_video, assign, s).items())}
        for v, recs in by_video.items():
            if assign[v] != s:
                continue
            for key, lines, img in recs:
                link(img, full / "images" / s / f"{key}{img.suffix}")
                lp = full / "labels" / s / f"{key}.txt"
                lp.parent.mkdir(parents=True, exist_ok=True)
                lp.write_text("\n".join(lines))
    write_yaml(full, ("train", "val", "test"))

    # ---- pilot subset
    rng = random.Random(a.seed)
    pilot = out / "pilot"
    train = [(k, l, i) for v, r in by_video.items() if assign[v] == "train" for (k, l, i) in r]
    val = [(k, l, i) for v, r in by_video.items() if assign[v] == "val" for (k, l, i) in r]
    take_all = frozenset(NAMES.index(c) for c in a.pilot_all_classes.split(",") if c)
    ptrain = per_class_sample(train, a.pilot_per_class, rng, take_all)
    pval = per_class_sample(val, a.pilot_val_per_class, rng)
    pc = {"train": Counter(), "val": Counter()}
    for s, items in (("train", ptrain), ("val", pval)):
        for key, lines, img in items:
            link(img, pilot / "images" / s / f"{key}{img.suffix}")
            lp = pilot / "labels" / s / f"{key}.txt"
            lp.parent.mkdir(parents=True, exist_ok=True)
            lp.write_text("\n".join(lines))
            pc[s].update(int(ln.split()[0]) for ln in lines)
    write_yaml(pilot, ("train", "val"))
    stats["pilot_train"] = {NAMES[k]: v for k, v in sorted(pc["train"].items())}
    stats["pilot_val"] = {NAMES[k]: v for k, v in sorted(pc["val"].items())}
    stats["images"] = {"train": len(train), "pilot_train": len(ptrain), "pilot_val": len(pval),
                       "val": sum(len(r) for v, r in by_video.items() if assign[v] == "val"),
                       "test": sum(len(r) for v, r in by_video.items() if assign[v] == "test")}
    stats["split_seed"] = seed
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    print("\ninstances per class")
    print(f"{'class':15s}" + "".join(f"{s:>12s}" for s in ("train", "val", "test", "pilot_train", "pilot_val")))
    for c in NAMES:
        print(f"{c:15s}" + "".join(f"{stats[s].get(c, 0):12d}" for s in ("train", "val", "test", "pilot_train", "pilot_val")))
    print("images:", stats["images"])
    empty = [c for c in NAMES if all(stats[s].get(c, 0) == 0 for s in ("train", "val", "test"))]
    if empty:
        print(f"\nWARNING: no instances at all for {empty}. They will show as missing in every result table.")
    thin = [c for c in NAMES if c not in empty and (stats["val"].get(c, 0) == 0 or stats["test"].get(c, 0) == 0)]
    if thin:
        print(f"WARNING: {thin} missing from val or test; try --split-tries 200 or accept it.")
    print(f"\nfull : {full / 'data.yaml'}\npilot: {pilot / 'data.yaml'}")


if __name__ == "__main__":
    main()
