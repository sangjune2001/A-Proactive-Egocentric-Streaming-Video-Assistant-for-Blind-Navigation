"""Pull the labelled frames out of the AI Hub 189 Polygon zips and save them as 640px JPEGs.

labels_all.jsonl lines look like
    {"k": "Polygon_0001__MP_SEL_P000002",
     "z": "/content/drive/MyDrive/sideguide/polygon/P1.zip",
     "m": "Polygon_0001/MP_SEL_P000002.jpg",
     "l": ["7 0.28 0.00 ...", ...]}
Each image is written to <out>/<k>.jpg (long side resized to --size, aspect kept, so the normalised
YOLO labels stay valid). Only the zip basename of "z" is used, so Colab paths do not matter.

    # zips already on local disk
    python extract_images.py --jsonl labels_all.jsonl --zips /data/sg/zips --out /data/sg/imgs
    # or stream them one at a time from Drive and delete each after use (peak disk ~ one zip, ~10 GB)
    python extract_images.py --jsonl labels_all.jsonl --remote gdrive:sideguide/polygon \
        --zips /data/sg/zips --out /data/sg/imgs --delete-zip

Resumable: images that already exist are skipped, finished zips are recorded in <out>/_done_zips.txt.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def load_index(jsonl: str) -> dict[str, list[tuple[str, str]]]:
    """zip basename -> [(member, key), ...]"""
    by_zip = defaultdict(list)
    n = 0
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            by_zip[Path(r["z"]).name].append((r["m"], r["k"]))
            n += 1
    print(f"{n} labelled frames in {len(by_zip)} zips: " +
          ", ".join(f"{z}={len(v)}" for z, v in sorted(by_zip.items(), key=lambda t: int(''.join(filter(str.isdigit, t[0])) or 0))),
          flush=True)
    return by_zip


def _resize_write(data: bytes, dst: Path, size: int) -> None:
    import cv2
    import numpy as np

    im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError("decode failed")
    h, w = im.shape[:2]
    s = size / max(h, w)
    if s < 1:
        im = cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    tmp = dst.with_suffix(".tmp.jpg")
    cv2.imwrite(str(tmp), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
    os.replace(tmp, dst)


def _work(args):
    zpath, items, out, size = args
    ok = skip = bad = 0
    with zipfile.ZipFile(zpath) as zf:
        names = set(zf.namelist())
        for member, key in items:
            dst = Path(out) / f"{key}.jpg"
            if dst.exists():
                skip += 1
                continue
            m = member if member in names else member.replace("\\", "/")
            if m not in names:
                bad += 1
                continue
            try:
                _resize_write(zf.read(m), dst, size)
                ok += 1
            except Exception:
                bad += 1
    return ok, skip, bad


def extract_zip(zpath: Path, items, out: Path, size: int, workers: int):
    chunks = [items[i::workers] for i in range(workers)]
    tot = [0, 0, 0]
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(_work, [(str(zpath), c, str(out), size) for c in chunks if c]):
            tot = [a + b for a, b in zip(tot, r)]
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--zips", required=True, help="local folder holding (or receiving) P*.zip")
    ap.add_argument("--out", required=True)
    ap.add_argument("--remote", default="", help="rclone path of the zip folder, e.g. gdrive:sideguide/polygon")
    ap.add_argument("--delete-zip", action="store_true", help="delete each zip after extracting it")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    a = ap.parse_args()

    out, zdir = Path(a.out), Path(a.zips)
    out.mkdir(parents=True, exist_ok=True)
    zdir.mkdir(parents=True, exist_ok=True)
    done_file = out / "_done_zips.txt"
    done = set(done_file.read_text().split()) if done_file.exists() else set()

    by_zip = load_index(a.jsonl)
    for zname in sorted(by_zip, key=lambda z: int(''.join(filter(str.isdigit, z)) or 0)):
        if zname in done:
            print(f"[skip] {zname} already extracted", flush=True)
            continue
        zpath = zdir / zname
        if not zpath.exists():
            if not a.remote:
                print(f"[miss] {zpath} not found and no --remote given", flush=True)
                continue
            t = time.time()
            print(f"[get ] {a.remote}/{zname}", flush=True)
            rc = subprocess.call(["rclone", "copy", f"{a.remote}/{zname}", str(zdir), "-P", "--stats-one-line"])
            if rc != 0 or not zpath.exists():
                print(f"[fail] download {zname} (rc={rc}); rerun later to retry", flush=True)
                continue
            print(f"       {zpath.stat().st_size / 1e9:.1f} GB in {time.time() - t:.0f}s", flush=True)
        t = time.time()
        try:
            ok, skip, bad = extract_zip(zpath, by_zip[zname], out, a.size, a.workers)
        except zipfile.BadZipFile:
            print(f"[fail] {zname} is corrupt; deleting so the next run downloads it again", flush=True)
            zpath.unlink(missing_ok=True)
            continue
        print(f"[done] {zname}: wrote {ok}, existed {skip}, missing/bad {bad} ({time.time() - t:.0f}s)", flush=True)
        with open(done_file, "a") as f:
            f.write(zname + "\n")
        if a.delete_zip:
            zpath.unlink(missing_ok=True)

    n = sum(1 for p in out.glob("*.jpg"))
    total = sum(len(v) for v in by_zip.values())
    print(f"\nimages on disk: {n} / {total} labelled frames", flush=True)
    if n < total:
        print("some frames are missing: rerun the same command (it resumes), or check the [fail]/[miss] lines")


if __name__ == "__main__":
    sys.exit(main())
