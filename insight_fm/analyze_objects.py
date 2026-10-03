"""Look at every labelled object in the actual images: is it cut off, small, blurry or dark?

    python analyze_objects.py --jsonl ~/sg/labels_all.jsonl --images ~/sg/imgs --out analysis/objects

Per object (all 10 classes, all splits) it measures, on the 640px training images:
    width/height in px, mask area, truncated (polygon touches the image border), sharpness (variance of the
    Laplacian inside the object box, higher = sharper), brightness and contrast inside the box,
    plus the same sharpness/brightness for the whole frame.
Writes objects.csv.gz (one row per object), object_quality.md (summary per class) and contact sheets
(crops of real objects per class and per problem) as JPEG so the data can be checked by eye.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import random
import statistics
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import build_dataset as bd
from analyze_data import load

NAMES = list(bd.NAMES)
FOCUS = ["scooter", "stairs", "obstacle", "other_vehicle", "traffic_light"]
EDGE = 0.002          # a polygon point this close to the border (relative) counts as touching it
TINY_PX = 16          # shorter box side below this: too few pixels to segment well at 640 input
BLUR_THR = 50.0       # Laplacian variance below this inside the box: visibly soft (checked on the sheets)
DARK_THR = 50         # mean gray level below this inside the box: dark


def measure(args):
    """one image -> rows for all its objects"""
    key, lines, split, img_path = args
    import cv2
    import numpy as np

    im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return []
    h, w = im.shape
    lap = cv2.Laplacian(im, cv2.CV_64F)
    rows = [None] * len(lines)
    frame_sharp, frame_bright = float(lap.var()), float(im.mean())
    for j, ln in enumerate(lines):
        p = ln.split()
        c = int(p[0])
        xs = np.array([float(v) for v in p[1::2]])
        ys = np.array([float(v) for v in p[2::2]])
        x0, x1 = int(np.clip(xs.min() * w, 0, w - 1)), int(np.clip(np.ceil(xs.max() * w), 1, w))
        y0, y1 = int(np.clip(ys.min() * h, 0, h - 1)), int(np.clip(np.ceil(ys.max() * h), 1, h))
        bw, bh = max(1, x1 - x0), max(1, y1 - y0)
        area = abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1))) / 2
        crop, lcrop = im[y0:y0 + bh, x0:x0 + bw], lap[y0:y0 + bh, x0:x0 + bw]
        edges = (xs.min() <= EDGE, xs.max() >= 1 - EDGE, ys.min() <= EDGE, ys.max() >= 1 - EDGE)
        rows[j] = {
            "key": key, "split": split, "cls": NAMES[c], "img_w": w, "img_h": h,
            "box_w": bw, "box_h": bh, "min_side": min(bw, bh), "area_pct": round(100 * area, 4),
            "truncated": int(any(edges)), "trunc_sides": "".join(s for s, e in zip("LRTB", edges) if e),
            "sharp": round(float(lcrop.var()), 1) if lcrop.size > 4 else 0.0,
            "bright": round(float(crop.mean()), 1), "contrast": round(float(crop.std()), 1),
            "frame_sharp": round(frame_sharp, 1), "frame_bright": round(frame_bright, 1),
            "box": f"{x0},{y0},{bw},{bh}",
        }
    return rows


def pct(rows, f):
    return round(100 * sum(1 for r in rows if f(r)) / len(rows), 1) if rows else 0.0


def med(rows, k):
    return round(statistics.median(r[k] for r in rows), 1) if rows else 0


def sheet(rows, images: Path, path: Path, title: str, n=40, seed=0):
    """grid of object crops (box + 25% context, object outlined) for eyeballing"""
    import cv2
    import numpy as np

    rows = random.Random(seed).sample(rows, min(n, len(rows)))
    if not rows:
        return False
    tile, cols = 128, 8
    grid = np.full(((len(rows) + cols - 1) // cols * (tile + 18) + 34, cols * tile, 3), 255, np.uint8)
    cv2.putText(grid, title, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
    for i, r in enumerate(rows):
        im = cv2.imread(str(images / f"{r['key']}.jpg"))
        x, y, bw, bh = map(int, r["box"].split(","))
        pad = max(4, int(0.25 * max(bw, bh)))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(im.shape[1], x + bw + pad), min(im.shape[0], y + bh + pad)
        crop = im[y0:y1, x0:x1].copy()
        cv2.rectangle(crop, (x - x0, y - y0), (x - x0 + bw - 1, y - y0 + bh - 1), (40, 120, 230), 1)
        s = tile / max(crop.shape[:2])
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))),
                          interpolation=cv2.INTER_NEAREST if s > 2 else cv2.INTER_AREA)
        gy, gx = 34 + (i // cols) * (tile + 18), (i % cols) * tile
        grid[gy:gy + crop.shape[0], gx:gx + crop.shape[1]] = crop
        lab = f"{r['box_w']}x{r['box_h']} s{int(r['sharp'])}" + (" T" if r["truncated"] else "")
        cv2.putText(grid, lab, (gx + 2, gy + tile + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 60, 60), 1,
                    cv2.LINE_AA)
    cv2.imwrite(str(path), grid)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="analysis/objects")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    a = ap.parse_args()
    out, images = Path(a.out), Path(a.images)
    out.mkdir(parents=True, exist_ok=True)

    by_video, _ = load(a.jsonl)
    _, _, assign = bd.choose_split(by_video, 50, 0)
    jobs = [(k, l, assign[v], str(images / f"{k}.jpg")) for v, recs in by_video.items() for k, l, _ in recs]
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(measure, jobs, chunksize=64)):
            rows += r
            if i % 10000 == 0:
                print(f"  {i}/{len(jobs)} images", flush=True)
    print(f"{len(rows)} objects measured")
    with gzip.open(out / "objects.csv.gz", "wt", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    by = defaultdict(list)
    for r in rows:
        by[r["cls"]].append(r)
    frames = {r["key"]: r for r in rows}.values()

    md = ["# 객체 품질 분석 (실제 이미지 기준)\n",
          "`python analyze_objects.py`로 생성. 학습에 쓰는 640px 이미지에서 모든 라벨 객체를 직접 측정.\n",
          f"- **잘림**: 폴리곤이 사진 가장자리에 닿음 (프레임 밖으로 잘린 객체)",
          f"- **아주 작음**: 객체 박스의 짧은 변 < {TINY_PX}px (640px 입력 기준)",
          f"- **흐림**: 객체 박스 안 Laplacian 분산 < {BLUR_THR:.0f} (낮을수록 흐림. 기준은 아래 예시 이미지로 눈으로 확인)",
          f"- **어두움**: 객체 박스 안 평균 밝기 < {DARK_THR} (0~255)\n",
          f"전체 프레임: 선명도 중앙값 {med(list(frames), 'frame_sharp')}, 흐린 프레임(< {BLUR_THR:.0f}) "
          f"{pct(list(frames), lambda r: r['frame_sharp'] < BLUR_THR)}%, "
          f"어두운 프레임(< {DARK_THR}) {pct(list(frames), lambda r: r['frame_bright'] < DARK_THR)}%\n",
          "| 클래스 | 객체 | 박스 짧은 변 중앙값(px) | 아주 작음(%) | 잘림(%) | 흐림(%) | 어두움(%) | 선명도 중앙값 |",
          "|---|---|---|---|---|---|---|---|"]
    order = sorted(NAMES, key=lambda c: (c not in FOCUS, c))
    for c in order:
        rs = by[c]
        md.append(f"| {c}{' ★' if c in FOCUS else ''} | {len(rs):,} | {med(rs, 'min_side')} | "
                  f"{pct(rs, lambda r: r['min_side'] < TINY_PX)} | {pct(rs, lambda r: r['truncated'])} | "
                  f"{pct(rs, lambda r: r['sharp'] < BLUR_THR)} | {pct(rs, lambda r: r['bright'] < DARK_THR)} | "
                  f"{med(rs, 'sharp')} |")
    md.append("\n★ = 5클래스 최종 모델의 대상 클래스\n")

    md.append("## 학습/평가 split별 (5클래스)\n\n| 클래스 | split | 객체 | 아주 작음(%) | 잘림(%) | 흐림(%) |\n|---|---|---|---|---|---|")
    for c in FOCUS:
        for s in ("train", "val", "test"):
            rs = [r for r in by[c] if r["split"] == s]
            md.append(f"| {c} | {s} | {len(rs):,} | {pct(rs, lambda r: r['min_side'] < TINY_PX)} | "
                      f"{pct(rs, lambda r: r['truncated'])} | {pct(rs, lambda r: r['sharp'] < BLUR_THR)} |")

    md.append("\n## 실제 객체 예시 (무작위 40개, 파란 박스 = 라벨, 아래 숫자 = 박스 크기 / 선명도 / T=잘림)\n")
    for c in FOCUS:
        if sheet(by[c], images, out / f"sheet_{c}.jpg", f"{c}: random 40"):
            md.append(f"### {c}\n\n![{c}](sheet_{c}.jpg)\n")
    probs = {
        "tiny": ("아주 작은 객체", lambda r: r["min_side"] < TINY_PX),
        "truncated": ("잘린 객체", lambda r: r["truncated"]),
        "blurry": ("흐린 객체", lambda r: r["sharp"] < BLUR_THR and r["min_side"] >= TINY_PX),
        "dark": ("어두운 객체", lambda r: r["bright"] < DARK_THR),
    }
    md.append("## 문제 유형별 예시 (5클래스)\n")
    for name, (title, f) in probs.items():
        rs = [r for r in rows if r["cls"] in FOCUS and f(r)]
        if sheet(rs, images, out / f"problem_{name}.jpg", f"{name}: random 40 of {len(rs)}"):
            md.append(f"### {title} ({len(rs):,}개)\n\n![{title}](problem_{name}.jpg)\n")

    (out / "object_quality.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {out / 'object_quality.md'}")


if __name__ == "__main__":
    main()
