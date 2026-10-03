"""Why is mAP low? Match every test object to the model's predictions and break hits / misses down by cause.

    python eval_errors.py --weights runs/final/E0_s0/weights/best.pt --data ~/sg/yolo5/full --out analysis/errors_E0

At one operating point (--conf, default 0.25) predictions are matched greedily to ground truth of the same class
(box IoU >= --iou, default 0.5). For every ground-truth object it records whether it was found, together with
its size (shorter box side in px at 640), truncation, brightness and camera; for every unmatched prediction
(false positive) it records class, confidence and whether it overlaps a ground-truth object of another class.
Writes recall tables (errors.md), gt.csv / fp.csv, and crop sheets of missed objects and of the most confident
false positives, so they can be checked by eye (missed label? wrong label? real error?).
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

SIZE_BINS = [(0, 8), (8, 16), (16, 32), (32, 64), (64, 10_000)]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def read_gt(label: Path, w: int, h: int):
    out = []
    if not label.exists():
        return out
    for ln in label.read_text().splitlines():
        p = ln.split()
        if len(p) < 7:
            continue
        xs, ys = [float(v) for v in p[1::2]], [float(v) for v in p[2::2]]
        edge = min(xs) <= 0.002 or max(xs) >= 0.998 or min(ys) <= 0.002 or max(ys) >= 0.998
        out.append((int(p[0]), (min(xs) * w, min(ys) * h, max(xs) * w, max(ys) * h), edge))
    return out


def sheet(rows, images: Path, path: Path, title: str, n=40, seed=0):
    import cv2
    import numpy as np

    rows = rows[:n] if rows and "conf" in rows[0] else random.Random(seed).sample(rows, min(n, len(rows)))
    if not rows:
        return False
    tile, cols = 128, 8
    grid = np.full(((len(rows) + cols - 1) // cols * (tile + 18) + 34, cols * tile, 3), 255, np.uint8)
    cv2.putText(grid, title, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
    for i, r in enumerate(rows):
        im = cv2.imread(str(images / r["file"]))
        x0, y0, x1, y1 = (int(float(v)) for v in r["box"].split(","))
        bw, bh = max(1, x1 - x0), max(1, y1 - y0)
        pad = max(6, int(0.4 * max(bw, bh)))
        cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
        cx1, cy1 = min(im.shape[1], x1 + pad), min(im.shape[0], y1 + pad)
        crop = im[cy0:cy1, cx0:cx1].copy()
        cv2.rectangle(crop, (x0 - cx0, y0 - cy0), (x1 - cx0, y1 - cy0), (40, 120, 230), 1)
        s = tile / max(crop.shape[:2])
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))),
                          interpolation=cv2.INTER_NEAREST if s > 2 else cv2.INTER_AREA)
        gy, gx = 34 + (i // cols) * (tile + 18), (i % cols) * tile
        grid[gy:gy + crop.shape[0], gx:gx + crop.shape[1]] = crop
        lab = f"{bw}x{bh}" + (f" c{float(r['conf']):.2f}" if "conf" in r else "") + (f" ~{r['overlaps']}" if r.get("overlaps") else "")
        cv2.putText(grid, lab[:22], (gx + 2, gy + tile + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return True


def pct(n, d):
    return f"{100 * n / d:.1f}" if d else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True, help="dataset root with images/test and labels/test")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    img_dir, lab_dir = Path(a.data) / "images" / a.split, Path(a.data) / "labels" / a.split
    files = sorted(p.name for p in img_dir.iterdir())
    model = YOLO(a.weights)
    names = model.names

    gt_rows, fp_rows = [], []
    def predictions():  # a list given to predict() is loaded as one batch, so feed it in chunks
        for k in range(0, len(files), 32):
            chunk = files[k:k + 32]
            for f, r in zip(chunk, model.predict([str(img_dir / f) for f in chunk], conf=a.conf, imgsz=640,
                                                 half=True, device=a.device, verbose=False)):
                yield f, r

    for i, (f, r) in enumerate(predictions()):
        h, w = r.orig_shape
        gray = cv2.cvtColor(r.orig_img, cv2.COLOR_BGR2GRAY)
        gts = read_gt(lab_dir / (Path(f).stem + ".txt"), w, h)
        preds = sorted(zip(r.boxes.cls.int().tolist(), r.boxes.xyxy.tolist(), r.boxes.conf.tolist()), key=lambda t: -t[2])
        hit = [False] * len(gts)
        cam = "ZED" if "__ZED" in f else "smartphone"
        for c, pb, cf in preds:
            best, bj = 0.0, -1
            for j, (gc, gb, _) in enumerate(gts):
                if gc == c and not hit[j]:
                    v = iou(pb, gb)
                    if v > best:
                        best, bj = v, j
            if best >= a.iou:
                hit[bj] = True
            else:
                other = max(((iou(pb, gb), names[gc]) for gc, gb, _ in gts if gc != c), default=(0, ""))
                fp_rows.append({"file": f, "cls": names[c], "conf": round(cf, 3), "box": ",".join(f"{v:.0f}" for v in pb),
                                "overlaps": other[1] if other[0] >= 0.3 else "", "camera": cam})
        for j, (gc, gb, edge) in enumerate(gts):
            x0, y0, x1, y1 = (int(v) for v in gb)
            crop = gray[max(0, y0):max(y0 + 1, y1), max(0, x0):max(x0 + 1, x1)]
            gt_rows.append({"file": f, "cls": names[gc], "found": int(hit[j]), "min_side": round(min(gb[2] - gb[0], gb[3] - gb[1]), 1),
                            "truncated": int(edge), "bright": round(float(crop.mean()), 1) if crop.size else 0.0,
                            "camera": cam, "box": ",".join(f"{v:.0f}" for v in gb)})
        if i % 2000 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    for name, rows in (("gt.csv", gt_rows), ("fp.csv", fp_rows)):
        with open(out / name, "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["file"])
            wr.writeheader()
            wr.writerows(rows)

    cls_list = [names[k] for k in sorted(names)]
    by = defaultdict(list)
    for g in gt_rows:
        by[g["cls"]].append(g)
    fp_by = defaultdict(list)
    for p in fp_rows:
        fp_by[p["cls"]].append(p)
    imgs_with = {c: len({g["file"] for g in by[c]}) for c in cls_list}

    md = [f"# 오류 분석: `{a.weights}`\n",
          f"{a.split} {len(files):,}장. 기준: confidence ≥ {a.conf}, 같은 클래스 박스 IoU ≥ {a.iou}면 '찾음'. "
          "AP는 모든 confidence를 훑지만, 여기서는 한 지점에서 찾은 것 / 놓친 것 / 잘못 찾은 것을 센다.\n",
          "## 1. 클래스별 요약\n",
          "| 클래스 | 정답 객체 | 찾음(recall %) | 오검출(FP) | 정답 1개당 FP | precision % | FP 중 다른 클래스 정답과 겹침 % |",
          "|---|---|---|---|---|---|---|"]
    for c in cls_list:
        g, fp = by[c], fp_by[c]
        tp = sum(x["found"] for x in g)
        md.append(f"| {c} | {len(g):,} | {tp:,} ({pct(tp, len(g))}) | {len(fp):,} | {len(fp) / max(1, len(g)):.2f} | "
                  f"{pct(tp, tp + len(fp))} | {pct(sum(1 for p in fp if p['overlaps']), len(fp))} |")

    md += ["\n## 2. 크기별 recall (박스 짧은 변, 640px 기준)\n",
           "| 클래스 | " + " | ".join(f"{lo}~{hi}px" if hi < 10_000 else f"{lo}px 이상" for lo, hi in SIZE_BINS) + " |",
           "|---|" + "---|" * len(SIZE_BINS)]
    for c in cls_list:
        cells = []
        for lo, hi in SIZE_BINS:
            g = [x for x in by[c] if lo <= x["min_side"] < hi]
            cells.append(f"{pct(sum(x['found'] for x in g), len(g))}% (n={len(g):,})" if g else "-")
        md.append(f"| {c} | " + " | ".join(cells) + " |")

    md += ["\n## 3. 잘림 · 밝기 · 카메라별 recall\n",
           "| 클래스 | 잘리지 않음 | 잘림 | 밝음(≥50) | 어두움(<50) | 스마트폰 | ZED |", "|---|---|---|---|---|---|---|"]
    for c in cls_list:
        def rc(f):
            g = [x for x in by[c] if f(x)]
            return f"{pct(sum(x['found'] for x in g), len(g))}% (n={len(g):,})" if g else "-"
        md.append(f"| {c} | {rc(lambda x: not x['truncated'])} | {rc(lambda x: x['truncated'])} | "
                  f"{rc(lambda x: x['bright'] >= 50)} | {rc(lambda x: x['bright'] < 50)} | "
                  f"{rc(lambda x: x['camera'] == 'smartphone')} | {rc(lambda x: x['camera'] == 'ZED')} |")

    md.append("\n## 4. 예시 (파란 박스. 놓친 것 = 정답 위치, 오검출 = 예측 위치, c = confidence, ~클래스 = 겹친 다른 정답)\n")
    for c in cls_list:
        miss = [x for x in by[c] if not x["found"]]
        if sheet(miss, img_dir, out / f"missed_{c}.jpg", f"{c}: missed, random 40 of {len(miss)}"):
            md.append(f"### {c}: 놓친 객체 ({len(miss):,}개 중 무작위 40)\n\n![](missed_{c}.jpg)\n")
        fps = sorted(fp_by[c], key=lambda p: -p["conf"])
        if sheet(fps, img_dir, out / f"fp_{c}.jpg", f"{c}: false positives, top 40 conf of {len(fps)}"):
            md.append(f"### {c}: 오검출 (confidence 상위 40 / {len(fps):,}개)\n\n![](fp_{c}.jpg)\n")
    md.append(f"\n테스트 이미지 중 클래스가 하나라도 있는 사진 수: " + ", ".join(f"{c} {imgs_with[c]:,}" for c in cls_list))
    (out / "errors.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:30]))


if __name__ == "__main__":
    main()
