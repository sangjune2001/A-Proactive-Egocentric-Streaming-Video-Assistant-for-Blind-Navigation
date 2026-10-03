"""Dataset analysis for labels_all.jsonl: how much data, how many objects, how they are split and sampled.

    python analyze_data.py --jsonl ~/sg/labels_all.jsonl --out analysis
    python analyze_data.py --jsonl labels_all.jsonl --out analysis --class-counts class_counts.csv   # + raw labels

Needs only the jsonl (no images). The video split and the pilot samples are rebuilt with the same functions and
seeds as build_dataset.py, so the numbers match the datasets the experiments were trained on.
Writes <out>/data_analysis.md, <out>/*.csv and, if matplotlib is installed, <out>/*.png.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import build_dataset as bd

NAMES10 = list(bd.NAMES)
COCO = {"person", "bicycle", "motorcycle", "car", "bus", "traffic_light"}     # also in the COCO pretraining
SPLITS = ("train", "val", "test")
# size buckets on relative mask area, COCO thresholds (32 px, 96 px) scaled to a 640 x 360 frame
SMALL, MEDIUM = 32 * 32 / (640 * 360), 96 * 96 / (640 * 360)


def poly_area(coords: list[float]) -> float:
    xs, ys = coords[0::2], coords[1::2]
    n = len(xs)
    return abs(sum(xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i] for i in range(n))) / 2


def load(jsonl: str):
    by_video, zip_of = defaultdict(list), {}
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            key, lines = bd.parse_record(r)
            by_video[bd.video_of(key)].append((key, lines, None))
            zip_of[key] = Path(r.get("z", "")).name
    return by_video, zip_of


def subset(items, keep: list[str]):
    """Keep only `keep` classes, renumbered like build_dataset.py --classes."""
    remap = {NAMES10.index(c): i for i, c in enumerate(keep)}
    out = []
    for key, lines, img in items:
        lines = [f"{remap[int(ln.split()[0])]} {ln.split(' ', 1)[1]}" for ln in lines if int(ln.split()[0]) in remap]
        out.append((key, lines, img))
    return out


def stats(items, names):
    """per class: objects, images, videos, objects per image (mean, max), relative area list"""
    obj, img, vids, per_img, areas = Counter(), Counter(), defaultdict(set), defaultdict(list), defaultdict(list)
    for key, lines, _ in items:
        cnt = Counter()
        for ln in lines:
            p = ln.split()
            c = int(p[0])
            cnt[c] += 1
            areas[c].append(poly_area([float(v) for v in p[1:]]))
        for c, k in cnt.items():
            obj[c] += k
            img[c] += 1
            vids[c].add(bd.video_of(key))
            per_img[c].append(k)
    rows = []
    for c, n in enumerate(names):
        a = areas[c]
        rows.append({
            "class": n, "objects": obj[c], "images": img[c], "videos": len(vids[c]),
            "obj_per_img_mean": round(obj[c] / img[c], 2) if img[c] else 0,
            "obj_per_img_max": max(per_img[c]) if per_img[c] else 0,
            "area_median_pct": round(100 * statistics.median(a), 3) if a else 0,
            "small_pct": round(100 * sum(x < SMALL for x in a) / len(a), 1) if a else 0,
            "medium_pct": round(100 * sum(SMALL <= x < MEDIUM for x in a) / len(a), 1) if a else 0,
            "large_pct": round(100 * sum(x >= MEDIUM for x in a) / len(a), 1) if a else 0,
        })
    return rows


def md_table(rows, cols, headers):
    out = "| " + " | ".join(headers) + " |\n|" + "---|" * len(headers) + "\n"
    for r in rows:
        out += "| " + " | ".join(f"{r[c]:,}" if isinstance(r[c], int) else str(r[c]) for c in cols) + " |\n"
    return out


def write_csv(path: Path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def charts(out: Path, overall, focus_rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except ImportError:
        print("matplotlib not installed: skipping charts")
        return []
    for f in ("/mnt/c/Windows/Fonts/malgun.ttf", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"):
        if Path(f).exists():
            font_manager.fontManager.addfont(f)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
            break
    ink, ink2, grid, s1, s2 = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"
    plt.rcParams.update({"axes.edgecolor": grid, "axes.labelcolor": ink2, "xtick.color": ink2,
                         "ytick.color": ink2, "text.color": ink, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.spines.left": False, "font.size": 10})
    files = []

    # 1) images and objects per class (log scale: 197 .. 294k)
    rows = sorted(overall, key=lambda r: r["objects"])
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    y = range(len(rows))
    ax.barh([i + 0.2 for i in y], [r["objects"] for r in rows], height=0.36, color=s1, label="객체 수")
    ax.barh([i - 0.2 for i in y], [r["images"] for r in rows], height=0.36, color=s2, label="사진 수")
    ax.set_yticks(list(y), [r["class"] for r in rows])
    ax.set_xscale("log")
    ax.grid(axis="x", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for i, r in enumerate(rows):
        ax.text(r["objects"] * 1.15, i + 0.2, f"{r['objects']:,}", va="center", fontsize=8, color=ink2)
        ax.text(r["images"] * 1.15, i - 0.2, f"{r['images']:,}", va="center", fontsize=8, color=ink2)
    ax.set_title("클래스별 객체 수와 사진 수 (전체 92,772장, 로그 축)", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "class_counts.png")
    plt.close(fig)
    files.append("class_counts.png")

    # 2) object size mix per class (share of small / medium / large masks)
    rows = sorted(overall, key=lambda r: -r["small_pct"])
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    y = list(range(len(rows)))
    sm = [r["small_pct"] for r in rows]
    md = [r["medium_pct"] for r in rows]
    lg = [r["large_pct"] for r in rows]
    ax.barh(y, sm, color="#2a78d6", label="small", edgecolor="white", linewidth=1)
    ax.barh(y, md, left=sm, color="#7fb0ea", label="medium", edgecolor="white", linewidth=1)
    ax.barh(y, lg, left=[a + b for a, b in zip(sm, md)], color="#c9def6", label="large", edgecolor="white",
            linewidth=1)
    for i, v in enumerate(sm):
        ax.text(1, i, f"{v:.0f}%", va="center", fontsize=8, color="white" if v > 8 else ink)
    ax.set_yticks(y, [r["class"] for r in rows])
    ax.set_xlim(0, 100)
    ax.set_xlabel("객체 비율 (%)")
    ax.set_title("클래스별 객체 크기 분포 (small < 32², medium < 96², 640×360 기준)", loc="left", fontsize=11)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    fig.savefig(out / "object_sizes.png")
    plt.close(fig)
    files.append("object_sizes.png")
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", default="analysis")
    ap.add_argument("--class-counts", default="", help="AI Hub raw label counts csv (데이터,라벨,객체 수,등장 이미지 수,...)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    by_video, zip_of = load(a.jsonl)
    _, seed, assign = bd.choose_split(by_video, 50, 0)
    split_items = {s: [t for v, r in by_video.items() if assign[v] == s for t in r] for s in SPLITS}
    every = [t for s in SPLITS for t in split_items[s]]

    # samples used by the experiments (same seeds as build_dataset.py)
    rng = random.Random(0)  # build_dataset.py draws train then val from one rng
    pilot1 = bd.per_class_sample(split_items["train"], 200, rng)
    pilot1_val = bd.per_class_sample(split_items["val"], 100, rng)
    take = frozenset(NAMES10.index(c) for c in ("scooter", "stairs", "traffic_light"))
    rng = random.Random(0)
    final_train = bd.per_class_sample(split_items["train"], 200, rng, take)
    final_val = bd.per_class_sample(split_items["val"], 100, rng)

    overall = stats(every, NAMES10)
    per_split = {s: stats(split_items[s], NAMES10) for s in SPLITS}
    final = {"train": stats(final_train, NAMES10), "val": stats(final_val, NAMES10), "test": per_split["test"]}

    write_csv(out / "class_stats_all.csv", overall)
    rows = []
    for s in SPLITS:
        rows += [{"split": s, **r} for r in per_split[s]]
    write_csv(out / "class_stats_by_split.csv", rows)
    rows = []
    for s in SPLITS:
        rows += [{"split": s, **r} for r in final[s]]
    write_csv(out / "final_stats.csv", rows)

    # ---------------------------------------------------------------- facts for the report
    zips = Counter(zip_of[k] for k, _, _ in every)
    n_obj = sum(len(l) for _, l, _ in every)
    empty = sum(1 for _, l, _ in every if not l)
    vids = {s: sum(1 for v in by_video if assign[v] == s) for s in SPLITS}
    stairs_zips = Counter(zip_of[k] for k, l, _ in every if any(int(x.split()[0]) == 7 + 1 for x in l))
    img_per_video = [len(r) for r in by_video.values()]
    co = Counter()
    for _, l, _ in final_train:
        co[len({int(x.split()[0]) for x in l} & set(take))] += 1

    figs = charts(out, overall, final["train"])

    by = {r["class"]: r for r in overall}
    rare = sorted(overall, key=lambda r: r["images"])[:3]
    md = f"""# 데이터 분석: AI Hub 189 인도 보행 영상 (10클래스 재라벨링)

`python analyze_data.py --jsonl labels_all.jsonl --out analysis`로 생성. 이미지 없이 라벨 파일만으로 계산하며,
train/val/test 분할과 pilot 샘플은 `build_dataset.py`와 같은 함수·seed로 다시 만들어 실험에 쓴 데이터와 숫자가 같다.

## 1. 전체 규모

| 항목 | 값 |
|---|---|
| 라벨된 사진 | {len(every):,}장 |
| 영상(폴더) | {len(by_video):,}개 (영상당 사진 평균 {statistics.mean(img_per_video):.1f}장, 최대 {max(img_per_video)}장) |
| 객체(폴리곤) | {n_obj:,}개 (사진당 평균 {n_obj / len(every):.1f}개) |
| 객체가 하나도 없는 사진 | {empty:,}장 |
| 원본 zip | Polygon P1~P14 {sum(v for z, v in zips.items() if z.startswith('P')):,}장, Surface S1 {zips.get('S1.zip', 0):,}장 |
| 해상도 | 원본에서 긴 변 640px로 줄여 학습 |

## 2. train / val / test 분할

영상 단위로 70 / 15 / 15 분할(같은 영상의 프레임이 서로 다른 split에 섞이지 않음). 50개 seed 중 val·test에서
가장 드문 클래스가 가장 많이 들어가는 seed({seed})를 자동 선택.

| split | 영상 | 사진 | 객체 |
|---|---|---|---|
""" + "".join(f"| {s} | {vids[s]:,} | {len(split_items[s]):,} | {sum(len(l) for _, l, _ in split_items[s]):,} |\n" for s in SPLITS) + f"""
## 3. 클래스별 분포 (전체)

객체 수는 폴리곤 개수, 사진 수는 그 클래스가 하나 이상 들어간 사진 수. 한 사진에 신호등이 3개 있으면 객체 3, 사진 1.

{md_table(sorted(overall, key=lambda r: -r['objects']), ['class', 'objects', 'images', 'videos', 'obj_per_img_mean', 'obj_per_img_max'], ['클래스', '객체', '사진', '영상', '사진당 객체(평균)', '사진당 객체(최대)'])}
""" + (f"![클래스별 객체 수와 사진 수](class_counts.png)\n\n" if "class_counts.png" in figs else "") + f"""- 가장 드문 클래스: {', '.join(f"{r['class']} (사진 {r['images']}장, 영상 {r['videos']}개)" for r in rare)}.
- 가장 많은 obstacle과 가장 적은 scooter의 객체 수 차이는 약 {by['obstacle']['objects'] // max(1, by['scooter']['objects']):,}배.
- COCO 사전학습에 이미 있는 클래스: {', '.join(c for c in NAMES10 if c in COCO)}. COCO에 없는 클래스: {', '.join(c for c in NAMES10 if c not in COCO)}.

### split별 (객체 / 사진)

| 클래스 | """ + " | ".join(SPLITS) + " |\n|---|" + "---|" * len(SPLITS) + "\n" + "".join(
        f"| {n} | " + " | ".join(f"{per_split[s][i]['objects']:,} / {per_split[s][i]['images']:,}" for s in SPLITS) + " |\n"
        for i, n in enumerate(NAMES10)) + f"""
## 4. 객체 크기

마스크 면적을 사진 면적 대비 비율로 계산. COCO 기준(32px, 96px)을 640×360 프레임에 맞춰 small < {100 * SMALL:.2f}%, medium < {100 * MEDIUM:.2f}%.

{md_table(sorted(overall, key=lambda r: -r['small_pct']), ['class', 'area_median_pct', 'small_pct', 'medium_pct', 'large_pct'], ['클래스', '면적 중앙값(%)', 'small(%)', 'medium(%)', 'large(%)'])}
""" + (f"![클래스별 객체 크기 분포](object_sizes.png)\n\n" if "object_sizes.png" in figs else "") + f"""## 5. 알아둘 점

- **stairs는 전부 Surface(S1.zip)에서 나옴** ({', '.join(f'{z} {n}장' for z, n in stairs_zips.items())}). Polygon 원본에는 계단 클래스가 없어서,
  Polygon 영상에 찍힌 계단은 배경으로 학습됨. stairs AP가 실제보다 낮거나 불안정할 수 있음.
- **scooter는 사진 {by['scooter']['images']}장, 영상 {by['scooter']['videos']}개뿐.** 영상 단위 분할이라 test에는 {per_split['test'][2]['images']}장만 들어가 test AP 변동이 큼.
- **traffic_light는 작은 객체 비율이 높음** (small {by['traffic_light']['small_pct']}%). 640px 입력에서 놓치기 쉬움.

## 6. 실험별로 실제 쓴 데이터

### 1차 pilot (10클래스, 2026-10-01)

클래스마다 그 클래스가 들어간 사진을 약 200장(val은 100장) 뽑음. 사진 수 기준이며 객체 수가 아님.
train {len(pilot1):,}장, val {len(pilot1_val):,}장. 결과는 [`README.md`](../README.md) 참고.

### 최종 학습 (10클래스, 2026-10-04)

- 라벨: 10개 클래스 모두 유지.
- train: scooter·stairs·traffic_light가 들어간 train 사진 **전부** → **{len(final_train):,}장**. 다른 클래스는 이 사진들에 이미 사진 200장 이상씩 들어 있어 추가 0장.
- val: 클래스당 사진 약 100장 → {len(final_val):,}장 (best epoch 선택용).
- test: 전체 test {len(split_items['test']):,}장.
- train 사진 중 scooter·stairs·traffic_light가 1개 있는 사진 {co[1]:,}장, 2개 이상 {sum(v for k, v in co.items() if k >= 2):,}장.

| 클래스 | train 객체 / 사진 | val 객체 / 사진 | test 객체 / 사진 |
|---|---|---|---|
""" + "".join(
        f"| {n} | " + " | ".join(f"{final[s][i]['objects']:,} / {final[s][i]['images']:,}" for s in SPLITS) + " |\n"
        for i, n in enumerate(NAMES10))

    if a.class_counts and Path(a.class_counts).exists():
        raw = list(csv.reader(open(a.class_counts, encoding="utf-8-sig")))[1:]
        md += "\n## 7. 원본 라벨 → 10클래스 매핑 (AI Hub 원본 개수)\n\n| 원본 라벨 | 객체 | 사진 | → 클래스 |\n|---|---|---|---|\n"
        for d, lab, o, i, *_ in raw:
            if d != "polygon":
                continue
            c = bd.MAP.get(lab)
            md += f"| {lab} | {int(o):,} | {int(i):,} | {NAMES10[c] if c is not None else '(사용 안 함)'} |\n"
        md += "\nSurface(S1)에서는 계단 라벨만 stairs로 사용.\n"

    (out / "data_analysis.md").write_text(md, encoding="utf-8")
    print(f"wrote {out / 'data_analysis.md'} (+ csv, {len(figs)} charts)")


if __name__ == "__main__":
    main()
