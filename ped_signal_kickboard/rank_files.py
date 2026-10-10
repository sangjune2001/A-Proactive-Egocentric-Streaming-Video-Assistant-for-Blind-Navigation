"""라벨만 받아서 원천 파일(파일키)별로 보행신호등/킥보드가 몇 장 있는지 세고 rank.csv에 쓴다.
원천은 받지 않는다. 결과를 보고 밀도 높은 파일키만 골라 받으면 된다."""
import csv
import sys
import pipeline as p

OUT = p.BASE / "rank.csv"
st = {"bytes": 0}


def size_gb(s):
    n, u = s.split()
    return float(n) * {"KB": 1e-6, "MB": 1e-3, "GB": 1, "TB": 1e3, "B": 1e-9}[u]


def count(ds, fk, test):
    imgs = objs = total = 0
    kind, gen = p.sniff_kind(p.concat_sources(ds, [fk], st, p.Meter(f"{ds} rank {fk}")))
    for name, chunks in p.iter_members(gen, kind):
        if not name.endswith(".json"):
            p.drain(chunks)
            continue
        d = p.load_json(b"".join(chunks)) or {}
        total += 1
        n = test(d)
        objs += n
        imgs += n > 0
    return total, imgs, objs


ped = lambda d: sum(1 for a in d.get("annotation", [])
                    if a.get("class") == "traffic_light" and a.get("type") == "pedestrian")
pm = lambda d: sum(1 for a in d.get("annotations", []) if a.get("category_id") == 99)

rows = []
targets = sys.argv[1:] or ["188", "187", "71784"]
new = not OUT.exists()
with open(OUT, "a", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    if new:
        w.writerow(["dataset", "split", "source_file", "source_filekey", "source_size_GB",
                    "images", "target_images", "target_objects", "target_images_per_GB"])
    for ds in targets:
        t = p.parse_tree(ds)
        if ds in ("188", "187"):
            labs = [(pa, fn, k) for pa, fn, s, k in t if any("라벨링" in x for x in pa)]
            srcs = {(("Validation" if any("Validation" in x for x in pa) else "Training"), fn): (k, s)
                    for pa, fn, s, k in t if any("원천" in x for x in pa)}
            test = ped
        else:  # 71784: 라벨 zip 1개 = 카메라 방향 1개, 원천은 같은 방향 zip 1~2개
            labs = [(pa, fn, k) for pa, fn, s, k in t if "02.라벨링데이터" in pa and "가시광" in fn]
            srcs = {}
            for pa, fn, s, k in t:
                if "01.원천데이터" in pa and "가시광" in fn:
                    split = "Validation" if "Validation" in pa else "Training"
                    key = (split, fn.replace("TS_", "TL_").replace("VS_", "VL_").rsplit("_", 1)[0]
                           if fn.rsplit("_", 1)[-1][0].isdigit() else fn.replace("TS_", "TL_").replace("VS_", "VL_"))
                    prev = srcs.get(key)
                    srcs[key] = (k if not prev else prev[0] + "+" + k,
                                 s if not prev else f"{size_gb(prev[1]) + size_gb(s):.0f} GB")
            test = pm
        for pa, fn, k in labs:
            split = "Validation" if any("Validation" in x for x in pa) else "Training"
            key = (split, fn if ds != "71784" else fn.removesuffix(".zip"))
            sk, ss = srcs.get(key) or srcs.get((split, fn), ("?", "0 GB"))
            try:
                total, imgs, objs = count(ds, k, test)
            except Exception as e:
                print(f"{ds} {fn} 실패: {e!r}", flush=True)
                continue
            gb = size_gb(ss) or 1
            w.writerow([ds, split, fn, sk, f"{gb:.1f}", total, imgs, objs, f"{imgs / gb:.0f}"])
            f.flush()
            print(f"{ds} {split} {fn}: {imgs:,}/{total:,} (원천 {sk}, {gb:.0f}GB)", flush=True)
print("RANK DONE", flush=True)
