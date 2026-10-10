"""v2 학습 목록: 출처(파일명 접두어)별 상한 + 반복으로 비율을 조정. 신호 변화 클립 강조도 유지.
사용: python3 make_list.py <ds> <clips csv 또는 shard 폴더> <출력 yaml>
       [--repeat 189_=10] [--cap 188_=10000 --cap 614_=5000] [--hot 5]
출처 접두어: 189_ (기존 인도보행), 188_, 614_, 71579_, rf_"""
import argparse
import csv
import hashlib
import io
import zipfile
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("ds", type=Path)
ap.add_argument("clips", type=Path)
ap.add_argument("out_yaml", type=Path)
ap.add_argument("--repeat", action="append", default=[], help="접두어=배수")
ap.add_argument("--cap", action="append", default=[], help="접두어=최대 장수")
ap.add_argument("--hot", type=int, default=5, help="신호 변화 클립 프레임 배수")
args = ap.parse_args()
rep = {k: int(v) for k, v in (s.split("=") for s in args.repeat)}
cap = {k: int(v) for k, v in (s.split("=") for s in args.cap)}

# 신호 변화 클립
text = None
if args.clips.is_file():
    text = args.clips.read_text(encoding="utf-8-sig")
else:
    for zp in sorted(args.clips.glob("*.zip")):
        with zipfile.ZipFile(zp) as z:
            hit = [n for n in z.namelist() if n.endswith("clips_pedestrian_signal.csv")]
            if hit:
                text = z.read(hit[0]).decode("utf-8-sig")
                break
hot = {r["clip"].split("_")[-1] for r in csv.DictReader(io.StringIO(text or "")) if int(r["ped_signal_changes"]) > 0}

train = sorted((args.ds / "images" / "train").glob("*.jpg"),
               key=lambda p: hashlib.md5(p.name.encode()).hexdigest())
groups = {}
for p in train:
    k = next((pre for pre in ("189_", "188_", "614_", "71579_", "rf_") if p.name.startswith(pre)), "etc")
    groups.setdefault(k, []).append(p)
lines, summary = [], {}
for k, ps in groups.items():
    ps = ps[: cap.get(k, len(ps))]
    n = rep.get(k, 1)
    for p in ps:
        m = n
        if k == "71579_" and p.stem.split("_")[3] in hot:
            m *= args.hot
        lines += [str(p.resolve())] * m
    summary[k] = (len(ps), sum(1 for _ in ps) * n)
lst = args.out_yaml.with_suffix(".txt")
lst.write_text("\n".join(lines) + "\n")
y = (args.ds / "data.yaml").read_text(encoding="utf-8").splitlines()
y = [(f"train: {lst}" if l.startswith("train:") else l) for l in y]
args.out_yaml.write_text("\n".join(y) + "\n", encoding="utf-8")
print("출처별 (장수, 반복 반영 줄수):", summary, "/ 총", len(lines), "줄, 변화 클립", len(hot))
