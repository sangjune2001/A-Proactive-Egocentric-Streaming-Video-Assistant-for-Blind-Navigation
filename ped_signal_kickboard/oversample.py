"""신호가 바뀌는 장면(71579에서 보행신호 red<->green 변화가 있는 클립)의 프레임을 학습 목록에 N번 반복 넣어 강조.
ds/train_list.txt 를 만들고 data.yaml 의 train 을 그 목록으로 바꾼다(라벨은 images->labels 경로로 자동 매칭).
사용: python3 oversample.py <ds> <clips_pedestrian_signal.csv> [--factor 5]"""
import argparse
import csv
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("ds", type=Path)
ap.add_argument("csv", type=Path)
ap.add_argument("--factor", type=int, default=5, help="변화 클립 프레임을 총 몇 번 넣을지")
args = ap.parse_args()

import io
import zipfile

if args.csv.exists():
    text = args.csv.read_text(encoding="utf-8-sig")
else:  # 예전 shard zip 안에 같이 들어가 있음
    text = None
    for zp in sorted(args.csv.parent.glob("*.zip")):
        with zipfile.ZipFile(zp) as z:
            hit = [n for n in z.namelist() if n.endswith("clips_pedestrian_signal.csv")]
            if hit:
                text = z.read(hit[0]).decode("utf-8-sig")
                break
    if text is None:
        raise SystemExit("clips_pedestrian_signal.csv 를 찾지 못함")
clips = set()
for row in csv.DictReader(io.StringIO(text)):
    if int(row["ped_signal_changes"]) > 0:
        clips.add(row["clip"].split("_")[-1])  # Clip_0997 -> 0997

train = sorted((args.ds / "images" / "train").glob("*.jpg"))
# 71579 이미지 이름: 71579_<번호>_<시간대>_<클립번호>_CF_<프레임> (예: 71579_512_ND_0997_CF_015)
hot = [p for p in train if p.name.startswith("71579_") and p.stem.split("_")[3] in clips]
lines = [str(p.resolve()) for p in train] + [str(p.resolve()) for p in hot] * (args.factor - 1)
(args.ds / "train_list.txt").write_text("\n".join(lines) + "\n")

y = (args.ds / "data.yaml").read_text(encoding="utf-8").splitlines()
y = [("train: train_list.txt" if l.startswith("train:") else l) for l in y]
(args.ds / "data.yaml").write_text("\n".join(y) + "\n", encoding="utf-8")
print(f"변화 클립 {len(clips)}개, 해당 프레임 {len(hot)}장 x{args.factor} / 전체 학습 목록 {len(lines)}줄")
