"""기존 189 seg 데이터(aihub189_yolo/labels + state.json manifest)를 YOLO 데이터셋 폴더에 '189_' 접두어로 추가.
사용: python build_189.py <드라이브 루트(MyDrive에 해당)> <ds 출력폴더>
manifest의 '/content/drive/MyDrive' 경로는 <드라이브 루트>로 바꿔 읽는다. 10%(해시) -> val."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

root, out = Path(sys.argv[1]), Path(sys.argv[2])
proj = root / "aihub189_yolo"
man = json.loads((proj / "state.json").read_text(encoding="utf-8"))["manifest"]
zips, n = {}, 0
for stem, (zpath, inner) in man.items():
    lab = proj / "labels" / f"{stem}.txt"
    if not lab.exists():
        continue
    sp = "val" if int(hashlib.md5(stem.encode()).hexdigest()[:8], 16) % 10 == 0 else "train"
    dst = out / "images" / sp / f"189_{stem}.jpg"
    if dst.exists():
        continue
    zp = Path(zpath.replace("/content/drive/MyDrive", str(root)))
    z = zips.get(zp) or zips.setdefault(zp, zipfile.ZipFile(zp))
    dst.parent.mkdir(parents=True, exist_ok=True)
    (out / "labels" / sp).mkdir(parents=True, exist_ok=True)
    dst.write_bytes(z.read(inner))
    (out / "labels" / sp / f"189_{stem}.txt").write_text(lab.read_text())
    n += 1
print("189 추가", n)
