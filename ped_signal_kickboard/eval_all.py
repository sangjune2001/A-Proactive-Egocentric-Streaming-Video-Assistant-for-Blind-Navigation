"""학습 끝난 뒤 자동 평가 → docs/RESULTS.md + 이미지.
- 새 모델 vs 기존 seg10: 같은 검증 부분집합(189 / 188 / 614 / Roboflow / 전체)에서 클래스별 Box mAP50, mAP50-95
- 상태 분류: top1, 혼동행렬
- 예측 예시: 탐지 + 상태 분류 결합 결과 몽타주
사용: python3 eval_all.py <ds> <새 best.pt> <seg10 best.pt> <cls best.pt> <출력 docs 폴더>"""
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ds, new_w, old_w, cls_w, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], Path(sys.argv[5])
img_out = out / "img" / "results"
img_out.mkdir(parents=True, exist_ok=True)
names = ["person", "bicycle", "scooter", "motorcycle", "car", "bus", "other_vehicle", "obstacle", "stairs",
         "traffic_light"]
val_imgs = sorted((ds / "images" / "val").glob("*.jpg"))
subsets = {
    "189 (기존 인도보행)": [p for p in val_imgs if p.name.startswith("189_")],
    "188 (보행신호 포함 도로)": [p for p in val_imgs if p.name.startswith("188_")],
    "614 (킥보드)": [p for p in val_imgs if p.name.startswith("614_")],
    "Roboflow (한국 보행신호+킥보드)": [p for p in val_imgs if p.name.startswith("rf_")],
    "전체": val_imgs,
}


def yaml_for(key, imgs):
    lst = ds / f"val_{abs(hash(key)) % 10**6}.txt"
    lst.write_text("\n".join(str(p) for p in imgs) + "\n")
    y = ds / f"val_{abs(hash(key)) % 10**6}.yaml"
    y.write_text(f"path: {ds}\ntrain: {lst}\nval: {lst}\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
    return y


lines = ["# 학습 결과 (자동 생성)", "",
         "평가 기준은 박스(Box)입니다. 기존 seg10과 새 모델을 **같은 검증 이미지**에서 비교했습니다. imgsz는 1280입니다.", ""]
for key, imgs in subsets.items():
    if not imgs:
        continue
    y = yaml_for(key, imgs)
    res = {}
    for tag, w in (("seg10 (기존)", old_w), ("새 모델", new_w)):
        m = YOLO(w).val(data=str(y), imgsz=1280, batch=16, split="val", plots=False, verbose=False)
        per = {names[c]: (m.box.ap50[i], m.box.ap[i]) for i, c in enumerate(m.box.ap_class_index)}
        res[tag] = (m.box.map50, m.box.map, per)
    lines += [f"## {key}: 검증 {len(imgs)}장", "",
              "| 클래스 | seg10 mAP50 | 새 mAP50 | seg10 mAP50-95 | 새 mAP50-95 |", "|---|---|---|---|---|"]
    o, n = res["seg10 (기존)"], res["새 모델"]
    lines.append(f"| **all** | {o[0]:.3f} | **{n[0]:.3f}** | {o[1]:.3f} | **{n[1]:.3f}** |")
    for c in names:
        if c in n[2] or c in o[2]:
            a, b = o[2].get(c, (0, 0)), n[2].get(c, (0, 0))
            lines.append(f"| {c} | {a[0]:.3f} | {b[0]:.3f} | {a[1]:.3f} | {b[1]:.3f} |")
    lines.append("")

# 상태 분류
cm = YOLO(cls_w).val(data=str(ds.parent / "cls_ds"), imgsz=96, split="val", plots=True, verbose=False)
lines += ["## 신호 상태 분류 (yolo11n-cls, 96px)", "", f"- top-1 정확도: **{cm.top1:.3f}**", ""]
cls_dir = Path(cls_w).parent.parent
for f in ("confusion_matrix_normalized.png", "results.png"):
    if (cls_dir / f).exists():
        shutil.copy(cls_dir / f, img_out / f"cls_{f}")
        lines.append(f"![](img/results/cls_{f})")
det_dir = Path(new_w).parent.parent
for f in ("results.png", "confusion_matrix_normalized.png", "BoxPR_curve.png"):
    if (det_dir / f).exists():
        shutil.copy(det_dir / f, img_out / f"det_{f}")
lines += ["", "## 탐지 학습 곡선 / 혼동행렬", "", "![](img/results/det_results.png)", "",
          "![](img/results/det_confusion_matrix_normalized.png)", ""]

# 예측 예시 (탐지 + 상태 분류)
det, cls = YOLO(new_w), YOLO(cls_w)
COL = {"red": (0, 0, 255), "green": (0, 200, 0), "off": (150, 150, 150), "vehicle": (255, 128, 0)}
picks = []
for key in ("188", "rf_", "614", "189", "71579"):
    picks += [p for p in val_imgs if p.name.startswith(key)][:3]
picks += [p for p in sorted((ds / "images" / "train").glob("71579_*.jpg"))][:2]
tiles = []
for p in picks[:16]:
    r = det.predict(str(p), imgsz=1280, verbose=False, conf=0.3)[0]
    img = r.orig_img.copy()
    for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist()):
        x1, y1, x2, y2 = map(int, b)
        label, col = names[int(c)], (255, 255, 0)
        if int(c) == 9:
            w, h = x2 - x1, y2 - y1
            crop = img[max(0, int(y1 - .3 * h)):int(y2 + .3 * h), max(0, int(x1 - .3 * w)):int(x2 + .3 * w)]
            if crop.size:
                q = cls.predict(crop, imgsz=96, verbose=False)[0]
                label = q.names[q.probs.top1]
                col = COL.get(label, col)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 3)
        cv2.putText(img, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    img = cv2.resize(img, (640, int(img.shape[0] * 640 / img.shape[1])))
    img = cv2.copyMakeBorder(img, 0, max(0, 480 - img.shape[0]), 0, 0, cv2.BORDER_CONSTANT)[:480]
    tiles.append(img)
while len(tiles) % 4:
    tiles.append(np.zeros_like(tiles[0]))
sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
cv2.imwrite(str(img_out / "pred_examples.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
lines += ["## 예측 예시 (탐지 + 상태 분류)", "",
          "traffic_light 박스는 상태 분류 결과(red/green/off/vehicle)로 표시했습니다. 색: 빨강=red, 초록=green, 회색=off, 주황=vehicle", "",
          "![](img/results/pred_examples.jpg)", ""]
(out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
print("RESULTS 작성 완료")
