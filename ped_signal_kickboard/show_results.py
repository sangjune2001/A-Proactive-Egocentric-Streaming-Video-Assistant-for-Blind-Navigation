"""학습 results.csv 요약: epoch별 mAP. 사용: python3 show_results.py <results.csv> [마지막 N개]"""
import csv
import sys

rows = list(csv.DictReader(open(sys.argv[1])))
n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
keys = [c for c in rows[0] if c.strip() == "epoch" or "mAP50" in c]
print(" | ".join(k.strip().replace("metrics/", "") for k in keys))
for r in rows[-n:]:
    print(" | ".join(r[k].strip()[:6] for k in keys))
best = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]))
print("best epoch", best["epoch"].strip(), "box mAP50", best["metrics/mAP50(B)"].strip()[:6],
      "mAP50-95", best["metrics/mAP50-95(B)"].strip()[:6])
