"""README 그림 Fig. 8: results/tts_bench/summary.md → assets/fig8_tts.png

    ~/egoenv/Scripts/python.exe code/readme_figs/make_tts_fig.py
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
summ = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "tts_bench" / "summary.md"
out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "assets"
out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 11,
                     "axes.edgecolor": "#c9c8c2", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.labelcolor": "#52514e"})
S1, S2, INK = "#2a78d6", "#eb6834", "#0b0b0b"
NAME = {"sapi": "Heami\n(Windows)", "supertonic_F1": "Supertonic\nF1", "supertonic_M1": "Supertonic\nM1",
        "melotts": "MeloTTS", "mms": "MMS"}
rows = {}
for ln in summ.read_text(encoding="utf-8").splitlines():
    c = [x.strip() for x in ln.strip("|").split("|")]
    if c and c[0] in NAME:
        rows[c[0]] = dict(w=float(c[1]), d=float(c[2]), t=float(c[5]))  # 5 = 합성 중앙
keys = [k for k in NAME if k in rows]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
for a in (a1, a2):
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.set_axisbelow(True)
x = np.arange(len(keys)); w = 0.36
w_ = [rows[k]["w"] for k in keys]; d_ = [rows[k]["d"] for k in keys]
a1.bar(x - w / 2 - 0.01, w_, w, color=S1, label="즉시 경고 (32문장)")
a1.bar(x + w / 2 + 0.01, d_, w, color=S2, label="VLM 설명 (23문장)")
for i in range(len(keys)):
    a1.text(i - w / 2, w_[i] + 0.004, f"{w_[i]:.3f}", ha="center", fontsize=8.5)
    a1.text(i + w / 2, d_[i] + 0.004, f"{d_[i]:.3f}", ha="center", fontsize=8.5)
a1.set_xticks(x, [NAME[k] for k in keys], fontsize=9.5); a1.set_ylabel("받아쓰기 글자 오류율 (CER, 낮을수록 좋음)")
a1.grid(axis="y", color="#ecebe6"); a1.legend(frameon=False, fontsize=9.5)
a1.set_title("정확도 — Whisper large-v3 받아쓰기", loc="left", fontsize=11.5, color=INK)
t_ = [rows[k]["t"] for k in keys]; y = np.arange(len(keys))[::-1]
a2.barh(y, t_, color="#a3a29c", height=0.55); a2.set_yticks(y, [NAME[k].replace("\n", " ") for k in keys], fontsize=9.5)
for yy, v in zip(y, t_):
    a2.text(v + 0.15, yy, f"{v:.2f} s", va="center", fontsize=9)
a2.set_xlim(0, max(t_) * 1.25); a2.grid(axis="x", color="#ecebe6"); a2.set_xlabel("문장당 합성 시간 중앙값 (s)")
a2.set_title("속도 — 노트북 CPU (i5-1035G4)", loc="left", fontsize=11.5, color=INK)
fig.tight_layout(); fig.savefig(out / "fig8_tts.png", dpi=200, bbox_inches="tight", facecolor="white")
print(rows)
