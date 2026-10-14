"""v5_11 기준 표 이미지 2장: ① Precision · Recall · 분당 오경보 (태규 B0 제외) ② GPU 단독 vs CPU+GPU 모듈별 시간."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle

font_manager.fontManager.addfont("C:/Windows/Fonts/malgun.ttf")
font_manager.fontManager.addfont("C:/Windows/Fonts/malgunbd.ttf")
plt.rcParams["font.family"] = "Malgun Gothic"

INK, MUTED, LINE, BG, PANEL = "#1d2420", "#5d6862", "#dfe3df", "#f6f7f5", "#ffffff"
GPU, CPU, ACC_SOFT, HL = "#13866f", "#c0781f", "#e3efe9", "#eef6f2"
OUT = "C:/Users/windo/Documents/trigger_proto/out/figs"
import os
os.makedirs(OUT, exist_ok=True)


def header(fig, title, sub):
    fig.text(0.04, 0.955, title, fontsize=19, fontweight="bold", color=INK, va="top")
    fig.text(0.04, 0.905, sub, fontsize=10.5, color=MUTED, va="top")


# ---------------- ① 채점 표 ----------------
rows = [  # (데이터, 알림, TP/FP/FN, P, R, F1, 분당 오경보, 강조)
    ("테스트 3편\n(라벨 12)", "위험만", "9 / 65 / 3", 0.122, 0.750, 0.209, 9.85, True),
    ("", "주의 + 위험", "11 / 113 / 1", 0.089, 0.917, 0.162, 17.13, False),
    ("학습 7편\n(라벨 16)", "위험만", "9 / 33 / 7", 0.214, 0.562, 0.310, 3.44, True),
    ("", "주의 + 위험", "9 / 73 / 7", 0.110, 0.562, 0.184, 7.60, False),
    ("학습 7편\n멈춘 탈것=장애물 인정", "위험만", "10 / 31 / 6", 0.244, 0.625, 0.351, 3.23, True),
    ("", "주의 + 위험", "10 / 69 / 6", 0.127, 0.625, 0.211, 7.19, False),
]
fig = plt.figure(figsize=(12.5, 6.6), dpi=170, facecolor=BG)
header(fig, "히트박스 트리거 채점 — v5_11 기준",
       "여유 기준: 라벨 구간 2초 전 ~ 0.5초 후 인정 · Precision ↑ Recall ↑ F1 ↑ 높을수록 좋음 · 분당 오경보 ↓ 낮을수록 좋음")
ax = fig.add_axes([0.04, 0.06, 0.92, 0.78]); ax.set_xlim(0, 100); ax.set_ylim(0, len(rows) + 1.4); ax.axis("off")
ax.add_patch(FancyBboxPatch((0, 0), 100, len(rows) + 1.4, boxstyle="round,pad=0,rounding_size=1.2", fc=PANEL, ec=LINE, lw=1))
cols = [(1.5, "데이터"), (19, "알림"), (29, "TP / FP / FN"), (41, "Precision ↑"), (56, "Recall ↑"), (71, "F1 ↑"), (85, "분당 오경보 ↓")]
ytop = len(rows) + 0.85
for x, c in cols:
    ax.text(x, ytop, c, fontsize=11, color=MUTED, va="center", fontweight="bold")
ax.plot([0.8, 99.2], [ytop - 0.4] * 2, color=LINE, lw=1.2)


def bar(x, y, v, vmax, w, color):
    ax.add_patch(Rectangle((x, y - 0.13), w, 0.26, fc=LINE, ec="none"))
    ax.add_patch(Rectangle((x, y - 0.13), w * v / vmax, 0.26, fc=color, ec="none"))


for i, (d, a, n, p, r, f1, fpm, hl) in enumerate(rows):
    y = len(rows) - i - 0.05
    if hl:
        ax.add_patch(Rectangle((0.8, y - 0.48), 98.4, 0.96, fc=HL, ec="none"))
    if d:
        ax.text(1.5, y - 0.5, d, fontsize=11.5, color=INK, va="center", fontweight="bold", linespacing=1.3)
        if i:
            ax.plot([0.8, 99.2], [y + 0.5] * 2, color=LINE, lw=1)
    ax.text(19, y, a, fontsize=11.5, color=INK, va="center", fontweight="bold" if hl else "normal")
    ax.text(29, y, n, fontsize=11.5, color=INK, va="center")
    for x, v in ((41, p), (56, r), (71, f1)):
        ax.text(x, y + 0.02, f"{v + 1e-9:.2f}", fontsize=12.5, color=INK, va="center", fontweight="bold" if hl else "normal")
        bar(x + 5.0, y, v, 1.0, 7.5, GPU)
    ax.text(85, y + 0.02, f"{fpm + 1e-9:.1f}", fontsize=12.5, color=INK, va="center", fontweight="bold" if hl else "normal")
    bar(90.5, y, fpm, 20, 7.8, CPU)
fig.text(0.04, 0.025, "B = 바닥 좌표 히트박스 트리거 v5_11 · 막대: Precision·Recall·F1은 0–1, 분당 오경보는 0–20회 기준 · 강조 행 = 위험만 음성으로 알릴 때",
         fontsize=9, color=MUTED)
fig.savefig(f"{OUT}/v5_11_채점표.png", facecolor=BG)
plt.close(fig)

# ---------------- ② GPU 단독 vs CPU+GPU ----------------
frame = [  # 모듈, (GPU 단독 곳, ms), (나눠 쓰기 곳, ms)
    ("프레임 읽기", ("CPU", 3.0), ("CPU", 3.0)),
    ("YOLO 검출 + 추적", ("GPU", 13.4), ("CPU×8", 39.7)),
    ("배경 광류 (걸음 속도 · 흔들림)", ("CPU", 43.0), ("CPU", 43.0)),
    ("트리거 판단", ("CPU", 2.5), ("CPU", 2.5)),
    ("신호 분류 (3프레임마다)", ("GPU", 1.5), ("CPU", 1.6)),
    ("바닥 분할 (5프레임마다)", ("GPU", 10.8), ("GPU", 10.8)),
]
event = [
    ("즉시 경고 (미리 합성한 조각 · 지금 방식)", "0.02 s", "0.05 s"),
    ("VLM 설명 문장 (Qwen2.5-VL-7B)", "0.93 s", "0.85 s"),
    ("TTS 합성 Supertonic (서버 CPU)", "1.34 s", "1.36 s"),
    ("설명까지 합계 (VLM + Supertonic)", "2.27 s", "2.21 s"),
    ("설명까지 합계 (VLM + Heami, 노트북)", "0.95 s", "0.87 s"),
]
fig = plt.figure(figsize=(12.5, 9.2), dpi=170, facecolor=BG)
header(fig, "GPU 단독 vs CPU+GPU 나눠 쓰기 — 모듈별 시간 (v5_11 기준)",
       "RunYourAI RTX A5000 · 중간값 · 나눠 쓰기 = VLM·바닥 분할만 GPU, YOLO·트리거·TTS는 CPU 8스레드(노트북급)")
ax = fig.add_axes([0.04, 0.38, 0.92, 0.5]); ax.set_xlim(0, 100); ax.set_ylim(0, len(frame) + 2.4); ax.axis("off")
ax.add_patch(FancyBboxPatch((0, 0), 100, len(frame) + 2.4, boxstyle="round,pad=0,rounding_size=1.2", fc=PANEL, ec=LINE, lw=1))
ytop = len(frame) + 1.8
ax.text(1.5, ytop, "A. 프레임마다 (ms / 프레임)", fontsize=12, color=INK, fontweight="bold", va="center")
ax.text(33, ytop, "GPU 단독", fontsize=11, color=GPU, fontweight="bold", va="center")
ax.text(66, ytop, "CPU+GPU 나눠 쓰기", fontsize=11, color=CPU, fontweight="bold", va="center")
ax.plot([0.8, 99.2], [ytop - 0.5] * 2, color=LINE, lw=1.2)
SC = 17 / 45.0      # 막대 길이: 45 ms = 17칸


def cell(x, y, where, ms, color):
    ax.text(x, y, where, fontsize=9.5, color=MUTED, va="center")
    bx = x + 8.5
    ax.add_patch(Rectangle((bx, y - 0.17), ms * SC, 0.34, fc=color, ec="none", alpha=0.9))
    ax.text(bx + ms * SC + 0.8, y, f"{ms:.1f}", fontsize=11.5, color=INK, va="center", fontweight="bold")


for i, (name, g, c) in enumerate(frame):
    y = len(frame) - i + 0.6
    ax.text(1.5, y, name, fontsize=11.5, color=INK, va="center")
    cell(33, y, g[0], g[1], GPU if g[0] == "GPU" else "#8fa39b")
    cell(66, y, c[0], c[1], CPU if c[0] != "GPU" else "#8fa39b")
ax.plot([0.8, 99.2], [0.95] * 2, color=LINE, lw=1.2)
ax.add_patch(Rectangle((0.8, 0.05), 98.4, 0.85, fc=HL, ec="none"))
ax.text(1.5, 0.48, "합계  (처리 가능 속도)", fontsize=12.5, color=INK, fontweight="bold", va="center")
ax.text(33, 0.48, "74.2 ms  (약 13 fps)", fontsize=13, color=GPU, fontweight="bold", va="center")
ax.text(66, 0.48, "100.6 ms  (약 10 fps)", fontsize=13, color=CPU, fontweight="bold", va="center")

ax2 = fig.add_axes([0.04, 0.06, 0.92, 0.29]); ax2.set_xlim(0, 100); ax2.set_ylim(0, len(event) + 1.4); ax2.axis("off")
ax2.add_patch(FancyBboxPatch((0, 0), 100, len(event) + 1.4, boxstyle="round,pad=0,rounding_size=1.2", fc=PANEL, ec=LINE, lw=1))
ytop = len(event) + 0.85
ax2.text(1.5, ytop, "B. 경고 한 번 (트리거 → 소리)", fontsize=12, color=INK, fontweight="bold", va="center")
ax2.text(60, ytop, "GPU 단독", fontsize=11, color=GPU, fontweight="bold", va="center")
ax2.text(80, ytop, "나눠 쓰기", fontsize=11, color=CPU, fontweight="bold", va="center")
ax2.plot([0.8, 99.2], [ytop - 0.45] * 2, color=LINE, lw=1.2)
for i, (name, g, c) in enumerate(event):
    y = len(event) - i - 0.1
    hl = name.startswith("설명까지 합계 (VLM + Supertonic")
    if hl:
        ax2.add_patch(Rectangle((0.8, y - 0.45), 98.4, 0.9, fc=HL, ec="none"))
    ax2.text(1.5, y, name, fontsize=11.5, color=INK, va="center", fontweight="bold" if hl else "normal")
    ax2.text(60, y, g, fontsize=12, color=INK, va="center", fontweight="bold" if hl or i == 0 else "normal")
    ax2.text(80, y, c, fontsize=12, color=INK, va="center", fontweight="bold" if hl or i == 0 else "normal")
fig.text(0.04, 0.022, "나눠 써도 즉시 경고는 +0.03초뿐 · YOLO를 CPU로 옮기면 프레임당 +26 ms (13 → 10 fps) · 가장 큰 비용은 원래 CPU인 배경 광류 · 모듈을 순서대로 돌린 합계",
         fontsize=9, color=MUTED)
fig.savefig(f"{OUT}/v5_11_GPU_CPU_속도비교.png", facecolor=BG)
plt.close(fig)
print("saved")
