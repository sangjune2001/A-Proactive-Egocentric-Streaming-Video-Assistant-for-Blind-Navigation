"""README 그림 Fig. 1–7 → assets/*.png (숫자는 docs/12 · results/vlm/_eval, c09 실행 결과에서 옮김)

    ~/egoenv/Scripts/python.exe code/readme_figs/make_figs.py
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "assets"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False,
                     "font.size": 11, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e"})
S1, S2, S3, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#ffffff"


def clean(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color="#ecebe6", lw=0.8); ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor=SURF); plt.close(fig)


def box(ax, x, y, w, h, text, fc, ec, fs=10.5, bold=False, color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=color, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, color=INK2, text=None, dy=0.13):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, text, ha="center", fontsize=8.5, color=INK2)


# ---------- Fig 1: architecture ----------
fig, ax = plt.subplots(figsize=(13, 5.8)); ax.set_xlim(0, 13); ax.set_ylim(-0.15, 5.6); ax.axis("off")
ax.add_patch(FancyBboxPatch((0.15, 2.55), 6.55, 2.75, boxstyle="round,pad=0.02,rounding_size=0.15",
                            fc="#f3f7fd", ec="#bcd3f0", lw=1, ls="--"))
ax.text(0.35, 5.08, "매 프레임 (15 fps) — 판단: 언제 · 무엇이 위험한가", fontsize=9.5, color=S1, fontweight="bold")
ax.add_patch(FancyBboxPatch((7.0, 0.25), 5.85, 4.95, boxstyle="round,pad=0.02,rounding_size=0.15",
                            fc="#fdf5f1", ec="#f2c6b3", lw=1, ls="--"))
ax.text(7.2, 4.98, "이벤트 때만 — 전달: 정확 · 빠르게 말하기", fontsize=9.5, color=S2, fontweight="bold")

box(ax, 0.35, 3.15, 1.55, 1.3, "① 카메라\n프레임\n(녹화 영상\n실제 속도)", "#ffffff", "#c9c8c2", 9.5)
box(ax, 2.25, 3.15, 2.0, 1.3, "② YOLO11s-seg\n+ ByteTrack\nid · 클래스 · bbox\n· 마스크", "#e3eefb", S1, 9.5)
box(ax, 4.6, 3.15, 1.9, 1.3, "③ 트리거\nstep(frame)\n→ [Event]\n(B0 → 임태규 규칙)", "#e3eefb", S1, 9.5)
arrow(ax, 1.9, 3.8, 2.25, 3.8); arrow(ax, 4.25, 3.8, 4.6, 3.8)

box(ax, 7.25, 3.45, 1.6, 1.0, "④ 분배기\n(우선순위)", "#ffffff", "#c9c8c2", 10, True)
arrow(ax, 6.5, 3.8, 7.25, 3.8, text="Event", dy=0.12)

box(ax, 9.25, 3.55, 3.4, 1.0, "⑤ 즉시 경고  \"왼쪽 오토바이\"\n미리 합성한 wav · VLM 안 기다림", "#fbe4da", S2, 9.5, True)
box(ax, 9.25, 1.95, 3.4, 1.25, "⑥ VLM (별도 스레드)  Qwen2.5-VL-7B\n과거 8장 · 빨간 박스 → JSON\n대상 · 움직임만 답함", "#fbe4da", S2, 9.2)
box(ax, 9.25, 0.5, 1.55, 1.05, "⑦ 하이브리드\n방향 = bbox\n행동 = 규칙표", "#fff4dc", "#eda100", 8.8)
box(ax, 11.1, 0.5, 1.55, 1.05, "⑧ TTS\n→ 한국어\n문장 음성", "#fbe4da", S2, 9.2)
arrow(ax, 8.85, 4.05, 9.25, 4.05); arrow(ax, 8.05, 3.45, 9.25, 2.6)
arrow(ax, 10.0, 1.95, 10.0, 1.55); arrow(ax, 10.8, 1.02, 11.1, 1.02)

box(ax, 0.35, 0.3, 6.3, 1.8, "", "#ffffff", "#c9c8c2")
ax.text(0.6, 1.75, "⑨ 재생 큐 (스피커)", fontsize=10.5, fontweight="bold", color=INK)
ax.text(0.6, 0.85, "• 경고가 설명을 끊는다 (경고 우선)\n• 새 경고가 나가면 이전 이벤트의 늦은 설명은 버린다\n• 모든 단계 시각을 로그 → 채점 · 시연 영상(소리+박스+자막)",
        fontsize=9, color=INK2, linespacing=1.5)
ax.annotate("", (6.7, 1.9), (9.25, 3.62), arrowprops=dict(arrowstyle="-|>", color=S2, lw=1.6))
ax.text(7.05, 2.95, "경고 ~0.1 s", fontsize=9, color=S2, fontweight="bold")
ax.plot([11.875, 11.875], [0.47, 0.37], color=S2, lw=1.6)
ax.annotate("", (6.7, 0.37), (11.875, 0.37), arrowprops=dict(arrowstyle="-|>", color=S2, lw=1.6))
ax.text(7.2, 0.05, "설명 ~1.4 s", fontsize=9, color=S2, fontweight="bold")
save(fig, "fig1_architecture.png")

# ---------- data (docs/12) ----------
models = ["Qwen2.5-VL-7B", "Qwen2.5-VL-3B", "InternVL3.5-4B", "Qwen3-VL-8B", "InternVL3.5-8B"]
full4 = [0.09, 0.00, 0.17, 0.09, 0.09]
hyb = [0.39, 0.26, 0.26, 0.22, 0.22]
lat = [0.94, 0.52, 0.64, 1.02, 1.05]
PT = 0.09

# ---------- Fig 2: 4-field vs hybrid ----------
fig, ax = plt.subplots(figsize=(10, 4.4)); clean(ax)
import numpy as np
x = np.arange(len(models)); w = 0.36
b1 = ax.bar(x - w / 2 - 0.01, full4, w, color=S1, label="VLM이 4칸 전부 답함")
b2 = ax.bar(x + w / 2 + 0.01, hyb, w, color=S2, label="하이브리드 (VLM 2칸 + 규칙 2칸)")
ax.axhline(PT, color=GRAY, ls="--", lw=1.4, label="PT (VLM 없이 규칙만) 0.09")
for b, v in zip(b2, hyb):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9.5, color=INK)
ax.set_xticks(x, models); ax.set_ylim(0, 0.48); ax.set_ylabel("완전 정답률 (4칸 모두 정답)")
ax.legend(frameon=False, loc="upper right", fontsize=9.5)
ax.set_title("VLM 역할을 줄이면 정확도가 오른다 (정답 시각 23개 · F8 · 사후 채점)", loc="left", fontsize=12, color=INK)
save(fig, "fig2_hybrid.png")

# ---------- Fig 3: per-field — who is better ----------
fields = ["대상", "움직임", "방향", "행동"]
vlm7 = [0.87, 0.61, 0.65, 0.22]; pt = [0.52, 0.09, 0.91, 0.83]
fig, ax = plt.subplots(figsize=(8.5, 4.2)); clean(ax)
x = np.arange(4)
ax.bar(x - w / 2 - 0.01, vlm7, w, color=S1, label="VLM (Qwen2.5-VL-7B)")
ax.bar(x + w / 2 + 0.01, pt, w, color=GRAY, label="규칙 · bbox (PT, VLM 없음)")
for i, (a, b) in enumerate(zip(vlm7, pt)):
    ax.text(i - w / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=9.5, fontweight="bold" if a > b else "normal")
    ax.text(i + w / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=9.5, fontweight="bold" if b > a else "normal")
ax.axvline(1.5, color="#c9c8c2", lw=1)
ax.text(0.5, 1.08, "VLM이 맡음", ha="center", color=S1, fontweight="bold")
ax.text(2.5, 1.08, "규칙이 맡음", ha="center", color=INK2, fontweight="bold")
ax.set_xticks(x, fields); ax.set_ylim(0, 1.15); ax.set_ylabel("칸별 정확도")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=9.5)
ax.set_title("칸마다 더 잘하는 쪽에 맡긴다 → 하이브리드", loc="left", fontsize=12, color=INK)
save(fig, "fig3_fields.png")

# ---------- Fig 4: accuracy vs latency ----------
fig, ax = plt.subplots(figsize=(7.5, 4.6)); clean(ax, "both")
cols = [S2, S1, S1, S1, S1]
for m, a, l, c in zip(models, hyb, lat, cols):
    ax.scatter(l, a, s=90, color=c, edgecolor="white", lw=2, zorder=3)
    off = {"Qwen2.5-VL-7B": (8, 6), "Qwen2.5-VL-3B": (8, 6), "InternVL3.5-4B": (8, -14),
           "Qwen3-VL-8B": (-10, 9), "InternVL3.5-8B": (-10, -16)}[m]
    ax.annotate(m, (l, a), xytext=off, textcoords="offset points", fontsize=9.5,
                ha="left" if off[0] > 0 else "right", color=INK)
ax.set_xlim(0.3, 1.2); ax.set_ylim(0.15, 0.45)
ax.set_xlabel("호출 지연 중앙값 (s, A5000, F8)"); ax.set_ylabel("하이브리드 완전 정답률")
ax.text(0.32, 0.43, "← 빠르고 정확할수록 왼쪽 위", fontsize=9, color=INK2)
ax.set_title("정확도 vs 지연 — 7B(정확) · 3B(속도 2배)", loc="left", fontsize=12, color=INK)
save(fig, "fig4_acc_latency.png")

# ---------- Fig 5: VIABench paper vs ours ----------
pm = ["InternVL3.5-8B", "InternVL3.5-4B", "Qwen2.5-VL-3B", "Qwen2.5-VL-7B"]
paper = [43.9, 41.4, 36.8, 34.9]
ours = {"InternVL3.5-8B": 0.22, "InternVL3.5-4B": 0.26, "Qwen2.5-VL-3B": 0.26, "Qwen2.5-VL-7B": 0.39}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
for a in (a1, a2):
    clean(a, "x")
y = np.arange(4)[::-1]
a1.barh(y, paper, color=GRAY, height=0.55); a1.set_yticks(y, pm); a1.set_xlim(0, 52)
for yy, v in zip(y, paper): a1.text(v + 0.8, yy, f"{v}", va="center", fontsize=9.5)
a1.set_title("VIABench Table 4 (자유 문장 · GPT 판정)", loc="left", fontsize=11, color=INK)
a1.set_xlabel("평균 점수 (MPS)")
ov = [ours[m] for m in pm]
a2.barh(y, ov, color=[S2 if m == "Qwen2.5-VL-7B" else S1 for m in pm], height=0.55)
a2.set_yticks(y, pm); a2.set_xlim(0, 0.48)
for yy, v in zip(y, ov): a2.text(v + 0.008, yy, f"{v:.2f}", va="center", fontsize=9.5)
a2.set_title("우리 조건 (박스 지정 · 칸 채점 · 하이브리드)", loc="left", fontsize=11, color=INK)
a2.set_xlabel("완전 정답률")
fig.suptitle("논문 1위가 우리 조건 1위는 아니다 → 직접 비교 실험이 필요했던 이유", x=0.01, ha="left", fontsize=12.5, color=INK)
fig.tight_layout()
save(fig, "fig5_paper_vs_ours.png")

# ---------- Fig 6: runtime timeline (c09) ----------
fig, ax = plt.subplots(figsize=(10, 2.9)); clean(ax, "x"); ax.spines["left"].set_visible(False)
rows = [("트리거 이벤트", 0, 0.03, S1), ("즉시 경고 재생", 0.11, 0.9, S2),
        ("VLM 호출 (1 s 흉내)", 0.05, 1.0, "#eda100"), ("TTS 합성", 1.05, 0.25, "#eda100"),
        ("설명 재생", 1.38, 1.6, S2)]
for i, (n, s, d, c) in enumerate(rows[::-1]):
    ax.barh(i, d, left=s, height=0.5, color=c)
    ax.text(-0.05, i, n, ha="right", va="center", fontsize=9.5, color=INK)
ax.axvline(0.11, color=S2, ls=":", lw=1); ax.axvline(1.38, color=S2, ls=":", lw=1)
ax.text(0.13, 4.45, "경고 0.11 s", fontsize=9, color=S2, fontweight="bold")
ax.text(1.40, 4.45, "설명 1.38 s", fontsize=9, color=S2, fontweight="bold")
ax.set_yticks([]); ax.set_xlim(-0.02, 3.1); ax.set_ylim(-0.5, 4.8)
ax.set_xlabel("트리거 시각 기준 경과 시간 (s) — 점선 = c09 실측 중앙값(노트북), 막대 길이는 개념도")
ax.set_title("경고는 VLM을 기다리지 않는다", loc="left", fontsize=12, color=INK)
save(fig, "fig6_timeline.png")

# ---------- Fig 7: experiment design flow ----------
fig, ax = plt.subplots(figsize=(13, 3.6)); ax.set_xlim(0, 13); ax.set_ylim(0, 3.6); ax.axis("off")
steps = [("1단계\nVLM 선택", "정답 시각에 호출\n→ 트리거 오류 제거\nS-M 모델 · S-F 프레임", "완료 (23개)\nVIABench 확대 예정", S1),
         ("2단계\n스트리밍 긴급 경고", "실제 속도 재생\n경고가 VLM을\n기다리지 않나", "뼈대 완료\n(VLM 흉내)", S1),
         ("3단계\nTTS 선택", "같은 55문장 합성\n받아쓰기 CER\n· 합성 시간", "노트북 진행 중\n최종은 A5000", "#eda100"),
         ("4단계\n출력 정합성", "JSON → 문장 → TTS\n→ 받아쓰기\n→ 칸 복원율", "예정", GRAY)]
for i, (t, d, s, c) in enumerate(steps):
    x0 = 0.2 + i * 3.2
    box(ax, x0, 2.1, 2.7, 1.2, t, "#ffffff", c, 10.5, True)
    ax.text(x0 + 1.35, 1.35, d, ha="center", va="center", fontsize=9, color=INK2, linespacing=1.4)
    ax.text(x0 + 1.35, 0.3, s, ha="center", va="center", fontsize=8.8, color=c if c != GRAY else INK2,
            fontweight="bold", linespacing=1.3)
    if i < 3:
        arrow(ax, x0 + 2.75, 2.7, x0 + 3.15, 2.7)
save(fig, "fig7_eval_stages.png")
print("ok", sorted(p.name for p in OUT.iterdir()))
