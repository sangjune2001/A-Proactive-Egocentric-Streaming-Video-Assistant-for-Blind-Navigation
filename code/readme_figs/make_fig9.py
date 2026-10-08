"""README 그림 Fig. 9: 최종 VLM → TTS 파이프라인 (10/9, A5000 실측) → assets/fig9_vlm_tts_pipeline.png

숫자 출처: results/runtime/final_1009/*/log.jsonl (c09 · c06 · c04, 이벤트 22개)
    ~/egoenv/Scripts/python.exe code/readme_figs/make_fig9.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets"
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
BLUE, ORANGE, YEL, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#eda100", "#a3a29c", "#0b0b0b", "#52514e"


def box(ax, x, y, w, h, title, body, ec, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=11, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.62, body, ha="center", va="top", fontsize=8.8, color=INK2, linespacing=1.45)


def arrow(ax, x0, y0, x1, y1, label=None, color=INK2, dy=0.12):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, label, ha="center", fontsize=8.3, color=color)


fig, ax = plt.subplots(figsize=(15, 7.4))
ax.set_xlim(0, 15)
ax.set_ylim(0, 7.4)
ax.axis("off")
ax.text(0.15, 7.15, "최종 VLM → TTS 파이프라인 (10/9, A5000 실측 · 시연 3편 이벤트 22개 중앙값)", fontsize=13.5,
        fontweight="bold", color=INK)

W, H, Y = 2.25, 2.55, 3.55
xs = [0.15, 2.65, 5.15, 7.65, 10.15, 12.65]
items = [
    ("① 트리거 이벤트", "임태규 규칙 (지금 B0)\n\n물체 번호 · 클래스\n상자 · 시나리오 · 시각\n\n= 위험 판단 · 언제", GRAY, "#f3f3f1"),
    ("② VLM 입력 만들기", "트리거 시각까지\n1 s 간격 8장\n대상에 빨간 박스\n긴 변 448 px\n입력 약 1,488 토큰\n영어 지시문 (h2)", BLUE, "#e3eefb"),
    ("③ VLM", "Qwen2.5-VL-7B\nvLLM · bf16 · GPU 0.80\nJSON 2칸 강제\n{대상, 움직임}\n출력 약 18 토큰\n1.00 s", BLUE, "#e3eefb"),
    ("④ 하이브리드", "대상 · 움직임 = VLM\n방향 = 상자 위치\n(화면 1/3씩)\n행동 = 규칙표\n위험 = 트리거\n< 1 ms", YEL, "#fff4dc"),
    ("⑤ 한국어 문장", "템플릿\n\"정면에 킥보드가\n다가오고 있어요.\n멈추세요.\"\n< 1 ms", YEL, "#fff4dc"),
    ("⑥ TTS", "Supertonic-3 M1\n(서버 CPU, ONNX)\n한국어 lang=ko\n2.28 s", ORANGE, "#fbe4da"),
]
for x, (t, b, ec, fc) in zip(xs, items):
    box(ax, x, Y, W, H, t, b, ec, fc)
for i in range(5):
    arrow(ax, xs[i] + W + 0.02, Y + H / 2, xs[i + 1] - 0.02, Y + H / 2)

# 즉시 경고 갈래
ax.add_patch(FancyBboxPatch((2.65, 6.3), 5.6, 0.62, boxstyle="round,pad=0.02,rounding_size=0.12", fc="#fbe4da", ec=ORANGE, lw=1.6))
ax.text(5.45, 6.61, "즉시 경고 \"왼쪽 오토바이\" · 미리 합성 32개 · VLM 안 기다림 · 0.25 s", ha="center", va="center",
        fontsize=9.6, color=INK, fontweight="bold")
ax.annotate("", (2.63, 6.61), (1.27, Y + H + 0.03), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.6,
            connectionstyle="arc3,rad=-0.25"))

# 재생 큐
ax.add_patch(FancyBboxPatch((8.6, 6.3), 6.2, 0.62, boxstyle="round,pad=0.02,rounding_size=0.12", fc="#ffffff", ec=INK2, lw=1.6))
ax.text(11.7, 6.61, "⑦ 재생 큐 → 스피커: 경고가 설명을 끊음 · 새 경고 뒤엔 이전 설명 버림", ha="center", va="center",
        fontsize=9.6, color=INK, fontweight="bold")
ax.annotate("", (8.27, 6.61), (8.58, 6.61), arrowprops=dict(arrowstyle="<|-", color=ORANGE, lw=1.6))
ax.annotate("", (13.77, 6.28), (13.77, Y + H + 0.03), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.6))

# 설명 지연 분해 막대
ax.text(0.15, 2.75, "설명 지연 = 트리거 → 설명 소리 시작, 중앙 3.7 s", fontsize=11, fontweight="bold", color=INK)
seg = [("VLM 1.00 s\n(입력 처리 0.54 + 글자 생성 0.43)", 1.00, BLUE), ("TTS 합성 2.28 s (CPU)", 2.28, ORANGE),
       ("재생 대기 약 0.4 s\n(경고가 끝나길)", 0.42, GRAY)]
x0, scale = 0.15, 2.45
for name, v, c in seg:
    ax.add_patch(plt.Rectangle((x0, 1.75), v * scale, 0.6, fc=c, ec="white", lw=2))
    ax.text(x0 + v * scale / 2, 1.42, name, ha="center", va="top", fontsize=8.8, color=INK2, linespacing=1.3)
    x0 += v * scale
ax.text(x0 + 0.15, 2.05, "3.7 s", va="center", fontsize=10.5, fontweight="bold", color=INK)

# 오른쪽 아래: 핵심 수치
notes = ["• 경고 21/22 재생, 지연 중앙 0.25 s", "• 설명 20/22 재생 시작 → 13 끝까지 (7 끊김)",
         "• VLM 형식 오류 0 · 출력 2칸이라 호출 절반 (G3)", "• 인식 YOLO+ByteTrack 프레임당 20–24 ms (15 fps 입력)",
         "• 남은 병목: TTS CPU 합성 2.3 s · 트리거 정답 1/6"]
for k, n in enumerate(notes):
    ax.text(10.4, 2.6 - k * 0.42, n, fontsize=9.6, color=INK)
OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "fig9_vlm_tts_pipeline.png", dpi=200, bbox_inches="tight", facecolor="white")
print("ok")
