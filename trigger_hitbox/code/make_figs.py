"""버전별 표 이미지: 채점표(v5_10 · v5_11) · GPU/CPU 속도표(v5_10 · v5_11) · v5_9–v5_11 비교표.
값은 서버 실행 출력과 eval JSON에서 옮김 (태규 B0 제외)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle

font_manager.fontManager.addfont("C:/Windows/Fonts/malgun.ttf")
font_manager.fontManager.addfont("C:/Windows/Fonts/malgunbd.ttf")
plt.rcParams["font.family"] = "Malgun Gothic"
INK, MUTED, LINE, BG, PANEL, HL = "#1d2420", "#5d6862", "#dfe3df", "#f6f7f5", "#ffffff", "#eef6f2"
GPU, CPU, NEU = "#13866f", "#c0781f", "#8fa39b"
OUT = "C:/Users/windo/Documents/trigger_proto/out/figs"
os.makedirs(OUT, exist_ok=True)

# 채점 (여유 기준): (TP/FP/FN, P, R, F1, 분당 오경보)
SCORE = {
    "v5_9": {"test": {"Bd": ("10 / 54 / 2", .156, .833, .263, 8.19), "B": ("11 / 96 / 1", .103, .917, .185, 14.55)},
             "train": {"Bd": ("7 / 29 / 9", .194, .438, .269, 3.02), "B": ("7 / 65 / 9", .097, .438, .159, 6.77)}},
    "v5_10": {"test": {"Bd": ("9 / 70 / 3", .114, .750, .198, 10.61), "B": ("10 / 124 / 2", .075, .833, .137, 18.80)},
              "train": {"Bd": ("9 / 39 / 7", .188, .562, .281, 4.06), "B": ("10 / 81 / 6", .110, .625, .187, 8.44)}},
    "v5_11": {"test": {"Bd": ("9 / 65 / 3", .122, .750, .209, 9.85), "B": ("11 / 113 / 1", .089, .917, .162, 17.13)},
              "train": {"Bd": ("9 / 33 / 7", .214, .562, .310, 3.44), "B": ("9 / 73 / 7", .110, .562, .184, 7.60)},
              "train_loose": {"Bd": ("10 / 31 / 6", .244, .625, .351, 3.23), "B": ("10 / 69 / 6", .127, .625, .211, 7.19)}},
}
TRIG = {"v5_9": (7.3, "측정 · test 1–3 평균"), "v5_10": (8.3, "측정 · test_2"), "v5_11": (8.3, "v5_10과 같은 로직 · 추정")}
LEAD = {"v5_9": (-0.83, -1.40), "v5_10": (-0.53, 0.00), "v5_11": (-0.81, -1.13)}   # 위험만, 라벨 시작 − 첫 트리거 (s)
BASE_GPU = 3.0 + 13.4 + 43.0 + 1.5 + 10.8     # 트리거 제외 (GPU 단독)
BASE_CPU = 3.0 + 39.7 + 43.0 + 1.6 + 10.8     # 트리거 제외 (나눠 쓰기)


def fr(v):
    return f"{v + 1e-9:.1f}"


def header(fig, title, sub):
    h = fig.get_figheight()             # 그림 높이와 상관없이 위에서 같은 거리 (인치)
    fig.text(0.04, 1 - 0.28 / h, title, fontsize=19, fontweight="bold", color=INK, va="top")
    fig.text(0.04, 1 - 0.72 / h, sub, fontsize=10.5, color=MUTED, va="top")


def panel(fig, rect, h):
    ax = fig.add_axes(rect); ax.set_xlim(0, 100); ax.set_ylim(0, h); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 100, h, boxstyle="round,pad=0,rounding_size=1.2", fc=PANEL, ec=LINE, lw=1))
    return ax


def hbar(ax, x, y, v, vmax, w, color):
    ax.add_patch(Rectangle((x, y - 0.13), w, 0.26, fc=LINE, ec="none"))
    ax.add_patch(Rectangle((x, y - 0.13), w * min(v, vmax) / vmax, 0.26, fc=color, ec="none"))


# ---------------- 채점표 ----------------
def score_fig(ver):
    S = SCORE[ver]
    rows = [("테스트 3편\n(라벨 12)", S["test"]), ("학습 7편\n(라벨 16)", S["train"])]
    if "train_loose" in S:
        rows.append(("학습 7편\n멈춘 탈것=장애물 인정", S["train_loose"]))
    lines = []
    for name, d in rows:
        lines.append((name, "위험만", *d["Bd"], True))
        lines.append(("", "주의 + 위험", *d["B"], False))
    H = len(lines) + 1.4
    fig = plt.figure(figsize=(12.5, 2.0 + 0.78 * len(lines)), dpi=170, facecolor=BG)
    header(fig, f"히트박스 트리거 채점 — {ver} 기준",
           "여유 기준: 라벨 구간 2초 전 ~ 0.5초 후 인정 · Precision ↑ Recall ↑ F1 ↑ 높을수록 좋음 · 분당 오경보 ↓ 낮을수록 좋음")
    hh = fig.get_figheight()
    ax = panel(fig, [0.04, 0.45 / hh, 0.92, 1 - 0.45 / hh - 1.15 / hh], H)
    cols = [(1.5, "데이터"), (19, "알림"), (29, "TP / FP / FN"), (41, "Precision ↑"), (56, "Recall ↑"), (71, "F1 ↑"), (85, "분당 오경보 ↓")]
    yt = len(lines) + 0.85
    for x, c in cols:
        ax.text(x, yt, c, fontsize=11, color=MUTED, va="center", fontweight="bold")
    ax.plot([0.8, 99.2], [yt - 0.4] * 2, color=LINE, lw=1.2)
    for i, (d, a, n, p, r, f1, fpm, hl) in enumerate(lines):
        y = len(lines) - i - 0.05
        if hl:
            ax.add_patch(Rectangle((0.8, y - 0.48), 98.4, 0.96, fc=HL, ec="none"))
        if d:
            ax.text(1.5, y - 0.5, d, fontsize=11.5, color=INK, va="center", fontweight="bold", linespacing=1.3)
            if i:
                ax.plot([0.8, 99.2], [y + 0.5] * 2, color=LINE, lw=1)
        wt = "bold" if hl else "normal"
        ax.text(19, y, a, fontsize=11.5, color=INK, va="center", fontweight=wt)
        ax.text(29, y, n, fontsize=11.5, color=INK, va="center")
        for x, v in ((41, p), (56, r), (71, f1)):
            ax.text(x, y + 0.02, f"{v + 1e-9:.2f}", fontsize=12.5, color=INK, va="center", fontweight=wt)
            hbar(ax, x + 5.0, y, v, 1.0, 7.5, GPU)
        ax.text(85, y + 0.02, fr(fpm), fontsize=12.5, color=INK, va="center", fontweight=wt)
        hbar(ax, 90.5, y, fpm, 20, 7.8, CPU)
    fig.text(0.04, 0.15 / hh, f"히트박스 트리거 {ver} · 막대: Precision·Recall·F1은 0–1, 분당 오경보는 0–20회 기준 · 강조 행 = 위험만 음성으로 알릴 때",
             fontsize=9, color=MUTED)
    fig.savefig(f"{OUT}/{ver}_score.png", facecolor=BG)
    plt.close(fig)


# ---------------- 속도표 ----------------
def speed_fig(ver):
    trig, note = TRIG[ver]
    frame = [("프레임 읽기", ("CPU", 3.0), ("CPU", 3.0)),
             ("YOLO 검출 + 추적", ("GPU", 13.4), ("CPU×8", 39.7)),
             ("배경 광류 (걸음 속도 · 흔들림)", ("CPU", 43.0), ("CPU", 43.0)),
             (f"트리거 판단 ({note})", ("CPU", trig), ("CPU", trig)),
             ("신호 분류 (3프레임마다)", ("GPU", 1.5), ("CPU", 1.6)),
             ("바닥 분할 (5프레임마다)", ("GPU", 10.8), ("GPU", 10.8))]
    tg, tc = BASE_GPU + trig, BASE_CPU + trig
    event = [("즉시 경고 (미리 합성한 조각 · 지금 방식)", "0.02 s", "0.05 s"),
             ("VLM 설명 문장 (Qwen2.5-VL-7B)", "0.93 s", "0.85 s"),
             ("TTS 합성 Supertonic (서버 CPU)", "1.34 s", "1.36 s"),
             ("설명까지 합계 (VLM + Supertonic)", "2.27 s", "2.21 s"),
             ("설명까지 합계 (VLM + Heami, 노트북)", "0.95 s", "0.87 s")]
    fig = plt.figure(figsize=(12.5, 9.2), dpi=170, facecolor=BG)
    header(fig, f"GPU 단독 vs CPU+GPU 나눠 쓰기 — 모듈별 시간 ({ver} 기준)",
           "RunYourAI RTX A5000 · 중간값 · 나눠 쓰기 = VLM·바닥 분할만 GPU, YOLO·트리거·TTS는 CPU 8스레드(노트북급)")
    ax = panel(fig, [0.04, 0.38, 0.92, 0.5], len(frame) + 2.4)
    yt = len(frame) + 1.8
    ax.text(1.5, yt, "A. 프레임마다 (ms / 프레임)", fontsize=12, color=INK, fontweight="bold", va="center")
    ax.text(36, yt, "GPU 단독", fontsize=11, color=GPU, fontweight="bold", va="center")
    ax.text(68, yt, "CPU+GPU 나눠 쓰기", fontsize=11, color=CPU, fontweight="bold", va="center")
    ax.plot([0.8, 99.2], [yt - 0.5] * 2, color=LINE, lw=1.2)
    sc = 16 / 45.0

    def cell(x, y, where, ms, color):
        ax.text(x, y, where, fontsize=9.5, color=MUTED, va="center")
        bx = x + 7.5
        ax.add_patch(Rectangle((bx, y - 0.17), ms * sc, 0.34, fc=color, ec="none", alpha=0.9))
        ax.text(bx + ms * sc + 0.8, y, fr(ms), fontsize=11.5, color=INK, va="center", fontweight="bold")
    for i, (name, g, c) in enumerate(frame):
        y = len(frame) - i + 0.6
        ax.text(1.5, y, name, fontsize=11 if len(name) > 22 else 11.5, color=INK, va="center")
        cell(36, y, g[0], g[1], GPU if g[0] == "GPU" else NEU)
        cell(68, y, c[0], c[1], CPU if c[0] != "GPU" else NEU)
    ax.plot([0.8, 99.2], [0.95] * 2, color=LINE, lw=1.2)
    ax.add_patch(Rectangle((0.8, 0.05), 98.4, 0.85, fc=HL, ec="none"))
    ax.text(1.5, 0.48, "합계  (처리 가능 속도)", fontsize=12.5, color=INK, fontweight="bold", va="center")
    ax.text(36, 0.48, f"{fr(tg)} ms  (약 {1000 / tg:.1f} fps)", fontsize=13, color=GPU, fontweight="bold", va="center")
    ax.text(68, 0.48, f"{fr(tc)} ms  (약 {1000 / tc:.1f} fps)", fontsize=13, color=CPU, fontweight="bold", va="center")
    ax2 = panel(fig, [0.04, 0.06, 0.92, 0.29], len(event) + 1.4)
    yt = len(event) + 0.85
    ax2.text(1.5, yt, "B. 경고 한 번 (트리거 → 소리)", fontsize=12, color=INK, fontweight="bold", va="center")
    ax2.text(60, yt, "GPU 단독", fontsize=11, color=GPU, fontweight="bold", va="center")
    ax2.text(80, yt, "나눠 쓰기", fontsize=11, color=CPU, fontweight="bold", va="center")
    ax2.plot([0.8, 99.2], [yt - 0.45] * 2, color=LINE, lw=1.2)
    for i, (name, g, c) in enumerate(event):
        y = len(event) - i - 0.1
        hl = name.startswith("설명까지 합계 (VLM + Supertonic")
        if hl:
            ax2.add_patch(Rectangle((0.8, y - 0.45), 98.4, 0.9, fc=HL, ec="none"))
        wt = "bold" if hl or i == 0 else "normal"
        ax2.text(1.5, y, name, fontsize=11.5, color=INK, va="center", fontweight="bold" if hl else "normal")
        ax2.text(60, y, g, fontsize=12, color=INK, va="center", fontweight=wt)
        ax2.text(80, y, c, fontsize=12, color=INK, va="center", fontweight=wt)
    fig.text(0.04, 0.022, f"나눠 써도 즉시 경고는 +0.03초뿐 · YOLO를 CPU로 옮기면 프레임당 +26 ms · 가장 큰 비용은 원래 CPU인 배경 광류 · 모듈을 순서대로 돌린 합계",
             fontsize=9, color=MUTED)
    fig.savefig(f"{OUT}/{ver}_speed.png", facecolor=BG)
    plt.close(fig)


# ---------------- v5_9 – v5_11 비교 (기준 설정 · 성능 · 처리 속도) ----------------
def compare_fig():
    vers = ["v5_9", "v5_10", "v5_11"]
    secs = [
        ("기준 설정", [
            ("영역 길이 (걸음 몇 초 앞)", ["4초", "5.5초", "4초"]),
            ("위험: 내 몸 폭 안 도달", ["3초", "4초", "3초"]),
            ("위험: 움직이는 물체 진입", ["2.5초", "3.5초", "2.5초"]),
            ('"멈추세요" 거리', ["2.5 m", "3.5 m", "2.5 m"]),
            ("판정 시작 대기", ["0.4초", "0.25초", "0.25초"]),
            ("켜짐 지연", ["0.2초", "0.1초", "0.1초"]),
            ("다가옴 확정", ["0.3초", "0.2초", "0.2초"]),
            ("경고 묶음", ["1.0초", "0.5초", "0.5초"]),
            ("고개 돌림 처리", ["없음", "0.7초 · 위험까지 보류", "0.4초 · 주의만 보류"]),
        ]),
        ("성능 (여유 기준 · 위험만 알림)", [
            ("테스트 Precision ↑", [f"{SCORE[v]['test']['Bd'][1]:.2f}" for v in vers]),
            ("테스트 Recall ↑", [f"{SCORE[v]['test']['Bd'][2]:.2f}" for v in vers]),
            ("테스트 분당 오경보 ↓", [fr(SCORE[v]['test']['Bd'][4]) for v in vers]),
            ("학습 Precision ↑", [f"{SCORE[v]['train']['Bd'][1]:.2f}" for v in vers]),
            ("학습 Recall ↑", [f"{SCORE[v]['train']['Bd'][2]:.2f}" for v in vers]),
            ("학습 분당 오경보 ↓", [fr(SCORE[v]['train']['Bd'][4]) for v in vers]),
            ("경고 시점 · 테스트 (라벨 시작 대비)", [f"{(-LEAD[v][0]) + 0.0:+.2f} s".replace("-0.00", "0.00").replace("+0.00", "0.00") for v in vers]),
            ("경고 시점 · 학습 (라벨 시작 대비)", [f"{(-LEAD[v][1]) + 0.0:+.2f} s".replace("-0.00", "0.00").replace("+0.00", "0.00") for v in vers]),
        ]),
        ("처리 속도 (RunYourAI RTX A5000, 중간값)", [
            ("트리거 판단 (ms / 프레임)", [fr(TRIG[v][0]) + (" *" if v == "v5_11" else "") for v in vers]),
            ("프레임 합계 · GPU 단독", [f"{fr(BASE_GPU + TRIG[v][0])} ms ({1000 / (BASE_GPU + TRIG[v][0]):.1f} fps)" for v in vers]),
            ("프레임 합계 · CPU+GPU 나눠 쓰기", [f"{fr(BASE_CPU + TRIG[v][0])} ms ({1000 / (BASE_CPU + TRIG[v][0]):.1f} fps)" for v in vers]),
            ("즉시 경고 지연 (트리거 → 소리)", ["0.02 s", "0.02 s", "0.02 s *"]),
        ]),
    ]
    # 성능 줄에서 가장 좋은 값 굵게
    better = {"Precision ↑": max, "Recall ↑": max, "오경보 ↓": min}
    n = sum(len(r) for _, r in secs) + len(secs)
    fig = plt.figure(figsize=(13, 0.43 * n + 2.2), dpi=170, facecolor=BG)
    header(fig, "버전 비교 — v5_9 / v5_10 / v5_11",
           "v5_10 = 경고 1–2초 앞당김 (영상으로 보니 너무 일찍 울림) · v5_11 = 거리·시간 기준은 v5_9로 되돌리고 판정 대기만 줄여 0.3–0.5초 앞당김")
    H = n + 1.2
    ax = panel(fig, [0.04, 0.05, 0.92, 1 - 0.05 - (1.5 / (0.43 * n + 2.2))], H)
    X = (40, 61, 82)
    yt = H - 0.6
    ax.text(1.5, yt, "항목", fontsize=11.5, color=MUTED, fontweight="bold", va="center")
    for x, v in zip(X, vers):
        ax.text(x, yt, v + ("  (이번)" if v == "v5_11" else ""), fontsize=12, color=GPU if v == "v5_11" else MUTED,
                fontweight="bold", va="center")
    ax.add_patch(Rectangle((79.5, 0.2), 19.7, H - 0.6, fc=HL, ec="none"))      # v5_11 열 강조
    y = yt - 1.0
    for title, rows in secs:
        ax.plot([0.8, 99.2], [y + 0.5] * 2, color=LINE, lw=1.2)
        ax.text(1.5, y, title, fontsize=12, color=INK, fontweight="bold", va="center")
        y -= 1.0
        for name, vals in rows:
            ax.text(3.0, y, name, fontsize=11.2, color=INK, va="center")
            pick = next((f for k_, f in better.items() if name.endswith(k_)), None)
            best = None
            if pick:
                nums = [float(v.split()[0]) for v in vals]
                best = pick(nums)
            for x, v in zip(X, vals):
                bold = best is not None and float(v.split()[0]) == best
                ax.text(x, y, v, fontsize=11.5, color=INK, va="center", fontweight="bold" if bold else "normal")
            y -= 1.0
    fig.text(0.04, 0.015, "굵게 = 세 버전 중 가장 좋은 값 · 경고 시점: + 는 라벨 시작보다 늦게, - 는 먼저 (중간값) · "
             "* v5_11은 서버 반납으로 직접 못 재서 같은 로직인 v5_10 값 · 프레임 합계 = 모듈을 순서대로 돌린 값",
             fontsize=8.8, color=MUTED)
    fig.savefig(f"{OUT}/compare_v5_9_v5_10_v5_11.png", facecolor=BG)
    plt.close(fig)


for v in ("v5_10", "v5_11"):
    score_fig(v)
    speed_fig(v)
compare_fig()
print("saved")
