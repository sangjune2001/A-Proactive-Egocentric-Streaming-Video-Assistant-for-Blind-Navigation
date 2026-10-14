"""히트박스 화면 설명도 2장: ① 화면 요소(막대·타원·점선·화살표 등) 역할·이유·규칙 ② 경고 판단 흐름. v5_11 기준."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, Circle

font_manager.fontManager.addfont("C:/Windows/Fonts/malgun.ttf")
font_manager.fontManager.addfont("C:/Windows/Fonts/malgunbd.ttf")
plt.rcParams["font.family"] = "Malgun Gothic"
INK, MUTED, LINE, BG, PANEL = "#1d2420", "#5d6862", "#dfe3df", "#f6f7f5", "#ffffff"
ZONE, EDGE, SAFE, WARN, DANG, B0 = "#2b9fd6", "#55b85a", "#3aa35b", "#d9a400", "#d23c2c", "#d15fb0"
GROUND, SKY = "#c9c4b8", "#dfe9ef"
OUT = "C:/Users/windo/Documents/trigger_proto/out/figs"
os.makedirs(OUT, exist_ok=True)

VP = (50.0, 84.0)            # 소실점 (걷는 방향 · 지평선)


def persp(x_m, z_m, f=55.0, h=1.77):
    """바닥 (옆 x m, 앞 z m) → 그림 좌표 (카메라 높이 h, 소실점 VP)."""
    return VP[0] + f * x_m / z_m * 1.15, VP[1] - f * h / z_m * 0.62


def poly_xz(pts):
    return [persp(x, z) for x, z in pts]


def badge(ax, x, y, n, color=INK):
    ax.add_patch(Circle((x, y), 1.9, fc=color, ec="white", lw=1.5, zorder=20))
    ax.text(x, y - 0.1, str(n), fontsize=10, color="white", ha="center", va="center", fontweight="bold", zorder=21)


# ================= ① 화면 요소 설명 =================
fig = plt.figure(figsize=(17, 11.5), dpi=150, facecolor=BG)
fig.text(0.03, 0.965, "히트박스 화면 읽는 법 — 무엇이 · 왜 · 어떤 규칙으로 (v5_11)", fontsize=20, fontweight="bold", color=INK, va="top")
fig.text(0.03, 0.928, "오른쪽 '판단 과정' 화면에 그려지는 요소를 번호로 표시했다. 색: 초록 = 안전 · 노랑 = 주의 · 빨강 = 위험",
         fontsize=11, color=MUTED, va="top")

ax = fig.add_axes([0.03, 0.05, 0.40, 0.85]); ax.set_xlim(0, 100); ax.set_ylim(0, 140); ax.axis("off")
ax.add_patch(Rectangle((0, VP[1]), 100, 140 - VP[1], fc=SKY, ec="none"))
ax.add_patch(Rectangle((0, 0), 100, VP[1], fc=GROUND, ec="none"))
# 인도 (가운데 밝은 바닥) · 인도 끝선 ⑨
side = poly_xz([(-2.6, 1.2), (2.4, 1.2), (2.4, 40), (-2.6, 40)])
ax.add_patch(Polygon(side, fc="#e7e2d6", ec="none"))
for x in (-2.6, 2.4):
    a, b = persp(x, 0.8), persp(x, 40)
    ax.plot([a[0], b[0]], [a[1], b[1]], color=EDGE, lw=3)
# 차도 (오른쪽) + 주차된 차
ax.add_patch(Polygon(poly_xz([(2.4, 1.2), (12, 1.2), (12, 40), (2.4, 40)]), fc="#b4b0a6", ec="none"))
cx, cy = persp(4.6, 9.0)
ax.add_patch(FancyBboxPatch((cx - 3, cy), 8, 4.5, boxstyle="round,pad=0.2", fc="#6c7a85", ec="none"))
ax.text(cx + 1, cy + 6.5, "인도 끝선 너머 차 = 무시", fontsize=8.5, color=MUTED, ha="center")
# ① 내 진행 영역
zone = poly_xz([(-1.3, 1.2), (1.3, 1.2), (1.3, 8.8), (-1.3, 8.8)])
ax.add_patch(Polygon(zone, fc=ZONE, alpha=0.18, ec="none"))
ax.add_patch(Polygon(zone, fill=False, ec=ZONE, lw=2.5, ls=(0, (6, 4))))
# ② 정면 막대 · ③ 곧 충돌 구역
ax.add_patch(Polygon(poly_xz([(-0.6, 1.2), (0.6, 1.2), (0.6, 2.9), (-0.6, 2.9)]), fc=DANG, alpha=0.28, ec="none"))
ax.add_patch(Polygon(poly_xz([(-0.25, 1.2), (0.25, 1.2), (0.25, 8.8), (-0.25, 8.8)]), fc="white", alpha=0.75,
                     ec="white", lw=1.5, ls=(0, (4, 3))))
# ⑧ 소실점 십자
ax.plot([VP[0] - 2.5, VP[0] + 2.5], [VP[1], VP[1]], color="#19b5c9", lw=2.5)
ax.plot([VP[0], VP[0]], [VP[1] - 2.5, VP[1] + 2.5], color="#19b5c9", lw=2.5)
# 물체 1: 정면 볼라드 (위험)
bx, by = persp(-0.45, 2.4)
ax.add_patch(Ellipse((bx, by), 30, 9, fc=DANG, alpha=0.25, ec=DANG, lw=2, ls=(0, (5, 3))))       # ④
ax.add_patch(Ellipse((bx + 3.5, by + 3.0), 30, 9, fill=False, ec="white", lw=1.8, ls=(0, (3, 3))))  # ⑤
ax.plot([bx, bx + 3.5], [by, by + 3.0], color="white", lw=1.6)
ax.add_patch(Rectangle((bx - 2.2, by), 4.4, 20, fc="#2d3134", ec="none"))
ax.add_patch(Rectangle((bx - 3.2, by - 0.8), 6.4, 21.6, fill=False, ec=DANG, lw=3))               # ⑪
ax.text(bx - 3.2, by + 22.5, "obstacle#801 static 3.2m", fontsize=8, color=DANG, fontweight="bold")
# 물체 2: 다가오는 자전거 (주의) + 경로 · 화살표
kx, ky = persp(1.5, 4.0)
ax.add_patch(Ellipse((kx, ky - 1.0), 22, 6, fc=WARN, alpha=0.25, ec=WARN, lw=2, ls=(0, (5, 3))))
ax.add_patch(Rectangle((kx - 2.6, ky), 5.2, 9, fill=False, ec=WARN, lw=2.5))
tr = [(kx + 9, ky + 9), (kx + 6, ky + 6), (kx + 3, ky + 3), (kx + 0.5, ky + 0.3)]
for i in range(len(tr) - 1):
    ax.plot([tr[i][0], tr[i + 1][0]], [tr[i][1], tr[i + 1][1]], color="#7a4fd6", lw=2 + 3 * i, solid_capstyle="round", alpha=0.85)  # ⑥
ax.add_patch(FancyArrowPatch((kx, ky), (kx - 6, ky - 6), arrowstyle="-|>", mutation_scale=22, color="#7a4fd6", lw=3))           # ⑦
# 물체 3: 멀어지는 사람 (안전, 위쪽 화살표)
px, py = persp(-2.0, 7.5)
ax.add_patch(Rectangle((px - 1.5, py), 3, 9, fill=False, ec=SAFE, lw=2))
ax.add_patch(FancyArrowPatch((px, py + 9.5), (px + 0.8, py + 15), arrowstyle="-|>", mutation_scale=16, color=SAFE, lw=2))
ax.text(px - 14, py + 14, "위쪽 화살표 = 멀어짐", fontsize=8.5, color=MUTED)
# 신호등 ⑫
ax.add_patch(Rectangle((74, 98), 3, 8, fc="#2d3134", ec="none"))
ax.add_patch(Rectangle((73.3, 97.3), 4.4, 9.4, fill=False, ec="#2bbf4f", lw=2.5))
ax.add_patch(Circle((75.5, 100), 1.0, fc="#2bbf4f"))
ax.text(79, 101.5, "신호 초록", fontsize=9, color="#2bbf4f", fontweight="bold")
# ⑮ 왼쪽 위 글자
ax.text(2, 134, "B: hitbox (cam 1.77m)  t=43.4s", fontsize=9.5, color=INK, fontweight="bold")
ax.text(2, 129.5, "DANGER obstacle#801 static", fontsize=9.5, color=DANG, fontweight="bold")
ax.text(2, 125, "B0 obstacle#136 approach", fontsize=9.5, color=B0, fontweight="bold")
# ⑭ 영역 만들기 상자 · ⑬ 미니맵
ax.add_patch(FancyBboxPatch((1.5, 96.5), 46, 21, boxstyle="round,pad=0.3", fc="#222", alpha=0.72, ec="none"))
for i, t in enumerate(["영역 만들기", "① 걸음 속도 1.16 m/s (바닥 광류)", "② 길이 = 2 + 1.16×4초 = 6.6 m",
                       "③ 길 유형: 인도 · 내 발밑: 인도", "   길 방향 -7° · 8° 넘게 벗어나면 무시", "④ 영역: 앞 6.6 m × 폭 ±1.3 m"]):
    ax.text(3, 115 - i * 3.3, t, fontsize=8.2, color="#9fd8ff" if i == 0 else "#f0f0f0")
ax.add_patch(Rectangle((1.5, 2), 20, 34, fc="#202422", ec="none"))
ax.add_patch(Rectangle((8.5, 5), 8, 21, fill=False, ec=ZONE, lw=1.5, ls=(0, (3, 2))))
ax.add_patch(Circle((12.5, 5), 1.2, fc="white"))
ax.add_patch(Ellipse((11.5, 10.5), 5, 3, fc=DANG, alpha=0.6))
ax.plot([12.5, 10.5], [5, 34], color="#b7f0b0", lw=1.5)
ax.text(17, 35, "top view", fontsize=6.5, color="#ccc")
# 번호
z1 = persp(1.3, 5.0); e9 = persp(2.4, 2.6); bar = persp(0.0, 7.5); imm = persp(0.6, 1.6)
for n, (x, y) in {1: (z1[0] + 4, z1[1]), 2: (bar[0] + 3.5, bar[1] + 1.5), 3: (imm[0] + 5, imm[1]), 4: (bx - 17, by - 2),
                  5: (bx + 19, by + 5), 6: (kx + 13, ky + 12), 7: (kx - 8, ky - 8), 8: (VP[0] + 4.5, VP[1] + 4),
                  9: (e9[0] + 4, e9[1] + 2), 10: (6, 31), 11: (bx + 6, by + 19), 12: (71, 106), 13: (19.5, 4),
                  14: (45.5, 115), 15: (52, 133)}.items():
    badge(ax, x, y, n)

# 오른쪽 설명
items = [
    (1, ZONE, "내 진행 영역 (하늘색 점선 띠)", "내가 몇 초 뒤 지나갈 바닥",
     "길이 = 2 m + 걸음 속도 × 4초 (4–15 m), 폭 ±1.3 m · 차도 옆 ×1.3·+0.3 m · 횡단보도 건너는 중 더 길게 · 물체 히트박스가 걸치면 '주의'"),
    (2, "#9aa3a0", "정면 막대 (흰 띠)", "'정면'인지 '11·1시'인지 가르는 기준",
     "길 중심선 ±0.25 m(사람 어깨 폭 절반). 카메라를 살짝 돌려 찍어도 길 방향 기준이라 흔들리지 않음"),
    (3, DANG, "곧 충돌 구역 (빨간 면)", "그대로 걸으면 1.5초 안에 부딪히는 곳",
     "내 몸 폭(±0.6 m) × 걸음 1.5초 거리. 여기 들어오면 '위험' + 피하세요 / 2.5 m 안이면 멈추세요"),
    (4, WARN, "물체 히트박스 (색 채운 타원)", "물체가 1.5초 동안 차지할 수 있는 바닥",
     "옆 여유 0.4 m, 움직이면 진행 방향으로 늘어남(정지 물체는 내 걸음 속도까지만). 발 위치에 붙여 그림, 색 = 판정"),
    (5, "#9aa3a0", "흰 점선 타원 + 흰 선", "같은 히트박스를 카메라 높이·화각대로만 투영",
     "④와 벌어질수록 거리 추정이 틀린 것 → 오차 확인용 (판정에는 안 씀)"),
    (6, "#7a4fd6", "경로 띠 (굵어지는 색 선)", "물체 발이 지나온 1.5초 길",
     "카메라 흔들림·회전만큼 매 프레임 옮겨서 '물체 자체의 움직임'만 보이게 함 · 물체마다 색 다름"),
    (7, "#7a4fd6", "화살표", "1초 뒤 갈 방향 (고정)",
     "첫 1초로 정하고 1초마다 확인, 35° 또는 속도 50% 넘게 바뀔 때만 교체 · 위쪽(±80°)=멀어짐 → 경고 안 함 · 옆=지나감 → 멈추세요"),
    (8, "#19b5c9", "하늘색 십자", "걷는 방향 · 지평선 (소실점)",
     "배경 광류로 구함. 거리 = 카메라 높이 1.77 m × 초점거리 ÷ (발 위치 - 지평선) · 걸음마다 출렁임 보정"),
    (9, EDGE, "인도 끝선 (연두 직선)", "내가 걷는 인도의 좌우 끝",
     "바닥 분할(인도·차도·연석·화단)로 끝점을 찾아 소실점 방향으로 이은 직선 · 너머의 차·버스는 다가와도 무시"),
    (10, "#7cc97a", "길 방향 (미니맵 연두 선)", "길이 휘는 방향",
     "인도 가운데선 기울기 · 걷는 방향과 길 방향 둘 다에서 8° 넘게 벗어난 정지 물체는 무시"),
    (11, DANG, "물체 상자 · 라벨", "판정 결과와 근거",
     "초록 안전 · 노랑 주의 · 빨강 위험 · 'class#ID 상태 TTC 거리' · 이름은 트랙 전체 투표 (멈춘 건 '장애물')"),
    (12, "#2bbf4f", "신호등 상자", "내 길 정면 보행 신호의 확정된 색",
     "길 방향 ±15° 안 신호만 · 0.5초 이상 같은 색이면 확정 · 1.5초 안 빨강↔초록만 '바뀜' 알림 · 아직 모르면 '신호 확인 중'"),
    (13, "#444", "위에서 본 지도", "미터 단위로 본 전체 상황",
     "흰 원 = 나 · 사각형 = 내 영역 · 색 점·타원 = 물체와 히트박스 · 연두 선 = 길 방향"),
    (14, "#444", "영역 만들기 상자", "영역 크기를 정한 과정",
     "① 걸음 속도 → ② 길이 계산 → ③ 길 유형(인도·차도 옆·횡단보도) · 내 발밑 · 길 방향 → ④ 최종 영역"),
    (15, B0, "왼쪽 위 글자", "지금 나는 경고 (최근 2초)",
     "DANGER·WARN = 히트박스 경고 · 분홍 'B0' = 태규 규칙 경고 (비교용)"),
]
ax2 = fig.add_axes([0.45, 0.04, 0.53, 0.87]); ax2.set_xlim(0, 100); ax2.set_ylim(0, len(items) * 6.0); ax2.axis("off")
for i, (n, col, name, what, rule) in enumerate(items):
    y = (len(items) - i) * 6.0 - 3.0
    if i % 2 == 0:
        ax2.add_patch(Rectangle((0, y - 2.9), 100, 5.8, fc="#eef1ee", ec="none"))
    badge(ax2, 2.2, y + 0.6, n)
    ax2.add_patch(Rectangle((5, y + 0.0), 1.6, 1.6, fc=col, ec="none"))
    ax2.text(7.5, y + 0.8, name, fontsize=11.5, color=INK, fontweight="bold", va="center")
    ax2.text(48, y + 0.8, what, fontsize=10.2, color=INK, va="center")
    ax2.text(5, y - 1.6, rule, fontsize=9, color=MUTED, va="center", wrap=True)
fig.savefig(f"{OUT}/explain_screen.png", facecolor=BG)
plt.close(fig)


# ================= ② 경고 판단 흐름 =================
fig = plt.figure(figsize=(17, 10.5), dpi=150, facecolor=BG)
fig.text(0.03, 0.965, "경고는 이렇게 정한다 — 히트박스 트리거 판단 흐름 (v5_11)", fontsize=20, fontweight="bold", color=INK, va="top")
fig.text(0.03, 0.928, "매 프레임 물체마다 ① → ⑦ 순서. 걸러내기에서 하나라도 걸리면 그 물체는 경고하지 않는다.", fontsize=11, color=MUTED, va="top")
ax = fig.add_axes([0.02, 0.03, 0.96, 0.87]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def box(x, y, w, h, title, lines, fc=PANEL, ec=LINE, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1.4))
    ax.text(x + 1.2, y + h - 2.2, title, fontsize=12.5, color=tc, fontweight="bold", va="top")
    for i, t in enumerate(lines):
        ax.text(x + 1.2, y + h - 6.6 - i * 3.6, t, fontsize=9.6, color=INK, va="top")


def arrow(a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=18, color=MUTED, lw=1.6))


box(1, 56, 22, 38, "① 찾기", ["YOLO 10클래스 + ByteTrack", "· conf 0.2 (0.5 미만은 영역에", "  걸칠 때만 판정)",
                               "· 80 % 겹치는 상자 = 하나", "· 사라진 트랙 근처 새 ID", "  = 같은 물체로 이어 붙임",
                               "· 이름 = 트랙 전체 투표", "신호등: 1920 해상도 + 상자 안 색"])
box(26, 56, 22, 38, "② 바닥 좌표로", ["발 위치 + 물체 키 → 거리 Z", "옆 거리 X (걷는 방향 기준)", "바닥 칼만 필터로 매끈하게",
                                      "", "물체 자체 속도 =", "1초 거리 변화 + 내 걸음 속도", "(주차 차 '다가옴' 오판 방지)",
                                      "12 m 넘으면 다가옴 판단 안 함"])
box(51, 56, 22, 38, "③ 움직임", ["다가옴: 자체 속도 0.8 m/s+", "  (사람 0.5 + 0.05·Z), 0.2초 유지", "지나감: 화살표가 옆 + 옆 속도",
                                 "  · 화면 끝에서 잘려 들어오는", "    차도 안쪽 끝 속도로", "멀어짐: 화살표 위쪽 ±80°",
                                 "정지: 그 외 (장애물은 항상)"])
box(76, 56, 23, 38, "④ 걸러내기 (무시)", ["· 내가 인도인데 물체가 차도 위", "· 인도 끝선 너머 차·버스", "· 길·걷는 방향 둘 다 8° 밖 (정지)",
                                       "· 멀어지는 물체", "· 빨간불 대기 중 지나가는 차", "· 고개 돌리는 중(0.4초) 주의만",
                                       "· 생긴 지 0.25초 미만", "· 사람: 막을 때·다가올 때만"], fc="#fbf4ec", ec="#e5cfae")
box(1, 6, 34, 42, "⑤ 주의 / 위험", ["주의 (노랑)", "· 물체 히트박스가 내 영역에 걸침", "· 움직이는 물체가 3초 안 영역에 들어옴(예측)",
                                    "· 앞 4 m 사람이 느려지거나 멈춤",
                                    "위험 (빨강)", "· 1.5초 히트박스가 내 몸에 닿음", "· 내 몸 폭 안 3초 안 도달",
                                    "· 2.5초 안 내 몸 폭으로 들어옴(예측)", "· 차도·횡단보도 건너는 중 영역에 들어오는 차",
                                    "켜짐 0.1초 · 꺼짐 0.6초 (깜빡임 방지)"], fc="#fdf2f0", ec="#efc9c2")
box(38, 6, 30, 42, "⑥ 행동", ["주의 → 위치만 말함", "위험이고 정면 충돌각(±0.9 m)일 때만:",
                               "· 멈추세요: 지나가거나 다가오는 차·사람,", "  계단, 2.5 m 안 바짝 접근",
                               "· 피하세요: 그 외 → 다른 물체가", "  없는(덜 막힌) 쪽으로", "",
                               "같은 트랙은 한 번만 말함", "(예외: 바짝 접근 · 서 있다 움직임)"])
box(71, 6, 28, 42, "⑦ 음성", ["\"1시 방향 자동차가 지나가요. 멈추세요.\"", "· 정면 막대 안이면 '정면'",
                               "· 조각 미리 합성 → 0.02초에 소리", "· 급하지 않은 건 0.5초 묶어 한 문장",
                               "· 줄지은 기둥: '이어져 있어요' 한 번", "· 횡단보도 · 볼라드 · 신호 바뀜 안내",
                               "· 재생 순서: 멈추세요 > 피하세요", "  > 위치 안내 (급한 게 끊고 먼저)"], fc="#eef6f2", ec="#bfdccf")
for a, b in [((23.6, 75), (25.6, 75)), ((48.6, 75), (50.6, 75)), ((73.6, 75), (75.6, 75)), ((87, 55.6), (25, 48.6)),
             ((35.6, 27), (37.6, 27)), ((68.6, 27), (70.6, 27))]:
    arrow(a, b)
fig.savefig(f"{OUT}/explain_flow.png", facecolor=BG)
plt.close(fig)
print("saved")
