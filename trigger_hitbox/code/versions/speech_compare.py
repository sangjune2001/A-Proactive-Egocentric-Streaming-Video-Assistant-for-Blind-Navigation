"""버전별 음성 출력(실제 재생된 경고 문장) 비교표 → Markdown.
usage: python speech_compare.py speech.json speech_compare.md
speech.json = {"v515|test_1": [[t, "문장"], ...], ...}  (run_demo 로그의 [경고] 줄)"""
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
VERS = ["v512", "v513", "v514", "v515", "v520", "v521", "v522", "v523", "v528", "v529"]
NAME = {v: "v5_" + v[2:] for v in VERS}
TTS = {"v512": "Supertonic", "v513": "Supertonic"}
BASE = "v515"
out = ["# 음성 출력 버전별 비교 (test_1 · test_2 · test_3)", "",
       "실제로 재생된 경고 문장을 시간 순으로 맞춰 나란히 놓았다 (같은 문장은 ±1.5 s, 다른 문장은 ±0.8 s 안이면 같은 줄). 같은 문장도 그대로 다 적음.", "",
       "- 빈 칸 = 그 버전은 그 시점에 말하지 않음 · **굵게** = v5_15(검출 기준판)와 다른 문장",
       "- 음성: v5_12·v5_13 = Supertonic, 나머지 = Heami",
       "- v5_12 = v5_7+고른 항목 / v5_13 = +사람 규칙 / v5_14·15 = +화살표 위쪽 제외 / v5_20·21 = 경로 추적·상대속도·버스 / v5_22 = 도로 방향 / v5_28·29 = v5_15 판정 + 음성 정리",
       "- 로그가 없는 버전·영상은 열에서 빠짐 (서버 꺼짐: v5_23, v5_28 test_1·2, v5_29)", ""]
for v in ("test_1", "test_2", "test_3"):
    vers = [x for x in VERS if f"{x}|{v}" in d]
    ev = sorted((t, x, s) for x in vers for t, s in d[f"{x}|{v}"])
    rows = []
    for t, x, s in ev:
        # 같은 문장이 1.5 s 안 줄에 있으면 그 줄, 아니면 0.8 s 안 빈 줄, 아니면 새 줄
        r = next((r for r in reversed(rows) if t - r["t0"] <= 1.5 and x not in r["c"] and s in r["c"].values()), None)
        if r is None:
            r = next((r for r in reversed(rows) if t - r["t0"] <= 0.8 and x not in r["c"]), None)
        if r is None:
            rows.append(dict(t0=t, c={x: s}))
        else:
            r["c"][x] = s
    out += [f"## {v}", "", "| 초 | " + " | ".join(NAME[x] + (" (ST)" if x in TTS else "") for x in vers) + " |",
            "|---|" + "---|" * len(vers)]
    for r in rows:
        base = r["c"].get(BASE)
        cells = ["" if x not in r["c"] else (r["c"][x] if x == BASE or r["c"][x] == base else f"**{r['c'][x]}**")
                 for x in vers]
        out.append(f"| {r['t0']:.1f} | " + " | ".join(c.replace("|", "/") for c in cells) + " |")
    same = [f"{NAME[x]} {sum(1 for r in rows if r['c'].get(x) is not None and r['c'].get(x) == r['c'].get(BASE))}/{len(d[f'{x}|{v}'])}"
            for x in vers if x != BASE]
    out += ["", f"v5_15({len(d.get(f'{BASE}|{v}', []))}문장)과 같은 문장 / 그 버전 문장 수: " + " · ".join(same), ""]
open(sys.argv[2], "w", encoding="utf-8").write("\n".join(out) + "\n")
print("ok")
