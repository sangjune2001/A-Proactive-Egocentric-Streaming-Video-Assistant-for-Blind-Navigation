"""TTS 채점: 합성 wav → Whisper large-v3 (faster-whisper, CPU int8) 받아쓰기 → 글자 오류율(CER) + 속도 요약.

    ~/ttsenv/asr/Scripts/python.exe asr_eval.py sapi supertonic_F1 melotts mms          # 노트북 CPU
    python asr_eval.py --device cuda --summary summary_server.md srv_...                 # A5000
→ results/tts_bench/<tag>/asr.json · results/tts_bench/<summary>

CER: 공백 · 문장부호를 빼고 글자(음절) 단위 편집거리 / 정답 글자 수. 문장 정확 = 받아쓰기가 원문과 완전히 같음.
'말하려던 단어가 들리는가'를 재는 지표 (자연스러움 MOS는 따로).
"""
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / "tts_bench"


def norm(s):
    return re.sub(r"[\s\.,!?·~\-'\"]", "", s)


def edit(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else None


def main():
    from faster_whisper import WhisperModel
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--summary", default="summary.md")
    a = ap.parse_args()
    tags = a.tags
    t0 = time.perf_counter()
    model = (WhisperModel("large-v3", device="cuda", compute_type="float16") if a.device == "cuda" else
             WhisperModel("large-v3", device="cpu", compute_type="int8", cpu_threads=os.cpu_count()))
    print(f"whisper large-v3 load {time.perf_counter() - t0:.0f}s", flush=True)
    lines = ["| 엔진 | 경고 CER | 설명 CER | 문장 정확 | 첫 소리 중앙 (s) | 합성 중앙 (s) | 합성 최대 (s) | RTF 중앙 | 로드 (s) | 설명 소리 길이 중앙 (s) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for tag in tags:
        tm = json.load(open(OUT / tag / "timing.json", encoding="utf-8"))
        rows = []
        for r in tm["rows"]:
            segs, _ = model.transcribe(str(OUT / tag / f"{r['id']}.wav"), language="ko", beam_size=5,
                                       vad_filter=False, condition_on_previous_text=False)
            hyp = "".join(s.text for s in segs).strip()
            ref_n, hyp_n = norm(r["text"]), norm(hyp)
            e = edit(ref_n, hyp_n)
            rows.append({**r, "asr": hyp, "cer": round(e / max(1, len(ref_n)), 4), "exact": ref_n == hyp_n})
            if e:
                print(f"  {tag} {r['id']} {r['text']}  →  {hyp}", flush=True)
        json.dump(rows, open(OUT / tag / "asr.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

        def cer(kind):
            rs = [x for x in rows if x["kind"] == kind]
            return sum(x["cer"] for x in rs) / len(rs)
        ex = sum(x["exact"] for x in rows) / len(rows)
        desc_dur = med([x["audio_s"] for x in rows if x["kind"] == "desc"])
        first = med([x.get("first_s", x["synth_s"]) for x in rows])
        lines.append(f"| {tag} | {cer('warn'):.3f} | {cer('desc'):.3f} | {ex:.2f} | {first:.3f} | {med([x['synth_s'] for x in rows]):.3f} | "
                     f"{max(x['synth_s'] for x in rows):.3f} | {med([x['rtf'] for x in rows]):.3f} | {tm['load_s']:.1f} | {desc_dur:.2f} |")
        print(lines[-1], flush=True)
    (OUT / a.summary).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
