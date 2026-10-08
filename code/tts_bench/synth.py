"""TTS 엔진 하나로 문장 세트 전부 합성 → results/tts_bench/<engine>/<id>.wav + timing.json

엔진마다 의존성이 달라서 각자 가상환경에서 이 파일을 실행한다 (프로젝트 코드 import 없음).
    ~/egoenv/Scripts/python.exe synth.py sapi
    ~/ttsenv/supertonic/Scripts/python.exe synth.py supertonic --voice F1
측정: 모델 로드 시간, 문장마다 합성 시간(= 비스트리밍 엔진의 첫 소리 시간), 소리 길이, RTF = 합성 / 소리 길이.
처음 2문장은 예열로 한 번 돌리고 버린다 (첫 호출의 그래프 컴파일 · 캐시 시간 제외, 로드 시간은 따로 기록).
스트리밍을 지원하는 엔진(CosyVoice)은 첫 조각까지의 시간 first_s를 따로 잰다 (없으면 first_s = 합성 전체).

참고 음성이 필요한 엔진(CosyVoice2/3 · Chatterbox: zero-shot 음성 복제)은 모두 같은 참고 음성을 쓴다:
  results/tts_bench/prompt_ko.wav + prompt_ko.txt  (make_prompt.py — Supertonic F1로 만든 한국어 문장, 저작권 문제 없음)
"""
import argparse
import json
import os
import time
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / "tts_bench"


def write_wav(path, x, sr):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


# ---------------------------------------------------------------- 엔진 어댑터: load() → synth(text, path)
def eng_sapi(a):
    import pythoncom
    import win32com.client
    pythoncom.CoInitialize()
    v = win32com.client.Dispatch("SAPI.SpVoice")
    for tok in v.GetVoices():
        if "Heami" in tok.GetDescription():
            v.Voice = tok

    def synth(text, path):
        fs = win32com.client.Dispatch("SAPI.SpFileStream")
        fmt = win32com.client.Dispatch("SAPI.SpAudioFormat")
        fmt.Type = 22
        fs.Format = fmt
        fs.Open(str(path), 3)
        v.AudioOutputStream = fs
        v.Speak(text)
        fs.Close()
    return synth, {"voice": "Microsoft Heami", "sr": 22050}


def eng_supertonic(a):
    from supertonic import TTS
    tts = TTS()
    style = tts.get_voice_style(a.voice or "F1")

    def synth(text, path):
        wav, _ = tts.synthesize(text, voice_style=style, lang="ko", total_steps=a.steps, speed=a.speed)
        write_wav(path, wav, tts.sample_rate)
    return synth, {"voice": a.voice or "F1", "steps": a.steps, "speed": a.speed, "sr": tts.sample_rate,
                   "threads": os.environ.get("OMP_NUM_THREADS")}


def eng_melotts(a):
    from melo.api import TTS
    m = TTS(language="KR", device=a.device)
    spk = m.hps.data.spk2id["KR"]

    def synth(text, path):
        m.tts_to_file(text, spk, str(path), speed=a.speed, quiet=True)
    return synth, {"voice": "KR", "speed": a.speed, "sr": m.hps.data.sampling_rate}


def eng_mms(a):
    import torch
    from transformers import AutoTokenizer, VitsModel
    tok = AutoTokenizer.from_pretrained("facebook/mms-tts-kor")
    m = VitsModel.from_pretrained("facebook/mms-tts-kor").eval().to(a.device)

    def synth(text, path):
        x = text
        if getattr(tok, "is_uroman", False):
            import uroman as ur
            x = ur.Uroman().romanize_string(text)
        with torch.no_grad():
            wav = m(**tok(x, return_tensors="pt").to(a.device)).waveform[0].float().cpu().numpy()
        write_wav(path, wav, m.config.sampling_rate)
    return synth, {"voice": "mms-tts-kor", "sr": m.config.sampling_rate}


def _prompt():
    wav = OUT / "prompt_ko.wav"
    return str(wav), (OUT / "prompt_ko.txt").read_text(encoding="utf-8").strip()


def eng_cosyvoice(a):
    """CosyVoice2-0.5B / Fun-CosyVoice3-0.5B — 공식 example.py의 inference_zero_shot 그대로. --repo = CosyVoice 클론 경로."""
    import sys
    import torch
    sys.path[:0] = [a.repo, os.path.join(a.repo, "third_party", "Matcha-TTS")]
    from cosyvoice.cli.cosyvoice import AutoModel
    m = AutoModel(model_dir=a.model_dir)
    pw, pt = _prompt()
    if "CosyVoice3" in a.model_dir:                    # v3는 프롬프트 앞에 시스템 문구 (example.py cosyvoice3_example)
        pt = "You are a helpful assistant.<|endofprompt|>" + pt
    assert m.add_zero_shot_spk(pt, pw, "ko_prompt") is True

    def synth(text, path):
        t0, first, chunks = time.perf_counter(), None, []
        for j in m.inference_zero_shot(text, "", "", zero_shot_spk_id="ko_prompt", stream=a.stream):
            if first is None:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                first = time.perf_counter() - t0
            chunks.append(j["tts_speech"].reshape(-1).float().cpu().numpy())
        write_wav(path, np.concatenate(chunks), m.sample_rate)
        return {"first_s": round(first, 4)}
    return synth, {"model_dir": a.model_dir, "stream": a.stream, "prompt": pt, "sr": m.sample_rate}


def eng_chatterbox(a):
    """Chatterbox Multilingual (README: ChatterboxMultilingualTTS.from_pretrained → generate(text, language_id, audio_prompt_path))."""
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    m = ChatterboxMultilingualTTS.from_pretrained(device=a.device)
    pw, _ = _prompt()

    def synth(text, path):
        wav = m.generate(text, language_id="ko", audio_prompt_path=pw)
        write_wav(path, wav.reshape(-1).float().cpu().numpy(), m.sr)
    return synth, {"prompt": pw, "sr": m.sr}


ENGINES = {"sapi": eng_sapi, "supertonic": eng_supertonic, "melotts": eng_melotts, "mms": eng_mms,
           "cosyvoice": eng_cosyvoice, "chatterbox": eng_chatterbox}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", choices=list(ENGINES))
    ap.add_argument("--voice", default=None)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--speed", type=float, default=1.05)
    ap.add_argument("--tag", default=None, help="결과 폴더 이름 (기본 = 엔진 이름)")
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--repo", default=os.path.expanduser("~/CosyVoice"), help="CosyVoice 클론 경로")
    ap.add_argument("--model-dir", default=None, help="CosyVoice 모델 폴더 (pretrained_models/CosyVoice2-0.5B 등)")
    ap.add_argument("--stream", action="store_true", help="CosyVoice 스트리밍 (첫 조각 시간 측정)")
    a = ap.parse_args()
    items = json.load(open(OUT / "sentences.json", encoding="utf-8"))
    od = OUT / (a.tag or a.engine)
    od.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    synth, meta = ENGINES[a.engine](a)
    load_s = time.perf_counter() - t0
    for it in items[:2]:                              # 예열
        synth(it["text"], od / "_warm.wav")
    rows = []
    for it in items:
        p = od / f"{it['id']}.wav"
        t0 = time.perf_counter()
        extra = synth(it["text"], p) or {}
        dt = time.perf_counter() - t0
        with wave.open(str(p)) as w:
            dur = w.getnframes() / w.getframerate()
        rows.append({**it, "synth_s": round(dt, 4), "first_s": extra.get("first_s", round(dt, 4)),
                     "audio_s": round(dur, 3), "rtf": round(dt / max(dur, 1e-6), 4)})
        print(f"{it['id']} {dt:6.3f}s  {dur:5.2f}s  {it['text']}", flush=True)
    (od / "_warm.wav").unlink(missing_ok=True)
    json.dump({"engine": a.engine, "tag": a.tag or a.engine, "load_s": round(load_s, 2), **meta, "rows": rows},
              open(od / "timing.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    s = sorted(r["synth_s"] for r in rows)
    print(f"load {load_s:.1f}s · 합성 중앙 {s[len(s) // 2]:.3f}s · 최대 {s[-1]:.3f}s")


if __name__ == "__main__":
    main()
