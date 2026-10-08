"""zero-shot 엔진(CosyVoice2/3 · Chatterbox) 공통 참고 음성 → results/tts_bench/prompt_ko.wav + prompt_ko.txt

모든 복제형 엔진이 같은 목소리를 따라 하게 해서 '참고 음성 차이'를 비교에서 뺀다.
Supertonic F1(로컬 CPU 후보 중 하나)로 합성 — 사람 녹음이 아니므로 저작권 · 동의 문제 없음. 평가 문장과 겹치지 않는 문장.
    ~/ttsenv/supertonic/Scripts/python.exe make_prompt.py
"""
from pathlib import Path

from supertonic import TTS

from synth import OUT, write_wav

TEXT = "안녕하세요. 저는 길 안내를 도와드리는 음성 도우미예요. 천천히 따라오세요."

tts = TTS()
wav, _ = tts.synthesize(TEXT, voice_style=tts.get_voice_style("F1"), lang="ko")
OUT.mkdir(parents=True, exist_ok=True)
write_wav(OUT / "prompt_ko.wav", wav, tts.sample_rate)
(OUT / "prompt_ko.txt").write_text(TEXT + "\n", encoding="utf-8")
print(f"→ {OUT / 'prompt_ko.wav'}  ({len(wav.reshape(-1)) / tts.sample_rate:.1f}s)")
