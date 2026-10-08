"""받아쓰기가 틀린 문장 듣기 페이지 → results/tts_bench/listen.html (브라우저로 열기)

문장마다 해당 엔진 소리 + 같은 문장의 Supertonic M1 소리(비교용)를 나란히 두고,
"TTS가 잘못 읽음 / 받아쓰기가 잘못 들음 / 애매" 판정을 고르면 CSV로 저장한다 (→ docs/15 6절).

    ~/egoenv/Scripts/python.exe code/tts_bench/make_listen.py
"""
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / "tts_bench"
ENGINES = [("supertonic_M1", "Supertonic M1"), ("supertonic_F1", "Supertonic F1"), ("sapi", "Windows Heami"),
           ("melotts", "MeloTTS"), ("mms", "MMS")]
REF = "supertonic_M1"

rows, n_eng = [], {}
for tag, name in ENGINES:
    items = json.load(open(OUT / tag / "asr.json", encoding="utf-8"))
    bad = [x for x in items if not x["exact"]]
    n_eng[name] = len(bad)
    for x in bad:
        rows.append((tag, name, x))

sec = []
for tag, name in ENGINES:
    rs = [r for r in rows if r[0] == tag]
    if not rs:
        continue
    open_ = "" if tag == "mms" else " open"
    body = []
    for _, _, x in rs:
        key = f"{tag}/{x['id']}"
        ref = "" if tag == REF else f'<div class="ref">비교: M1 <audio controls preload="none" src="{REF}/{x["id"]}.wav"></audio></div>'
        body.append(f"""
<div class="card" data-key="{key}">
  <div class="meta">{x['id']} · {'경고' if x['kind'] == 'warn' else '설명'} · CER {x['cer']:.2f}</div>
  <div class="txt"><span class="lab">원문</span>{html.escape(x['text'])}</div>
  <div class="txt"><span class="lab">받아쓰기</span><b>{html.escape(x['asr'])}</b></div>
  <audio controls preload="none" src="{tag}/{x['id']}.wav"></audio>{ref}
  <div class="opts">
    <label><input type="radio" name="{key}" value="tts"> TTS가 잘못 읽음</label>
    <label><input type="radio" name="{key}" value="asr"> 받아쓰기가 잘못 들음 (소리는 맞음)</label>
    <label><input type="radio" name="{key}" value="unclear"> 애매</label>
  </div>
</div>""")
    sec.append(f'<details{open_}><summary>{name} — 틀린 문장 {len(rs)}개</summary>{"".join(body)}</details>')

page = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>TTS 오류 듣기</title>
<style>
:root{{--bg:#fcfcfb;--card:#fff;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3de;--acc:#2a78d6}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1a1a19;--card:#242423;--ink:#fff;--ink2:#c3c2b7;--line:#3a3a38;--acc:#3987e5}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 "Malgun Gothic",system-ui,sans-serif}}
main{{max-width:860px;margin:0 auto;padding:24px 16px 80px}}
h1{{font-size:20px;margin:0 0 4px}} p{{color:var(--ink2);margin:4px 0 16px}}
details{{margin:14px 0;border:1px solid var(--line);border-radius:10px;background:var(--card)}}
summary{{cursor:pointer;padding:12px 14px;font-weight:700}}
.card{{border-top:1px solid var(--line);padding:12px 14px}}
.meta{{font-size:12px;color:var(--ink2)}} .lab{{display:inline-block;width:64px;color:var(--ink2);font-size:13px}}
.txt{{margin:2px 0}} audio{{height:32px;margin:6px 0;max-width:100%}} .ref{{font-size:13px;color:var(--ink2)}}
.opts{{display:flex;flex-wrap:wrap;gap:6px 16px;margin-top:4px;font-size:14px}}
.bar{{position:fixed;left:0;right:0;bottom:0;background:var(--card);border-top:1px solid var(--line);padding:10px 16px;display:flex;gap:12px;align-items:center;justify-content:center}}
button{{background:var(--acc);color:#fff;border:0;border-radius:8px;padding:8px 14px;font:inherit;cursor:pointer}}
</style></head><body><main>
<h1>TTS 오류 듣기</h1>
<p>받아쓰기(Whisper large-v3)가 원문과 다르게 적은 문장 {len(rows)}개. 소리를 듣고 누가 틀렸는지 고른 뒤 아래 <b>CSV 저장</b>을 누르세요.
선택은 이 브라우저에 자동 저장됩니다. MMS는 탈락 후보라 접어 두었습니다.</p>
{''.join(sec)}
</main>
<div class="bar"><span id="cnt"></span><button id="save">CSV 저장</button></div>
<script>
const K='tts_listen_v1';let st={{}};try{{st=JSON.parse(localStorage.getItem(K)||'{{}}')}}catch(e){{}}
const cards=[...document.querySelectorAll('.card')];
function upd(){{document.getElementById('cnt').textContent=Object.keys(st).length+' / '+cards.length+' 판정';}}
cards.forEach(c=>{{const k=c.dataset.key;if(st[k]){{const r=c.querySelector(`input[value="${{st[k]}}"]`);if(r)r.checked=true}}
 c.querySelectorAll('input').forEach(i=>i.addEventListener('change',()=>{{st[k]=i.value;try{{localStorage.setItem(K,JSON.stringify(st))}}catch(e){{}};upd()}}))}});
upd();
document.getElementById('save').onclick=()=>{{const L=['engine,id,text,asr,cer,verdict'];
 cards.forEach(c=>{{const k=c.dataset.key,[e,id]=k.split('/'),t=c.querySelectorAll('.txt');
  const q=s=>'"'+s.replace(/"/g,'""')+'"';
  L.push([e,id,q(t[0].lastChild.textContent),q(t[1].lastChild.textContent),c.querySelector('.meta').textContent.split('CER ')[1],st[k]||''].join(','))}});
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob(['\\ufeff'+L.join('\\n')],{{type:'text/csv'}}));a.download='listen_verdicts.csv';a.click()}};
</script></body></html>"""
(OUT / "listen.html").write_text(page, encoding="utf-8")
print(f"{len(rows)}개 → {OUT / 'listen.html'}", n_eng)
