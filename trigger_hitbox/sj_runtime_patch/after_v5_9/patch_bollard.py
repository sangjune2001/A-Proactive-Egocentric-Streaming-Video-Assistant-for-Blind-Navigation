from pathlib import Path
p = Path("/home/ubuntu/sj/code/runtime/dispatcher.py")
s = p.read_text()
old = '''               "주변 사람들이 건너기 시작했어요. 신호를 확인하세요."]'''
assert old in s
s = s.replace(old, '''               "주변 사람들이 건너기 시작했어요. 신호를 확인하세요.",
               "신호 대기 중이에요. 전방에 볼라드를 조심하세요.", "횡단보도 앞이에요. 전방에 볼라드를 조심하세요."]''', 1)
p.write_text(s); print("patched")
