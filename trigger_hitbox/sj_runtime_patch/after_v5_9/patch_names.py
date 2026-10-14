from pathlib import Path
p = Path("/home/ubuntu/sj/code/p0/ko_template.py")
s = p.read_text()
if '"wall"' not in s:
    s = s.replace('"traffic_light": "신호등", "other": "장애물", "obstacle": "장애물"',
                  '"traffic_light": "신호등", "other": "장애물", "obstacle": "장애물", "wall": "벽"', 1)
    assert '"wall": "벽"' in s
    p.write_text(s)
p = Path("/home/ubuntu/sj/code/runtime/dispatcher.py")
s = p.read_text()
s = s.replace('for c in CLASSES + ["bollard"]', 'for c in CLASSES + ["bollard", "pole", "wall"]')
p.write_text(s)
print("patched", s.count('"pole", "wall"'))
