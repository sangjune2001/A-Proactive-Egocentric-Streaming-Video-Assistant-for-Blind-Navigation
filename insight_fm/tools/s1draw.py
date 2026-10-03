import random, re, zipfile, cv2, numpy as np
z = zipfile.ZipFile("/home/ubuntu/sg/s1chk/S1.zip")
items = []
for f in (f for f in z.namelist() if f.endswith(".xml")):
    s = z.read(f).decode("utf-8", "ignore")
    for img in re.findall(r"<image [^>]*>.*?</image>", s, re.S):
        name = re.search(r'name="([^"]+)"', img).group(1)
        st = [pts for pts, body in re.findall(r'<polygon label="caution_zone"[^>]*points="([^"]+)"[^>]*>(.*?)</polygon>', img, re.S)
              if ">stairs<" in body]
        if st: items.append((f.split("/")[0], name, st))
random.seed(3); pick = random.sample(items, 12)
names = set(z.namelist()); tiles = []
for folder, name, st in pick:
    cand = [n for n in (f"{folder}/{name}", f"{folder}/{name.rsplit('/',1)[-1]}") if n in names]
    if not cand: continue
    im = cv2.imdecode(np.frombuffer(z.read(cand[0]), np.uint8), cv2.IMREAD_COLOR)
    for pts in st:
        p = np.array([[float(a) for a in q.split(",")] for q in pts.split(";")], np.int32)
        cv2.polylines(im, [p], True, (0, 0, 255), 6)
    im = cv2.resize(im, (480, 270)); cv2.putText(im, name.rsplit("/",1)[-1][:24], (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 1)
    tiles.append(im)
while len(tiles) % 3: tiles.append(np.full_like(tiles[0], 255))
cv2.imwrite("/home/ubuntu/s1_stairs_orig.jpg", np.vstack([np.hstack(tiles[i:i+3]) for i in range(0, len(tiles), 3)]))
print(len(items), "frames with stairs in xml")
