import collections, json, re, zipfile
z = zipfile.ZipFile("/home/ubuntu/sg/s1chk/S1.zip")
n = z.namelist()
x = [f for f in n if f.endswith(".xml")]
print(len(n), "files, xml:", len(x), x[:2])
keys = set()
for line in open("/home/ubuntu/sg/labels_all.jsonl", encoding="utf-8"):
    if '"S1.zip"' in line or "S1.zip" in line:
        r = json.loads(line)
        if r["z"].endswith("S1.zip"):
            keys.add(r["m"].rsplit("/", 1)[-1])
print("labelled S1 frames:", len(keys))
lab_all, lab_used = collections.Counter(), collections.Counter()
for f in x:
    s = z.read(f).decode("utf-8", "ignore")
    for img in re.findall(r"<image [^>]*>.*?</image>", s, re.S):
        name = re.search(r'name="([^"]+)"', img).group(1).rsplit("/", 1)[-1]
        for lab, body in re.findall(r'<polygon label="([^"]+)"[^>]*>(.*?)</polygon>', img, re.S):
            attrs = ";".join(f"{a}={v}" for a, v in re.findall(r'<attribute name="([^"]+)">([^<]*)</attribute>', body))
            k = lab + (" [" + attrs + "]" if attrs else "")
            lab_all[k] += 1
            if name in keys:
                lab_used[k] += 1
print("== all labels in S1 xml")
for k, v in lab_all.most_common(30):
    print(v, k)
print("== labels on the 372 frames used")
for k, v in lab_used.most_common(30):
    print(v, k)
