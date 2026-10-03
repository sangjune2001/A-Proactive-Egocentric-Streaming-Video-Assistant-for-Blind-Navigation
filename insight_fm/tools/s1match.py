# For each S1 frame: do the jsonl stairs polygons (class 8) coincide with the XML caution_zone[stairs] polygons,
# or with some other XML polygon (manhole, grating, ...)?
import collections, json, re, zipfile
z = zipfile.ZipFile("/home/ubuntu/sg/s1chk/S1.zip")
xmlp = {}
for f in (f for f in z.namelist() if f.endswith(".xml")):
    s = z.read(f).decode("utf-8", "ignore")
    for img in re.findall(r"<image [^>]*>.*?</image>", s, re.S):
        head = re.search(r"<image [^>]*>", img).group(0)
        name = re.search(r'name="([^"]+)"', head).group(1).rsplit("/", 1)[-1]
        w, h = float(re.search(r'width="([\d.]+)"', head).group(1)), float(re.search(r'height="([\d.]+)"', head).group(1))
        polys = []
        for lab, pts, body in re.findall(r'<polygon label="([^"]+)"[^>]*points="([^"]+)"[^>]*>(.*?)</polygon>', img, re.S):
            at = re.findall(r'<attribute name="[^"]+">([^<]*)</attribute>', body)
            xy = [tuple(map(float, p.split(","))) for p in pts.split(";")]
            polys.append((lab + ("/" + at[0] if at else ""), [(x / w, y / h) for x, y in xy]))
        xmlp[(f.split("/")[0], name)] = polys

def box(p):
    xs, ys = [a for a, _ in p], [b for _, b in p]
    return min(xs), min(ys), max(xs), max(ys)

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i / u if u else 0

best_lab = collections.Counter(); n = 0; examples = collections.defaultdict(list)
for line in open("/home/ubuntu/sg/labels_all.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if not r["z"].endswith("S1.zip"):
        continue
    folder, name = r["m"].split("/")[0], r["m"].rsplit("/", 1)[-1]
    polys = xmlp.get((folder, name))
    if polys is None:
        best_lab["(frame not in xml)"] += 1; continue
    for ln in r["l"]:
        p = ln.split()
        if int(p[0]) != 8:
            continue
        v = list(map(float, p[1:])); jb = box(list(zip(v[0::2], v[1::2])))
        sc = sorted(((iou(jb, box(pp)), lab) for lab, pp in polys), reverse=True)
        lab = sc[0][1] if sc and sc[0][0] > 0.5 else f"(no match, best {sc[0][1]} iou {sc[0][0]:.2f})" if sc else "(none)"
        best_lab[lab if "no match" not in lab else "(no polygon with IoU>0.5)"] += 1
        n += 1
        if len(examples[lab]) < 3: examples[lab].append(f"{folder}/{name}")
print("stairs polygons checked:", n)
for k, v in best_lab.most_common(): print(v, k, examples.get(k, [])[:2])
