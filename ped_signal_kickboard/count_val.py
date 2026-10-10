"""Validation 라벨만 받아서 남게 될 이미지/객체 수를 센다 (저장하지 않음)."""
import pipeline as p

st = {"bytes": 0}
jobs = {(ds, split): lab for ds, split, lab, _ in p.build_jobs()}

# 188 Validation
imgs = objs = total = 0
for fk in jobs[("188", "Validation")]:
    kind, gen = p.sniff_kind(p.concat_sources("188", [fk], st, p.Meter("188 count")))
    for name, chunks in p.iter_members(gen, kind):
        if not name.endswith(".json"):
            p.drain(chunks)
            continue
        d = p.load_json(b"".join(chunks)) or {}
        total += 1
        n = sum(1 for a in d.get("annotation", []) if a.get("class") == "traffic_light" and a.get("type") == "pedestrian")
        objs += n
        imgs += n > 0
print(f"188 Validation: 전체 {total:,}장 중 보행신호등 이미지 {imgs:,}장, 보행신호등 {objs:,}개", flush=True)

# 71784 Validation (가시광)
imgs = objs = total = 0
for fk in jobs[("71784", "Validation")]:
    kind, gen = p.sniff_kind(p.concat_sources("71784", [fk], st, p.Meter("71784 count")))
    for name, chunks in p.iter_members(gen, kind):
        if not name.endswith(".json"):
            p.drain(chunks)
            continue
        d = p.load_json(b"".join(chunks)) or {}
        total += 1
        n = sum(1 for a in d.get("annotations", []) if a.get("category_id") == 99)
        objs += n
        imgs += n > 0
print(f"71784 Validation: 전체 {total:,}장 중 킥보드 이미지 {imgs:,}장, 킥보드 {objs:,}개", flush=True)
