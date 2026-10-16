"""pass2_trigger_v5_29.py → pass2_label.py: 같은 판정에 프레임별 라벨 출력만 추가.

라벨 (트랙 · 프레임마다, OUT_DIR/labels_frames.jsonl):
  motion: approaching(다가옴) / receding(멀어짐) / static(가만히 있음) / crossing(옆으로 지나감)
  speak : must(말해야 됨)    — 히트박스 판정이 주의·위험 (v5_29가 실제로 경고하는 단계)
          may(말해도 됨)     — 걸러지지 않았고 내 진행 영역 근처지만 경고 단계는 아님 (옆에서 조용히 추적 등)
          must_not(말하면 안 됨) — 차도 위 · 경로 밖 · 멀어짐 · 빨간불 대기 차 · 붐빔 속 사람 · 지나가지 않는 버스 등
usage: python make_label_pass2.py  (같은 폴더에 pass2_label.py 생성)
"""
from pathlib import Path

here = Path(__file__).parent
src = (here / "pass2_trigger_v5_29.py").read_text(encoding="utf-8")


def sub(old, new):
    global src
    assert src.count(old) == 1, old[:80]
    src = src.replace(old, new)


# 1) 바닥 좌표가 있는 물체: 판정 변수로 speak 라벨 계산
sub("""                dist = float(math.hypot(X, Z))
            else:
                dist = None
""", """                dist = float(math.hypot(X, Z))
                # --- 라벨 (make_label_pass2) ---
                filtered = (going_up or red_wait or off_road or off_path or (low_conf and not touch_zone)
                            or (cls == "person" and (crowd_now or not person_appr))
                            or (cls == "bus" and state != "crossing"))
                if lvB != "safe":
                    spk = "must"
                elif filtered:
                    spk = "must_not"
                elif in_zone or touch_zone or enter_w or (lat < zone_w + 1.0 and 0 < Z < zone_z * 1.5):
                    spk = "may"
                else:
                    spk = "must_not"
            else:
                dist = None
                spk = "must" if lvB != "safe" else "must_not"
""")
# 2) 그림 정보에 라벨 같이 저장
sub("""            draws.append(dict(box=d["box"], cls=cls, tid=tid, state=state, ttc=ttc, lvA=lvA, lvB=lvB,
""", """            draws.append(dict(spk=spk, box=d["box"], cls=cls, tid=tid, state=state, ttc=ttc, lvA=lvA, lvB=lvB,
""")
# 3) 끝나면 프레임별 라벨 파일
sub("""    for tag, evs in (("A", evA), ("B", evB), ("light", evL)):
""", """    with open(out / "labels_frames.jsonl", "w", encoding="utf-8") as fh_:
        for fr_, pf_ in zip(frames, per_frame_draw):
            objs_ = [dict(tid=d_["tid"], cls=d_["cls"], box=d_["box"], motion=d_["state"], speak=d_["spk"],
                          level=d_["lvB"], dist_m=None if d_["dist"] is None else round(d_["dist"], 2),
                          X=None if d_["g"] is None else round(d_["g"][0], 2))
                     for d_ in pf_["draws"]]
            fh_.write(json.dumps(dict(f=fr_["f"], t=round(pf_["t"], 3), surface=pf_["surf"], road=pf_["road"],
                                      walk_mps=round(float(pf_["walk"]), 2), objs=objs_), ensure_ascii=False) + "\\n")
    for tag, evs in (("A", evA), ("B", evB), ("light", evL)):
""")
(here / "pass2_label.py").write_text(src, encoding="utf-8")
print("wrote pass2_label.py")
