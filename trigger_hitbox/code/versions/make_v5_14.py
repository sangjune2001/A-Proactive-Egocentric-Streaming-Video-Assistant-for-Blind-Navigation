"""화살표 위쪽(멀어짐) 경고 제외 보강 (10/15 사용자): v5_12 → v5_14, v5_13 → v5_15.
화살표 방향은 트랙 생기고 1 s 뒤에 정해지는데 경고는 0.25 s부터 나가서, 방향이 정해지기 전에 경고하고
나중에 화살표가 위로 그려지던 문제 (test_1 6건 · test_3 4건). 방향이 아직 없으면 최근 0.3 s 이상 발 경로로
임시 방향을 구해 위쪽(±80°)이면 경고 안 함.
usage: python make_v5_14.py SRC DST VERSION_NAME"""
import sys
from pathlib import Path

src, dst, ver = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
s = src.read_text(encoding="utf-8")
old = '''                lk = tr.lock_v
                going_up = (lk is not None'''
assert s.count(old) == 1
s = s.replace(old, '''                lk = tr.lock_v
                if lk is None:              # 방향이 아직 안 정해짐 → 최근 발 경로로 임시 방향 ({ver})
                    pp_ = [p for p in tr.hist if t - p[0] <= LOCK_WIN_S]
                    if len(pp_) >= 4 and pp_[-1][0] - pp_[0][0] >= 0.3:
                        a_ = np.asarray(pp_, float)
                        tt_ = a_[:, 0] - a_[:, 0].mean()
                        den_ = float((tt_ ** 2).sum())
                        if den_ > 0:
                            lk = (float((tt_ * (a_[:, 1] - a_[:, 1].mean())).sum() / den_),
                                  float((tt_ * (a_[:, 2] - a_[:, 2].mean())).sum() / den_))
                going_up = (lk is not None'''.replace("{ver}", ver))
s = s.replace('"""v5_1', f'"""{ver} (10/15): 아래 버전 + 화살표 위쪽이면 방향 확정 전에도 경고 안 함.\nv5_1', 1)
dst.write_text(s, encoding="utf-8")
print("ok", ver)
