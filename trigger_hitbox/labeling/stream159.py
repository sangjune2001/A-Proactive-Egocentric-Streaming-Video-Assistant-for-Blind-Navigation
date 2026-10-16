"""AI Hub 159(1인칭 시점 보행영상) BBOX 중 평균 키(Average_stature, Average_staure_2) 실외(out)만
원천 + 라벨을 gd745800:Video Dataset/AIHub_159_1인칭보행영상/ 로 스트리밍 업로드 (stream189.py 방식 재사용).
실행: python -u -X utf8 stream159.py
"""
import json
import os
import re
import threading
import time

import stream189 as s

s.URL = "https://api.aihub.or.kr/down/0.6/159.do?fileSn={fk}"
s.REMOTE = "gd745800:Video Dataset/AIHub_159_1인칭보행영상"
STATE = s.BASE / "state_159up.json"
log = s.log
LOCK = threading.Lock()


def parse_tree():
    jobs, stack = [], []
    for line in (s.BASE / "trees" / "159.txt").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([\s│]*)[├└]─(.+?)\s*$", line)
        if not m:
            continue
        # 들여쓰기 폭이 단계마다 3~4칸으로 달라서 (들여쓰기, 이름) 스택으로 부모를 찾는다
        indent, name = len(m.group(1)), m.group(2)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        f = re.match(r"(.+?) \| [\d.]+ (?:GB|MB|KB) \| (\d+)$", name)
        if not f:
            stack.append((indent, name))
            continue
        # stack: [109.1인칭.., 01.데이터, 1.Training|2.Validation, BBOX, 원천|라벨, (추가_비식별화이미지_0730), 키, in/out]
        parts = [n for _, n in stack[2:]]
        if (len(parts) >= 4 and parts[1] == "BBOX"
                and any(p in ("Average_stature", "Average_staure_2", "Average_stature_2") for p in parts)
                and parts[-1] == "out"):
            jobs.append(("/".join(parts), f.group(1).strip(), int(f.group(2))))
    # 라벨(작음) 먼저, 그다음 원천
    jobs.sort(key=lambda j: "원천" in j[0])
    return jobs


def main():
    # 여러 줄 동시 실행: 각 줄은 claims/<fk> 를 먼저 만든 묶음만 받는다. 완료 기록은 state_159up.json 하나에 합침.
    claims = s.BASE / "claims159"
    claims.mkdir(exist_ok=True)
    for folder, name, fk in parse_tree():
        st = json.loads(STATE.read_text()) if STATE.exists() else {"done": []}
        if fk in st["done"]:
            continue
        try:
            os.close(os.open(claims / str(fk), os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            continue
        for attempt in range(1, 6):
            log(f"시작 {folder}/{name} (fk {fk}, 시도 {attempt})")
            try:
                n = s.upload_one(folder, name, fk)
            except Exception as e:
                log(f"  실패: {e!r}"[:300])
                time.sleep(60)
                continue
            with LOCK:
                st = json.loads(STATE.read_text()) if STATE.exists() else {"done": []}
                st["done"].append(fk)
                STATE.write_text(json.dumps(st))
            log(f"완료 {folder}/{name} {n / 1e9:.2f} GB")
            break
        else:
            log(f"포기 {folder}/{name}")
            (claims / str(fk)).unlink(missing_ok=True)
    log("전체 끝")


if __name__ == "__main__":
    n = int(os.environ.get("WORKERS", "1"))
    ts = [threading.Thread(target=main) for _ in range(n)]
    for t in ts:
        t.start()
        time.sleep(5)
    for t in ts:
        t.join()
