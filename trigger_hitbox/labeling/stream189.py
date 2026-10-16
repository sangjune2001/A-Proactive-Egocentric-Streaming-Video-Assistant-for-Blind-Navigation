"""AI Hub 189(인도보행영상) 원본 zip을 PC 디스크에 저장하지 않고 그대로 gd745800:Video Dataset/ 로 스트리밍 업로드.

AI Hub 응답(tar 안에 .part0, .part1 ...)을 이어 붙여 rclone rcat 의 stdin으로 넘긴다.
Range 미지원이라 실패하면 그 파일은 처음부터 다시. 끝난 파일키는 state_189up.json 에 기록.
실행: python -u -X utf8 stream189.py
"""
import io
import json
import logging
import os
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import requests

BASE = Path(__file__).resolve().parent
API_KEY = (BASE / "aihub_apikey.txt").read_text().strip()
RCLONE = os.environ.get("RCLONE") or r"C:\Users\windo\AppData\Local\Microsoft\WinGet\Packages\Rclone.Rclone_Microsoft.Winget.Source_8wekyb3d8bbwe\rclone-v1.75.1-windows-amd64\rclone.exe"
REMOTE = "gd745800:Video Dataset/AIHub_189_인도보행영상"
URL = "https://api.aihub.or.kr/down/0.6/189.do?fileSn={fk}"
STATE = BASE / "state_189up.json"
CHUNK = 8 << 20
# 업로드 순서: 바운딩박스(영상) → 뎁스 → 서피스 → 폴리곤
ORDER = ["바운딩박스", "뎁스프리딕션", "서피스마스킹", "폴리곤세그멘테이션"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    handlers=[logging.FileHandler(BASE / f"{Path(sys.argv[0]).stem or 'stream189'}.log", encoding="utf-8"),
                              logging.StreamHandler(sys.stdout)])
log = logging.info


def parse_tree():
    jobs, folder = [], None
    for line in (BASE / "trees" / "189.txt").read_text(encoding="utf-8").splitlines():
        m = re.search(r"[├└]─(.+?) \| (\d+) GB \| (\d+)", line)
        if m:
            jobs.append((folder, m.group(1).strip(), int(m.group(3))))
            continue
        m = re.search(r"[├└]─(\S+)\s*$", line)
        if m and m.group(1) in ORDER:
            folder = m.group(1)
    jobs.sort(key=lambda j: ORDER.index(j[0]))
    return jobs


class GenReader(io.RawIOBase):
    def __init__(self, gen):
        self.gen, self.buf = gen, b""

    def readable(self):
        return True

    def readinto(self, b):
        while not self.buf:
            try:
                self.buf = next(self.gen)
            except StopIteration:
                return 0
        n = min(len(b), len(self.buf))
        b[:n], self.buf = self.buf[:n], self.buf[n:]
        return n


def resumable(fk):
    """연결이 끊기면 다시 요청해서 이미 넘긴 바이트(pos)까지 버리고 이어서 내보낸다(Range 미지원)."""
    pos, fails = 0, 0
    while True:
        try:
            r = requests.get(URL.format(fk=fk), headers={"apikey": API_KEY}, stream=True, timeout=120)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            skip = pos
            for c in r.iter_content(CHUNK):
                if skip:
                    if len(c) <= skip:
                        skip -= len(c)
                        continue
                    c, skip = c[skip:], 0
                pos += len(c)
                yield c
            return
        except (requests.ConnectionError, requests.Timeout,
                requests.exceptions.ChunkedEncodingError) as e:
            fails += 1
            if fails > 50:
                raise
            log(f"  연결 끊김 ({pos / 1e9:.2f} GB 지점), 20초 후 이어받기: {e!r}"[:200])
            time.sleep(20)


def upload_one(folder, name, fk):
    dst = f"{REMOTE}/{folder}/{name}"
    p = subprocess.Popen([RCLONE, "rcat", dst, "--drive-chunk-size", "64M", "--retries", "5"],
                         stdin=subprocess.PIPE)
    n, t0, last = 0, time.time(), 0
    try:
        outer = tarfile.open(fileobj=io.BufferedReader(GenReader(resumable(fk)), CHUNK), mode="r|")
        expect = 0
        for m in outer:
            if not m.isfile():
                continue
            pm = re.search(r"\.part(\d+)$", m.name)
            # partN 의 N은 순번 또는 바이트 오프셋(part0, part1073741824, ...)
            if pm and int(pm.group(1)) not in (expect, n):
                raise RuntimeError(f"part 순서 이상: {m.name} (기대 part{expect} 또는 part{n})")
            expect += 1
            f = outer.extractfile(m)
            while True:
                c = f.read(CHUNK)
                if not c:
                    break
                p.stdin.write(c)
                n += len(c)
                if n - last > 1 << 30:
                    last = n
                    log(f"  {name}: {n / 1e9:.1f} GB, {n / 1e6 / (time.time() - t0):.1f} MB/s")
        p.stdin.close()
        if p.wait() != 0:
            raise RuntimeError(f"rclone rcat 실패 (code {p.returncode})")
    except BaseException:
        p.kill()
        raise
    return n


def main():
    st = json.loads(STATE.read_text()) if STATE.exists() else {"done": []}
    for folder, name, fk in parse_tree():
        if fk in st["done"]:
            continue
        for attempt in range(1, 6):
            log(f"시작 {folder}/{name} (fk {fk}, 시도 {attempt})")
            try:
                n = upload_one(folder, name, fk)
            except Exception as e:
                log(f"  실패: {e!r}"[:300])
                time.sleep(60)
                continue
            st["done"].append(fk)
            STATE.write_text(json.dumps(st))
            log(f"완료 {folder}/{name} {n / 1e9:.2f} GB")
            break
        else:
            log(f"포기 {folder}/{name}")
    log("전체 끝")


if __name__ == "__main__":
    main()
