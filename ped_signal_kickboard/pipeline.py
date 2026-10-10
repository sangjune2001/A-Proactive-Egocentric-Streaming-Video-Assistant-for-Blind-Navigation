"""
AI Hub 스트리밍 필터 파이프라인
- 원천 압축파일을 디스크에 저장하지 않고 받으면서 바로 읽어서, 필요한 이미지/라벨만 staging에 쓰고
- 별도 스레드가 rclone으로 staging -> 구글 드라이브로 옮긴다(move).

남기는 대상
  71579 : 보행신호등(pedestrian_signal)이 한 번이라도 나오는 클립 전체 (신호 변화 학습용)
  188   : traffic_light 중 type == pedestrian 이 있는 이미지
  71784 : Personal Mobility(킥보드, category 99)가 있는 가시광 이미지

재실행하면 state.json 기준으로 끝난 파일키는 건너뛴다.
"""
import csv
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import zipfile
from pathlib import Path

import requests
from stream_unzip import stream_unzip

BASE = Path(__file__).resolve().parent
# Colab: AIHUB_OUT=/content/drive/MyDrive/aihub_신호등_킥보드 로 주면 드라이브에 바로 쓰고 rclone 업로드는 하지 않는다
STAGING = Path(os.environ.get("AIHUB_OUT", BASE / "staging"))
DIRECT_TO_DRIVE = "AIHUB_OUT" in os.environ
# 병렬 실행: AIHUB_TAG=71579 AIHUB_JOBS=71579 처럼 주면 진행기록(state/saved/log)을 따로 쓰고 업로드는 기본 작업에 맡김
TAG = os.environ.get("AIHUB_TAG", "")
SUFFIX = f"_{TAG}" if TAG else ""
ONLY_JOBS = [x for x in os.environ.get("AIHUB_JOBS", "").split(",") if x]
STATE_PATH = BASE / f"state{SUFFIX}.json"
KEEP_DIR = BASE / "keep"
CLAIMS = BASE / "claims"
CLAIMS.mkdir(exist_ok=True)
_key_file = BASE / "aihub_apikey.txt"
API_KEY = (_key_file if _key_file.exists() else Path.home() / ".aihub_apikey").read_text().strip()
DOWN_URL = "https://api.aihub.or.kr/down/0.6/{ds}.do?fileSn={fk}"
RCLONE = os.environ.get("RCLONE", str(Path.home() / "AppData/Local/Microsoft/WinGet/Packages/"
                         "Rclone.Rclone_Microsoft.Winget.Source_8wekyb3d8bbwe/rclone-v1.75.1-windows-amd64/rclone.exe"))
REMOTE = "gdrive:aihub_신호등_킥보드"
MIN_FREE = 2 * 1024**3          # C: 남은 용량이 이보다 적으면 업로드될 때까지 쓰기 대기
CHUNK = 1 << 20

OUT_NAME = {"71579": "71579_신호등신호정보", "188": "188_신호등표지판", "71784": "71784_생활도로_킥보드",
            "614": "614_개인형이동장치_킥보드"}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(BASE / f"pipeline{SUFFIX}.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger()


# ---------------------------------------------------------------- 파일 트리

def parse_tree(ds):
    """aihubshell 파일트리 텍스트 -> [(경로리스트, 파일명, 크기, 파일키)]"""
    leaves, stack = [], []  # stack: [(열 위치, 폴더명)]
    for line in (BASE / "trees" / f"{ds}.txt").read_text(encoding="utf-8").splitlines():
        m = re.search(r"[├└]─", line)
        if not m:
            continue
        col = m.start()
        name = line[m.end():].strip()
        while stack and stack[-1][0] >= col:
            stack.pop()
        if " | " in name:
            fname, size, key = [x.strip() for x in name.split(" | ")]
            leaves.append(([n for _, n in stack], fname, size, key))
        else:
            stack.append((col, name))
    return leaves


def build_jobs():
    """데이터셋별 (split, 라벨 파일키들, 원천 그룹들) 목록. 원천 그룹 = 이어 붙여야 하는 파일키 리스트."""
    """순서: 614 킥보드 Validation -> 188 Validation -> 188 Training(1280_720만, 보행신호 밀도 최고)
    -> 71579 Validation(나머지). 71579 Training, 71784 는 효율이 낮아 제외."""
    jobs = []
    # 614 개인형 이동장치(킥보드) Validation
    t = parse_tree("614")
    lab = [k for p, f, s, k in t if "2.Validation" in p and "라벨링데이터" in p]
    src = [[k] for p, f, s, k in t if "2.Validation" in p and "원천데이터" in p]
    jobs.append(("614", "Validation", lab, src))
    # 188
    t = parse_tree("188")
    for split, only in (("2.Validation", ""), ("1.Training", "1280_720")):
        lab = [k for p, f, s, k in t if split in p and any("라벨링데이터" in x for x in p) and only in f]
        src = [[k] for p, f, s, k in t if split in p and any("원천데이터" in x for x in p) and only in f]
        jobs.append(("188", split.split(".")[1], lab, src))
    # 71579 Validation
    t = parse_tree("71579")
    lab = [k for p, f, s, k in t if "Validation" in p and "02.라벨링데이터" in p]
    src = [k for p, f, s, k in t if "Validation" in p and "01.원천데이터" in p]
    jobs.append(("71579", "Validation", lab, [src]))
    if ONLY_JOBS:
        jobs = [j for j in jobs if j[0] in ONLY_JOBS or f"{j[0]}:{j[1]}" in ONLY_JOBS]  # 예: 188:Training
    return jobs


# ---------------------------------------------------------------- 상태

def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"done_labels": [], "done_sources": [], "bytes": 0}


def save_state(st):
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATE_PATH)


# ---------------------------------------------------------------- 스트리밍

class GenReader(io.RawIOBase):
    """바이트 청크 제너레이터를 파일 객체처럼 읽게 해준다 (tarfile 스트림 모드용)."""

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


class Meter:
    def __init__(self, label):
        self.label, self.n, self.t0, self.last = label, 0, time.time(), time.time()

    def add(self, k, st):
        self.n += k
        st["bytes"] += k
        if time.time() - self.last > 60:
            self.last = time.time()
            mbps = self.n / (time.time() - self.t0) / 1e6
            log.info(f"  [{self.label}] {self.n / 1e9:.1f} GB 받음, {mbps:.2f} MB/s")


def download_chunks(ds, fk, st, meter):
    """AI Hub 다운로드 응답(tar)을 읽어서, 안에 든 실제 파일(.partN 이어붙임)의 바이트를 순서대로 내보낸다."""
    url = DOWN_URL.format(ds=ds, fk=fk)

    def counted():
        """연결이 끊기면 다시 요청해서 이미 넘긴 바이트(pos)까지는 버리고 이어서 내보낸다.
        (AI Hub는 Range 요청을 지원하지 않아 처음부터 다시 받아야 함)"""
        pos, fails = 0, 0
        while True:
            try:
                r = requests.get(url, headers={"apikey": API_KEY}, stream=True, timeout=120)
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                skip = pos
                for c in r.iter_content(CHUNK):
                    meter.add(len(c), st)
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
                if fails > 30:
                    raise
                log.info(f"  연결 끊김 ({pos / 1e9:.2f} GB 지점), 30초 후 이어받기: {e!r}"[:300])
                time.sleep(30)

    outer = tarfile.open(fileobj=io.BufferedReader(GenReader(counted()), CHUNK), mode="r|")
    for m in outer:
        if not m.isfile():
            continue
        f = outer.extractfile(m)
        while True:
            c = f.read(CHUNK)
            if not c:
                break
            yield c


def concat_sources(ds, fks, st, meter):
    first = True
    for fk in fks:
        for c in download_chunks(ds, fk, st, meter):
            if first:
                first = False
                if c[:4] == b"PK\x07\x08":  # 분할 zip 시작 표시 제거
                    c = c[4:]
            yield c


def iter_members(gen, kind):
    """(이름, 바이트제너레이터) 를 차례로. kind: 'zip' | 'tar'"""
    if kind == "zip":
        for name, size, chunks in stream_unzip(gen, chunk_size=CHUNK):
            try:
                name = name.decode("utf-8")
            except UnicodeDecodeError:
                name = name.decode("cp949", "replace")
            yield name, chunks
    else:
        t = tarfile.open(fileobj=io.BufferedReader(GenReader(gen), CHUNK), mode="r|")
        for m in t:
            if m.isfile():
                f = t.extractfile(m)
                yield m.name, iter(lambda: f.read(CHUNK), b"")


def sniff_kind(gen):
    """첫 청크를 보고 zip/tar 판별 후, 그 청크를 다시 앞에 붙인 제너레이터 반환."""
    first = next(gen)
    kind = "zip" if first[:2] == b"PK" else "tar"

    def again():
        yield first
        yield from gen
    return kind, again()


# ---------------------------------------------------------------- 저장/업로드

def wait_for_space():
    while shutil.disk_usage(STAGING.resolve().anchor).free < MIN_FREE:
        log.info("  디스크 여유 부족 -> 업로드 대기")
        time.sleep(30)


SHARD_BYTES = 1024**3
SAVED_PATH = BASE / f"saved{SUFFIX}.txt"
SAVED = set()
for _p in BASE.glob("saved*.txt"):  # 다른 병렬 작업이 저장한 것도 중복 저장하지 않음
    SAVED |= set(_p.read_text(encoding="utf-8").replace("\\", "/").split("\n"))
_saved_f = open(SAVED_PATH, "a", encoding="utf-8")


class Shards:
    """구글 드라이브는 파일 개수가 많으면 매우 느려서, <데이터셋>/<split>/shard_*.zip (약 1GB, 무압축) 으로 묶는다.
    zip 안의 경로는 원래 폴더 구조(images/..., labels/...)를 유지한다."""

    def __init__(self):
        self.open = {}  # (ds_dir, split) -> [zipfile, tmp경로, 크기]

    def add(self, rel, chunks):
        rel = Path(rel)
        key = rel.parts[:2]
        arc = "/".join(rel.parts[2:])
        rid = rel.as_posix()  # 윈도우/리눅스(Colab) 공통 표기
        if rid in SAVED:
            drain(chunks)
            return
        if key not in self.open:
            d = STAGING.joinpath(*key)
            d.mkdir(parents=True, exist_ok=True)
            tmp = d / f"shard{SUFFIX}_{time.strftime('%Y%m%d_%H%M%S')}_{len(SAVED) % 100000:05d}.zip.tmp"
            self.open[key] = [zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED, allowZip64=True), tmp, 0]
        z = self.open[key]
        with z[0].open(arc, "w", force_zip64=True) as f:
            for c in chunks:
                f.write(c)
                z[2] += len(c)
        SAVED.add(rid)
        _saved_f.write(rid + "\n")
        if z[2] >= SHARD_BYTES:
            self.close(key)

    def close(self, key):
        z, tmp, _ = self.open.pop(key)
        z.close()
        _saved_f.flush()
        tmp.replace(tmp.with_suffix(""))  # .zip.tmp -> .zip (업로더가 가져감)

    def close_all(self):
        for key in list(self.open):
            self.close(key)


SHARDS = Shards()


def write_file(rel, chunks):
    SHARDS.add(rel, chunks)


def drain(chunks):
    for _ in chunks:
        pass


class Uploader(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop_flag = threading.Event()

    def run_once(self):
        if DIRECT_TO_DRIVE or TAG:
            return
        cmd = [RCLONE, "move", str(STAGING), REMOTE, "--exclude", "*.tmp", "--transfers", "2",
               "--drive-chunk-size", "64M",
               "--min-age", "10s", "--delete-empty-src-dirs", "--retries", "5", "--log-level", "ERROR"]
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if p.returncode != 0:
            log.info(f"  rclone 오류: {p.stderr[-500:]}")

    def run(self):
        while not self.stop_flag.is_set():
            self.run_once()
            self.stop_flag.wait(60)

    def finish(self):
        self.stop_flag.set()
        self.join()
        time.sleep(11)
        self.run_once()


# ---------------------------------------------------------------- 라벨 판정

def stem(name):
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    for ext in (".png.json", ".jpg.json", ".json", ".jpg", ".png", ".jpeg"):
        if base.lower().endswith(ext):
            return base[: -len(ext)]
    return base


def load_json(raw):
    for enc in ("utf-8-sig", "cp949"):
        try:
            return json.loads(raw.decode(enc))
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
    return None


def scan_labels(ds, split, label_keys, st):
    """라벨을 받아서 남길 이미지 stem 집합을 만들고, 남길 라벨 파일은 바로 staging에 저장."""
    keep_path = KEEP_DIR / f"{ds}_{split}.json"
    if keep_path.exists():
        return set(json.loads(keep_path.read_text(encoding="utf-8")))
    out = Path(OUT_NAME[ds]) / split
    keep = set()
    clip_rows = {}   # 71579: 클립별 보행신호 상태 기록
    pending = {}     # 71579: 클립 단위로 모았다가 보행신호가 있으면 라벨 저장
    for fk in label_keys:
        meter = Meter(f"{ds} {split} 라벨 {fk}")
        kind, gen = sniff_kind(concat_sources(ds, [fk], st, meter))
        for name, chunks in iter_members(gen, kind):
            if not name.endswith(".json"):
                drain(chunks)
                continue
            raw = b"".join(chunks)
            d = load_json(raw)
            if d is None:
                continue
            s = stem(name)
            if ds == "188":
                if any(a.get("class") == "traffic_light" and a.get("type") == "pedestrian"
                       for a in d.get("annotation", [])):
                    keep.add(s)
                    wait_for_space()
                    write_file(out / "labels" / name.lstrip("./"), [raw])
            elif ds == "614":
                ann = d.get("annotations") or {}
                if isinstance(ann, dict) and ann.get("PM"):
                    keep.add(s)
                    wait_for_space()
                    write_file(out / "labels" / name.lstrip("./"), [raw])
            elif ds == "71784":
                if any(a.get("category_id") == 99 for a in d.get("annotations", [])):
                    keep.add(stem(d.get("png_filename", s)))
                    wait_for_space()
                    write_file(out / "labels" / name.lstrip("./"), [raw])
            elif ds == "71579":
                clip = name.replace("\\", "/").split("/")[0]
                ped = [o["attribute"].get("signal") for o in d.get("objects", [])
                       if o.get("class_name") == "pedestrian_signal"]
                pending.setdefault(clip, []).append((s, name, raw, ped, d.get("flags", {}).get("v2")))
        log.info(f"  라벨 {fk} 완료 (지금까지 남길 이미지 {len(keep):,}장)")

    if ds == "71579":
        for clip, frames in pending.items():
            frames.sort()
            if not any(f[3] for f in frames):
                continue
            states = [",".join(sorted(f[3])) if f[3] else "-" for f in frames]
            changes = sum(1 for a, b in zip(states, states[1:]) if a != b and a != "-" and b != "-")
            clip_rows[clip] = [clip, len(frames), changes, " > ".join(dedup(states)),
                               ",".join(sorted({str(f[4]) for f in frames}))]
            for s, name, raw, _, _ in frames:
                keep.add(s)
                keep.add(clip_key(s))  # 라벨 없는 중간 프레임도 같은 클립이면 남긴다
                write_file(out / "labels" / name, [raw])
        csv_path = STAGING / out / "clips_pedestrian_signal.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["clip", "frames", "ped_signal_changes", "ped_signal_sequence", "v2_flags"])
            w.writerows(sorted(clip_rows.values(), key=lambda r: -r[2]))
        log.info(f"  71579 {split}: 보행신호 클립 {len(clip_rows):,}개, 그중 신호 변화 있는 클립 "
                 f"{sum(1 for r in clip_rows.values() if r[2] > 0):,}개")

    KEEP_DIR.mkdir(exist_ok=True)
    keep_path.write_text(json.dumps(sorted(keep)), encoding="utf-8")
    return keep


def clip_key(s):
    """512_DD_0034_CF_015 -> CLIP:512_DD_0034_CF"""
    return "CLIP:" + s.rsplit("_", 1)[0]


def dedup(seq):
    out = []
    for x in seq:
        if not out or out[-1] != x:
            out.append(x)
    return out


# ---------------------------------------------------------------- 원천 추출

def extract_sources(ds, split, group, keep, st):
    out = Path(OUT_NAME[ds]) / split / "images"
    meter = Meter(f"{ds} {split} 원천 {'+'.join(group)}")
    kind, gen = sniff_kind(concat_sources(ds, group, st, meter))
    kept = seen = 0
    for name, chunks in iter_members(gen, kind):
        seen += 1
        s = stem(name)
        hit = s in keep or (ds == "71579" and clip_key(s) in keep)
        if hit and name.lower().endswith((".jpg", ".jpeg", ".png")):
            wait_for_space()
            write_file(out / name.lstrip("./"), chunks)
            kept += 1
        else:
            drain(chunks)
    log.info(f"  원천 {'+'.join(group)} 완료: {seen:,}개 중 {kept:,}개 저장")


def main():
    STAGING.mkdir(exist_ok=True)
    st = load_state()
    up = Uploader()
    up.start()
    try:
        for ds, split, lab, groups in build_jobs():
            log.info(f"=== {ds} {split}: 라벨 {len(lab)}개, 원천 그룹 {len(groups)}개")
            keep = None
            for attempt in range(5):
                try:
                    keep = scan_labels(ds, split, lab, st)
                    break
                except Exception as e:  # 네트워크 끊김 등
                    log.info(f"  라벨 처리 실패({attempt + 1}/5): {e!r}")
                    time.sleep(60)
            SHARDS.close_all()
            if keep is None:
                continue
            save_state(st)
            log.info(f"  남길 이미지 {len(keep):,}장")
            if not keep:
                continue
            for group in groups:
                gid = f"{ds}:{'+'.join(group)}"
                if gid in st["done_sources"]:
                    continue
                # 여러 작업이 같은 목록을 나눠 받을 때: 먼저 claim 파일을 만든 작업만 이 묶음을 받는다
                claim = CLAIMS / gid.replace(":", "_").replace("+", "_")
                try:
                    os.close(os.open(claim, os.O_CREAT | os.O_EXCL))
                except FileExistsError:
                    continue
                for attempt in range(5):
                    try:
                        extract_sources(ds, split, group, keep, st)
                        SHARDS.close_all()
                        st["done_sources"].append(gid)
                        save_state(st)
                        break
                    except Exception as e:
                        log.info(f"  원천 {gid} 실패({attempt + 1}/5): {e!r}")
                        time.sleep(60)
                else:
                    claim.unlink(missing_ok=True)  # 5번 다 실패 → 다른 작업이 다시 시도할 수 있게
        log.info("전체 완료")
    finally:
        SHARDS.close_all()
        save_state(st)
        up.finish()


if __name__ == "__main__":
    main()
