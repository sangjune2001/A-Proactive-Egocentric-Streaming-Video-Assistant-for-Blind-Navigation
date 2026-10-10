"""
AI Hub 데이터셋에서 신호등 라벨과 영상(미디어) 유형을 점검하는 스크립트.

사용법:
    python inspect_traffic_light.py <데이터셋_루트폴더> [--max-labels 5000] [--samples 3]

예:
    python inspect_traffic_light.py D:/aihub/71579
    python inspect_traffic_light.py D:/aihub/188 --max-labels 20000

하는 일:
  1) 파일 확장자별 개수 -> 실제 동영상(mp4/avi 등)인지, 프레임 이미지인지 확인
  2) 폴더별로 연속 번호 프레임이 있는지 -> "클립(시퀀스)" 형태인지 판정
  3) JSON/XML 라벨을 훑어 신호등 객체를 찾고, 키/클래스/신호 상태 값 분포 출력
  4) 클립 단위로 프레임별 신호 상태를 이어 붙여 '신호 변화(예: green->yellow)'가 있는 클립을 찾음
"""
import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".ts", ".h264", ".mpg", ".mpeg"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
LABEL_EXT = {".json", ".xml", ".csv"}

# 신호등으로 판단할 키워드 (데이터셋마다 표기가 다름)
#  188: class="traffic_light" / 71579: class_name="vehicular_signal" 등
#  71784: categories name="Traffic Light" / 71786: name(시설물 종류)
TL_PATTERN = re.compile(r"traffic[_ ]?light|signal|신호등", re.IGNORECASE)
# 신호 상태 값으로 볼 만한 키
STATE_KEYS = ("signal", "state", "color", "light", "attribute", "status", "type")


def scan_files(root: Path):
    ext_count = Counter()
    frames_by_dir = defaultdict(list)
    labels = []
    videos = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        ext_count[ext] += 1
        if ext in IMAGE_EXT:
            frames_by_dir[p.parent].append(p.name)
        elif ext in VIDEO_EXT:
            videos.append(p)
        elif ext in LABEL_EXT:
            labels.append(p)
    return ext_count, frames_by_dir, labels, videos


def report_media(ext_count, frames_by_dir, videos, samples):
    print("\n=== 1. 확장자별 파일 수 ===")
    for ext, n in ext_count.most_common():
        kind = ("동영상" if ext in VIDEO_EXT else "이미지" if ext in IMAGE_EXT
                else "라벨" if ext in LABEL_EXT else "기타")
        print(f"  {ext or '(없음)':8s} {n:>10,}  [{kind}]")

    print("\n=== 2. 영상 유형 판정 ===")
    if videos:
        print(f"  실제 동영상 파일 {len(videos):,}개 발견")
        for v in videos[:samples]:
            print("   -", v, video_info(v))
    else:
        print("  동영상 파일 없음 -> 이미지(프레임) 기반 데이터")

    # 같은 폴더 안에 숫자만 다른 파일이 여러 개면 연속 프레임(클립)으로 간주
    seq_dirs = {}
    for d, names in frames_by_dir.items():
        stems = Counter(re.sub(r"\d+(?=\D*$)", "#", Path(n).stem) for n in names)
        stem, cnt = stems.most_common(1)[0]
        if cnt >= 5:
            seq_dirs[d] = cnt
    total_dirs = len(frames_by_dir)
    print(f"  이미지가 있는 폴더 {total_dirs:,}개 중 연속 프레임(클립) 형태로 보이는 폴더: {len(seq_dirs):,}개")
    if seq_dirs:
        lens = sorted(seq_dirs.values())
        print(f"  클립당 프레임 수: 최소 {lens[0]}, 중앙 {lens[len(lens)//2]}, 최대 {lens[-1]}")
        for d in list(seq_dirs)[:samples]:
            names = sorted(frames_by_dir[d])
            print(f"   - {d}  ({len(names)}장)  {names[0]} ... {names[-1]}")
    return seq_dirs


def video_info(path: Path):
    try:
        import cv2
    except ImportError:
        return "(opencv 미설치: pip install opencv-python 로 fps/길이 확인 가능)"
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    dur = n / fps if fps else 0
    return f"{int(w)}x{int(h)}, {fps:.1f}fps, {int(n)}프레임, {dur:.1f}초"


# ---------- 라벨 파싱 ----------

def load_label(path: Path):
    if path.suffix.lower() == ".json":
        for enc in ("utf-8", "utf-8-sig", "cp949"):
            try:
                with open(path, encoding=enc) as f:
                    return json.load(f)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        return None
    if path.suffix.lower() == ".xml":
        try:
            return xml_to_dict(ET.parse(path).getroot())
        except ET.ParseError:
            return None
    return None  # CSV는 아래에서 헤더만 따로 출력


def xml_to_dict(el):
    d = dict(el.attrib)
    for child in el:
        d.setdefault(child.tag, []).append(xml_to_dict(child))
    if el.text and el.text.strip():
        d["_text"] = el.text.strip()
    return d


def find_traffic_lights(obj, category_names=None):
    """라벨 구조를 재귀로 돌며 신호등으로 보이는 dict를 모두 찾는다."""
    found = []
    if isinstance(obj, dict):
        # COCO 형식(71784 등): category_id -> 이름 매핑으로 판정
        if category_names is not None and "category_id" in obj:
            if TL_PATTERN.search(str(category_names.get(obj["category_id"], ""))):
                found.append(obj)
        elif any(isinstance(v, str) and TL_PATTERN.search(v) for v in obj.values()):
            found.append(obj)
        for v in obj.values():
            found.extend(find_traffic_lights(v, category_names))
    elif isinstance(obj, list):
        for v in obj:
            found.extend(find_traffic_lights(v, category_names))
    return found


def coco_categories(data):
    if isinstance(data, dict) and isinstance(data.get("categories"), list):
        return {c.get("id"): c.get("name") for c in data["categories"] if isinstance(c, dict)}
    return None


def state_of(tl: dict):
    """신호 상태를 문자열 하나로 요약 (프레임 간 비교용)."""
    parts = []
    for k in STATE_KEYS:
        if k in tl and k != "type":
            v = tl[k]
            parts.append(f"{k}={json.dumps(v, ensure_ascii=False, sort_keys=True)}")
    return ";".join(parts) or None


def report_labels(labels, max_labels, samples):
    print("\n=== 3. 라벨 파일 분석 ===")
    csvs = [p for p in labels if p.suffix.lower() == ".csv"]
    parsable = [p for p in labels if p.suffix.lower() != ".csv"]
    print(f"  JSON/XML {len(parsable):,}개, CSV {len(csvs):,}개 (분석 대상 최대 {max_labels:,}개)")

    for c in csvs[:samples]:
        try:
            with open(c, encoding="utf-8-sig", errors="replace") as f:
                print(f"   CSV 헤더 [{c.name}]: {f.readline().strip()[:200]}")
        except OSError:
            pass

    key_count = Counter()
    value_count = defaultdict(Counter)
    files_with_tl = 0
    tl_total = 0
    sample_shown = 0
    # 클립(폴더)별 -> [(파일명, 상태들)] : 신호 변화 탐지용
    clip_states = defaultdict(list)
    scenario_values = Counter()

    for path in sorted(parsable)[:max_labels]:
        data = load_label(path)
        if data is None:
            continue
        tls = find_traffic_lights(data, coco_categories(data))
        # 71579의 v2(신호등 변화 시나리오)처럼 파일 단위 메타가 있으면 수집
        if isinstance(data, dict):
            for k in ("v2", "scenario", "signal_change"):
                if k in data:
                    scenario_values[f"{k}={data[k]}"] += 1
        if not tls:
            continue
        files_with_tl += 1
        tl_total += len(tls)
        states = []
        for tl in tls:
            key_count.update(tl.keys())
            for k, v in tl.items():
                if isinstance(v, (str, int, float, bool)) and len(str(v)) < 40:
                    value_count[k][str(v)] += 1
            s = state_of(tl)
            if s:
                states.append(s)
        clip_states[path.parent].append((path.name, tuple(sorted(states))))
        if sample_shown < samples:
            print(f"\n  [샘플] {path}")
            print("   ", json.dumps(tls[0], ensure_ascii=False)[:600])
            sample_shown += 1

    print(f"\n  신호등 포함 라벨 파일: {files_with_tl:,}개 / 신호등 객체: {tl_total:,}개")
    if not tl_total:
        print("  신호등 객체를 못 찾음 -> 샘플 라벨을 직접 열어 키 이름을 확인하고 TL_PATTERN을 수정하세요.")
        return clip_states

    print("\n  신호등 객체의 키 빈도:")
    for k, n in key_count.most_common(30):
        print(f"    {k:20s} {n:>8,}")
    print("\n  주요 키별 값 분포 (상위 10개):")
    for k in key_count:
        if k in value_count and 1 < len(value_count[k]) <= 200:
            top = ", ".join(f"{v}({n})" for v, n in value_count[k].most_common(10))
            print(f"    {k}: {top}")
    if scenario_values:
        print("\n  파일 단위 시나리오 값:")
        for v, n in scenario_values.most_common(20):
            print(f"    {v}: {n}")
    return clip_states


def report_transitions(clip_states, samples):
    print("\n=== 4. 클립 내 신호 변화 탐지 ===")
    multi = {d: v for d, v in clip_states.items() if len(v) >= 2}
    if not multi:
        print("  같은 폴더에 신호등 라벨이 2개 이상인 클립이 없음 -> 단일 이미지 데이터로 보임")
        return
    changed = {}
    for d, seq in multi.items():
        seq.sort()
        trans = [(a[0], b[0], a[1], b[1]) for a, b in zip(seq, seq[1:]) if a[1] != b[1] and a[1] and b[1]]
        if trans:
            changed[d] = trans
    print(f"  라벨이 연속으로 있는 클립 {len(multi):,}개 중 신호 상태가 바뀌는 클립: {len(changed):,}개")
    for d, trans in list(changed.items())[:samples]:
        print(f"   - {d}  (변화 {len(trans)}회)")
        f1, f2, s1, s2 = trans[0]
        print(f"       {f1} -> {f2}")
        print(f"       {list(s1)[:2]}  =>  {list(s2)[:2]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--max-labels", type=int, default=5000, help="파싱할 라벨 파일 최대 개수")
    ap.add_argument("--samples", type=int, default=3)
    args = ap.parse_args()
    if not args.root.is_dir():
        sys.exit(f"폴더가 없음: {args.root}")

    print(f"데이터셋 루트: {args.root}")
    ext_count, frames_by_dir, labels, videos = scan_files(args.root)
    report_media(ext_count, frames_by_dir, videos, args.samples)
    clip_states = report_labels(labels, args.max_labels, args.samples)
    report_transitions(clip_states, args.samples)


if __name__ == "__main__":
    main()
