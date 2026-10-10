"""Roboflow 프로젝트 정보(이미지 수, 클래스, 버전) 조회. 사용: python rf_info.py ws/proj [ws/proj ...]"""
import json
import sys
from pathlib import Path

import requests

KEY = (Path.home() / ".roboflow_key").read_text().strip()
for slug in sys.argv[1:]:
    r = requests.get(f"https://api.roboflow.com/{slug}", params={"api_key": KEY}, timeout=60)
    if r.status_code != 200:
        print(slug, "HTTP", r.status_code, r.text[:200])
        continue
    d = r.json()
    p = d.get("project", {})
    vers = [(v.get("id", "").rsplit("/", 1)[-1], v.get("images"), v.get("exports")) for v in d.get("versions", [])]
    print(json.dumps({"slug": slug, "name": p.get("name"), "type": p.get("type"), "images": p.get("images"),
                      "classes": p.get("classes"), "license": p.get("license"), "versions": vers[:5]},
                     ensure_ascii=False))
