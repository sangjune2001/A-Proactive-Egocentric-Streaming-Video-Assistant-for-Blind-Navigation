"""Roboflow Universe 검색. 사용: python rf_search.py "검색어" ..."""
import json
import sys
from pathlib import Path

import requests

KEY = (Path.home() / ".roboflow_key").read_text().strip()
for q in sys.argv[1:]:
    r = requests.get("https://api.roboflow.com/universe/search", params={"api_key": KEY, "q": q, "limit": 30}, timeout=60)
    print("==", q, r.status_code)
    if r.status_code != 200:
        print(r.text[:300])
        continue
    d = r.json()
    for it in d.get("results", d if isinstance(d, list) else [])[:30]:
        print(json.dumps({k: it.get(k) for k in ("url", "name", "images", "classes", "id")}, ensure_ascii=False)[:300])
