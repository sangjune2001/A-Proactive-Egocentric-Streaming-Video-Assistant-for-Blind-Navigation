"""188t(구버전 코드, PID 28452)가 지금 받는 묶음(49279)을 끝내면 멈추고, claim 방식 새 코드로 다시 띄운다.
이후 188 Validation / 71579 가 끝난 자리에서도 188 Training 남은 묶음을 나눠 받는다(queue_after)."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
OLD_PID, WAIT_FOR = 28452, "원천 49279 완료"
log = BASE / "pipeline_188t.log"
while WAIT_FOR not in log.read_text(encoding="utf-8", errors="ignore"):
    time.sleep(20)
subprocess.run(["taskkill", "/F", "/PID", str(OLD_PID)], capture_output=True)
time.sleep(3)
subprocess.run([sys.executable, "-X", "utf8", str(BASE / "migrate.py"), "188t"], cwd=BASE)

claims = BASE / "claims"
claims.mkdir(exist_ok=True)
for sp in BASE.glob("state*.json"):
    for gid in json.loads(sp.read_text(encoding="utf-8")).get("done_sources", []):
        (claims / gid.replace(":", "_").replace("+", "_")).touch()

env = dict(os.environ, AIHUB_TAG="188t", AIHUB_JOBS="188:Training")
flags = subprocess.CREATE_NO_WINDOW
with open(BASE / "stdout_188t.log", "a") as so, open(BASE / "stderr_188t.log", "a") as se:
    p = subprocess.Popen([sys.executable, "-u", "-X", "utf8", str(BASE / "pipeline.py")], env=env, cwd=BASE,
                         stdout=so, stderr=se, creationflags=flags)
(BASE / "switch_188t.done").write_text(str(p.pid))
p.wait()
