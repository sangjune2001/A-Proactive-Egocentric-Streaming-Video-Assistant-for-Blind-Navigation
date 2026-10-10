"""지정한 PID(앞 작업)가 끝나면 다음 작업들을 차례로 실행한다.
사용: python queue_after.py <PID> <TAG:JOBS> [<TAG:JOBS> ...]   예) python queue_after.py 39236 188v=188:Validation"""
import os
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
pid = int(sys.argv[1])
while "python" in subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True,
                                 encoding="cp949", errors="ignore").stdout:
    time.sleep(60)
for spec in sys.argv[2:]:
    tag, jobs = spec.split("=", 1)
    env = dict(os.environ, AIHUB_TAG=tag, AIHUB_JOBS=jobs)
    with open(BASE / f"stdout_{tag}.log", "a") as so, open(BASE / f"stderr_{tag}.log", "a") as se:
        subprocess.run([sys.executable, "-u", "-X", "utf8", str(BASE / "pipeline.py")], env=env, stdout=so, stderr=se,
                       cwd=BASE)
