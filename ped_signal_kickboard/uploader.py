"""staging 의 완성된 shard(.zip)를 1분마다 구글 드라이브로 옮긴다(병렬 작업들은 업로드하지 않음)."""
import subprocess
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
RC = str(Path.home() / "AppData/Local/Microsoft/WinGet/Packages/Rclone.Rclone_Microsoft.Winget.Source_8wekyb3d8bbwe"
         "/rclone-v1.75.1-windows-amd64/rclone.exe")
while True:
    subprocess.run([RC, "move", str(BASE / "staging"), "gdrive:aihub_신호등_킥보드", "--exclude", "*.tmp",
                    "--transfers", "2", "--drive-chunk-size", "64M", "--min-age", "10s",
                    "--delete-empty-src-dirs", "--retries", "5", "--log-level", "ERROR",
                    "--log-file", str(BASE / "uploader.log")])
    time.sleep(60)
