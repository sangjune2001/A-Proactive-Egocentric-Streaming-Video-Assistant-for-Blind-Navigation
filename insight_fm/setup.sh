#!/usr/bin/env bash
# One-time setup on the Mondrian (RTX A5000) instance. Re-running is safe.
#   bash setup.sh            # everything
#   STEP=data bash setup.sh  # only one step: deps | data | hf | prefetch | selftest | build
set -euo pipefail

# ---------------------------------------------------------------- edit these if your Drive layout differs
RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive:sideguide}"   # Drive folder that holds labels_all.jsonl
JSONL_NAME="${JSONL_NAME:-labels_all.jsonl}"
SHARDS_SUBDIR="${SHARDS_SUBDIR:-shards}"              # Drive subfolder with the 640px image tar shards
DATA="${DATA:-/data/sg}"                              # local disk on the instance (NOT a Drive mount)
export HF_HOME="${HF_HOME:-$DATA/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-$DATA/torch_cache}"
# ----------------------------------------------------------------------------------------------------------

HERE="$(cd "$(dirname "$0")" && pwd)"
STEP="${STEP:-all}"
mkdir -p "$DATA" "$HF_HOME" "$TORCH_HOME"
run() { [[ "$STEP" == "all" || "$STEP" == "$1" ]]; }

if run deps; then
  echo "== deps"
  nvidia-smi --query-gpu=name,memory.total --format=csv
  pip install -U "ultralytics>=8.3.0" "transformers>=4.56" timm huggingface_hub einops
  command -v rclone >/dev/null || curl -fsSL https://rclone.org/install.sh | sudo bash
  command -v tmux >/dev/null || (sudo apt-get update -y && sudo apt-get install -y tmux) || true
  python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
fi

if run data; then
  echo "== data from $RCLONE_REMOTE"
  rclone listremotes | grep -q "^${RCLONE_REMOTE%%:*}:" || {
    echo "rclone remote '${RCLONE_REMOTE%%:*}' not configured."
    echo "Copy it from WSL:  scp ~/.config/rclone/rclone.conf <server>:~/.config/rclone/rclone.conf"
    exit 1; }
  rclone copy "$RCLONE_REMOTE/$JSONL_NAME" "$DATA" -P
  rclone copy "$RCLONE_REMOTE/$SHARDS_SUBDIR" "$DATA/shards" -P --transfers 8 --checkers 16
  mkdir -p "$DATA/imgs"
  for f in "$DATA"/shards/*.tar; do
    [[ -e "$f.extracted" ]] && continue
    echo "  extracting $(basename "$f")"
    tar -xf "$f" -C "$DATA/imgs" && touch "$f.extracted"
  done
  echo "  images: $(find "$DATA/imgs" -type f \( -iname '*.jpg' -o -iname '*.png' \) | wc -l)"
fi

if run hf; then
  echo "== Hugging Face login (DINOv3 is gated: accept the licence on its HF page first)"
  if [[ -n "${HF_TOKEN:-}" ]]; then
    python -c "import os; from huggingface_hub import login; login(token=os.environ['HF_TOKEN'])"
  else
    echo "  HF_TOKEN not set -> skipping login (DINOv3 runs will fail until you set it)"
  fi
fi

if run prefetch; then
  echo "== downloading all encoder weights once (also a smoke test on the GPU)"
  (cd "$HERE" && python encoders.py --prefetch all)
  (cd "$HERE" && python -c "from ultralytics import YOLO; YOLO('yolo11s-seg.pt')")
fi

if run selftest; then
  echo "== code self-test on synthetic data"
  (cd "$HERE" && FM_RANDOM_INIT=1 python selftest.py > "$DATA/selftest.log" 2>&1 && tail -3 "$DATA/selftest.log") \
    || { echo "selftest failed, see $DATA/selftest.log"; exit 1; }
fi

if run build; then
  echo "== build YOLO dataset (inspect first)"
  (cd "$HERE" && python build_dataset.py --jsonl "$DATA/$JSONL_NAME" --images "$DATA/imgs" --inspect)
  (cd "$HERE" && python build_dataset.py --jsonl "$DATA/$JSONL_NAME" --images "$DATA/imgs" --out "$DATA/yolo")
fi

echo "== setup done. Next:  tmux new -s exp   then   python run_all.py --phase all"
