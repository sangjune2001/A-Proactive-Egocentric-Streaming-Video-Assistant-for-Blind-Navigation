#!/usr/bin/env bash
# One-time setup on the Mondrian (RTX A5000) instance. Re-running is safe.
#   bash setup.sh            # everything
#   STEP=data bash setup.sh  # only one step: deps | data | hf | prefetch | selftest | build
set -euo pipefail

# ---------------------------------------------------------------- edit these if your Drive layout differs
JSONL_REMOTE="${JSONL_REMOTE:-gdrive:aihub189_yolo/labels_all.jsonl}"  # relabelled 10-class labels
ZIP_REMOTE="${ZIP_REMOTE:-gdrive:sideguide/polygon}"                    # AI Hub 189 P1.zip ... P14.zip
DATA="${DATA:-$HOME/sg}"                                               # local disk on the instance
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
  echo "== data: labels from $JSONL_REMOTE, images from $ZIP_REMOTE"
  rclone listremotes | grep -q "^${ZIP_REMOTE%%:*}:" || {
    echo "rclone remote '${ZIP_REMOTE%%:*}' not configured."
    echo "From the laptop:  scp -i <key.pem> ~/.config/rclone/rclone.conf ubuntu@<server>:~/.config/rclone/"
    exit 1; }
  rclone copy "$JSONL_REMOTE" "$DATA" -P
  echo "  frames in jsonl: $(wc -l < "$DATA/labels_all.jsonl")   free disk: $(df -h "$DATA" | awk 'NR==2{print $4}')"
  # one zip at a time: download -> extract labelled frames at 640px -> delete zip (resumable)
  (cd "$HERE" && python extract_images.py --jsonl "$DATA/labels_all.jsonl" --remote "$ZIP_REMOTE" \
      --zips "$DATA/zips" --out "$DATA/imgs" --delete-zip)
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
  (cd "$HERE" && python build_dataset.py --jsonl "$DATA/labels_all.jsonl" --images "$DATA/imgs" --inspect)
  (cd "$HERE" && python build_dataset.py --jsonl "$DATA/labels_all.jsonl" --images "$DATA/imgs" --out "$DATA/yolo")
fi

echo "== setup done. Next:  tmux new -s exp   then   python run_all.py --phase all"
