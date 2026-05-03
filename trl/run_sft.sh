#!/usr/bin/env bash
# Launch SFT training. Logs are tee'd into /workspace/trl/logs/.
set -euo pipefail

VENV_PY="/workspace/tinker-cookbook-arisohn/tinker_cookbook/distillation/.venv/bin/python"
SCRIPT="/workspace/trl/train_sft.py"
LOG_DIR="/workspace/trl/logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/train_${TS}.log"

export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1

echo "[run_sft] logging to $LOG_FILE"
"$VENV_PY" "$SCRIPT" 2>&1 | tee "$LOG_FILE"
