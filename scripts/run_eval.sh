#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="$ROOT_DIR/configs/config.yaml"
EVAL_SCRIPT="$ROOT_DIR/run_eval.py"
RESULTS_PATH="$ROOT_DIR/results.json"

CHECKPOINTS=(
    "checkpoints/checkpoint.pth"
    "checkpoints/model_best.pth"
)

METHODS=(
    "knn"
    "linear"
)

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

log() {
    echo "[INFO] $*"
}

[[ -f "$CONFIG_PATH" ]] || die "Config file not found at $CONFIG_PATH"
[[ -f "$EVAL_SCRIPT" ]] || die "Evaluation script not found at $EVAL_SCRIPT"

for ckpt in "${CHECKPOINTS[@]}"; do
    if [[ ! -f "$ckpt" ]]; then
        log "Checkpoint not found at $ckpt, skipping..."
        continue
    fi

    log "Found checkpoint at $ckpt, starting evaluation..."

    for method in "${METHODS[@]}"; do
        log "Evaluating method='$method' on checkpoint $ckpt..."
        python "$EVAL_SCRIPT" --config "$CONFIG_PATH" --simclr_path "$ckpt" --method "$method"
    done
done

log "Evaluation completed, results saved at $RESULTS_PATH"
