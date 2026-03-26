#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_PATH="${SCRIPT_PATH:-$PROJECT_DIR/train.py}"
REQ_PATH="${REQ_PATH:-$PROJECT_DIR/requirements.txt}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"

CONFIG_PATH="$PROJECT_DIR/configs/config.yaml"
USE_VENV=false
START_TENSORBOARD=false
TB_LOGDIR="$PROJECT_DIR/runs"
TB_PORT=6006

usage() {
    cat <<EOF
Usage:
    $0 [options] [-- extra python args]

Options:
    --config PATH          Path to the yaml config file (default: $CONFIG_PATH)
    --venv                 Create/use local virtual environment in $VENV_DIR
    --tb                   Start TensorBoard
    --tb-logdir PATH       TensorBoard log directory (default: $TB_LOGDIR)
    --tb-port PORT         TensorBoard port (default: $TB_PORT)
    --script PATH          Path to the training script (default: $SCRIPT_PATH)
    --python PATH          Python executable to use (default: $PYTHON_BIN)
    --requirements PATH    Path to requirements.txt (default: $REQ_PATH)
    -h, --help             Show this help message and exit

Examples:
    $0 --config configs/config.yaml
    $0 --venv --tb --tb-logdir runs --config configs/config.yaml
    $0 --venv --config configs/config.yaml -- --some_future_arg value
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --venv)
            USE_VENV=true
            shift
            ;;
        --tb)
            START_TENSORBOARD=true
            shift
            ;;
        --tb-logdir)
            TB_LOGDIR="$2"
            shift 2
            ;;
        --tb-port)
            TB_PORT="$2"
            shift 2
            ;;
        --script)
            SCRIPT_PATH="$2"
            shift 2
            ;;
        --python)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --requirements)
            REQ_PATH="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

EXTRA_ARGS=("${EXTRA_ARGS[@]:-}")

if [[ ! -f "$REQ_PATH" ]]; then
    echo "[ERROR] Requirements file not found at $REQ_PATH"
    exit 1
fi

if [[ ! -f "$SCRIPT_PATH" ]]; then
    echo "[ERROR] Training script not found at $SCRIPT_PATH"
    exit 1
fi

if [[ "$USE_VENV" == true ]]; then
    if [[ ! -d "$VENV_DIR" ]]; then
        echo "[INFO] Creating virtual environment in $VENV_DIR"
        python3 -m venv "$VENV_DIR"
    fi
    source "$VENV_DIR/bin/activate"

    PYTHON_BIN="python"
fi

echo "[INFO] Installing dependencies from $REQ_PATH"
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -r "$REQ_PATH"

if [[ "$START_TENSORBOARD" == true ]]; then
    if ! command -v tensorboard &> /dev/null; then
        echo "[ERROR] TensorBoard not found in the current environment. Please install it to use --tb option."
        exit 1
    fi

    echo "[INFO] Starting TensorBoard on port $TB_PORT with logdir $TB_LOGDIR"
    tensorboard --logdir "$TB_LOGDIR" --port "$TB_PORT" >/tmp/tensorboard.log 2>&1 &
    TB_PID=$!
    echo "[INFO] TensorBoard started with PID $TB_PID"
fi

echo "[INFO] Starting training with config $CONFIG_PATH"
"$PYTHON_BIN" "$SCRIPT_PATH" --config "$CONFIG_PATH" "${EXTRA_ARGS[@]}"

if [[ "${TB_PID:-}" != "" ]]; then
    echo "[INFO] TensorBoard is running with PID $TB_PID. To stop it, run: kill $TB_PID"
fi
