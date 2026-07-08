#!/bin/bash
set -e

PROJECT_DIR="/home/elf/Neurolink_Project"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
SCRIPT_NAME="RK3588_MLP_System.py"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/rk3588_runtime.log"

mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"

rm -rf __pycache__

export PYTHONUNBUFFERED=1
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export PULSE_RUNTIME_PATH="${PULSE_RUNTIME_PATH:-$XDG_RUNTIME_DIR/pulse}"
export PULSE_SERVER="${PULSE_SERVER:-unix:$XDG_RUNTIME_DIR/pulse/native}"

exec "$PYTHON_BIN" "$SCRIPT_NAME" >> "$LOG_FILE" 2>&1

