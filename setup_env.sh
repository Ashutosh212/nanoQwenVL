#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${QWEN25_VENV_PATH:-${PROJECT_DIR}/.venv}"
TORCH_BACKEND="${QWEN25_TORCH_BACKEND:-auto}"

if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
elif [[ -x /opt/venv/bin/uv ]]; then
    UV_BIN=/opt/venv/bin/uv
else
    echo "uv was not found. Install uv, then rerun this script." >&2
    exit 1
fi

if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
    if command -v python3.11 >/dev/null 2>&1; then
        BASE_PYTHON="$(command -v python3.11)"
    elif command -v python3.10 >/dev/null 2>&1; then
        BASE_PYTHON="$(command -v python3.10)"
    else
        BASE_PYTHON="$(command -v python3)"
    fi
    "${UV_BIN}" venv "${VENV_PATH}" --python "${BASE_PYTHON}"
fi

"${UV_BIN}" pip install \
    --python "${VENV_PATH}/bin/python" \
    --torch-backend "${TORCH_BACKEND}" \
    --editable "${PROJECT_DIR}[dev]"

"${VENV_PATH}/bin/python" "${PROJECT_DIR}/scripts/check_environment.py"

echo
echo "Environment ready: ${VENV_PATH}"
echo "Activate with: source ${VENV_PATH}/bin/activate"

