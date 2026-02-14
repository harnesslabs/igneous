#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

VENV_PATH="${1:-${ROOT_DIR}/notes/hodge/.venv_ref}"
PYTHON_BIN="${PYTHON_BIN:-/opt/homebrew/bin/python3.13}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  if command -v python3.13 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.13)"
  else
    echo "python3.13 not found. Install Python 3.13 or set PYTHON_BIN." >&2
    exit 1
  fi
fi

if [[ ! -d "${VENV_PATH}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_PATH}"
fi

# shellcheck source=/dev/null
source "${VENV_PATH}/bin/activate"

python -m pip -q install --upgrade pip
python -m pip -q install numpy scipy scikit-learn opt_einsum matplotlib
python -m pip -q install -e "${ROOT_DIR}/DiffusionGeometry"

echo "${VENV_PATH}"
