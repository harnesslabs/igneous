#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${1:-${ROOT_DIR}/build}"
LINT_SCOPE="${IGNEOUS_LINT_SCOPE:-src}"

pick_tool() {
  local tool
  for tool in "$@"; do
    if command -v "${tool}" >/dev/null 2>&1; then
      command -v "${tool}"
      return 0
    fi
  done
  return 1
}

if [[ ! -f "${BUILD_DIR}/compile_commands.json" ]]; then
  echo "error: ${BUILD_DIR}/compile_commands.json not found."
  echo "hint: run 'cmake --preset default-local' first."
  exit 1
fi

CLANG_TIDY_BIN="$(pick_tool clang-tidy clang-tidy-19 clang-tidy-18 clang-tidy-17)" || {
  echo "error: clang-tidy not found in PATH."
  exit 1
}

SEARCH_DIRS=()
case "${LINT_SCOPE}" in
  src)
    SEARCH_DIRS=(src)
    ;;
  all)
    SEARCH_DIRS=(src tests benches)
    ;;
  *)
    echo "error: unsupported IGNEOUS_LINT_SCOPE='${LINT_SCOPE}'. Use 'src' or 'all'."
    exit 1
    ;;
esac

mapfile -t TRANSLATION_UNITS < <(
  cd "${ROOT_DIR}" &&
    rg --files "${SEARCH_DIRS[@]}" | rg '\.(cpp|cc|cxx)$'
)

if [[ ${#TRANSLATION_UNITS[@]} -eq 0 ]]; then
  echo "No translation units found under src/tests/benches."
  exit 0
fi

cd "${ROOT_DIR}"
status=0
EXTRA_ARGS=()
if [[ "$(uname -s)" == "Darwin" ]]; then
  SDKROOT="$(xcrun --show-sdk-path)"
  EXTRA_ARGS+=(--extra-arg=-isysroot --extra-arg="${SDKROOT}")
fi

for file in "${TRANSLATION_UNITS[@]}"; do
  if ! "${CLANG_TIDY_BIN}" -quiet "${EXTRA_ARGS[@]}" -p "${BUILD_DIR}" "${file}" 2>&1 |
    sed '/^[0-9][0-9]* warnings generated\.$/d'; then
    status=1
  fi
done
exit "${status}"
