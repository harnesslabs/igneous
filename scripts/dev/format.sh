#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-apply}"

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

CLANG_FORMAT_BIN="$(pick_tool clang-format clang-format-19 clang-format-18 clang-format-17)" || {
  echo "error: clang-format not found in PATH."
  exit 1
}

mapfile -t SOURCE_FILES < <(
  cd "${ROOT_DIR}" &&
    rg --files include src tests benches | rg '\.(hpp|h|hh|cpp|cc|cxx|mm)$'
)

if [[ ${#SOURCE_FILES[@]} -eq 0 ]]; then
  echo "No C++ source files found under include/src/tests/benches."
  exit 0
fi

cd "${ROOT_DIR}"
if ! "${CLANG_FORMAT_BIN}" --style=file --dump-config >/dev/null 2>&1; then
  echo "error: unable to parse .clang-format."
  exit 1
fi

if [[ "${MODE}" == "--check" ]]; then
  status=0
  declare -a NEEDS_FORMAT=()
  for file in "${SOURCE_FILES[@]}"; do
    if ! diff -q "${file}" <("${CLANG_FORMAT_BIN}" "${file}") >/dev/null; then
      NEEDS_FORMAT+=("${file}")
      status=1
    fi
  done
  if [[ ${status} -ne 0 ]]; then
    echo "The following files are not clang-formatted:"
    printf '  %s\n' "${NEEDS_FORMAT[@]}"
    echo "Run 'make format' to apply formatting."
  fi
  exit "${status}"
fi

if [[ "${MODE}" != "apply" ]]; then
  echo "error: unsupported mode '${MODE}'. Use 'apply' or '--check'."
  exit 1
fi

"${CLANG_FORMAT_BIN}" -i "${SOURCE_FILES[@]}"
