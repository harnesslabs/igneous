#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${1:-${ROOT_DIR}/build}"
LINT_SCOPE="${IGNEOUS_LINT_SCOPE:-src}"
LINT_HEADERS="${IGNEOUS_LINT_HEADERS:-1}"
LINT_CHANGED_ONLY="${IGNEOUS_LINT_CHANGED_ONLY:-0}"
LINT_JOBS="${IGNEOUS_LINT_JOBS:-}"

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

compute_default_jobs() {
  local cores=1
  if command -v nproc >/dev/null 2>&1; then
    cores="$(nproc)"
  elif [[ "$(uname -s)" == "Darwin" ]]; then
    cores="$(sysctl -n hw.logicalcpu 2>/dev/null || echo 1)"
  fi

  if [[ "${cores}" -gt 8 ]]; then
    echo 8
  else
    echo "${cores}"
  fi
}

validate_bool() {
  local name="$1"
  local value="$2"
  case "${value}" in
    0 | 1) ;;
    *)
      echo "error: ${name} must be 0 or 1 (got '${value}')."
      exit 1
      ;;
  esac
}

if [[ ! -f "${BUILD_DIR}/compile_commands.json" ]]; then
  echo "error: ${BUILD_DIR}/compile_commands.json not found."
  echo "hint: run 'cmake --preset default-local' first."
  exit 1
fi

validate_bool "IGNEOUS_LINT_HEADERS" "${LINT_HEADERS}"
validate_bool "IGNEOUS_LINT_CHANGED_ONLY" "${LINT_CHANGED_ONLY}"

if [[ -z "${LINT_JOBS}" ]]; then
  LINT_JOBS="$(compute_default_jobs)"
fi
if ! [[ "${LINT_JOBS}" =~ ^[0-9]+$ ]] || [[ "${LINT_JOBS}" -lt 1 ]]; then
  echo "error: IGNEOUS_LINT_JOBS must be a positive integer (got '${LINT_JOBS}')."
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

collect_translation_units() {
  if command -v rg >/dev/null 2>&1; then
    rg --files "${SEARCH_DIRS[@]}" | rg '\.(cpp|cc|cxx)$'
  else
    find "${SEARCH_DIRS[@]}" -type f \( -name '*.cpp' -o -name '*.cc' -o -name '*.cxx' \)
  fi
}

collect_headers() {
  if command -v rg >/dev/null 2>&1; then
    rg --files include/igneous | rg '\.(hpp|h|hh)$'
  else
    find include/igneous -type f \( -name '*.hpp' -o -name '*.h' -o -name '*.hh' \)
  fi
}

extract_header_compile_args() {
  python3 - "$1" <<'PY'
import json
import shlex
import sys

path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as f:
        db = json.load(f)
except Exception:
    sys.exit(0)

def command_tokens(entry):
    if "arguments" in entry and isinstance(entry["arguments"], list):
        return entry["arguments"]
    if "command" in entry and isinstance(entry["command"], str):
        try:
            return shlex.split(entry["command"])
        except ValueError:
            return []
    return []

def score(tokens):
    text = " ".join(tokens)
    s = 0
    if "vcpkg_installed" in text:
        s += 8
    if any(tok.startswith("-std=") for tok in tokens):
        s += 4
    if "-std" in tokens:
        s += 4
    if any(tok == "-isystem" or tok.startswith("-isystem") for tok in tokens):
        s += 2
    if any(tok == "-I" or tok.startswith("-I") for tok in tokens):
        s += 1
    return s

best = []
best_score = -1
for entry in db:
    tokens = command_tokens(entry)
    if not tokens:
        continue
    s = score(tokens)
    if s > best_score:
        best = tokens
        best_score = s

if not best:
    sys.exit(0)

keep = []
i = 0
while i < len(best):
    tok = best[i]
    if tok in ("-I", "-isystem", "-D", "-U", "-std"):
        if i + 1 < len(best):
            keep.append(tok)
            keep.append(best[i + 1])
            i += 2
            continue
    if tok.startswith(("-I", "-D", "-U", "-std=", "-isystem")):
        keep.append(tok)
    i += 1

seen = set()
for tok in keep:
    if tok in seen:
        continue
    seen.add(tok)
    print(tok)
PY
}

collect_changed_paths() {
  (
    cd "${ROOT_DIR}" || exit 1
    if ! command -v git >/dev/null 2>&1 || ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
      return 1
    fi
    {
      git diff --name-only --diff-filter=ACMRTUXB HEAD -- "${SEARCH_DIRS[@]}" include/igneous
      git ls-files --others --exclude-standard -- "${SEARCH_DIRS[@]}" include/igneous
    } | sort -u
  )
}

filter_cpp_paths() {
  while IFS= read -r path; do
    case "${path}" in
      *.cpp | *.cc | *.cxx)
        echo "${path}"
        ;;
    esac
  done
}

filter_header_paths() {
  while IFS= read -r path; do
    case "${path}" in
      include/igneous/*)
        case "${path}" in
          *.h | *.hh | *.hpp)
            echo "${path}"
            ;;
        esac
      ;;
    esac
  done
}

TRANSLATION_UNITS=()
HEADER_FILES=()

if [[ "${LINT_CHANGED_ONLY}" == "1" ]]; then
  mapfile -t CHANGED_PATHS < <(collect_changed_paths || true)
  if [[ ${#CHANGED_PATHS[@]} -gt 0 ]]; then
    mapfile -t TRANSLATION_UNITS < <(printf '%s\n' "${CHANGED_PATHS[@]}" | filter_cpp_paths)
    if [[ "${LINT_HEADERS}" == "1" ]]; then
      mapfile -t HEADER_FILES < <(printf '%s\n' "${CHANGED_PATHS[@]}" | filter_header_paths)
    fi
  fi
else
  mapfile -t TRANSLATION_UNITS < <(cd "${ROOT_DIR}" && collect_translation_units)
  if [[ "${LINT_HEADERS}" == "1" ]]; then
    mapfile -t HEADER_FILES < <(cd "${ROOT_DIR}" && collect_headers)
  fi
fi

if [[ ${#TRANSLATION_UNITS[@]} -eq 0 && ( "${LINT_HEADERS}" != "1" || ${#HEADER_FILES[@]} -eq 0 ) ]]; then
  if [[ "${LINT_CHANGED_ONLY}" == "1" ]]; then
    echo "No changed C++ files found for lint."
  else
    echo "No translation units found under src/tests/benches."
  fi
  exit 0
fi

cd "${ROOT_DIR}"
status=0
EXTRA_ARGS=()
if [[ "$(uname -s)" == "Darwin" ]]; then
  SDKROOT="$(xcrun --show-sdk-path)"
  EXTRA_ARGS+=(--extra-arg=-isysroot --extra-arg="${SDKROOT}")
fi

HEADER_EXTRA_ARGS=("${EXTRA_ARGS[@]}")
mapfile -t HEADER_COMPILE_ARGS < <(extract_header_compile_args "${BUILD_DIR}/compile_commands.json")
for arg in "${HEADER_COMPILE_ARGS[@]}"; do
  HEADER_EXTRA_ARGS+=(--extra-arg="${arg}")
done

lint_file() {
  local kind="$1"
  local file="$2"
  local -a cmd=("${CLANG_TIDY_BIN}" -quiet)
  if [[ "${kind}" == "header" ]]; then
    cmd+=("${HEADER_EXTRA_ARGS[@]}")
    cmd+=(-checks='misc-include-cleaner')
  else
    cmd+=("${EXTRA_ARGS[@]}")
  fi
  cmd+=(-p "${BUILD_DIR}" "${file}")

  "${cmd[@]}" 2>&1 | sed '/^[0-9][0-9]* warnings generated\.$/d'
}

run_parallel_lint() {
  local kind="$1"
  shift
  local -a files=("$@")
  local -a pids=()
  local failed=0

  if [[ ${#files[@]} -eq 0 ]]; then
    return 0
  fi

  for file in "${files[@]}"; do
    (
      lint_file "${kind}" "${file}"
    ) &
    pids+=("$!")

    if [[ ${#pids[@]} -ge ${LINT_JOBS} ]]; then
      if ! wait "${pids[0]}"; then
        failed=1
      fi
      pids=("${pids[@]:1}")
    fi
  done

  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done

  return "${failed}"
}

echo "Lint config: scope=${LINT_SCOPE} jobs=${LINT_JOBS} headers=${LINT_HEADERS} changed_only=${LINT_CHANGED_ONLY}"

if ! run_parallel_lint "tu" "${TRANSLATION_UNITS[@]}"; then
  status=1
fi

if [[ "${LINT_HEADERS}" == "1" ]]; then
  if ! run_parallel_lint "header" "${HEADER_FILES[@]}"; then
    status=1
  fi
fi

exit "${status}"
