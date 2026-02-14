#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RESULTS_ROOT="${RESULTS_ROOT:-${ROOT_DIR}/notes/hodge/results}"
ROUND_TAG="${ROUND_TAG:-$(date -u +%Y%m%d-%H%M%S)}"
ROUND_DIR="${RESULTS_ROOT}/round_${ROUND_TAG}"

INPUT_DIR="${ROUND_DIR}/input"
REFERENCE_DIR="${ROUND_DIR}/reference"
CPP_DIR="${ROUND_DIR}/cpp"
REPORT_DIR="${ROUND_DIR}/report"
mkdir -p "${INPUT_DIR}" "${REFERENCE_DIR}" "${CPP_DIR}" "${REPORT_DIR}"

VENV_PATH="${VENV_PATH:-${ROOT_DIR}/notes/hodge/.venv_ref}"
if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
  "${SCRIPT_DIR}/setup_reference_env.sh" "${VENV_PATH}" >/dev/null
fi
# shellcheck source=/dev/null
source "${VENV_PATH}/bin/activate"
PYTHON_BIN="${VENV_PATH}/bin/python"

BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
if [[ ! -x "${BUILD_DIR}/igneous-hodge" ]]; then
  cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}"
  cmake --build "${BUILD_DIR}" -j8
fi

N_POINTS="${N_POINTS:-1000}"
MAJOR_RADIUS="${MAJOR_RADIUS:-2.0}"
MINOR_RADIUS="${MINOR_RADIUS:-1.0}"
SEED="${SEED:-0}"

N_FUNCTION_BASIS="${N_FUNCTION_BASIS:-50}"
N_COEFFICIENTS="${N_COEFFICIENTS:-50}"
K_NEIGHBORS="${K_NEIGHBORS:-32}"
KNN_BANDWIDTH="${KNN_BANDWIDTH:-8}"
BANDWIDTH_VARIABILITY="${BANDWIDTH_VARIABILITY:--0.5}"
C_PARAM="${C_PARAM:-0.0}"
CIRCULAR_LAMBDA="${CIRCULAR_LAMBDA:-1.0}"

FORM_INDICES="${FORM_INDICES:-0,1}"
MODE_INDICES="${MODE_INDICES:-0,1}"

INPUT_CSV="${INPUT_DIR}/torus.csv"
"${PYTHON_BIN}" "${SCRIPT_DIR}/generate_input_torus.py" \
  --output "${INPUT_CSV}" \
  --n "${N_POINTS}" \
  --major-radius "${MAJOR_RADIUS}" \
  --minor-radius "${MINOR_RADIUS}" \
  --seed "${SEED}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_reference_hodge.py" \
  --input-csv "${INPUT_CSV}" \
  --output-dir "${REFERENCE_DIR}" \
  --n-function-basis "${N_FUNCTION_BASIS}" \
  --n-coefficients "${N_COEFFICIENTS}" \
  --k-neighbors "${K_NEIGHBORS}" \
  --knn-bandwidth "${KNN_BANDWIDTH}" \
  --bandwidth-variability "${BANDWIDTH_VARIABILITY}" \
  --c "${C_PARAM}" \
  --circular-lambda "${CIRCULAR_LAMBDA}" \
  --form-indices "${FORM_INDICES}" \
  --mode-indices "${MODE_INDICES}"

CIRCULAR_MODE_0="$(echo "${MODE_INDICES}" | cut -d',' -f1)"
CIRCULAR_MODE_1="$(echo "${MODE_INDICES}" | cut -d',' -f2)"

N_BASIS="${N_FUNCTION_BASIS}" \
K_NEIGHBORS="${K_NEIGHBORS}" \
KNN_BANDWIDTH="${KNN_BANDWIDTH}" \
BANDWIDTH_VARIABILITY="${BANDWIDTH_VARIABILITY}" \
C_PARAM="${C_PARAM}" \
CIRCULAR_LAMBDA="${CIRCULAR_LAMBDA}" \
CIRCULAR_MODE_0="${CIRCULAR_MODE_0}" \
CIRCULAR_MODE_1="${CIRCULAR_MODE_1}" \
BUILD_DIR="${BUILD_DIR}" \
"${SCRIPT_DIR}/run_cpp_hodge.sh" "${INPUT_CSV}" "${CPP_DIR}"

COMPARE_ARGS=(
  --reference-dir "${REFERENCE_DIR}"
  --cpp-dir "${CPP_DIR}"
  --output-markdown "${REPORT_DIR}/parity_report.md"
  --output-json "${REPORT_DIR}/parity_report.json"
  --label "Hodge parity round ${ROUND_TAG}"
)

if [[ -n "${PREVIOUS_REPORT_JSON:-}" ]]; then
  COMPARE_ARGS+=(--previous-json "${PREVIOUS_REPORT_JSON}")
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/compare_hodge_outputs.py" "${COMPARE_ARGS[@]}"

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

report = Path("${REPORT_DIR}/parity_report.json")
payload = json.loads(report.read_text())
summary = payload["summary"]
gates = payload["gates"]
print(f"round_dir={Path('${ROUND_DIR}')}")
print(f"composite_score={summary['composite_score']:.6f}")
print(f"harmonic_subspace_max_angle_deg={summary['harmonic_subspace_max_angle_deg']:.6f}")
print(f"harmonic_procrustes_rel_error={summary['harmonic_procrustes_rel_error']:.6f}")
print(f"circular_complex_correlation_min={summary['circular_complex_correlation_min']:.6f}")
print(f"circular_wrapped_p95_rad_max={summary['circular_wrapped_p95_rad_max']:.6f}")
print(f"final_pass={gates['final_pass']}")
print(f"commit_gate_pass={gates['commit_gate_pass']}")
PY

echo "Wrote report: ${REPORT_DIR}/parity_report.md"
echo "Wrote report: ${REPORT_DIR}/parity_report.json"
