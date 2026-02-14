#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RESULTS_ROOT="${RESULTS_ROOT:-${ROOT_DIR}/notes/diffgeo_ops/results}"
ROUND_TAG="${ROUND_TAG:-$(date -u +%Y%m%d-%H%M%S)}"
ROUND_DIR="${RESULTS_ROOT}/round_${ROUND_TAG}"

INPUT_DIR="${ROUND_DIR}/input"
REFERENCE_ROOT="${ROUND_DIR}/reference"
CPP_ROOT="${ROUND_DIR}/cpp"
REPORT_DIR="${ROUND_DIR}/report"
mkdir -p "${INPUT_DIR}" "${REFERENCE_ROOT}" "${CPP_ROOT}" "${REPORT_DIR}"

VENV_PATH="${VENV_PATH:-${ROOT_DIR}/notes/diffgeo_ops/.venv_ref}"
if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
  "${SCRIPT_DIR}/setup_reference_env.sh" "${VENV_PATH}" >/dev/null
fi
# shellcheck source=/dev/null
source "${VENV_PATH}/bin/activate"
PYTHON_BIN="${VENV_PATH}/bin/python"

BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
if [[ ! -x "${BUILD_DIR}/igneous-diffusion-topology" ]]; then
  cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}"
  cmake --build "${BUILD_DIR}" -j8
fi

N_POINTS="${N_POINTS:-1000}"
MAJOR_RADIUS="${MAJOR_RADIUS:-2.0}"
MINOR_RADIUS="${MINOR_RADIUS:-1.0}"
SPHERE_RADIUS="${SPHERE_RADIUS:-1.0}"
SEED="${SEED:-0}"

N_FUNCTION_BASIS="${N_FUNCTION_BASIS:-50}"
N_COEFFICIENTS="${N_COEFFICIENTS:-50}"
K_NEIGHBORS="${K_NEIGHBORS:-32}"
KNN_BANDWIDTH="${KNN_BANDWIDTH:-8}"
BANDWIDTH_VARIABILITY="${BANDWIDTH_VARIABILITY:--0.5}"
C_PARAM="${C_PARAM:-0.0}"
HARMONIC_TOLERANCE="${HARMONIC_TOLERANCE:-1e-3}"
CIRCULAR_LAMBDA="${CIRCULAR_LAMBDA:-1.0}"
CIRCULAR_MODE_0="${CIRCULAR_MODE_0:-0}"
CIRCULAR_MODE_1="${CIRCULAR_MODE_1:-1}"

for DATASET in torus sphere; do
  INPUT_CSV="${INPUT_DIR}/${DATASET}.csv"
  if [[ "${DATASET}" == "torus" ]]; then
    "${PYTHON_BIN}" "${SCRIPT_DIR}/generate_input_manifolds.py" \
      --output "${INPUT_CSV}" \
      --kind torus \
      --n "${N_POINTS}" \
      --major-radius "${MAJOR_RADIUS}" \
      --minor-radius "${MINOR_RADIUS}" \
      --seed "${SEED}"
  else
    "${PYTHON_BIN}" "${SCRIPT_DIR}/generate_input_manifolds.py" \
      --output "${INPUT_CSV}" \
      --kind sphere \
      --n "${N_POINTS}" \
      --sphere-radius "${SPHERE_RADIUS}" \
      --seed "${SEED}"
  fi

  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_reference_diffgeo_ops.py" \
    --input-csv "${INPUT_CSV}" \
    --output-dir "${REFERENCE_ROOT}/${DATASET}" \
    --n-function-basis "${N_FUNCTION_BASIS}" \
    --n-coefficients "${N_COEFFICIENTS}" \
    --k-neighbors "${K_NEIGHBORS}" \
    --knn-bandwidth "${KNN_BANDWIDTH}" \
    --bandwidth-variability "${BANDWIDTH_VARIABILITY}" \
    --c "${C_PARAM}" \
    --harmonic-tolerance "${HARMONIC_TOLERANCE}" \
    --circular-lambda "${CIRCULAR_LAMBDA}" \
    --circular-mode-0 "${CIRCULAR_MODE_0}" \
    --circular-mode-1 "${CIRCULAR_MODE_1}"

  N_BASIS="${N_FUNCTION_BASIS}" \
  N_COEFFICIENTS="${N_COEFFICIENTS}" \
  K_NEIGHBORS="${K_NEIGHBORS}" \
  KNN_BANDWIDTH="${KNN_BANDWIDTH}" \
  BANDWIDTH_VARIABILITY="${BANDWIDTH_VARIABILITY}" \
  C_PARAM="${C_PARAM}" \
  HARMONIC_TOLERANCE="${HARMONIC_TOLERANCE}" \
  CIRCULAR_LAMBDA="${CIRCULAR_LAMBDA}" \
  CIRCULAR_MODE_0="${CIRCULAR_MODE_0}" \
  CIRCULAR_MODE_1="${CIRCULAR_MODE_1}" \
  BUILD_DIR="${BUILD_DIR}" \
  "${SCRIPT_DIR}/run_cpp_diffgeo_ops.sh" "${INPUT_CSV}" "${CPP_ROOT}/${DATASET}"
done

COMPARE_ARGS=(
  --reference-root "${REFERENCE_ROOT}"
  --cpp-root "${CPP_ROOT}"
  --output-markdown "${REPORT_DIR}/parity_report.md"
  --output-json "${REPORT_DIR}/parity_report.json"
  --label "Diffusion topology parity round ${ROUND_TAG}"
)

if [[ -n "${PREVIOUS_REPORT_JSON:-}" ]]; then
  COMPARE_ARGS+=(--previous-json "${PREVIOUS_REPORT_JSON}")
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/compare_diffgeo_ops.py" "${COMPARE_ARGS[@]}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/diagnostics/plot_diffgeo_ops.py" --round-dir "${ROUND_DIR}"

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
report = Path("${REPORT_DIR}/parity_report.json")
payload = json.loads(report.read_text())
print(f"round_dir={Path('${ROUND_DIR}')}")
print(f"composite_score={payload['summary']['composite_score']:.6f}")
print(f"harmonic1_max_angle={payload['summary']['harmonic1_subspace_max_angle_deg']:.6f}")
print(f"harmonic2_max_angle={payload['summary']['harmonic2_subspace_max_angle_deg']:.6f}")
print(f"wedge_rel_error={payload['summary']['wedge_rel_error']:.6f}")
print(f"form2_spectrum_rel_error={payload['summary']['form2_spectrum_rel_error']:.6f}")
print(f"final_pass={payload['gates']['final_pass']}")
print(f"commit_gate_pass={payload['gates']['commit_gate_pass']}")
PY

echo "Wrote report: ${REPORT_DIR}/parity_report.md"
echo "Wrote report: ${REPORT_DIR}/parity_report.json"
