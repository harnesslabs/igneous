#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

INPUT_CSV="${1:?first arg must be input csv path}"
OUTPUT_DIR="${2:?second arg must be output dir path}"

BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
EXE="${BUILD_DIR}/igneous-diffusion-geometry"

N_BASIS="${N_BASIS:-50}"
N_COEFFICIENTS="${N_COEFFICIENTS:-50}"
K_NEIGHBORS="${K_NEIGHBORS:-32}"
KNN_BANDWIDTH="${KNN_BANDWIDTH:-8}"
BANDWIDTH_VARIABILITY="${BANDWIDTH_VARIABILITY:--0.5}"
C_PARAM="${C_PARAM:-0.0}"
HARMONIC_TOLERANCE="${HARMONIC_TOLERANCE:-1e-3}"
CIRCULAR_LAMBDA="${CIRCULAR_LAMBDA:-1.0}"
CIRCULAR_MODE_0="${CIRCULAR_MODE_0:-0}"
CIRCULAR_MODE_1="${CIRCULAR_MODE_1:-1}"

if [[ ! -x "${EXE}" ]]; then
  echo "Executable not found: ${EXE}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

"${EXE}" \
  --input-csv "${INPUT_CSV}" \
  --output-dir "${OUTPUT_DIR}" \
  --n-basis "${N_BASIS}" \
  --n-coefficients "${N_COEFFICIENTS}" \
  --k-neighbors "${K_NEIGHBORS}" \
  --knn-bandwidth "${KNN_BANDWIDTH}" \
  --bandwidth-variability "${BANDWIDTH_VARIABILITY}" \
  --c "${C_PARAM}" \
  --harmonic-tolerance "${HARMONIC_TOLERANCE}" \
  --circular-lambda "${CIRCULAR_LAMBDA}" \
  --circular-mode-0 "${CIRCULAR_MODE_0}" \
  --circular-mode-1 "${CIRCULAR_MODE_1}"
