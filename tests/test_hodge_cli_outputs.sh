#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <path-to-igneous-hodge>" >&2
  exit 2
fi

HODGE_EXE="$1"
if [[ ! -x "${HODGE_EXE}" ]]; then
  echo "missing executable: ${HODGE_EXE}" >&2
  exit 2
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT
OUT_DIR="${TMP_DIR}/out"

"${HODGE_EXE}" \
  --output-dir "${OUT_DIR}" \
  --n-points 160 \
  --n-basis 24 \
  --k-neighbors 24 \
  --knn-bandwidth 8 \
  --bandwidth-variability -0.5 \
  --c 0.0 \
  --circular-lambda 1.0 \
  --circular-mode-0 0 \
  --circular-mode-1 1 \
  --no-ply >/dev/null

required_files=(
  "points.csv"
  "hodge_spectrum.csv"
  "harmonic_coeffs.csv"
  "harmonic_ambient.csv"
  "circular_coordinates.csv"
  "circular_modes.csv"
)

for rel in "${required_files[@]}"; do
  if [[ ! -f "${OUT_DIR}/${rel}" ]]; then
    echo "missing required output file: ${rel}" >&2
    exit 1
  fi
done

forbidden_files=(
  "function_gram.csv"
  "laplacian0_weak.csv"
  "function_basis.csv"
  "circular_operator_form0_x_weak.csv"
  "circular_operator_form1_x_weak.csv"
  "circular_operator_form0_operator_weak.csv"
  "circular_operator_form1_operator_weak.csv"
  "circular_operator_form0_evals.csv"
  "circular_operator_form1_evals.csv"
)

for rel in "${forbidden_files[@]}"; do
  if [[ -f "${OUT_DIR}/${rel}" ]]; then
    echo "found unexpected diagnostic output file: ${rel}" >&2
    exit 1
  fi
done

echo "hodge CLI output contract check passed"
