#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <path-to-igneous-diffusion-topology>" >&2
  exit 2
fi

EXE="$1"
if [[ ! -x "${EXE}" ]]; then
  echo "missing executable: ${EXE}" >&2
  exit 2
fi

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

INPUT_CSV="${WORK_DIR}/input.csv"
python3 - <<PY
import csv, math
from pathlib import Path
p = Path("${INPUT_CSV}")
with p.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["x","y","z"])
    n = 180
    for i in range(n):
        t = (i / n) * 2.0 * math.pi
        x = (2.0 + 0.8 * math.cos(3.0 * t)) * math.cos(t)
        y = (2.0 + 0.8 * math.cos(3.0 * t)) * math.sin(t)
        z = 0.8 * math.sin(3.0 * t)
        w.writerow([x,y,z])
PY

OUT_DIR="${WORK_DIR}/out"
"${EXE}" \
  --input-csv "${INPUT_CSV}" \
  --output-dir "${OUT_DIR}" \
  --n-basis 24 \
  --n-coefficients 16 \
  --k-neighbors 20 \
  --knn-bandwidth 8 \
  --bandwidth-variability -0.5 \
  --c 0.0 \
  --harmonic-tolerance 1e-3

required=(
  "points.csv"
  "form1_spectrum.csv"
  "form2_spectrum.csv"
  "harmonic1_coeffs.csv"
  "harmonic1_ambient.csv"
  "harmonic2_coeffs.csv"
  "harmonic2_ambient.csv"
  "wedge_h1h1_coeffs.csv"
  "wedge_h1h1_ambient.csv"
  "circular_coordinates.csv"
)
for f in "${required[@]}"; do
  if [[ ! -f "${OUT_DIR}/${f}" ]]; then
    echo "missing required output: ${f}" >&2
    exit 1
  fi
done

debug_only=(
  "function_gram.csv"
  "laplacian0_weak.csv"
  "function_basis.csv"
)
for f in "${debug_only[@]}"; do
  if [[ -f "${OUT_DIR}/${f}" ]]; then
    echo "unexpected debug output present: ${f}" >&2
    exit 1
  fi
done

echo "diffgeo CLI output contract check passed"
