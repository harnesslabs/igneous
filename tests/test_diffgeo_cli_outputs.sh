#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <path-to-igneous-diffusion-geometry>" >&2
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
  "circular_theta_0.ply"
  "circular_theta_1.ply"
  "harmonic1_form_0.ply"
  "harmonic1_form_1.ply"
  "harmonic2_form_0.ply"
  "wedge_h1h1_dual.ply"
)
for f in "${required[@]}"; do
  if [[ ! -f "${OUT_DIR}/${f}" ]]; then
    echo "missing required output: ${f}" >&2
    exit 1
  fi
done

if compgen -G "${OUT_DIR}/*.csv" >/dev/null; then
  echo "unexpected csv outputs present in ${OUT_DIR}" >&2
  ls -1 "${OUT_DIR}"/*.csv >&2
  exit 1
fi

echo "diffgeo CLI output contract check passed"
