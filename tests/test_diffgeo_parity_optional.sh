#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQUIRE_PARITY="${IGNEOUS_REQUIRE_PARITY:-0}"

if [[ "${REQUIRE_PARITY}" != "1" ]]; then
  echo "Skipping optional diffusion geometry parity test: CLI now emits PLY-only outputs"
  exit 0
fi

if [[ ! -d "${ROOT_DIR}/DiffusionGeometry" ]]; then
  echo "DiffusionGeometry/ is required but missing" >&2
  exit 1
fi

run_output="$("${ROOT_DIR}/scripts/diffgeo/run_parity_round.sh")"
echo "${run_output}"

report_json="$(echo "${run_output}" | awk -F= '/^round_dir=/{print $2}' | tail -n1)/report/parity_report.json"
if [[ ! -f "${report_json}" ]]; then
  echo "Missing parity report JSON: ${report_json}" >&2
  exit 1
fi

python3 - <<PY
import json
from pathlib import Path
report = Path("${report_json}")
payload = json.loads(report.read_text())
if not payload["gates"]["final_pass"]:
    raise SystemExit("final parity gate failed")
PY

echo "optional diffusion geometry parity test passed"
