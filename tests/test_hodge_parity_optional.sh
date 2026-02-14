#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQUIRE_PARITY="${IGNEOUS_REQUIRE_PARITY:-0}"
REF_DIR="${ROOT_DIR}/DiffusionGeometry"

if [[ ! -d "${REF_DIR}" && "${REQUIRE_PARITY}" != "1" ]]; then
  echo "optional parity test skipped: ${REF_DIR} not found"
  exit 0
fi

if [[ ! -d "${REF_DIR}" && "${REQUIRE_PARITY}" == "1" ]]; then
  echo "parity required but reference repo is missing: ${REF_DIR}" >&2
  exit 1
fi

run_output="$("${ROOT_DIR}/scripts/hodge/run_parity_round.sh")"
echo "${run_output}"

round_dir="$(echo "${run_output}" | awk -F= '/^round_dir=/{print $2}' | tail -n1)"
if [[ -z "${round_dir}" ]]; then
  echo "failed to parse round_dir from parity output" >&2
  exit 1
fi

report_json="${round_dir}/report/parity_report.json"
if [[ ! -f "${report_json}" ]]; then
  echo "missing parity report: ${report_json}" >&2
  exit 1
fi

python3 - "${report_json}" <<'PY'
import json
import pathlib
import sys

report_path = pathlib.Path(sys.argv[1])
payload = json.loads(report_path.read_text())
gates = payload.get("gates", {})
summary = payload.get("summary", {})

if not gates.get("final_pass", False):
    score = summary.get("composite_score", "nan")
    corr = summary.get("circular_complex_correlation_min", "nan")
    p95 = summary.get("circular_wrapped_p95_rad_max", "nan")
    raise SystemExit(
        f"parity final gate failed: composite={score}, corr_min={corr}, p95={p95}"
    )

print("optional parity regression check passed")
PY
