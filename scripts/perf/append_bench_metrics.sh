#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <bench_json_file> [benchmark_id_prefix]"
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required to parse benchmark JSON output"
  exit 1
fi

JSON_FILE="$1"
PREFIX="${2:-bench_dod}"
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f "$JSON_FILE" ]]; then
  echo "File not found: $JSON_FILE"
  exit 1
fi

TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
COMMIT="$(git rev-parse --short HEAD)"

jq -r --arg ts "$TS" --arg c "$COMMIT" --arg p "$PREFIX" '
  .benchmarks
  | map(select(.run_type == "aggregate" and (.aggregate_name == "mean" or .aggregate_name == "median" or .aggregate_name == "stddev"))
        | .base_name = (.name | sub("_(mean|median|stddev)$"; "")))
  | group_by(.base_name)
  | .[]
  | {
      name: .[0].base_name,
      mean: (map(select(.aggregate_name=="mean"))[0].real_time // 0),
      median: (map(select(.aggregate_name=="median"))[0].real_time // 0),
      stddev: (map(select(.aggregate_name=="stddev"))[0].real_time // 0)
    }
  | [$ts, $c, "Release", ($p + "_" + .name), "synthetic", (.mean|tostring), (.median|tostring), (.stddev|tostring), "", "google-benchmark aggregate"]
  | @csv
' "$JSON_FILE" >> notes/perf/metrics.csv

echo "Appended metrics from: $JSON_FILE"
