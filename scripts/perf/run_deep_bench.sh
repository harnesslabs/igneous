#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p notes/perf/results

TS="$(date +%Y%m%d-%H%M%S)"
OUT_DOD_JSON="notes/perf/results/bench_dod_${TS}.json"
OUT_DOD_TXT="notes/perf/results/bench_dod_${TS}.txt"
OUT_PIPE_JSON="notes/perf/results/bench_pipelines_${TS}.json"
OUT_PIPE_TXT="notes/perf/results/bench_pipelines_${TS}.txt"

IGNEOUS_BENCH_MODE=1 ./build/bench_dod \
  --benchmark_min_time=0.2s \
  --benchmark_repetitions=10 \
  --benchmark_report_aggregates_only=true \
  --benchmark_out="$OUT_DOD_JSON" \
  --benchmark_out_format=json \
  | tee "$OUT_DOD_TXT"

IGNEOUS_BENCH_MODE=1 ./build/bench_pipelines \
  --benchmark_min_time=0.2s \
  --benchmark_repetitions=10 \
  --benchmark_report_aggregates_only=true \
  --benchmark_out="$OUT_PIPE_JSON" \
  --benchmark_out_format=json \
  | tee "$OUT_PIPE_TXT"

echo "Wrote: $OUT_DOD_JSON"
echo "Wrote: $OUT_DOD_TXT"
echo "Wrote: $OUT_PIPE_JSON"
echo "Wrote: $OUT_PIPE_TXT"
