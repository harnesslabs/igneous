#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <binary> [args...]"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

NOTES_DIR="${IGNEOUS_PERF_NOTES_DIR:-notes/perf}"
TS="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="${NOTES_DIR}/profiles/${TS}"
mkdir -p "$OUT_DIR"

xctrace record \
  --template "Time Profiler" \
  --output "$OUT_DIR/time-profiler.trace" \
  --launch -- "$@"

echo "Wrote: $OUT_DIR/time-profiler.trace"
