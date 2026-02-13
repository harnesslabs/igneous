#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <binary> [args...]"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="notes/perf/profiles/$TS"
mkdir -p "$OUT_DIR"

xctrace record \
  --template "CPU Counters" \
  --output "$OUT_DIR/cpu-counters.trace" \
  --launch -- "$@"

echo "Wrote: $OUT_DIR/cpu-counters.trace"
