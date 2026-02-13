# Performance Journal

Use one entry per optimization hypothesis.

## Entry Template
- Timestamp:
- Commit:
- Hypothesis:
- Files touched:
- Benchmark commands:
- Smoke results:
- Deep results:
- Profile traces:
- Numeric checks:
- Decision: `kept` | `rejected` | `deferred`
- Notes:

## 2026-02-13 Initial Refactor Baseline
- Timestamp: 2026-02-13T15:54:44Z
- Commit: e761562 (working tree with uncommitted changes)
- Hypothesis: SoA geometry + explicit triangle adjacency + workspace kernels remove gather/allocation overhead in hot loops.
- Files touched: data buffers/topology/mesh, curvature/flow/geometry/hodge ops, mains, CMake, tests, benchmark suite.
- Benchmark commands:
  - `./build/bench_geometry`
  - `./build/bench_dod --benchmark_min_time=0.1s --benchmark_repetitions=3 --benchmark_report_aggregates_only=true`
- Smoke results:
  - `bench_geometry` 1M grid: curvature ~46.231 ms, flow ~3.982 ms.
- Deep results:
  - Captured in `notes/perf/metrics.csv` and console output from `bench_dod`.
- Profile traces: not yet captured.
- Numeric checks: all doctest suites pass (`7/7`).
- Decision: `kept`
- Notes: major throughput gain on mesh kernels; topology build is now the dominant mesh-side cost.
- Deep benchmark artifact:
  - `notes/perf/results/bench_dod_20260213-085501.json`
  - `notes/perf/results/bench_dod_20260213-085501.txt`
- Profiling script validation traces:
  - `notes/perf/profiles/20260213-085833/time-profiler.trace`
  - `notes/perf/profiles/20260213-085849/cpu-counters.trace`
