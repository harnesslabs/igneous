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

## 2026-02-13 Diffusion CSR Hot-Path Refactor
- Timestamp: 2026-02-13T16:12:58Z
- Commit: 1c1caf1 (working tree with uncommitted changes)
- Hypothesis: Build diffusion CSR row slices directly in `DiffusionTopology::build` and route `carre_du_champ` through CSR row traversal to reduce sparse-iterator overhead in diffusion-derived operators.
- Files touched:
  - `include/igneous/data/topology.hpp`
  - `include/igneous/ops/geometry.hpp`
  - `tests/test_topology_diffusion.cpp`
- Benchmark commands:
  - `./scripts/perf/run_deep_bench.sh`
  - `IGNEOUS_BENCH_MODE=1 ./build/bench_dod --benchmark_filter='bench_diffusion_build/2000|bench_markov_step/2000|bench_1form_gram/2000/16|bench_weak_derivative/2000/16|bench_curl_energy/2000/16' --benchmark_min_time=0.2s --benchmark_repetitions=10 --benchmark_report_aggregates_only=true`
- Smoke results:
  - `bench_diffusion_build/2000`: `3.47 ms` mean
  - `bench_markov_step/2000`: `30.7 us` mean
  - `bench_1form_gram/2000/16`: `0.361 ms` mean
  - `bench_weak_derivative/2000/16`: `1.67 ms` mean
  - `bench_curl_energy/2000/16`: `6.95 ms` mean
- Deep results (vs `bench_dod_20260213-085501.json`):
  - `bench_1form_gram/2000/16`: `-23.23%`
  - `bench_weak_derivative/2000/16`: `-32.13%`
  - `bench_curl_energy/2000/16`: `-30.99%`
  - `bench_eigenbasis/2000/16`: `-4.75%`
  - `bench_diffusion_build/2000`: `+6.26%` on deep run, `+2.70%` on targeted rerun
  - `bench_markov_step/2000`: `+4.38%` on deep run, `+3.53%` on targeted rerun
- Profile traces:
  - `notes/perf/profiles/20260213-091206/time-profiler.trace`
  - `notes/perf/profiles/20260213-091246/cpu-counters.trace`
- Numeric checks: all doctest suites pass (`7/7`).
- Decision: `kept`
- Notes: Large gains across diffusion-derived Hodge kernels justify the retained CSR build cost.

## 2026-02-13 Diffusion Markov-Step Kernel
- Timestamp: 2026-02-13T16:17:39Z
- Commit: f782968 (working tree with uncommitted changes)
- Hypothesis: Replace `P * u` hot-loop calls with an explicit CSR row traversal kernel (`apply_markov_transition`) to reduce sparse matvec overhead in iterative diffusion.
- Files touched:
  - `include/igneous/ops/geometry.hpp`
  - `src/main_diffusion.cpp`
  - `benches/bench_dod.cpp`
  - `tests/test_topology_diffusion.cpp`
- Benchmark commands:
  - `./scripts/perf/run_deep_bench.sh`
  - `IGNEOUS_BENCH_MODE=1 ./build/bench_dod --benchmark_filter='bench_markov_step/2000|bench_diffusion_build/2000|bench_1form_gram/2000/16|bench_weak_derivative/2000/16|bench_curl_energy/2000/16' --benchmark_min_time=0.2s --benchmark_repetitions=10 --benchmark_report_aggregates_only=true`
- Smoke results:
  - `bench_markov_step/2000`: `25.3 us` mean
  - `bench_diffusion_build/2000`: `3.42 ms` mean
- Deep results (vs `bench_dod_20260213-090922.json`):
  - `bench_markov_step/2000`: `-18.26%`
  - `bench_diffusion_build/2000`: `-3.89%`
  - Other benchmark groups moved within expected run-to-run noise under concurrent load.
- Profile traces:
  - `notes/perf/profiles/20260213-091713/time-profiler.trace`
  - `notes/perf/profiles/20260213-091725/cpu-counters.trace`
- Numeric checks: all doctest suites pass (`7/7`), including CSR-vs-sparse-product equality.
- Decision: `kept`
- Notes: This optimization improves both isolated Markov stepping and end-to-end diffusion iteration cost while preserving spectral compatibility through retained `P`.
