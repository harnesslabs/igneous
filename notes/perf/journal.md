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

## 2026-02-13 Diffusion Build Scratch Reuse (Rejected)
- Timestamp: 2026-02-13T16:19:08Z
- Commit: 79b2fd3 (working tree with uncommitted changes)
- Hypothesis: Reuse `DiffusionTopology::build` scratch allocations (triplets + KNN buffers) to reduce repeated build overhead.
- Files touched:
  - `include/igneous/data/topology.hpp` (reverted)
- Benchmark commands:
  - `IGNEOUS_BENCH_MODE=1 ./build/bench_dod --benchmark_filter='bench_diffusion_build/2000|bench_markov_step/2000' --benchmark_min_time=0.2s --benchmark_repetitions=10 --benchmark_report_aggregates_only=true`
- Smoke results:
  - `bench_diffusion_build/2000`: `3.52 ms` mean
  - `bench_markov_step/2000`: `25.6 us` mean
- Numeric checks: all doctest suites pass (`7/7`).
- Decision: `rejected`
- Notes: No meaningful throughput gain on primary diffusion metrics; change was reverted.

## 2026-02-13 Diffusion Markov Inner-Loop Unroll
- Timestamp: 2026-02-13T16:21:33Z
- Commit: 79b2fd3 (working tree with uncommitted changes)
- Hypothesis: Replace index-heavy CSR accumulation with pointer-based unrolled accumulation inside `apply_markov_transition` to reduce per-edge overhead.
- Files touched:
  - `include/igneous/ops/geometry.hpp`
- Benchmark commands:
  - `./scripts/perf/run_deep_bench.sh`
  - `IGNEOUS_BENCH_MODE=1 ./build/bench_dod --benchmark_filter='bench_diffusion_build/2000|bench_markov_step/2000' --benchmark_min_time=0.2s --benchmark_repetitions=10 --benchmark_report_aggregates_only=true`
- Smoke results:
  - `bench_markov_step/2000`: `17.6 us` mean
  - `bench_diffusion_build/2000`: `3.44 ms` mean
- Deep results (vs `bench_dod_20260213-091622.json`):
  - `bench_markov_step/2000`: `-31.79%`
  - `bench_diffusion_build/2000`: `-2.56%`
- Profile traces:
  - `notes/perf/profiles/20260213-092110/time-profiler.trace`
  - `notes/perf/profiles/20260213-092120/cpu-counters.trace`
- Numeric checks: all doctest suites pass (`7/7`).
- Decision: `kept`
- Notes: High-confidence Markov-step throughput gain with no API churn.

## 2026-02-13 Triangle Neighbor CSR Rebuild
- Timestamp: 2026-02-13T16:26:30Z
- Commit: a7b5c3d (working tree with uncommitted changes)
- Hypothesis: Replace sort/unique directed-edge neighbor construction with a two-pass stamped CSR build over vertex-incident faces to remove `O(E log E)` overhead in triangle topology assembly.
- Files touched:
  - `include/igneous/data/topology.hpp`
- Benchmark commands:
  - `IGNEOUS_BENCH_MODE=1 ./build/bench_dod --benchmark_filter='bench_mesh_topology_build/400|bench_curvature_kernel/400|bench_flow_kernel/400' --benchmark_min_time=0.2s --benchmark_repetitions=10 --benchmark_report_aggregates_only=true`
  - `./scripts/perf/run_deep_bench.sh`
- Smoke results:
  - `bench_mesh_topology_build/400`: `4.70 ms` mean
  - `bench_curvature_kernel/400`: `7.32 ms` mean
  - `bench_flow_kernel/400`: `0.611 ms` mean
- Deep results (vs `bench_dod_20260213-092022.json`):
  - `bench_mesh_topology_build/400`: `-87.26%`
  - `bench_curvature_kernel/400`: `-0.48%`
  - `bench_flow_kernel/400`: `+0.19%`
- Profile traces:
  - `notes/perf/profiles/20260213-092552/time-profiler.trace`
  - `notes/perf/profiles/20260213-092603/cpu-counters.trace`
- Numeric checks: all doctest suites pass (`7/7`).
- Decision: `kept`
- Notes: Very large topology-build gain; this removes the dominant mesh pipeline bottleneck identified in previous runs.

## 2026-02-13 Curvature SoA Fast Path (Rejected)
- Timestamp: 2026-02-13T16:30:05Z
- Commit: 2ccddc0 (working tree with uncommitted changes)
- Hypothesis: Specialize triangle curvature kernel to direct SoA arithmetic and avoid `get_vec3`/operator object churn inside inner loops.
- Files touched:
  - `include/igneous/ops/curvature.hpp` (reverted)
- Benchmark commands:
  - `IGNEOUS_BENCH_MODE=1 ./build/bench_dod --benchmark_filter='bench_mesh_topology_build/400|bench_curvature_kernel/400|bench_flow_kernel/400' --benchmark_min_time=0.2s --benchmark_repetitions=10 --benchmark_report_aggregates_only=true`
- Smoke results:
  - `bench_curvature_kernel/400`: `7.27 ms` mean (no material improvement)
- Numeric checks: all doctest suites pass (`7/7`).
- Decision: `rejected`
- Notes: Throughput change was below acceptance threshold; code reverted.

## 2026-02-13 Spectral Adaptive NCV (Rejected)
- Timestamp: 2026-02-13T16:30:54Z
- Commit: 2ccddc0 (working tree with uncommitted changes)
- Hypothesis: Use smaller Arnoldi subspace (`ncv`) first with fallback to cut eigensolver runtime while preserving convergence.
- Files touched:
  - `include/igneous/ops/spectral.hpp` (reverted)
- Benchmark commands:
  - `IGNEOUS_BENCH_MODE=1 ./build/bench_dod --benchmark_filter='bench_eigenbasis/2000/16|bench_1form_gram/2000/16|bench_weak_derivative/2000/16|bench_curl_energy/2000/16|bench_hodge_solve/2000/16' --benchmark_min_time=0.2s --benchmark_repetitions=10 --benchmark_report_aggregates_only=true`
- Smoke results:
  - `bench_eigenbasis/2000/16`: `20.25 ms` mean (`+32%` regression)
- Numeric checks: all doctest suites pass (`7/7`).
- Decision: `rejected`
- Notes: Large regression in primary eigenbasis benchmark; code reverted.

## 2026-02-13 Flow SoA Triangle Path
- Timestamp: 2026-02-13T16:32:12Z
- Commit: 2ccddc0 (working tree with uncommitted changes)
- Hypothesis: Specialize triangle mean-curvature flow to direct SoA+CSR loops to reduce neighbor gather/set overhead.
- Files touched:
  - `include/igneous/ops/flow.hpp`
- Benchmark commands:
  - `IGNEOUS_BENCH_MODE=1 ./build/bench_dod --benchmark_filter='bench_flow_kernel/400|bench_curvature_kernel/400|bench_mesh_topology_build/400' --benchmark_min_time=0.2s --benchmark_repetitions=10 --benchmark_report_aggregates_only=true`
  - `./scripts/perf/run_deep_bench.sh`
- Smoke results:
  - `bench_flow_kernel/400`: `0.583 ms` mean
- Deep results (vs `bench_dod_20260213-092505.json`):
  - `bench_flow_kernel/400`: `-3.99%`
  - `bench_mesh_topology_build/400`: `+0.30%`
  - `bench_curvature_kernel/400`: `+0.30%`
- Profile traces:
  - `notes/perf/profiles/20260213-093150/time-profiler.trace`
  - `notes/perf/profiles/20260213-093200/cpu-counters.trace`
- Numeric checks: all doctest suites pass (`7/7`).
- Decision: `kept`
- Notes: Cleared the acceptance threshold for the primary flow kernel target.
