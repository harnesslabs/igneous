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

## 2026-02-13 Curl-Energy Manual Dot Loop (Rejected)
- Timestamp: 2026-02-13T16:34:02Z
- Commit: 7b409e9 (working tree with uncommitted changes)
- Hypothesis: Remove Eigen temporary vectors in curl-energy assembly by switching to a manual fused weighted loop.
- Files touched:
  - `include/igneous/ops/hodge.hpp` (reverted)
- Benchmark commands:
  - `IGNEOUS_BENCH_MODE=1 ./build/bench_dod --benchmark_filter='bench_curl_energy/2000/16|bench_weak_derivative/2000/16|bench_hodge_solve/2000/16|bench_1form_gram/2000/16' --benchmark_min_time=0.2s --benchmark_repetitions=10 --benchmark_report_aggregates_only=true`
- Smoke results:
  - `bench_curl_energy/2000/16`: `8.08 ms` mean (regression)
- Numeric checks: all doctest suites pass (`7/7`).
- Decision: `rejected`
- Notes: Scalar manual loop lost Eigen vectorization and regressed the primary target.

## 2026-02-13 Diffusion Sparse Assembly From CSR
- Timestamp: 2026-02-13T16:37:59Z
- Commit: 7b409e9 (working tree with uncommitted changes)
- Hypothesis: Build `P` by materializing a compressed row-major sparse matrix directly from existing diffusion CSR buffers, then convert once to col-major, avoiding `setFromTriplets`.
- Files touched:
  - `include/igneous/data/topology.hpp`
  - `include/igneous/ops/geometry.hpp`
- Benchmark commands:
  - `IGNEOUS_BENCH_MODE=1 ./build/bench_dod --benchmark_filter='bench_diffusion_build/2000|bench_markov_step/2000|bench_eigenbasis/2000/16' --benchmark_min_time=0.2s --benchmark_repetitions=10 --benchmark_report_aggregates_only=true`
  - `./scripts/perf/run_deep_bench.sh`
- Smoke results:
  - `bench_diffusion_build/2000`: `3.10 ms` mean
  - `bench_eigenbasis/2000/16`: `15.02 ms` mean
- Deep results (vs `bench_dod_20260213-093101.json`):
  - `bench_diffusion_build/2000`: `-10.68%`
  - `bench_eigenbasis/2000/16`: `-0.11%`
  - `bench_markov_step/2000`: `+0.08%`
- Profile traces:
  - `notes/perf/profiles/20260213-093726/time-profiler.trace`
  - `notes/perf/profiles/20260213-093737/cpu-counters.trace`
- Numeric checks: all doctest suites pass (`7/7`).
- Decision: `kept`
- Notes: Significant diffusion build improvement with neutral downstream spectral/Hodge performance.

## 2026-02-13 Hodge Nested Workspace Reuse (Rejected)
- Timestamp: 2026-02-13T16:39:08Z
- Commit: dbce9fb (working tree with uncommitted changes)
- Hypothesis: Replace nested `assign` calls in Hodge workspace setup with in-place resize reuse to reduce per-call allocation churn.
- Files touched:
  - `include/igneous/ops/hodge.hpp` (reverted)
- Benchmark commands:
  - `IGNEOUS_BENCH_MODE=1 ./build/bench_dod --benchmark_filter='bench_weak_derivative/2000/16|bench_curl_energy/2000/16|bench_hodge_solve/2000/16|bench_1form_gram/2000/16' --benchmark_min_time=0.2s --benchmark_repetitions=10 --benchmark_report_aggregates_only=true`
- Smoke results:
  - `bench_weak_derivative/2000/16`: `1.669 ms` mean
  - `bench_curl_energy/2000/16`: `6.990 ms` mean
- Numeric checks: all doctest suites pass (`7/7`).
- Decision: `rejected`
- Notes: Improvements stayed below threshold and were within noise.

## 2026-02-13 Flow Workspace Zero-Fill Removal
- Timestamp: 2026-02-13T16:40:59Z
- Commit: dbce9fb (working tree with uncommitted changes)
- Hypothesis: Eliminate full `workspace.displacements.assign(...)` zero-fill in flow and only write displacements per vertex, zeroing only isolated vertices.
- Files touched:
  - `include/igneous/ops/flow.hpp`
- Benchmark commands:
  - `IGNEOUS_BENCH_MODE=1 ./build/bench_dod --benchmark_filter='bench_flow_kernel/400|bench_curvature_kernel/400|bench_mesh_topology_build/400' --benchmark_min_time=0.2s --benchmark_repetitions=10 --benchmark_report_aggregates_only=true`
  - `./scripts/perf/run_deep_bench.sh`
- Smoke results:
  - `bench_flow_kernel/400`: `0.524 ms` mean
- Deep results (vs `bench_dod_20260213-093639.json`):
  - `bench_flow_kernel/400`: `-9.90%`
  - `bench_mesh_topology_build/400`: `+0.28%`
  - `bench_curvature_kernel/400`: `+0.02%`
- Profile traces:
  - `notes/perf/profiles/20260213-094042/time-profiler.trace`
  - `notes/perf/profiles/20260213-094052/cpu-counters.trace`
- Numeric checks: all doctest suites pass (`7/7`).
- Decision: `kept`
- Notes: Material flow kernel gain from removing redundant per-call initialization work.

## 2026-02-13 Spectral Largest-Real Sort (Rejected)
- Timestamp: 2026-02-13T16:42:34Z
- Commit: e1c3135 (working tree with uncommitted changes)
- Hypothesis: Switch spectral solve sort rule from largest-magnitude to largest-real to reduce eigensolver time on stochastic operators.
- Files touched:
  - `include/igneous/ops/spectral.hpp` (reverted)
- Benchmark commands:
  - `IGNEOUS_BENCH_MODE=1 ./build/bench_dod --benchmark_filter='bench_eigenbasis/2000/16|bench_1form_gram/2000/16|bench_weak_derivative/2000/16|bench_curl_energy/2000/16|bench_hodge_solve/2000/16' --benchmark_min_time=0.2s --benchmark_repetitions=10 --benchmark_report_aggregates_only=true`
- Smoke results:
  - `bench_eigenbasis/2000/16`: `14.89 ms` mean (near-noise change)
- Numeric checks: all doctest suites pass (`7/7`).
- Decision: `rejected`
- Notes: Did not reach threshold and introduced behavior drift risk in eigenmode ordering.

## 2026-02-13 Curvature Face-Accumulation Rewrite (Rejected)
- Timestamp: 2026-02-13T16:44:01Z
- Commit: e1c3135 (working tree with uncommitted changes)
- Hypothesis: Recompute curvature contributions in a face-streaming accumulation pass to improve locality and remove vertex-face indirection.
- Files touched:
  - `include/igneous/ops/curvature.hpp` (reverted)
- Benchmark commands:
  - `IGNEOUS_BENCH_MODE=1 ./build/bench_dod --benchmark_filter='bench_curvature_kernel/400|bench_flow_kernel/400|bench_mesh_topology_build/400' --benchmark_min_time=0.2s --benchmark_repetitions=10 --benchmark_report_aggregates_only=true`
- Smoke results:
  - `bench_curvature_kernel/400`: `19.6 ms` mean (major regression)
- Numeric checks: all doctest suites pass (`7/7`).
- Decision: `rejected`
- Notes: Additional accumulation arrays and corner recomputation overwhelmed locality gains.

## 2026-02-13 Carre-Du-Champ CSR Unroll
- Timestamp: 2026-02-13T16:46:22Z
- Commit: e1c3135 (working tree with uncommitted changes)
- Hypothesis: Unroll and pointer-specialize CSR traversal inside `carre_du_champ` to cut per-edge arithmetic/index overhead in diffusion-derived operators.
- Files touched:
  - `include/igneous/ops/geometry.hpp`
- Benchmark commands:
  - `IGNEOUS_BENCH_MODE=1 ./build/bench_dod --benchmark_filter='bench_1form_gram/2000/16|bench_weak_derivative/2000/16|bench_curl_energy/2000/16|bench_hodge_solve/2000/16|bench_markov_step/2000' --benchmark_min_time=0.2s --benchmark_repetitions=10 --benchmark_report_aggregates_only=true`
  - `./scripts/perf/run_deep_bench.sh`
- Smoke results:
  - `bench_1form_gram/2000/16`: `0.343 ms` mean
  - `bench_weak_derivative/2000/16`: `1.520 ms` mean
  - `bench_curl_energy/2000/16`: `6.266 ms` mean
- Deep results (vs `bench_dod_20260213-093951.json`):
  - `bench_1form_gram/2000/16`: `-6.73%`
  - `bench_weak_derivative/2000/16`: `-8.71%`
  - `bench_curl_energy/2000/16`: `-10.39%`
- Profile traces:
  - `notes/perf/profiles/20260213-094555/time-profiler.trace`
  - `notes/perf/profiles/20260213-094609/cpu-counters.trace`
- Numeric checks: all doctest suites pass (`7/7`).
- Decision: `kept`
- Notes: High-confidence gain across all primary diffusion-derived geometry/Hodge kernels.

## 2026-02-13 KD-Tree Leaf Size Tuning
- Timestamp: 2026-02-13T16:48:46Z
- Commit: b9f4ce0 (working tree with uncommitted changes)
- Hypothesis: Increase nanoflann KD-tree leaf size from `10` to `32` to reduce total diffusion topology build cost for repeated KNN assembly workloads.
- Files touched:
  - `include/igneous/data/topology.hpp`
- Benchmark commands:
  - `IGNEOUS_BENCH_MODE=1 ./build/bench_dod --benchmark_filter='bench_diffusion_build/2000|bench_markov_step/2000|bench_eigenbasis/2000/16' --benchmark_min_time=0.2s --benchmark_repetitions=10 --benchmark_report_aggregates_only=true`
  - `./scripts/perf/run_deep_bench.sh`
- Smoke results:
  - `bench_diffusion_build/2000`: `2.94 ms` mean
  - `bench_eigenbasis/2000/16`: `14.95 ms` mean
- Deep results (vs `bench_dod_20260213-094503.json`):
  - `bench_diffusion_build/2000`: `-7.27%`
  - `bench_eigenbasis/2000/16`: `-0.80%`
  - `bench_markov_step/2000`: `+1.89%`
- Profile traces:
  - `notes/perf/profiles/20260213-094817/time-profiler.trace`
  - `notes/perf/profiles/20260213-094827/cpu-counters.trace`
- Numeric checks: all doctest suites pass (`7/7`).
- Decision: `kept`
- Notes: Strong diffusion topology build gain with neutral downstream impact.

## 2026-02-13 KD-Tree Leaf Size 64 Sweep (Rejected)
- Timestamp: 2026-02-13T16:49:58Z
- Commit: 867417c (working tree with uncommitted changes)
- Hypothesis: Increase KD-tree leaf size from `32` to `64` for further diffusion build reduction.
- Files touched:
  - `include/igneous/data/topology.hpp` (reverted)
- Benchmark commands:
  - `IGNEOUS_BENCH_MODE=1 ./build/bench_dod --benchmark_filter='bench_diffusion_build/2000|bench_markov_step/2000|bench_eigenbasis/2000/16' --benchmark_min_time=0.2s --benchmark_repetitions=10 --benchmark_report_aggregates_only=true`
- Smoke results:
  - `bench_diffusion_build/2000`: `3.12 ms` mean (`+6%` vs leaf-size-32 baseline)
- Numeric checks: all doctest suites pass (`7/7`).
- Decision: `rejected`
- Notes: Larger leaf size regressed diffusion build throughput.

## 2026-02-13 Unsorted KNN Retrieval (Rejected)
- Timestamp: 2026-02-13T16:51:41Z
- Commit: 867417c (working tree with uncommitted changes)
- Hypothesis: Use `findNeighbors(..., SearchParameters(sorted=false))` to skip per-query sort overhead during diffusion build.
- Files touched:
  - `include/igneous/data/topology.hpp` (reverted)
- Benchmark commands:
  - `IGNEOUS_BENCH_MODE=1 ./build/bench_dod --benchmark_filter='bench_diffusion_build/2000|bench_markov_step/2000|bench_eigenbasis/2000/16' --benchmark_min_time=0.2s --benchmark_repetitions=10 --benchmark_report_aggregates_only=true`
- Smoke results:
  - `bench_diffusion_build/2000`: `2.94 ms` mean (neutral to leaf-32 baseline)
  - `bench_eigenbasis/2000/16`: `15.42 ms` mean (`+3%` regression)
- Numeric checks: all doctest suites pass (`7/7`).
- Decision: `rejected`
- Notes: Eigenbasis path regressed despite build-time similarity.

## 2026-02-13 Face-Unpack Skip On Rebuild (Rejected)
- Timestamp: 2026-02-13T16:52:28Z
- Commit: 867417c (working tree with uncommitted changes)
- Hypothesis: Skip `faces_to_vertices` unpack when face-array sizes already match to reduce repeated triangle topology build work.
- Files touched:
  - `include/igneous/data/topology.hpp` (reverted)
- Benchmark commands:
  - `IGNEOUS_BENCH_MODE=1 ./build/bench_dod --benchmark_filter='bench_mesh_topology_build/400|bench_curvature_kernel/400|bench_flow_kernel/400' --benchmark_min_time=0.2s --benchmark_repetitions=10 --benchmark_report_aggregates_only=true`
- Smoke results:
  - `bench_mesh_topology_build/400`: `4.60 ms` mean (`~1%` gain)
- Numeric checks: all doctest suites pass (`7/7`).
- Decision: `rejected`
- Notes: Improvement was below threshold and relied on stale-face assumptions.

## 2026-02-13 Weak-Derivative Matrix Coupling
- Timestamp: 2026-02-13T16:55:39Z
- Commit: 867417c (working tree with uncommitted changes)
- Hypothesis: Replace per-row dot-product assembly in weak derivative with dense coupling matrices (`weighted_u^T * gamma_x_phi`) to better exploit Eigen kernels.
- Files touched:
  - `include/igneous/ops/hodge.hpp`
- Benchmark commands:
  - `IGNEOUS_BENCH_MODE=1 ./build/bench_dod --benchmark_filter='bench_weak_derivative/2000/16|bench_curl_energy/2000/16|bench_hodge_solve/2000/16|bench_1form_gram/2000/16' --benchmark_min_time=0.2s --benchmark_repetitions=10 --benchmark_report_aggregates_only=true`
  - `./scripts/perf/run_deep_bench.sh`
- Smoke results:
  - `bench_weak_derivative/2000/16`: `1.47 ms` mean
- Deep results (vs `bench_dod_20260213-094728.json`):
  - `bench_weak_derivative/2000/16`: `-3.25%`
  - `bench_1form_gram/2000/16`: `-1.12%`
  - `bench_curl_energy/2000/16`: `-0.90%`
- Profile traces:
  - `notes/perf/profiles/20260213-095516/time-profiler.trace`
  - `notes/perf/profiles/20260213-095528/cpu-counters.trace`
- Numeric checks: all doctest suites pass (`7/7`).
- Decision: `kept`
- Notes: Cleared threshold on the primary target (`weak_derivative`) with no material regressions elsewhere.

## 2026-02-13 Curl-Energy Array Fusion
- Timestamp: 2026-02-13T18:19:26Z
- Commit: 1ab1f43 (working tree with uncommitted changes)
- Hypothesis: Replace temporary-vector assembly in curl-energy accumulation with a fused Eigen array expression to keep SIMD packetization and reduce temporary traffic.
- Files touched:
  - `include/igneous/ops/hodge.hpp`
- Benchmark commands:
  - `IGNEOUS_BENCH_MODE=1 ./build/bench_dod --benchmark_filter='bench_1form_gram/2000/16|bench_weak_derivative/2000/16|bench_curl_energy/2000/16|bench_hodge_solve/2000/16' --benchmark_min_time=0.2s --benchmark_repetitions=10 --benchmark_report_aggregates_only=true`
  - `./scripts/perf/run_deep_bench.sh`
- Smoke results:
  - `bench_curl_energy/2000/16`: `5.94 ms` mean
  - `bench_weak_derivative/2000/16`: `1.43 ms` mean
- Deep results (vs `bench_dod_20260213-095422.json`):
  - `bench_curl_energy/2000/16`: `-3.43%`
  - `bench_weak_derivative/2000/16`: `-0.33%`
  - `bench_1form_gram/2000/16`: `-0.03%`
  - `bench_hodge_solve/2000/16`: `+1.51%`
- Profile traces:
  - `notes/perf/profiles/20260213-111704/time-profiler.trace`
  - `notes/perf/profiles/20260213-111715/cpu-counters.trace`
- Numeric checks: all doctest suites pass (`7/7`).
- Decision: `kept`
- Notes: Cleared keep threshold on primary target (`curl_energy`) with no material regressions on downstream kernels.

## 2026-02-13 Curl-Energy Preweighted Dot Path (Rejected)
- Timestamp: 2026-02-13T18:21:45Z
- Commit: 7be6233 (working tree with uncommitted changes)
- Hypothesis: Precompute `mu`-weighted gamma vectors (`gamma_xx_mu`, `gamma_phi_x_mu`) and replace fused elementwise expression with two dot products to increase SIMD throughput.
- Files touched:
  - `include/igneous/ops/hodge.hpp` (reverted)
- Benchmark commands:
  - `IGNEOUS_BENCH_MODE=1 ./build/bench_dod --benchmark_filter='bench_1form_gram/2000/16|bench_weak_derivative/2000/16|bench_curl_energy/2000/16|bench_hodge_solve/2000/16' --benchmark_min_time=0.2s --benchmark_repetitions=10 --benchmark_report_aggregates_only=true`
  - `./scripts/perf/run_deep_bench.sh`
- Smoke results:
  - `bench_curl_energy/2000/16`: `5.93 ms` mean
- Deep results (vs `bench_dod_20260213-111843.json`):
  - `bench_curl_energy/2000/16`: `-1.23%`
  - `bench_weak_derivative/2000/16`: `-0.13%`
  - `bench_1form_gram/2000/16`: `+0.28%`
  - `bench_hodge_solve/2000/16`: `-1.90%`
- Numeric checks: all doctest suites pass (`7/7`).
- Decision: `rejected`
- Notes: Primary gain was below `3%` keep threshold; retained the simpler fused expression.

## 2026-02-13 Diffusion Eigen-Array Exp Path (Rejected)
- Timestamp: 2026-02-13T18:23:25Z
- Commit: 7be6233 (working tree with uncommitted changes)
- Hypothesis: Replace per-neighbor scalar kernel accumulation with Eigen array `exp`/mask/sum to increase SIMD utilization in `DiffusionTopology::build`.
- Files touched:
  - `include/igneous/data/topology.hpp` (reverted)
- Benchmark command:
  - `IGNEOUS_BENCH_MODE=1 ./build/bench_dod --benchmark_filter='bench_diffusion_build/2000|bench_markov_step/2000|bench_eigenbasis/2000/16|bench_1form_gram/2000/16|bench_weak_derivative/2000/16|bench_curl_energy/2000/16' --benchmark_min_time=0.2s --benchmark_repetitions=10 --benchmark_report_aggregates_only=true`
- Smoke results:
  - `bench_diffusion_build/2000`: `3.02 ms` mean (regression vs current baseline class ~`2.90 ms`)
  - `bench_markov_step/2000`: `17.76 us` mean
- Numeric checks: all doctest suites pass (`7/7`).
- Decision: `rejected`
- Notes: Vectorized array expression introduced overhead and worsened diffusion build throughput.

## 2026-02-13 1-Form Gram Block GEMM Path (Rejected)
- Timestamp: 2026-02-13T18:25:09Z
- Commit: 7be6233 (working tree with uncommitted changes)
- Hypothesis: Replace pairwise Gram assembly with block matrix products (`U^T * diag(mu*gamma_ab) * U`) and compute only unique `(a,b)` blocks to exploit SIMD/BLAS kernels.
- Files touched:
  - `include/igneous/ops/geometry.hpp` (reverted)
- Benchmark commands:
  - `IGNEOUS_BENCH_MODE=1 ./build/bench_dod --benchmark_filter='bench_1form_gram/2000/16|bench_weak_derivative/2000/16|bench_curl_energy/2000/16|bench_hodge_solve/2000/16' --benchmark_min_time=0.2s --benchmark_repetitions=10 --benchmark_report_aggregates_only=true`
  - `./scripts/perf/run_deep_bench.sh`
  - `IGNEOUS_BENCH_MODE=1 ./build/bench_dod --benchmark_filter='bench_diffusion_build/2000|bench_1form_gram/2000/16' --benchmark_min_time=0.3s --benchmark_repetitions=30 --benchmark_report_aggregates_only=true` (A/B check)
- Smoke results:
  - `bench_1form_gram/2000/16`: `0.290 ms` mean
- Deep results (vs `bench_dod_20260213-111843.json`):
  - `bench_1form_gram/2000/16`: `-13.31%`
  - `bench_diffusion_build/2000`: `+3.96%`
  - `bench_eigenbasis/2000/16`: `+1.50%`
  - `bench_curl_energy/2000/16`: `+0.07%`
- A/B confirmation:
  - baseline (`no patch`) `bench_diffusion_build/2000`: `2.888 ms` mean (`30 reps`)
  - with patch `bench_diffusion_build/2000`: `2.980 ms` mean (`30 reps`)
- Numeric checks: all doctest suites pass (`7/7`).
- Decision: `rejected`
- Notes: Strong local gain for `1form_gram`, but rejected due repeatable diffusion-build regression on a primary benchmark.

## 2026-02-13 Flow Displacement SoA Workspace (Rejected)
- Timestamp: 2026-02-13T18:28:47Z
- Commit: 7be6233 (working tree with uncommitted changes)
- Hypothesis: Switch flow displacement workspace from `Vec3` AoS to `dx/dy/dz` SoA to improve contiguous access and vectorization in update loops.
- Files touched:
  - `include/igneous/ops/flow.hpp` (reverted)
- Benchmark command:
  - `IGNEOUS_BENCH_MODE=1 ./build/bench_dod --benchmark_filter='bench_flow_kernel/400|bench_curvature_kernel/400|bench_mesh_topology_build/400' --benchmark_min_time=0.2s --benchmark_repetitions=10 --benchmark_report_aggregates_only=true`
- Smoke results:
  - `bench_flow_kernel/400`: `0.587 ms` mean (major regression)
- Numeric checks: all doctest suites pass (`7/7`).
- Decision: `rejected`
- Notes: Extra scalar-array traffic and fallback conversion overhead outweighed any SoA benefits.

## 2026-02-13 Curl-Energy Coupling-Matrix Precompute (Rejected)
- Timestamp: 2026-02-13T18:30:12Z
- Commit: 7be6233 (working tree with uncommitted changes)
- Hypothesis: Precompute dense `gamma_phi_x` coupling matrices to remove per-(k,l,a,b) inner-loop dot products in curl-energy assembly.
- Files touched:
  - `include/igneous/ops/hodge.hpp` (reverted)
- Benchmark command:
  - `IGNEOUS_BENCH_MODE=1 ./build/bench_dod --benchmark_filter='bench_curl_energy/2000/16|bench_weak_derivative/2000/16|bench_hodge_solve/2000/16|bench_1form_gram/2000/16' --benchmark_min_time=0.2s --benchmark_repetitions=10 --benchmark_report_aggregates_only=true`
- Smoke results:
  - `bench_curl_energy/2000/16`: `5.905 ms` mean (`<3%` improvement)
- Numeric checks: all doctest suites pass (`7/7`).
- Decision: `rejected`
- Notes: Gain did not meet keep threshold and added substantial workspace complexity.

## 2026-02-13 Circular-Coordinate Matrix Rewrite
- Timestamp: 2026-02-13T18:31:53Z
- Commit: 7be6233 (working tree with uncommitted changes)
- Hypothesis: Rewrite circular-coordinate assembly from nested `(s,t,k,a)` scalar loops to matrix form (`q = U * alpha`, then `X_op.col(t) = U^T * (mu .* sum_a(gamma_x_phi[a,t] .* q_a))`) to unlock SIMD/BLAS kernels and remove temporary-vector churn.
- Files touched:
  - `include/igneous/ops/hodge.hpp`
- Benchmark commands:
  - `IGNEOUS_BENCH_MODE=1 /usr/bin/time -p ./build/igneous-hodge` (3-run before/after)
  - `./scripts/perf/run_deep_bench.sh`
- Smoke pipeline results (`igneous-hodge`, 3 runs):
  - baseline: `1.25 s`, `1.24 s`, `1.25 s`
  - candidate: `0.33 s`, `0.32 s`, `0.32 s`
  - mean delta: `-74.07%`
- Deep results (vs `bench_dod_20260213-111843.json`):
  - `bench_1form_gram/2000/16`: `+0.05%`
  - `bench_weak_derivative/2000/16`: `+0.20%`
  - `bench_curl_energy/2000/16`: `-0.13%`
  - `bench_hodge_solve/2000/16`: `-1.21%`
- Profile traces:
  - `notes/perf/profiles/20260213-113237/time-profiler.trace`
  - `notes/perf/profiles/20260213-113245/cpu-counters.trace`
- Numeric checks: all doctest suites pass (`7/7`).
- Decision: `kept`
- Notes: Large real-pipeline throughput gain with neutral deep-kernel impact.

## 2026-02-13 Circular-Coordinate Workspace Cache (Rejected)
- Timestamp: 2026-02-13T18:34:44Z
- Commit: 594341d (working tree with uncommitted changes)
- Hypothesis: Add workspace-cached circular-coordinate overload and reuse `gamma_x_phi` precompute across repeated calls in `main_hodge`.
- Files touched:
  - `include/igneous/ops/hodge.hpp` (reverted)
  - `src/main_hodge.cpp` (reverted)
- Benchmark commands:
  - `IGNEOUS_BENCH_MODE=1 /usr/bin/time -p ./build/igneous-hodge` (3-run before/after)
  - `./scripts/perf/run_deep_bench.sh`
  - `IGNEOUS_BENCH_MODE=1 ./build/bench_dod --benchmark_filter='bench_1form_gram/2000/16|bench_weak_derivative/2000/16|bench_curl_energy/2000/16|bench_hodge_solve/2000/16' --benchmark_min_time=0.3s --benchmark_repetitions=20 --benchmark_report_aggregates_only=true`
- Smoke pipeline results (`igneous-hodge`, 3 runs):
  - baseline (post-`594341d`): `0.33 s`, `0.32 s`, `0.32 s`
  - candidate: `0.32 s`, `0.31 s`, `0.31 s`
  - mean delta: `-3.09%`
- Deep results (vs `bench_dod_20260213-113152.json`):
  - `bench_weak_derivative/2000/16`: `+2.69%`
  - `bench_curl_energy/2000/16`: `+3.78%`
  - `bench_hodge_solve/2000/16`: `+2.65%`
  - `bench_1form_gram/2000/16`: `+2.47%`
- Focused confirmation (20 reps):
  - `bench_weak_derivative/2000/16`: `1.511 ms`
  - `bench_curl_energy/2000/16`: `6.233 ms`
  - `bench_hodge_solve/2000/16`: `74.84 us`
- Numeric checks: all doctest suites pass (`7/7`).
- Decision: `rejected`
- Notes: Small pipeline gain did not justify repeatable regressions across primary Hodge kernels.

## 2026-02-13 Coordinate Copy Fast Path (Rejected)
- Timestamp: 2026-02-13T18:37:07Z
- Commit: 594341d (working tree with uncommitted changes)
- Hypothesis: Bypass per-vertex `get_vec3()` gathers in `fill_coordinate_vectors` by copying directly from SoA `geometry.x/y/z` buffers.
- Files touched:
  - `include/igneous/ops/geometry.hpp` (reverted)
- Benchmark commands:
  - `IGNEOUS_BENCH_MODE=1 ./build/bench_dod --benchmark_filter='bench_diffusion_build/2000|bench_1form_gram/2000/16|bench_weak_derivative/2000/16|bench_curl_energy/2000/16|bench_hodge_solve/2000/16' --benchmark_min_time=0.2s --benchmark_repetitions=10 --benchmark_report_aggregates_only=true`
  - `IGNEOUS_BENCH_MODE=1 /usr/bin/time -p ./build/igneous-hodge` (3 runs)
- Smoke results:
  - `bench_1form_gram/2000/16`: `0.337 ms` mean (neutral)
  - `bench_curl_energy/2000/16`: `6.01 ms` mean (noise-level)
  - `igneous-hodge`: `0.35 s`, `0.33 s`, `0.33 s` (no improvement)
- Numeric checks: all doctest suites pass (`7/7`).
- Decision: `rejected`
- Notes: Direct copy path did not deliver a measurable throughput improvement.

## 2026-02-13 Circular Reconstruction GEMV Path (Rejected)
- Timestamp: 2026-02-13T18:38:21Z
- Commit: 594341d (working tree with uncommitted changes)
- Hypothesis: Replace scalar complex accumulation per vertex with two real GEMV passes (`U*Re`, `U*Im`) plus `atan2` in circular-coordinate reconstruction.
- Files touched:
  - `include/igneous/ops/hodge.hpp` (reverted)
- Benchmark command:
  - `IGNEOUS_BENCH_MODE=1 /usr/bin/time -p ./build/igneous-hodge` (3 runs)
- Smoke results (`igneous-hodge`):
  - candidate: `0.33 s`, `0.32 s`, `0.33 s`
  - baseline class (post-`594341d`): `0.33 s`, `0.32 s`, `0.32 s`
- Numeric checks: all doctest suites pass (`7/7`).
- Decision: `rejected`
- Notes: Runtime was effectively unchanged; no measurable throughput gain.

## 2026-02-13 Markov Unroll-8 (Rejected)
- Timestamp: 2026-02-13T18:40:09Z
- Commit: 594341d (working tree with uncommitted changes)
- Hypothesis: Increase `apply_markov_transition` CSR inner-loop unroll from `4` to `8` to improve diffusion Markov throughput via wider ILP/SIMD scheduling.
- Files touched:
  - `include/igneous/ops/geometry.hpp` (reverted)
- Benchmark commands:
  - `IGNEOUS_BENCH_MODE=1 ./build/bench_dod --benchmark_filter='bench_markov_step/2000|bench_1form_gram/2000/16|bench_curl_energy/2000/16' --benchmark_min_time=0.3s --benchmark_repetitions=30 --benchmark_report_aggregates_only=true` (A/B)
- A/B results (`30 reps`):
  - with unroll-8: `markov 16.24 us`, `1form 347.32 us`, `curl 6.07 ms`
  - baseline: `markov 17.87 us`, `1form 338.23 us`, `curl 5.99 ms`
  - deltas: `markov -9.10%`, `1form +2.69%`, `curl +1.28%`
- Numeric checks: all doctest suites pass (`7/7`).
- Decision: `rejected`
- Notes: Strong isolated Markov win, but rejected due regressions on other primary diffusion-derived kernels.

## 2026-02-13 Real-World Pipeline Harness
- Timestamp: 2026-02-13T18:48:01Z
- Commit: 88ea1af (working tree with uncommitted changes)
- Hypothesis: Add an end-to-end benchmark harness mirroring `main_diffusion`, `main_spectral`, and `main_hodge` to target dominant real-world phases instead of only micro-kernels.
- Files touched:
  - `benches/bench_pipelines.cpp`
  - `CMakeLists.txt`
- Benchmark command:
  - `IGNEOUS_BENCH_MODE=1 ./build/bench_pipelines --benchmark_min_time=0.1s --benchmark_repetitions=5 --benchmark_report_aggregates_only=true`
- Baseline characterization (initial run):
  - `bench_pipeline_hodge_main`: `323.133 ms`
  - `bench_hodge_phase_eigenbasis`: `123.883 ms`
  - `bench_hodge_phase_curl_energy`: `144.710 ms`
- Numeric checks: all doctest suites pass (`7/7`).
- Decision: `kept`
- Notes: Harness exposed real bottlenecks (`eigenbasis` and `curl_energy`) and now guides optimization priority.

## 2026-02-13 Curl-Energy Coupling Matrix (Real-World Recheck, Rejected)
- Timestamp: 2026-02-13T18:49:55Z
- Commit: 88ea1af (working tree with uncommitted changes)
- Hypothesis: Precompute `gamma_phi_x` coupling matrices and use matrix lookups in `compute_curl_energy_matrix` to remove one vector dot term from the hot inner loop.
- Files touched:
  - `include/igneous/ops/hodge.hpp` (reverted)
- Benchmark command:
  - `IGNEOUS_BENCH_MODE=1 ./build/bench_pipelines --benchmark_filter='bench_hodge_phase_curl_energy|bench_pipeline_hodge_main' --benchmark_min_time=0.2s --benchmark_repetitions=10 --benchmark_report_aggregates_only=true` (A/B)
- A/B CPU-time results (`10 reps`):
  - with patch: `bench_hodge_phase_curl_energy=143.680 ms`, `bench_pipeline_hodge_main=318.897 ms`
  - baseline: `bench_hodge_phase_curl_energy=145.345 ms`, `bench_pipeline_hodge_main=321.781 ms`
  - deltas: `curl_energy -1.15%`, `pipeline_hodge -0.90%`
- Numeric checks: all doctest suites pass (`7/7`).
- Decision: `rejected`
- Notes: Improvement remained below keep threshold on primary real-world target.

## 2026-02-13 Adaptive Arnoldi NCV (Large Basis)
- Timestamp: 2026-02-13T18:58:25Z
- Commit: 88ea1af (working tree with uncommitted changes)
- Hypothesis: Use a compact Arnoldi subspace only for large basis solves (`n_eigenvectors >= 32`) with automatic fallback to the full subspace if convergence is incomplete.
- Files touched:
  - `include/igneous/ops/spectral.hpp`
- Benchmark commands:
  - `IGNEOUS_BENCH_MODE=1 ./build/bench_pipelines --benchmark_filter='bench_pipeline_hodge_main|bench_hodge_phase_eigenbasis|bench_pipeline_spectral_main' --benchmark_min_time=0.1s --benchmark_repetitions=5 --benchmark_report_aggregates_only=true` (A/B off/on)
  - `IGNEOUS_BENCH_MODE=1 /usr/bin/time -p ./build/igneous-hodge` (5-run off/on)
- A/B CPU-time results (`bench_pipelines`, 5 reps):
  - `bench_pipeline_hodge_main`: `320.505 ms -> 306.240 ms` (`-4.45%`)
  - `bench_hodge_phase_eigenbasis`: `125.386 ms -> 108.015 ms` (`-13.85%`)
- App-level A/B (`igneous-hodge`, 5 runs):
  - baseline mean: `0.322 s`
  - candidate mean: `0.302 s`
  - delta: `-6.21%`
- Numeric checks: all doctest suites pass (`7/7`).
- Decision: `kept`
- Notes: Cleared keep threshold on real-world throughput target with stable convergence via fallback path.

## 2026-02-13 Adaptive NCV `k+8` Sweep (Rejected)
- Timestamp: 2026-02-13T18:52:58Z
- Commit: 88ea1af (working tree with uncommitted changes)
- Hypothesis: Tighten compact Arnoldi space to `n_eigenvectors + 8` for additional speed.
- Files touched:
  - `include/igneous/ops/spectral.hpp` (reverted)
- Benchmark command:
  - `IGNEOUS_BENCH_MODE=1 ./build/bench_pipelines --benchmark_filter='bench_pipeline_hodge_main|bench_hodge_phase_eigenbasis' --benchmark_min_time=0.1s --benchmark_repetitions=5 --benchmark_report_aggregates_only=true`
- Results:
  - `bench_hodge_phase_eigenbasis`: `180.231 ms` (major regression)
  - `bench_pipeline_hodge_main`: `385.794 ms` (major regression)
- Numeric checks: all doctest suites pass (`7/7`).
- Decision: `rejected`
- Notes: Compact space was too small and convergence cost exploded.

## 2026-02-13 Adaptive NCV `k+12` Sweep (Rejected)
- Timestamp: 2026-02-13T18:53:36Z
- Commit: 88ea1af (working tree with uncommitted changes)
- Hypothesis: Use `n_eigenvectors + 12` compact Arnoldi size as a middle ground.
- Files touched:
  - `include/igneous/ops/spectral.hpp` (reverted)
- Benchmark command:
  - `IGNEOUS_BENCH_MODE=1 ./build/bench_pipelines --benchmark_filter='bench_pipeline_hodge_main|bench_hodge_phase_eigenbasis' --benchmark_min_time=0.1s --benchmark_repetitions=5 --benchmark_report_aggregates_only=true`
- Results:
  - `bench_hodge_phase_eigenbasis`: `118.798 ms`
  - `bench_pipeline_hodge_main`: `316.084 ms`
- Numeric checks: all doctest suites pass (`7/7`).
- Decision: `rejected`
- Notes: Better than baseline but dominated by the `k+16` variant.

## 2026-02-13 Spectral Gram Eigenvalues-Only (Rejected)
- Timestamp: 2026-02-13T18:59:36Z
- Commit: 88ea1af (working tree with uncommitted changes)
- Hypothesis: Use `Eigen::EigenvaluesOnly` for Gram condition estimation in spectral pipeline to skip unnecessary eigenvector computation.
- Files touched:
  - `src/main_spectral.cpp` (reverted)
  - `benches/bench_pipelines.cpp` (reverted)
- Benchmark command:
  - `IGNEOUS_BENCH_MODE=1 ./build/bench_pipelines --benchmark_filter='bench_pipeline_spectral_main' --benchmark_min_time=0.1s --benchmark_repetitions=10 --benchmark_report_aggregates_only=true` (A/B)
- A/B CPU-time results:
  - with change: `34.238 ms`
  - baseline: `33.858 ms`
  - delta: `+1.12%`
- Numeric checks: all doctest suites pass (`7/7`).
- Decision: `rejected`
- Notes: No reliable throughput gain.

## 2026-02-13 Threaded Curl-Energy Assembly + Backend Control
- Timestamp: 2026-02-13T19:07:36Z
- Commit: 9e02b50 (working tree with uncommitted changes)
- Hypothesis: Parallelize `compute_curl_energy_matrix` over basis indices with thread-local temporaries; add runtime backend controls so users can choose serial CPU vs threaded CPU/GPU mode.
- Files touched:
  - `include/igneous/core/parallel.hpp`
  - `include/igneous/ops/hodge.hpp`
- Runtime controls introduced:
  - `IGNEOUS_BACKEND=cpu` (forces serial CPU)
  - `IGNEOUS_BACKEND=parallel` (or unset; uses threaded CPU)
  - `IGNEOUS_BACKEND=gpu` (currently routes to threaded CPU path)
  - `IGNEOUS_NUM_THREADS=<N>` (caps worker count)
- Benchmark commands:
  - `IGNEOUS_BENCH_MODE=1 ./build/bench_pipelines --benchmark_filter='bench_pipeline_hodge_main|bench_hodge_phase_curl_energy|bench_hodge_phase_weak_derivative|bench_hodge_phase_eigenbasis' --benchmark_min_time=0.1s --benchmark_repetitions=5 --benchmark_report_aggregates_only=true` (threaded)
  - `IGNEOUS_BACKEND=cpu IGNEOUS_BENCH_MODE=1 ./build/bench_pipelines --benchmark_filter='bench_pipeline_hodge_main|bench_hodge_phase_curl_energy|bench_hodge_phase_weak_derivative|bench_hodge_phase_eigenbasis' --benchmark_min_time=0.1s --benchmark_repetitions=5 --benchmark_report_aggregates_only=true` (serial baseline)
  - `./scripts/perf/run_deep_bench.sh`
  - `IGNEOUS_BENCH_MODE=1 /usr/bin/time -p ./build/igneous-hodge` (5-run before/after)
- A/B results (`bench_pipelines`, serial CPU vs threaded default):
  - `bench_pipeline_hodge_main`: `303.402 ms -> 161.566 ms` (`-46.75%`)
  - `bench_hodge_phase_curl_energy`: `142.376 ms -> 0.859 ms` (`-99.40%` CPU-time metric)
  - `bench_hodge_phase_weak_derivative`: `12.260 ms -> 12.165 ms` (`-0.78%`)
  - `bench_hodge_phase_eigenbasis`: `106.716 ms -> 111.086 ms` (`+4.09%`)
- App-level throughput (`igneous-hodge`, 5 runs):
  - baseline mean: `0.308 s`
  - threaded mean: `0.180 s`
  - delta: `-41.56%`
- Deep benchmark highlight (vs `bench_dod_20260213-115708.json`):
  - `bench_curl_energy/2000/16` CPU-time: `6.030 ms -> 0.552 ms` (`-90.85%`)
- Profile traces:
  - `notes/perf/profiles/20260213-120846/time-profiler.trace` (serial CPU)
  - `notes/perf/profiles/20260213-120826/time-profiler.trace` (threaded)
  - `notes/perf/profiles/20260213-120904/cpu-counters.trace` (serial CPU)
  - `notes/perf/profiles/20260213-120847/cpu-counters.trace` (threaded)
- Numeric checks: all doctest suites pass (`7/7`).
- Decision: `kept`
- Notes: Real-world throughput gain is large; users can force serial CPU behavior when needed.

## 2026-02-13 Threaded Circular-Coordinate Assembly
- Timestamp: 2026-02-13T19:13:00Z
- Commit: a816dc6 (working tree with uncommitted changes)
- Hypothesis: Extend threaded backend to `compute_circular_coordinates` by parallelizing `gamma_x_phi` precompute and per-column `X_op` assembly to reduce the next-largest Hodge phase after threaded curl-energy.
- Files touched:
  - `include/igneous/ops/hodge.hpp`
- Benchmark commands:
  - `IGNEOUS_BENCH_MODE=1 ./build/bench_pipelines --benchmark_filter='bench_pipeline_hodge_main|bench_hodge_phase_circular|bench_hodge_phase_curl_energy|bench_hodge_phase_eigenbasis|bench_hodge_phase_weak_derivative' --benchmark_min_time=0.1s --benchmark_repetitions=5 --benchmark_report_aggregates_only=true`
  - `IGNEOUS_BACKEND=cpu IGNEOUS_BENCH_MODE=1 ./build/bench_pipelines --benchmark_filter='bench_pipeline_hodge_main|bench_hodge_phase_circular|bench_hodge_phase_curl_energy|bench_hodge_phase_eigenbasis|bench_hodge_phase_weak_derivative' --benchmark_min_time=0.1s --benchmark_repetitions=5 --benchmark_report_aggregates_only=true`
  - `./scripts/perf/run_deep_bench.sh`
  - `IGNEOUS_BENCH_MODE=1 /usr/bin/time -p ./build/igneous-hodge` (5 runs)
- A/B results (`bench_pipelines`, serial CPU vs threaded default):
  - `bench_pipeline_hodge_main`: `311.486 ms -> 141.164 ms` (`-54.68%`)
  - `bench_hodge_phase_circular`: `27.441 ms -> 2.815 ms` (`-89.74%`)
  - `bench_hodge_phase_curl_energy`: `151.985 ms -> 0.848 ms` (`-99.44%`)
  - `bench_hodge_phase_eigenbasis`: `110.787 ms -> 109.603 ms` (`-1.07%`)
- App-level throughput (`igneous-hodge`, 5 runs):
  - previous threaded mean: `0.180 s`
  - candidate mean: `0.160 s`
  - delta: `-11.11%`
- Deep benchmark highlight (vs `bench_dod_20260213-120652.json`):
  - `bench_1form_gram/2000/16`: `-3.91%`
  - `bench_curl_energy/2000/16`: `+2.29%` (still far below pre-thread baseline class)
- Profile traces:
  - `notes/perf/profiles/20260213-121355/time-profiler.trace` (serial CPU)
  - `notes/perf/profiles/20260213-121356/time-profiler.trace` (threaded)
  - `notes/perf/profiles/20260213-121411/cpu-counters.trace` (serial CPU)
  - `notes/perf/profiles/20260213-121412/cpu-counters.trace` (threaded)
- Numeric checks: all doctest suites pass (`7/7`).
- Decision: `kept`
- Notes: Additional real-world Hodge throughput gain while preserving backend switchability.

## 2026-02-13 Threaded Geometry + Diffusion Topology Build (Global CPU Pass)
- Timestamp: 2026-02-13T19:35:18Z
- Commit: 73a8b91 (working tree with uncommitted changes)
- Hypothesis: Extend threaded backend beyond Hodge by parallelizing high-cardinality geometry kernels (`curvature`, `flow`) and `DiffusionTopology::build`; tune scheduler granularity for mixed regular/irregular kernels.
- Files touched:
  - `include/igneous/core/parallel.hpp`
  - `include/igneous/data/topology.hpp`
  - `include/igneous/ops/curvature.hpp`
  - `include/igneous/ops/flow.hpp`
- Benchmark commands:
  - `IGNEOUS_BENCH_MODE=1 ./build/bench_geometry`
  - `IGNEOUS_BENCH_MODE=1 ./build/bench_dod --benchmark_min_time=0.2s --benchmark_repetitions=10 --benchmark_report_aggregates_only=true --benchmark_out=notes/perf/results/bench_dod_20260213-123249.json --benchmark_out_format=json`
  - `IGNEOUS_BENCH_MODE=1 ./build/bench_pipelines --benchmark_min_time=0.12s --benchmark_repetitions=10 --benchmark_report_aggregates_only=true --benchmark_out=notes/perf/results/bench_pipelines_20260213-1233-threaded.json --benchmark_out_format=json`
  - `IGNEOUS_BACKEND=cpu IGNEOUS_BENCH_MODE=1 ./build/bench_pipelines --benchmark_min_time=0.12s --benchmark_repetitions=10 --benchmark_report_aggregates_only=true --benchmark_out=notes/perf/results/bench_pipelines_20260213-1233-cpu.json --benchmark_out_format=json`
  - `ctest --test-dir build --output-on-failure`
- A/B results (`bench_pipelines`, 10 reps, serial CPU vs threaded default, wall time):
  - `bench_pipeline_diffusion_main/20`: `4.228 ms -> 1.342 ms` (`-68.25%`)
  - `bench_pipeline_diffusion_main/100`: `5.927 ms -> 3.093 ms` (`-47.82%`)
  - `bench_pipeline_spectral_main`: `33.384 ms -> 30.437 ms` (`-8.83%`)
  - `bench_pipeline_hodge_main`: `302.312 ms -> 148.748 ms` (`-50.80%`)
  - `bench_hodge_phase_topology`: `6.136 ms -> 1.389 ms` (`-77.37%`)
  - `bench_hodge_phase_eigenbasis`: `107.208 ms -> 106.645 ms` (`-0.53%`)
  - `bench_hodge_phase_curl_energy`: `143.066 ms -> 14.958 ms` (`-89.54%`)
  - `bench_hodge_phase_circular`: `25.405 ms -> 5.308 ms` (`-79.11%`)
- App-level throughput (`main_*`, 3 runs, `/usr/bin/time -p`, bench mode):
  - `igneous-diffusion assets/bunny.obj`: `0.01 s -> 0.00 s`
  - `igneous-spectral assets/bunny.obj`: `0.04 s -> 0.03 s`
  - `igneous-hodge`: `0.30 s -> 0.15 s`
- Delta vs pre-change threaded baseline (`bench_pipelines_20260213-current.txt`, wall time):
  - `bench_pipeline_diffusion_main/100`: `5.914 ms -> 3.093 ms` (`-47.71%`)
  - `bench_pipeline_spectral_main`: `33.395 ms -> 30.437 ms` (`-8.86%`)
  - `bench_pipeline_hodge_main`: `154.395 ms -> 148.748 ms` (`-3.66%`)
- Deep `bench_dod` delta vs pre-change baseline (wall time):
  - `bench_curvature_kernel/400`: `7.239 ms -> 0.998 ms` (`-86.21%`)
  - `bench_flow_kernel/400`: `0.519 ms -> 0.335 ms` (`-35.44%`)
  - `bench_diffusion_build/2000`: `2.904 ms -> 0.770 ms` (`-73.48%`)
- Geometry benchmark highlights (`bench_geometry`, pre-change vs latest threaded run):
  - `Grid 500x500`: `Curv 12.125 -> 1.421 ms` (`-88.28%`), `Flow 0.826 -> 0.413 ms` (`-50.00%`)
  - `Grid 1000x1000`: `Curv 46.532 -> 5.180 ms` (`-88.87%`), `Flow 3.397 -> 0.856 ms` (`-74.80%`)
- Numeric checks: all doctest suites pass (`7/7`).
- Decision: `kept`
- Notes:
  - Static-range scheduler prototype was tested and rejected due imbalance on irregular triangular work; final scheduler uses coarse dynamic chunking.
  - Small flow workloads (`<= 62.5k verts`) remain better on serial CPU due thread startup overhead; large grids and pipeline workloads are strongly improved.

## 2026-02-13 TriangleTopology Neighbor Build Dual-Path
- Timestamp: 2026-02-13T19:51:26Z
- Commit: e6f4bde (working tree with uncommitted changes)
- Hypothesis: Replace the stamp-based neighbor build with a dedupe-based path for large meshes and keep the stamp path for small meshes to improve topology throughput without tiny-mesh regressions.
- Files touched:
  - `include/igneous/data/topology.hpp`
- Implementation notes:
  - Threaded unpack of `faces_to_vertices -> face_v{0,1,2}`.
  - New large-mesh neighbor path: per-vertex unique-neighbor gather with thread-local scratch vectors, then count/prefix/write passes.
  - Small-mesh fallback (`num_vertices < 50000`): retained original stamp algorithm for best low-overhead behavior.
- Benchmark commands:
  - `IGNEOUS_BENCH_MODE=1 ./build/bench_geometry`
  - `IGNEOUS_BENCH_MODE=1 ./build/bench_dod --benchmark_min_time=0.2s --benchmark_repetitions=5 --benchmark_report_aggregates_only=true --benchmark_out=notes/perf/results/bench_dod_20260213-tri-nei-h3.json --benchmark_out_format=json`
- Results vs pre-hypothesis threaded baseline:
  - `bench_mesh_topology_build/400`: `4.566 ms -> 2.966 ms` (`-35.04%`)
  - `bench_curvature_kernel/400`: `0.984 ms -> 0.977 ms` (`-0.71%`)
  - `bench_flow_kernel/400`: `0.334 ms -> 0.330 ms` (`-1.33%`)
  - `bench_diffusion_build/2000`: `0.753 ms -> 0.757 ms` (`+0.42%`)
- Geometry benchmark topology deltas:
  - `Grid 100x100`: `0.403 -> 0.346 ms` (`-14.14%`)
  - `Grid 250x250`: `2.308 -> 1.862 ms` (`-19.32%`)
  - `Grid 500x500`: `7.816 -> 5.028 ms` (`-35.67%`)
  - `Grid 1000x1000`: `31.840 -> 17.853 ms` (`-43.93%`)
- Numeric checks: all doctest suites pass (`7/7`).
- Decision: `kept`
- Notes:
  - First dedupe-only attempt was rejected for small meshes; dual-path resolved the regression.
