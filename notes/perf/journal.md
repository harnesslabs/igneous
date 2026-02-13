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
