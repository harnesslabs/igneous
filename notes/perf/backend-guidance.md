# Backend Guidance (2026-02-13)

## Current backend modes
- `IGNEOUS_BACKEND=cpu`: serial CPU execution.
- `IGNEOUS_BACKEND=parallel` (default): threaded CPU execution.
- `IGNEOUS_BACKEND=gpu`: enables Metal diffusion kernels when workload size clears offload gate.

## GPU controls
- `IGNEOUS_GPU_MIN_ROWS=<int>`: minimum vertex count for GPU offload (default: `8192`).
- `IGNEOUS_GPU_FORCE=1`: force GPU offload (debug/experiments only).

## Measured guidance (Apple M3 Max, Release)
From `notes/perf/results/bench_pipelines_20260213-workerpool-cpuparallel-v2.txt` and `notes/perf/results/bench_pipelines_20260213-gpu-*.txt` (5 reps):
- `bench_pipeline_diffusion_main/100`: `cpuparallel 3.002 ms`, `gpu-gated 2.986 ms`, `gpu-forced 19.627 ms`.
- `bench_pipeline_spectral_main`: `cpuparallel 23.120 ms`, `gpu-gated 22.217 ms`.
- `bench_pipeline_hodge_main`: `cpuparallel 115.609 ms`, `gpu-gated 116.474 ms`.
- `bench_markov_step/2000` (`bench_dod`): `cpuparallel 17.24 us`, `gpu-forced 178.75 us`.

App-level (`notes/perf/results/main_timings_20260213-gpu-gating-h1.txt`, 3 runs):
- `igneous-diffusion assets/bunny.obj`: cpuparallel/gpu-gated `~0.03-0.04 s`; gpu-forced `~0.04-0.05 s`.
- `igneous-hodge`: cpuparallel/gpu-gated `~0.14-0.15 s`; gpu-forced `~0.21-0.22 s`.
- `igneous-spectral assets/bunny.obj`: cpuparallel/gpu-gated `~0.05-0.06 s`; gpu-forced `~0.06 s`.

## Recommended usage
- Use `parallel` for current real workloads and all tested 2k-4k point pipelines.
- Use `gpu` only when working on large diffusion graphs (default gate prevents small-workload regressions).
- Keep `IGNEOUS_GPU_FORCE=1` for profiling/debug only; it regresses current real-world benchmarks.
- Use `cpu` for deterministic debugging or tiny kernels where threading overhead dominates.

## GPU roadmap candidates
- Diffusion graph construction (`DiffusionTopology::build`) for very large point clouds.
- Batched/vector-resident diffusion iterations to amortize host-device synchronization.
- Dense linear algebra in Hodge phases after data transfer and batching strategy is in place.
