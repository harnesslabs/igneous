# Backend Guidance (2026-02-13)

## Current backend modes
- `IGNEOUS_BACKEND=cpu`: serial CPU execution.
- `IGNEOUS_BACKEND=parallel` (default): threaded CPU execution.
- `IGNEOUS_BACKEND=gpu`: enables Metal diffusion kernels when workload size clears offload gate.

## GPU controls
- `IGNEOUS_GPU_MIN_ROWS=<int>`: minimum vertex count for GPU offload (default: `8192`).
- `IGNEOUS_GPU_MIN_ROW_STEPS=<int>`: minimum `rows * steps` work for multi-step GPU offload (default: `200000`).
- `IGNEOUS_GPU_FORCE=1`: force GPU offload (debug/experiments only).

## Measured guidance (Apple M3 Max, Release)
From `notes/perf/results/bench_dod_20260213-gpu-steps-*.txt` and `notes/perf/results/bench_pipelines_20260213-gpu-policy-h1.txt` (5 reps):
- `bench_pipeline_diffusion_main/100`: `cpuparallel 2.900 ms`, `gpu-gated(policy) 2.214 ms` (`-23.15%`).
- `bench_pipeline_diffusion_main/20`: `cpuparallel 1.136 ms`, `gpu-gated(policy) 1.158 ms` (`+1.02%`, noise-level drift).
- `bench_markov_multi_step/20000/20`: `cpuparallel 4.718 ms`, `gpu-gated(policy) 0.763 ms` (`-83.83%` wall time).
- `bench_markov_step/2000`: `cpuparallel 17.38 us`, `gpu-forced 181.23 us` (single-step small workload still CPU-favored).

App-level (`notes/perf/results/main_timings_20260213-gpu-gating-h1.txt`, 3 runs):
- `igneous-diffusion assets/bunny.obj`: cpuparallel/gpu-gated `~0.03-0.04 s`; gpu-forced `~0.04-0.05 s`.
- `igneous-hodge`: cpuparallel/gpu-gated `~0.14-0.15 s`; gpu-forced `~0.21-0.22 s`.
- `igneous-spectral assets/bunny.obj`: cpuparallel/gpu-gated `~0.05-0.06 s`; gpu-forced `~0.06 s`.

## Recommended usage
- Use `parallel` for current real workloads and all tested 2k-4k point pipelines.
- Use `gpu` for diffusion workloads with larger graph/step products (e.g. long-step diffusion, larger point clouds).
- Keep `IGNEOUS_GPU_FORCE=1` for profiling/debug only; it still regresses small single-step kernels.
- Use `cpu` for deterministic debugging or tiny kernels where threading overhead dominates.

## GPU roadmap candidates
- Diffusion graph construction (`DiffusionTopology::build`) for very large point clouds.
- Batched/vector-resident diffusion iterations to amortize host-device synchronization.
- Dense linear algebra in Hodge phases after data transfer and batching strategy is in place.
