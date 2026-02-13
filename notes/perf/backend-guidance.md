# Backend Guidance (2026-02-13)

## Current backend modes
- `IGNEOUS_BACKEND=cpu`: serial CPU execution.
- `IGNEOUS_BACKEND=parallel` (default): threaded CPU execution.
- `IGNEOUS_BACKEND=gpu`: currently mapped to threaded CPU (no native GPU kernels yet).

## Measured guidance (Apple M3 Max, Release)
From `notes/perf/results/bench_pipelines_20260213-1233-*.txt` (10 reps):
- `bench_pipeline_diffusion_main/100`: `cpu 5.927 ms`, `parallel 3.093 ms`.
- `bench_pipeline_spectral_main`: `cpu 33.384 ms`, `parallel 30.437 ms`.
- `bench_pipeline_hodge_main`: `cpu 302.312 ms`, `parallel 148.748 ms`.

## Recommended usage
- Use `parallel` for real workloads (diffusion/spectral/hodge pipelines, large meshes).
- Use `cpu` for tiny meshes, deterministic debugging, or when thread startup overhead dominates.
- Treat `gpu` as a future-facing selector only until native GPU kernels are implemented.

## GPU roadmap candidates
- Diffusion graph construction (`DiffusionTopology::build`) for very large point clouds.
- Repeated diffusion operators (`apply_markov_transition`, `carre_du_champ`) in spectral/hodge loops.
- Dense linear algebra in Hodge phases after data transfer and batching strategy is in place.
