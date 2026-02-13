# Main vs Branch Performance Summary

Baseline commit: `e7615627872a53010b006d69775174113cdbc467` (`main`, Friday, February 13, 2026)
Current branch snapshot: `fedc87b`

## Coverage notes

- Strict `main` comparison is available for `bench_geometry` and app-level `main_*` executables.
- `bench_pipelines` did not exist at `e761562`; those metrics are reported as branch-era evolution only.
- `bench_dod` baseline comes from the branch-point measurement artifact tagged with `e761562` in `notes/perf/results/bench_dod_20260213-085501.json`.

## Shared benchmark deltas (main -> branch)

### `bench_geometry` (`notes/perf/results/bench_geometry_20260213-main-e761562.txt` -> `notes/perf/results/bench_geometry_20260213-final-head.txt`)

| Benchmark | Baseline | Current | Delta |
| --- | ---: | ---: | ---: |
| `bench_geometry_frame_ms/1000x1000` | `59.567 ms` | `26.291 ms` | `-55.86%` |
| `bench_geometry_frame_ms/500x500` | `15.013 ms` | `7.699 ms` | `-48.72%` |
| `bench_geometry_frame_ms/250x250` | `3.724 ms` | `2.533 ms` | `-31.98%` |
| `bench_geometry_frame_ms/100x100` | `0.565 ms` | `0.656 ms` | `+16.11%` |

Topology/curvature/flow split on `1000x1000`:
- `topology`: `7.666 ms -> 18.261 ms` (`+138.21%`)
- `curvature`: `42.624 ms -> 6.921 ms` (`-83.76%`)
- `flow`: `9.277 ms -> 1.109 ms` (`-88.05%`)

### `bench_dod` branch-point baseline (`notes/perf/results/bench_dod_20260213-085501.json` -> `notes/perf/results/bench_dod_20260213-final-head.json`)

| Benchmark | Baseline | Current | Delta |
| --- | ---: | ---: | ---: |
| `bench_mesh_topology_build/400` | `33.979 ms` | `3.092 ms` | `-90.90%` |
| `bench_curl_energy/2000/16` | `10.115 ms` | `1.166 ms` | `-88.47%` |
| `bench_weak_derivative/2000/16` | `2.476 ms` | `0.387 ms` | `-84.35%` |
| `bench_diffusion_build/2000` | `3.381 ms` | `0.657 ms` | `-80.56%` |
| `bench_curvature_kernel/400` | `7.237 ms` | `1.164 ms` | `-83.91%` |
| `bench_flow_kernel/400` | `0.600 ms` | `0.233 ms` | `-61.13%` |
| `bench_markov_step/2000` | `0.030 ms` | `0.017 ms` | `-41.67%` |
| `bench_eigenbasis/2000/16` | `15.800 ms` | `11.833 ms` | `-25.11%` |
| `bench_1form_gram/2000/16` | `0.468 ms` | `0.252 ms` | `-46.07%` |
| `bench_hodge_solve/2000/16` | `0.074 ms` | `0.071 ms` | `-4.32%` |

## Pipeline benchmarks (branch-era evolution, not strict `main` comparison)

`notes/perf/results/bench_pipelines_20260213-current.json` -> `notes/perf/results/bench_pipelines_20260213-final-head.json`

| Benchmark | Baseline | Current | Delta |
| --- | ---: | ---: | ---: |
| `bench_pipeline_diffusion_main/20` | `4.119 ms` | `1.145 ms` | `-72.20%` |
| `bench_pipeline_diffusion_main/100` | `5.914 ms` | `2.777 ms` | `-53.05%` |
| `bench_pipeline_spectral_main` | `33.395 ms` | `16.430 ms` | `-50.80%` |
| `bench_pipeline_hodge_main` | `154.395 ms` | `108.460 ms` | `-29.75%` |
| `bench_hodge_phase_eigenbasis` | `107.877 ms` | `75.064 ms` | `-30.42%` |
| `bench_hodge_phase_gram` | `6.449 ms` | `1.229 ms` | `-80.95%` |
| `bench_hodge_phase_weak_derivative` | `12.059 ms` | `2.908 ms` | `-75.89%` |
| `bench_hodge_phase_topology` | `6.040 ms` | `1.167 ms` | `-80.68%` |
| `bench_hodge_phase_curl_energy` | `14.821 ms` | `19.539 ms` | `+31.83%` |
| `bench_hodge_phase_solve` | `1.856 ms` | `2.131 ms` | `+14.79%` |
| `bench_hodge_phase_circular` | `4.997 ms` | `5.247 ms` | `+4.99%` |

## App-level strict baseline timings (`main_*` binaries)

`notes/perf/results/main_timings_20260213-main-e761562.txt` -> `notes/perf/results/main_timings_20260213-current-head.txt`

| App | Baseline mean real | Current mean real | Delta |
| --- | ---: | ---: | ---: |
| `igneous-diffusion assets/bunny.obj` | `0.210 s` | `0.223 s` | `+6.35%` |
| `igneous-spectral assets/bunny.obj` | `0.070 s` | `0.080 s` | `+14.29%` |
| `igneous-hodge` | `1.353 s` | `0.170 s` | `-87.44%` |

## Top takeaways

- Largest strict `main` win is end-to-end `igneous-hodge` (`-87.44%` wall time).
- Large-grid geometry frame time improved materially (`1000x1000`: `-55.86%`) even with slower topology stage due much faster curvature/flow kernels.
- Branch-era pipeline harness shows major throughput gains in dominant phases, but curl-energy/circular/solve remain active hotspots.
