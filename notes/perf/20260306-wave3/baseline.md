# Wave 3 Baseline

This file records the first Release baseline for the Wave 3 campaign.

## Status
- Release tree rebuilt with `cmake --preset release-local`.
- Full suite passes: `14/14` via `ctest --test-dir build --output-on-failure -j8`.
- Baseline artifact stamp: `20260306-170545`.

## Planned benchmark groups
- `bench_geometry`
- `bench_dod`
- `bench_pipelines`
- `bench_diffgeo` once added

## Captured artifacts
- `results/bench_geometry_20260306-170545.txt`
- `results/bench_dod_20260306-170545.json`
- `results/bench_dod_20260306-170545.txt`
- `results/bench_pipelines_20260306-170545.json`
- `results/bench_pipelines_20260306-170545.txt`
- `results/bench_diffgeo_20260306-170545.json`
- `results/bench_diffgeo_20260306-170545.txt`

## Release snapshot
- `bench_geometry`:
  - `Grid 1000x1000`: topology `246.040 ms`, curvature `100.111 ms`, flow `17.066 ms`
- `bench_dod`:
  - `bench_diffusion_build/2000`: `76.986 ms` CPU mean
  - `bench_eigenbasis/2000/16`: `60.300 ms`
  - `bench_1form_gram/2000/16`: `1.931 ms`
  - `bench_weak_derivative/2000/16`: `3.460 ms`
  - `bench_curl_energy/2000/16`: `7.077 ms`
- `bench_pipelines`:
  - `bench_pipeline_diffusion_main/20`: `93.933 ms`
  - `bench_pipeline_spectral_main`: `216.192 ms`
  - `bench_pipeline_hodge_main`: `3275.117 ms`
  - `bench_hodge_phase_eigenbasis`: `637.631 ms`
  - `bench_hodge_phase_circular`: `2401.417 ms`
  - `bench_hodge_phase_curl_energy`: `101.242 ms`
- `bench_diffgeo`:
  - `bench_diffgeo_pipeline/1000/50/32/32`: `742.725 ms`
  - `bench_diffgeo_pipeline/4000/64/32/32`: `4169.532 ms`
  - `bench_diffgeo_phase_structure_build/4000/64/32/32`: `145.745 ms`
  - `bench_diffgeo_phase_eigenbasis/4000/64/32/32`: `583.928 ms`
  - `bench_diffgeo_phase_circular/4000/64/32/32`: `2416.567 ms`
  - `bench_diffgeo_phase_k1_up/4000/64/32/32`: `325.643 ms`
  - `bench_diffgeo_phase_k2_up/4000/64/32/32`: `425.540 ms`
  - `bench_diffgeo_phase_k2_down/4000/64/32/32`: `0.188 ms`

## App-level timing
- `igneous-diffusion-geometry --n-points 1000 --output-dir <tempdir>`: `real 1.40`, `user 1.22`, `sys 0.07`

## Current hotspot order
- Diffusion-geometry pipeline:
  - `circular`
  - `eigenbasis`
  - `k2_up`
  - `k1_up`
  - `structure_build`
- Legacy Hodge pipeline:
  - `circular`
  - `eigenbasis`
  - `structure_build`
  - `curl_energy`
  - `weak_derivative`

## Notes
- This baseline was captured after the first kept Wave 3 diffgeo optimization, because the harness and the first reuse patch were developed together.
- The new `bench_diffgeo_phase_k2_down` benchmark intentionally measures the hot reused stage after the 1-form operators are already available, which matches the target use inside the pipeline.
