# Final Performance Report

Date: Friday, February 13, 2026
Branch: `perf/bench-suite-optimizations`
Head snapshot: `fedc87b`

## 1) Baseline definition

- Strict branch point baseline: `main` commit `e7615627872a53010b006d69775174113cdbc467`.
- Baseline artifacts used:
  - `notes/perf/results/bench_geometry_20260213-main-e761562.txt`
  - `notes/perf/results/main_timings_20260213-main-e761562.txt`
- Branch-point campaign baseline for new DOD harness:
  - `notes/perf/results/bench_dod_20260213-085501.json` (tagged as `e761562` in the log stream).

## 2) Methodology

- Platform: Apple M3 Max (local), Release builds.
- Correctness gates: `ctest` suite (`7/7`) after accepted changes.
- Micro/mid-level suites:
  - `bench_geometry`
  - `bench_dod`
- Real-world pipeline suite:
  - `bench_pipelines` (mirrors `main_diffusion`, `main_spectral`, `main_hodge` stage structure).
- Runtime app checks:
  - `/usr/bin/time -p ./build/igneous-*` on representative inputs.
- Policy:
  - one optimization hypothesis per commit,
  - keep non-trivial changes only when deep benchmark gain >=3% on primary targets,
  - report-only perf policy in CI (no hard fail on hosted-runner perf noise).

## 3) Accepted optimizations (from journal)

Accepted: `31` entries.

Key accepted commits with measured impact:

| Commit | Optimization | Representative delta |
| --- | --- | --- |
| `a7b5c3d` | Triangle neighbor CSR rebuild | `bench_mesh_topology_build/400`: `-87.26%` |
| `79b2fd3` | Markov inner-loop unroll | `bench_markov_step/2000`: `-31.79%` |
| `e1c3135` | `carre_du_champ` CSR unroll | `bench_weak_derivative/2000/16`: `-9.85%` (campaign deep run) |
| `867417c` | KD-tree leaf-size tuning | `bench_diffusion_build/2000`: `-7.22%` |
| `1ab1f43` | Weak-derivative matrix coupling | `bench_weak_derivative/2000/16`: `-11.99%` |
| `7be6233` | Circular-coordinate matrix rewrite | `igneous-hodge` wall time: `~0.35s -> ~0.15s` (campaign stage) |
| `9e02b50` | Threaded curl-energy assembly | `bench_pipeline_hodge_main`: `303.402 ms -> 161.566 ms` (`-46.75%`) |
| `a816dc6` | Threaded circular-coordinate assembly | `bench_pipeline_hodge_main`: `311.486 ms -> 141.164 ms` (`-54.68%`) |
| `d17a954` | Threaded geometry + diffusion topology | `bench_pipeline_hodge_main`: `302.312 ms -> 148.748 ms` (`-50.80%`) |
| `c136055` | Spectral CSR matvec operator | `bench_hodge_phase_eigenbasis`: `-12.53%` |
| `ae7dbf9` | Spectral sort `LargestReal` | `bench_pipeline_hodge_main`: `-3.25%` |
| `91cf268` | Parallel 1-form Gram assembly | `bench_hodge_phase_gram`: `-81.66%` |
| `0742e0f` | Persistent worker pool | `bench_pipeline_diffusion_main/20`: additional threaded gain logged in wave2 |
| `7984261` | Metal diffusion backend (gated) | large diffusion offload path enabled (policy-gated) |
| `2cb7579` | Batched multi-step GPU Markov | `bench_markov_multi_step/20000/20`: `4.718 ms -> 0.734 ms` (`-84.46%`) |
| `76c48be` | GPU offload policy (`rows*steps`) | `bench_pipeline_diffusion_main/100`: `2.881 ms -> 2.214 ms` (`-23.15%`) |
| `6e2b2fb` | Symmetric diffusion eigensolver path | `bench_pipeline_spectral_main`: `-22.42%`, `bench_pipeline_hodge_main`: `-10.84%` |
| `266e4b6` | Diffusion CSR-only topology storage | `bench_hodge_phase_topology`: `-27.00%`, `bench_pipeline_diffusion_main/20`: `-12.13%` |

## 4) Rejected optimizations (from journal)

Rejected: `31` entries.

Consolidated rejection reasons:

- Sub-threshold gain on primary target (`<3%`), even when secondary kernels improved.
- Regression in dominant real-world path (`bench_pipeline_hodge_main`, `bench_hodge_phase_eigenbasis`, or diffusion topology build).
- Solver-parameter changes that destabilized runtime (`ncv` sweeps, `mu`-init, low-threshold spectral parallelization).
- Data-layout rewrites that looked promising in isolation but regressed end-to-end (curl matrix layout, coupling precompute variants).
- Forcing GPU outside policy envelope caused severe regressions in Hodge phases.

Examples:
- `Spectral Compact NCV k+8`: `bench_pipeline_hodge_main +87.55%` (rejected).
- `Spectral matvec threshold lowering`: `bench_pipeline_spectral_main +68.40%` (rejected).
- `Hybrid brute-force kNN`: diffusion topology path regressed strongly (rejected).
- `Hodge gamma cache reuse`: only `-1.03%` on primary deep target (rejected by threshold).

## 5) Main vs branch improvements

See canonical log: `notes/perf/main-vs-branch.md`.

### Strict `main` comparable highlights (`e761562` -> `fedc87b`)

| Benchmark | Baseline | Current | Delta |
| --- | ---: | ---: | ---: |
| `bench_geometry_frame_ms/1000x1000` | `59.567 ms` | `26.291 ms` | `-55.86%` |
| `bench_geometry_frame_ms/500x500` | `15.013 ms` | `7.699 ms` | `-48.72%` |
| `bench_geometry_frame_ms/250x250` | `3.724 ms` | `2.533 ms` | `-31.98%` |
| `igneous-hodge` (real wall time mean) | `1.353 s` | `0.170 s` | `-87.44%` |

Strict app-level note:
- `igneous-diffusion assets/bunny.obj`: `0.210 s -> 0.223 s` (`+6.35%`).
- `igneous-spectral assets/bunny.obj`: `0.070 s -> 0.080 s` (`+14.29%`).
- Those two app timings include full default runtime behavior (including non-kernel overhead); compute-focused `bench_pipelines` still shows substantial throughput wins.

### Branch-era pipeline harness evolution (not strict `main` comparison)

| Benchmark | Baseline | Current | Delta |
| --- | ---: | ---: | ---: |
| `bench_pipeline_diffusion_main/20` | `4.119 ms` | `1.145 ms` | `-72.20%` |
| `bench_pipeline_diffusion_main/100` | `5.914 ms` | `2.777 ms` | `-53.05%` |
| `bench_pipeline_spectral_main` | `33.395 ms` | `16.430 ms` | `-50.80%` |
| `bench_pipeline_hodge_main` | `154.395 ms` | `108.460 ms` | `-29.75%` |

## 6) Current pipeline status and backend guidance

Current pipeline timing snapshot (`notes/perf/results/bench_pipelines_20260213-final-head.json`):

- `bench_pipeline_diffusion_main/20`: `1.145 ms`
- `bench_pipeline_diffusion_main/100`: `2.777 ms`
- `bench_pipeline_spectral_main`: `16.430 ms`
- `bench_pipeline_hodge_main`: `108.460 ms`

Backend guidance (`notes/perf/backend-guidance.md`):

- Default to `IGNEOUS_BACKEND=parallel` for current 2k-4k workloads.
- Use `IGNEOUS_BACKEND=gpu` for large diffusion row-step workloads (especially multi-step diffusion).
- Keep conservative gating (`IGNEOUS_GPU_MIN_ROWS`, `IGNEOUS_GPU_MIN_ROW_STEPS`); do not globally force low-threshold GPU for Hodge workloads.
- Reserve `IGNEOUS_GPU_FORCE=1` for profiling/debug experiments.

## 7) Plateau rationale and next opportunities

Plateau status:
- We still get localized wins, but broad cross-suite gains are now harder; several recent hypotheses improved one phase while regressing another.
- Current dominant residual hotspot in the Hodge pipeline is curl-energy/circular tail behavior under full end-to-end runs.

High-value next opportunities:

1. Targeted curl-energy algorithmic reformulation that preserves exactness but reduces dense coupling cost in the full pipeline path.
2. GPU extension beyond diffusion Markov steps: staged/batched Hodge sub-operators only when transfer amortization is proven.
3. Deeper topology-build split (small vs large graph strategy) to recover strict `bench_geometry` topology regressions while preserving current curvature/flow wins.
4. CI trend dashboards from nightly deep artifacts (historical deltas over time, not just run-local comparisons).
