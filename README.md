# Igneous

[![CI](https://github.com/harnesslabs/igneous/actions/workflows/ci.yml/badge.svg)](https://github.com/harnesslabs/igneous/actions/workflows/ci.yml)
[![Perf Smoke](https://github.com/harnesslabs/igneous/actions/workflows/perf-smoke.yml/badge.svg)](https://github.com/harnesslabs/igneous/actions/workflows/perf-smoke.yml)
[![Perf Deep](https://github.com/harnesslabs/igneous/actions/workflows/perf-deep.yml/badge.svg)](https://github.com/harnesslabs/igneous/actions/workflows/perf-deep.yml)
[![CodeQL](https://github.com/harnesslabs/igneous/actions/workflows/codeql.yml/badge.svg)](https://github.com/harnesslabs/igneous/actions/workflows/codeql.yml)
[![Release](https://github.com/harnesslabs/igneous/actions/workflows/release.yml/badge.svg)](https://github.com/harnesslabs/igneous/actions/workflows/release.yml)

Data-oriented C++23 geometry and structure engine with a throughput-first CPU pipeline.

## Highlights

- SoA geometry storage (`x`, `y`, `z`) for cache-friendly traversal.
- `DiscreteExteriorCalculus` structure with explicit face arrays and CSR adjacency (faces + vertex neighbors).
- `DiffusionGeometry` structure with k-NN graph construction (`nanoflann`) and sparse Markov chain (`Eigen`).
- Spectral and Hodge operator stack (Gram, weak exterior derivative, curl energy, Hodge spectrum).
- Doctest correctness suite and Google Benchmark performance suite.

## Dependencies

Managed through `vcpkg` manifest:

- `fmt`
- `range-v3`
- `doctest`
- `xsimd`
- `eigen3`
- `nanoflann`
- `spectra`
- `benchmark`

## Build

```bash
cmake --preset default-local -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
```

## Run Examples

```bash
./build/igneous-mesh assets/bunny.obj
./build/igneous-point assets/bunny.obj
./build/igneous-diffusion assets/bunny.obj
./build/igneous-spectral assets/bunny.obj
./build/igneous-hodge
./build/igneous-diffusion-geometry
```

## Visualizations

Per-main output viewers are available under `visualizations/`.

```bash
python3 visualizations/view_main_point.py --run --open
python3 visualizations/view_main_mesh.py --run --open
python3 visualizations/view_main_diffusion.py --run --open
python3 visualizations/view_main_spectral.py --run --open
python3 visualizations/view_main_hodge.py --run --open
python3 visualizations/view_main_diffusion_geometry.py --run --open
```

Detailed usage is documented in `visualizations/README.md`.

## Reference Implementation

The diffusion/Hodge parity work in this repository is aligned against the Python
reference implementation:

- [DiffusionGeometry](https://github.com/Iolo-Jones/DiffusionGeometry)

## Hodge Parity Workflow

Standard parity round:

```bash
./scripts/hodge/run_parity_round.sh
```

## Diffusion Geometry Parity Workflow

Standard torus+sphere parity round:

```bash
./scripts/diffgeo/run_parity_round.sh
```

By default, `test_diffgeo_parity_optional` reports parity metrics but only hard-fails
when strict enforcement is requested:

```bash
IGNEOUS_REQUIRE_PARITY=1 ctest --test-dir build -R test_diffgeo_parity_optional --output-on-failure
```

Set `IGNEOUS_BENCH_MODE=1` to disable heavy export paths in runtime apps.

Runtime backend controls:

- `IGNEOUS_BACKEND=cpu` for single-thread CPU execution.
- `IGNEOUS_BACKEND=parallel` (default) for threaded CPU execution.
- `IGNEOUS_BACKEND=gpu` enables Metal diffusion kernels on Apple platforms.
- `IGNEOUS_NUM_THREADS=<N>` to override worker count for threaded kernels.
- `IGNEOUS_GPU_MIN_ROWS=<N>` sets minimum vertex count for GPU offload (default `8192`).
- `IGNEOUS_GPU_MIN_ROW_STEPS=<N>` sets minimum `rows*steps` for multi-step GPU offload (default `200000`).
- `IGNEOUS_GPU_FORCE=1` forces GPU offload for debugging/profiling.
- Workload guidance for backend choice: `notes/perf/backend-guidance.md`.

## Tests

```bash
ctest --test-dir build --output-on-failure --verbose
```

Current suites:

- `test_algebra`
- `test_structure_dec`
- `test_structure_diffusion_geometry`
- `test_ops_curvature_flow`
- `test_ops_spectral_geometry`
- `test_ops_hodge`
- `test_ops_diffusion_basis`
- `test_ops_diffusion_forms`
- `test_ops_diffusion_wedge`
- `test_io_meshes`
- `test_hodge_cli_outputs`
- `test_hodge_parity_optional` (skips unless `DiffusionGeometry/` is available, unless forced)
- `test_diffgeo_cli_outputs`
- `test_diffgeo_parity_optional` (strict-fail only when `IGNEOUS_REQUIRE_PARITY=1`)

## Benchmarks

### Existing mesh benchmark

```bash
./build/bench_geometry
```

### Google Benchmark suite

```bash
./build/bench_dod --benchmark_min_time=0.1s --benchmark_repetitions=5 --benchmark_report_aggregates_only=true
./build/bench_pipelines --benchmark_min_time=0.1s --benchmark_repetitions=5 --benchmark_report_aggregates_only=true
```

PR-smoke style local run:

```bash
IGNEOUS_BENCH_MODE=1 IGNEOUS_BACKEND=parallel IGNEOUS_NUM_THREADS=8 \
./build/bench_dod --benchmark_min_time=0.05s --benchmark_repetitions=5 --benchmark_report_aggregates_only=true
IGNEOUS_BENCH_MODE=1 IGNEOUS_BACKEND=parallel IGNEOUS_NUM_THREADS=8 \
./build/bench_pipelines --benchmark_min_time=0.05s --benchmark_repetitions=5 --benchmark_report_aggregates_only=true
```

Nightly-deep style local run:

```bash
./scripts/perf/run_deep_bench.sh
IGNEOUS_BENCH_MODE=1 IGNEOUS_BACKEND=parallel IGNEOUS_NUM_THREADS=8 \
./build/bench_pipelines --benchmark_min_time=0.2s --benchmark_repetitions=10 --benchmark_report_aggregates_only=true
```

Benchmark groups:

- `bench_mesh_structure_build`
- `bench_curvature_kernel`
- `bench_flow_kernel`
- `bench_diffusion_build`
- `bench_markov_step`
- `bench_markov_multi_step`
- `bench_eigenbasis`
- `bench_1form_gram`
- `bench_weak_derivative`
- `bench_curl_energy`
- `bench_hodge_solve`

Pipeline benchmark groups:

- `bench_pipeline_diffusion_main`
- `bench_pipeline_spectral_main`
- `bench_pipeline_hodge_main`
- `bench_hodge_phase_structure_build`
- `bench_hodge_phase_eigenbasis`
- `bench_hodge_phase_gram`
- `bench_hodge_phase_weak_derivative`
- `bench_hodge_phase_curl_energy`
- `bench_hodge_phase_solve`
- `bench_hodge_phase_circular`

## CI/CD Workflows

- `.github/workflows/ci.yml`: Linux/macOS build + tests, sanitizer pass, compile commands artifact.
- `.github/workflows/perf-smoke.yml`: PR smoke benchmark report against baseline (report-only).
- `.github/workflows/perf-deep.yml`: nightly/manual deep benchmark capture and summary (report-only).
- `.github/workflows/release.yml`: tag-triggered `v*` release packaging and GitHub Release asset publish.
- `.github/workflows/codeql.yml`: weekly/manual C++ CodeQL scan.

## Performance Workflow Artifacts

- Journal: `notes/perf/journal.md`
- Metrics log: `notes/perf/metrics.csv`
- Main-vs-branch summary: `notes/perf/main-vs-branch.md`
- Results: `notes/perf/results/`
- Profiles: `notes/perf/profiles/`
- Report: `notes/perf/final-report.md`
- Migration notes: `notes/perf/migration.md`

Helper scripts:

- `scripts/perf/run_deep_bench.sh`
- `scripts/perf/compare_against_main.py`
- `scripts/perf/profile_time.sh`
- `scripts/perf/profile_counters.sh`
- `scripts/perf/download_datasets.sh`

Regenerate `main-vs-branch.md`:

```bash
python3 scripts/perf/compare_against_main.py \
  --baseline notes/perf/results/bench_dod_20260213-085501.json \
  --current notes/perf/results/bench_dod_20260213-current-latest.json \
  --baseline-commit e7615627872a53010b006d69775174113cdbc467 \
  --label "Main vs branch: bench_dod" \
  --output-markdown notes/perf/main-vs-branch-bench_dod.md \
  --output-json notes/perf/main-vs-branch-bench_dod.json
```

Release tagging convention:

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

## API Example (Curvature + Flow)

```cpp
#include <igneous/igneous.hpp>

using Mesh = igneous::data::Space<igneous::data::DiscreteExteriorCalculus>;

int main() {
  Mesh mesh;
  igneous::io::load_obj(mesh, "assets/bunny.obj");

  std::vector<float> H;
  std::vector<float> K;

  igneous::ops::dec::CurvatureWorkspace<igneous::data::DiscreteExteriorCalculus> curvature_ws;
  igneous::ops::dec::FlowWorkspace<igneous::data::DiscreteExteriorCalculus> flow_ws;

  igneous::ops::dec::compute_curvature_measures(mesh, H, K, curvature_ws);
  igneous::ops::dec::integrate_mean_curvature_flow(mesh, 0.01f, flow_ws);
}
```

## Makefile Shortcuts

```bash
make release
make build
make test
make bench
make bench-deep
```
