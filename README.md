# Igneous

Data-oriented C++23 geometry and topology engine with a throughput-first CPU pipeline.

## Highlights

- SoA geometry storage (`x`, `y`, `z`) for cache-friendly traversal.
- Triangle topology with explicit face arrays and CSR adjacency (faces + vertex neighbors).
- Diffusion topology with k-NN graph construction (`nanoflann`) and sparse Markov chain (`Eigen`).
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
```

Set `IGNEOUS_BENCH_MODE=1` to disable heavy export paths in runtime apps.

## Tests

```bash
ctest --test-dir build --output-on-failure --verbose
```

Current suites:

- `test_algebra`
- `test_topology_triangle`
- `test_topology_diffusion`
- `test_ops_curvature_flow`
- `test_ops_spectral_geometry`
- `test_ops_hodge`
- `test_io_meshes`

## Benchmarks

### Existing mesh benchmark

```bash
./build/bench_geometry
```

### Google Benchmark suite

```bash
./build/bench_dod --benchmark_min_time=0.1s --benchmark_repetitions=5 --benchmark_report_aggregates_only=true
```

Benchmark groups:

- `bench_mesh_topology_build`
- `bench_curvature_kernel`
- `bench_flow_kernel`
- `bench_diffusion_build`
- `bench_markov_step`
- `bench_eigenbasis`
- `bench_1form_gram`
- `bench_weak_derivative`
- `bench_curl_energy`
- `bench_hodge_solve`

## Performance Workflow Artifacts

- Journal: `notes/perf/journal.md`
- Metrics log: `notes/perf/metrics.csv`
- Results: `notes/perf/results/`
- Profiles: `notes/perf/profiles/`
- Report: `notes/perf/final-report.md`
- Migration notes: `notes/perf/migration.md`

Helper scripts:

- `scripts/perf/run_deep_bench.sh`
- `scripts/perf/profile_time.sh`
- `scripts/perf/profile_counters.sh`
- `scripts/perf/download_datasets.sh`

## API Example (Curvature + Flow)

```cpp
#include <igneous/igneous.hpp>

using Sig = igneous::core::Euclidean3D;
using Mesh = igneous::data::Mesh<Sig, igneous::data::TriangleTopology>;

int main() {
  Mesh mesh;
  igneous::io::load_obj(mesh, "assets/bunny.obj");

  std::vector<float> H;
  std::vector<float> K;

  igneous::ops::CurvatureWorkspace<Sig, igneous::data::TriangleTopology> curvature_ws;
  igneous::ops::FlowWorkspace<Sig, igneous::data::TriangleTopology> flow_ws;

  igneous::ops::compute_curvature_measures(mesh, H, K, curvature_ws);
  igneous::ops::integrate_mean_curvature_flow(mesh, 0.01f, flow_ws);
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
