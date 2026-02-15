# Structure Refactor Journal

## Entry 0001
- Timestamp: 2026-02-15
- Hypothesis: Hard cutover to structure-first API (`Space` + `Structure`) simplifies data and ops surfaces while preserving behavior.
- Structural Difference Targeted: Replace mesh/topology/buffer model with split structures and flattened space geometry.
- Baseline Verification:
  - `ctest --test-dir build --output-on-failure` -> `14/14` passed.
- Decisions:
  - `Topology` -> `Structure`, `SurfaceTopology` -> `SurfaceStructure`
  - `TriangleTopology` -> `DiscreteExteriorCalculus`
  - `DiffusionTopology` -> `DiffusionGeometry`
  - `Mesh` -> `Space`, no compatibility aliases
  - Remove `PointTopology` and `GeometryBuffer`
  - Explicit structure build policy in IO callers
  - CLI rename to `diffusion-geometry`
- Status: in_progress

## Entry 0002
- Timestamp: 2026-02-15
- Structural Difference Targeted: Complete hard cutover in data and ops layers.
- Files Added:
  - `include/igneous/data/structure.hpp`
  - `include/igneous/data/space.hpp`
  - `include/igneous/data/structures/discrete_exterior_calculus.hpp`
  - `include/igneous/data/structures/diffusion_geometry.hpp`
  - `include/igneous/ops/dec/curvature.hpp`
  - `include/igneous/ops/dec/flow.hpp`
- Files Removed:
  - `include/igneous/data/topology.hpp`
  - `include/igneous/data/mesh.hpp`
  - `include/igneous/data/buffers.hpp`
  - `include/igneous/ops/curvature.hpp`
  - `include/igneous/ops/flow.hpp`
- Decisions:
  - Keep `igneous::ops` root; structure ops grouped under `ops::dec` and `ops::diffusion`.
  - Remove sparse-matrix `structure.P` fallback paths and standardize spectral flow on CSR arrays in `DiffusionGeometry`.
- Verification:
  - `cmake --build build --parallel` -> pass.

## Entry 0003
- Timestamp: 2026-02-15
- Structural Difference Targeted: Enforce explicit build policy after load and propagate API cutover through callsites/tests.
- Callsite Policy Updates:
  - `io::load_obj` remains load-only.
  - DEC and diffusion callers explicitly invoke `space.structure.build(...)` when downstream ops require built structures.
- Key Test Update:
  - `tests/test_io_meshes.cpp` now asserts load-only state first, then explicit `structure.build(...)` for DEC and diffusion checks.
- Verification:
  - `ctest --test-dir build --output-on-failure` initially failed at `test_io_meshes` due to old implicit-build expectation.
  - After explicit-build updates, `ctest --test-dir build --output-on-failure` -> `14/14` passed.

## Entry 0004
- Timestamp: 2026-02-15
- Structural Difference Targeted: Rename topology-oriented CLI and structure tests to final terminology.
- Renames:
  - `src/main_diffusion_topology.cpp` -> `src/main_diffusion_geometry.cpp`
  - `tests/test_topology_triangle.cpp` -> `tests/test_structure_dec.cpp`
  - `tests/test_topology_diffusion.cpp` -> `tests/test_structure_diffusion_geometry.cpp`
  - `visualizations/view_main_diffusion_topology.py` -> `visualizations/view_main_diffusion_geometry.py`
  - CMake target `igneous-diffusion-topology` -> `igneous-diffusion-geometry`
- Dependent Surfaces Updated:
  - `tests/test_diffgeo_cli_outputs.sh`
  - `scripts/diffgeo/run_cpp_diffgeo_ops.sh`
  - `scripts/diffgeo/run_parity_round.sh`
  - `visualizations/README.md`
  - `README.md`
  - Diffgeo parity helper messaging/scripts under `scripts/diffgeo/` and `tests/`.
- Verification:
  - `cmake -S . -B build -G Ninja && cmake --build build --parallel` -> pass.
  - `ctest --test-dir build --output-on-failure` -> `14/14` passed.
  - Benchmark smoke:
    - `IGNEOUS_BENCH_MODE=1 ./build/bench_geometry` -> pass
    - `IGNEOUS_BENCH_MODE=1 ./build/bench_dod --benchmark_min_time=0.02s --benchmark_repetitions=1 --benchmark_report_aggregates_only=true` -> pass
    - `IGNEOUS_BENCH_MODE=1 ./build/bench_pipelines --benchmark_min_time=0.02s --benchmark_repetitions=1 --benchmark_report_aggregates_only=true` -> pass
- Status: completed

## Entry 0005
- Timestamp: 2026-02-15
- Structural Difference Targeted: Remove remaining non-historical `topology` terminology from active surfaces.
- Updates:
  - Renamed benchmark labels/functions to `structure` terminology where applicable.
  - Updated `README.md` benchmark list names to match renamed benchmark labels.
  - Updated perf comparator metric key `bench_geometry_topology_ms` -> `bench_geometry_structure_ms`.
- Verification:
  - `cmake --build build --parallel && ctest --test-dir build --output-on-failure` -> pass.
  - Benchmark smoke rerun passed:
    - `IGNEOUS_BENCH_MODE=1 ./build/bench_geometry`
    - `IGNEOUS_BENCH_MODE=1 ./build/bench_dod --benchmark_min_time=0.02s --benchmark_repetitions=1 --benchmark_report_aggregates_only=true`
    - `IGNEOUS_BENCH_MODE=1 ./build/bench_pipelines --benchmark_min_time=0.02s --benchmark_repetitions=1 --benchmark_report_aggregates_only=true`

## Entry 0006
- Timestamp: 2026-02-15
- Structural Difference Targeted: Improve API discoverability with inline C++ docs and generated reference docs.
- Documentation Additions:
  - Added Doxygen comments across `include/igneous` headers, including internal/private state fields for key workspace and runtime classes.
  - Added docs generation tooling:
    - `docs/Doxyfile.in`
    - `docs/mainpage.md`
    - CMake `docs` target (when Doxygen is available)
    - `Makefile` `docs` shortcut
  - Added README docs section with generation instructions and output path.
- Decisions:
  - Use Doxygen as the canonical C++ API doc generator (closest to Rust `cargo doc` workflow).
  - Keep extraction of private members enabled in docs config (`EXTRACT_PRIVATE=YES`) to match maintenance needs.

## Entry 0007
- Timestamp: 2026-02-15
- Structural Difference Targeted: Add docs quality checks to CI and deploy generated docs.
- CI Additions:
  - Added `docs` job to `.github/workflows/ci.yml`:
    - installs Doxygen + Graphviz
    - configures CMake with `IGNEOUS_BUILD_DOCS=ON`
    - builds `docs` target
    - verifies `build-docs/docs/html/index.html` exists
- Deployment Additions:
  - Added `.github/workflows/docs.yml`:
    - triggers on `push` to `main` and manual dispatch
    - builds docs artifact
    - deploys to GitHub Pages using `actions/deploy-pages`
- Documentation:
  - Added README note for docs deployment URL pattern.

## Entry 0008
- Timestamp: 2026-02-15
- Structural Difference Targeted: Add pedantic static analysis and formatting workflow for maintainability.
- Tooling Additions:
  - Added `.clang-tidy` with a practical pedantic checks profile (`bugprone`, `performance`, selected `modernize/readability`).
  - Added `.clang-format` as repo-wide formatter policy.
  - Added helper scripts:
    - `scripts/dev/lint.sh`
    - `scripts/dev/format.sh`
  - Added `Makefile` targets:
    - `make lint`
    - `make lint-all`
    - `make format`
    - `make format-check`
- Documentation:
  - Added lint/format usage section to `README.md`.
- Verification:
  - `make lint` -> pass; reports actionable warnings from current source.
  - `make lint-all` -> pass; reports additional warnings across tests/benches.
  - `make format-check` -> fail on existing formatting drift, as expected before first repo-wide `make format`.
  - `ctest --test-dir build --output-on-failure` -> `14/14` passed after tooling integration.

## Entry 0009
- Timestamp: 2026-02-15
- Structural Difference Targeted: Ensure lint catches unused direct includes in headers.
- Lint Fixes:
  - Updated `scripts/dev/lint.sh` to run a dedicated header pass with `misc-include-cleaner` over `include/igneous/**`.
  - Added `std::cstddef` include and removed unused includes in `include/igneous/ops/diffusion/basis.hpp`.
- Verification:
  - `make lint` now surfaces include-cleaner diagnostics from headers (including `basis.hpp`) and reports missing-direct-include issues.

## Entry 0010
- Timestamp: 2026-02-15
- Structural Difference Targeted: Establish a repo-wide canonical C++ formatting baseline.
- Formatting Changes:
  - Ran `make format` across `include/`, `src/`, `tests/`, and `benches/` using `.clang-format`.
  - Applied style-only edits (spacing/wrapping/alignment) with no intended behavioral changes.
- Verification:
  - `make format-check` -> pass.
  - `ctest --test-dir build --output-on-failure` -> `14/14` passed after formatting sweep.

## Entry 0011
- Timestamp: 2026-02-15
- Structural Difference Targeted: Make lint/format helpers resilient when ripgrep is unavailable.
- Fixes:
  - Updated `scripts/dev/format.sh` to fall back to `find` when `rg` is not installed.
  - Updated `scripts/dev/lint.sh` to fall back to `find` for both translation units and header discovery.
- Verification:
  - `PATH="/opt/homebrew/opt/llvm/bin:/usr/bin:/bin" /opt/homebrew/bin/bash ./scripts/dev/format.sh --check` -> pass (no ripgrep in PATH).
  - `PATH="/opt/homebrew/opt/llvm/bin:/usr/bin:/bin" /opt/homebrew/bin/bash ./scripts/dev/lint.sh build` -> pass with expected lint warnings.
