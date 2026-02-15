.PHONY: all debug release build clean test test-all test-algebra test-structure test-ops bench bench-memory bench-geometry bench-dod bench-deep run-mesh run-diffusion run-spectral run-hodge

all: build

debug:
	cmake --preset default-local -DCMAKE_BUILD_TYPE=Debug

release:
	cmake --preset default-local -DCMAKE_BUILD_TYPE=Release

build:
	cmake --build build

clean:
	rm -rf build

test: debug
	cmake --build build --target test_algebra test_structure_dec test_structure_diffusion_geometry test_ops_curvature_flow test_ops_spectral_geometry test_ops_hodge test_io_meshes
	ctest --test-dir build --output-on-failure --verbose

test-all: test

test-algebra: debug
	cmake --build build --target test_algebra
	./build/test_algebra

test-structure: debug
	cmake --build build --target test_structure_dec test_structure_diffusion_geometry
	./build/test_structure_dec
	./build/test_structure_diffusion_geometry

test-ops: debug
	cmake --build build --target test_ops_curvature_flow test_ops_spectral_geometry test_ops_hodge
	./build/test_ops_curvature_flow
	./build/test_ops_spectral_geometry
	./build/test_ops_hodge

bench-memory: release
	cmake --build build --target bench_memory
	./build/bench_memory

bench-geometry: release
	cmake --build build --target bench_geometry
	./build/bench_geometry

bench-dod: release
	cmake --build build --target bench_dod
	./build/bench_dod --benchmark_min_time=0.1s --benchmark_repetitions=5 --benchmark_report_aggregates_only=true

bench-deep: release
	cmake --build build --target bench_dod
	./scripts/perf/run_deep_bench.sh

bench: bench-memory bench-geometry bench-dod

run-mesh: release
	cmake --build build --target igneous-mesh
	./build/igneous-mesh assets/bunny.obj

run-diffusion: release
	cmake --build build --target igneous-diffusion
	./build/igneous-diffusion assets/bunny.obj

run-spectral: release
	cmake --build build --target igneous-spectral
	./build/igneous-spectral assets/bunny.obj

run-hodge: release
	cmake --build build --target igneous-hodge
	./build/igneous-hodge
