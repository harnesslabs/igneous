.PHONY: all config build test test-all test-algebra test-simplex clean run bench bench-algebra bench-memory release debug

# Default: Build everything
all: build

# --- Configuration ---
debug:
	cmake --preset default-local -DCMAKE_BUILD_TYPE=Debug

release:
	cmake --preset default-local -DCMAKE_BUILD_TYPE=Release

config: debug

# --- Building ---
# Builds everything (library + all tests + all benchmarks)
build:
	cmake --build build

# --- Testing ---

# 1. Run ALL tests (via CTest)
test: debug
	cmake --build build --target test_algebra test_simplex
	ctest --test-dir build --output-on-failure --verbose

# 2. Run ONLY Algebra tests (Fast iteration)
test-algebra: debug
	cmake --build build --target test_algebra
	./build/test_algebra

# 3. Run ONLY Simplex tests (Fast iteration)
test-simplex: debug
	cmake --build build --target test_simplex
	./build/test_simplex

# --- Benchmarking ---
bench-algebra: release
	cmake --build build --target bench_algebra
	./build/bench_algebra

bench-memory: release
	cmake --build build --target bench_memory
	./build/bench_memory

# Run all benchmarks
bench: bench-algebra bench-memory

# --- Utilities ---
check:
	cmake --build build --target igneous

clean:
	rm -rf build