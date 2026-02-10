.PHONY: all config build test clean run check bench release debug

# Default: Build in whatever mode is currently active
all: build

# --- Configuration Helpers ---

# Switch to Debug Mode (Best for coding/testing)
debug:
	cmake --preset default-local -DCMAKE_BUILD_TYPE=Debug

# Switch to Release Mode (Best for benchmarking)
release:
	cmake --preset default-local -DCMAKE_BUILD_TYPE=Release

# --- Standard Targets ---

# 1. Configure (Default to Debug if fresh)
config: debug

# 2. Build
build:
	cmake --build build

# 3. Test
# We enforce Debug mode because debugging optimized tests is painful.
test: debug
	cmake --build build --target igneous-test
	ctest --test-dir build --output-on-failure --verbose

# 4. Run (Test Runner)
run: test
	./build/igneous-test

# 5. Bench
# CRITICAL CHANGE: We force the 'release' config first.
# This rewrites the Ninja files to use -O3 optimizations.
bench: release
	cmake --build build --target igneous-bench
	./build/igneous-bench

# 6. Check (Syntax check only)
check:
	cmake --build build --target igneous

# 7. Clean
clean:
	rm -rf build