.PHONY: all config build test clean run check bench

# Default target: build everything
all: build

# 1. Configure (cargo init/update)
config:
	cmake --preset default-local

# 2. Build (cargo build)
# Now explicitly builds the benchmark too
build:
	cmake --build build

# 3. Test (cargo test)
# DEPENDS ON BUILD: Ensures we are testing the latest code.
# --verbose: Shows the nice doctest output (green/red text) instead of just "Passed"
test: build
	ctest --test-dir build --output-on-failure --verbose

# 4. Run (cargo run)
# Runs the main test runner directly (useful for debugging specific flags)
run: build
	./build/igneous-test

# 5. Bench (cargo bench)
# builds and runs your benchmark in Release mode (important!)
bench:
	cmake --build build --target igneous-bench --config Release
	./build/igneous-bench

# 6. Check (cargo check)
# Just compiles the library interface to check for errors
check:
	cmake --build build --target igneous

# 7. Clean (cargo clean)
clean:
	rm -rf build