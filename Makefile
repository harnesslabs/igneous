.PHONY: all config build test clean run check

# Default target: just build
all: build

# 1. Configure (cargo init/update)
# Uses your local preset to ensure vcpkg/ninja are found
config:
	cmake --preset default-local

# 2. Build (cargo build)
build:
	cmake --build build

# 3. Test (cargo test)
# --output-on-failure makes it verbose only if something breaks
test:
	ctest --test-dir build --output-on-failure

# 4. Run (cargo run)
# Runs your test runner executable directly
run: build
	./build/igneous-test

# 5. Check (cargo check)
# Just compiles the code without linking (faster)
check:
	cmake --build build --target igneous

# 6. Clean (cargo clean)
clean:
	rm -rf build