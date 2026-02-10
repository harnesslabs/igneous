cmake_minimum_required(VERSION 3.25)
project(igneous LANGUAGES CXX)

# --- Compiler Settings ---
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_compile_options(-Wall -Wextra -Wpedantic)

# --- Dependencies ---
find_package(fmt CONFIG REQUIRED)
find_package(range-v3 CONFIG REQUIRED)
find_package(doctest CONFIG REQUIRED)
find_package(xsimd CONFIG REQUIRED)

# --- 1. The Library Target ---
add_library(igneous INTERFACE)
target_include_directories(igneous INTERFACE include)
target_link_libraries(igneous INTERFACE fmt::fmt range-v3::range-v3 xsimd)

# --- 2. Testing Setup ---
enable_testing()

# Helper function to add a test easily
function(add_igneous_test name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE igneous doctest::doctest)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

# --- Test Suites ---
add_igneous_test(test_algebra tests/algebra.cpp)
add_igneous_test(test_simplex tests/simplex.cpp)

# --- 3. Benchmarks ---
add_executable(bench_algebra tests/benchmark_algebra.cpp)
target_link_libraries(bench_algebra PRIVATE igneous fmt::fmt)

add_executable(bench_memory tests/benchmark_memory.cpp)
target_link_libraries(bench_memory PRIVATE igneous fmt::fmt)