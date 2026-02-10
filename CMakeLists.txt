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

# --- 1. The Library Target ---
# INTERFACE means "Header Only" (Common for templates)
add_library(igneous INTERFACE)

# Where are the headers?
target_include_directories(igneous INTERFACE include)

# Dependencies needed by the library itself
target_link_libraries(igneous INTERFACE fmt::fmt range-v3::range-v3)

# --- 2. The Test Runner ---
# This enables CTest (standard testing tool)
enable_testing()

add_executable(igneous-test tests/main.cpp)

# Link the test runner to our library and doctest
target_link_libraries(igneous-test 
    PRIVATE 
    igneous 
    doctest::doctest
)

# Register the test so 'ctest' command works
add_test(NAME unit_tests COMMAND igneous-test)

# --- 3. The Benchmark Runner (NEW) ---
# This creates the 'igneous-bench' executable
add_executable(igneous-bench tests/benchmark.cpp)

# Link to our library and fmt (for printing timing results)
target_link_libraries(igneous-bench 
    PRIVATE 
    igneous 
    fmt::fmt
)