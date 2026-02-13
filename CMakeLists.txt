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
find_package(Eigen3 CONFIG REQUIRED)
find_package(nanoflann CONFIG REQUIRED)
find_package(Spectra CONFIG REQUIRED)

# --- 1. The Library Target ---
add_library(igneous INTERFACE)
target_include_directories(igneous INTERFACE include)
target_link_libraries(igneous INTERFACE fmt::fmt range-v3::range-v3 xsimd Eigen3::Eigen nanoflann::nanoflann Spectra::Spectra)

# --- 2. Testing Setup ---
enable_testing()

# Helper function to add a test easily
function(add_igneous_test name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE igneous doctest::doctest)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

# --- Test Suites ---
add_igneous_test(test_algebra tests/test_algebra.cpp)

# --- 3. Benchmarks ---
add_executable(bench_memory benches/bench_memory.cpp)
target_link_libraries(bench_memory PRIVATE igneous fmt::fmt)

add_executable(bench_geometry benches/bench_geometry.cpp)
target_link_libraries(bench_geometry PRIVATE igneous fmt::fmt)

# --- THE MAIN APPLICATION ---
add_executable(igneous-mesh src/main_mesh.cpp)
target_link_libraries(igneous-mesh 
    PRIVATE 
    igneous 
    fmt::fmt
)

add_executable(igneous-point src/main_point.cpp)
target_link_libraries(igneous-point 
    PRIVATE 
    igneous 
    fmt::fmt
)

add_executable(igneous-diffusion src/main_diffusion.cpp)
target_link_libraries(igneous-diffusion 
    PRIVATE 
    igneous 
    fmt::fmt
)

add_executable(igneous-spectral src/main_spectral.cpp)
target_link_libraries(igneous-spectral 
    PRIVATE 
    igneous 
    fmt::fmt
)

add_executable(igneous-hodge src/main_hodge.cpp)
target_link_libraries(igneous-hodge 
    PRIVATE 
    igneous 
    fmt::fmt
)