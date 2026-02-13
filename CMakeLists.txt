cmake_minimum_required(VERSION 3.25)
project(igneous LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(fmt CONFIG REQUIRED)
find_package(range-v3 CONFIG REQUIRED)
find_package(doctest CONFIG REQUIRED)
find_package(xsimd CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(nanoflann CONFIG REQUIRED)
find_package(Spectra CONFIG REQUIRED)
find_package(benchmark CONFIG REQUIRED)

add_library(igneous INTERFACE)
target_include_directories(igneous INTERFACE include)
target_link_libraries(
  igneous
  INTERFACE fmt::fmt
            range-v3::range-v3
            xsimd
            Eigen3::Eigen
            nanoflann::nanoflann
            Spectra::Spectra)

enable_testing()

function(add_igneous_test name source)
  add_executable(${name} ${source})
  target_link_libraries(${name} PRIVATE igneous doctest::doctest)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

add_igneous_test(test_algebra tests/test_algebra.cpp)
add_igneous_test(test_topology_triangle tests/test_topology_triangle.cpp)
add_igneous_test(test_topology_diffusion tests/test_topology_diffusion.cpp)
add_igneous_test(test_ops_curvature_flow tests/test_ops_curvature_flow.cpp)
add_igneous_test(test_ops_spectral_geometry tests/test_ops_spectral_geometry.cpp)
add_igneous_test(test_ops_hodge tests/test_ops_hodge.cpp)
add_igneous_test(test_io_meshes tests/test_io_meshes.cpp)

add_executable(bench_memory benches/bench_memory.cpp)
target_link_libraries(bench_memory PRIVATE igneous fmt::fmt)

add_executable(bench_geometry benches/bench_geometry.cpp)
target_link_libraries(bench_geometry PRIVATE igneous fmt::fmt)

add_executable(bench_dod benches/bench_dod.cpp)
target_link_libraries(bench_dod PRIVATE igneous benchmark::benchmark_main)

add_executable(igneous-mesh src/main_mesh.cpp)
target_link_libraries(igneous-mesh PRIVATE igneous fmt::fmt)

add_executable(igneous-point src/main_point.cpp)
target_link_libraries(igneous-point PRIVATE igneous fmt::fmt)

add_executable(igneous-diffusion src/main_diffusion.cpp)
target_link_libraries(igneous-diffusion PRIVATE igneous fmt::fmt)

add_executable(igneous-spectral src/main_spectral.cpp)
target_link_libraries(igneous-spectral PRIVATE igneous fmt::fmt)

add_executable(igneous-hodge src/main_hodge.cpp)
target_link_libraries(igneous-hodge PRIVATE igneous fmt::fmt)
