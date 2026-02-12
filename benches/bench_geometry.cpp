#include "igneous/core/algebra.hpp"
#include "igneous/core/blades.hpp"
#include "igneous/data/topology.hpp"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// Include your Engine
#include <igneous/data/mesh.hpp>
#include <igneous/ops/curvature.hpp>
#include <igneous/ops/flow.hpp>
#include <igneous/ops/transform.hpp>

using namespace igneous;
using Clock = std::chrono::high_resolution_clock;
using Sig = igneous::core::Euclidean3D;
using igneous::data::Mesh;

// ==============================================================================
// 1. HELPER: Procedural Mesh Generator
// ==============================================================================
Mesh<Sig> generate_benchmark_mesh(int side_length) {
  Mesh<Sig> mesh;
  mesh.name = "Benchmark_Grid_" + std::to_string(side_length);

  // Pre-allocate memory using the new API
  // Estimate: side_length^2 vertices
  mesh.geometry.reserve(side_length * side_length, 0, 0);

  // 1. Generate Vertices (z = sin(x) + cos(y))
  float scale = 10.0f / side_length;
  for (int y = 0; y < side_length; ++y) {
    for (int x = 0; x < side_length; ++x) {
      float px = x * scale;
      float py = y * scale;
      float pz = std::sin(px) + std::cos(py); // Add curvature

      auto v = core::Vec3{px, py, pz};
      mesh.geometry.push_point(v);
    }
  }

  // 2. Generate Topology (Quads -> Triangles)
  std::vector<uint32_t> indices;
  indices.reserve((side_length - 1) * (side_length - 1) * 6);

  for (int y = 0; y < side_length - 1; ++y) {
    for (int x = 0; x < side_length - 1; ++x) {
      uint32_t i0 = y * side_length + x;
      uint32_t i1 = y * side_length + (x + 1);
      uint32_t i2 = (y + 1) * side_length + x;
      uint32_t i3 = (y + 1) * side_length + (x + 1);

      // Triangle 1
      indices.push_back(i0);
      indices.push_back(i1);
      indices.push_back(i2);
      // Triangle 2
      indices.push_back(i1);
      indices.push_back(i3);
      indices.push_back(i2);
    }
  }

  mesh.topology.faces_to_vertices = std::move(indices);
  return mesh;
}

// ==============================================================================
// 2. BENCHMARK RUNNER
// ==============================================================================
struct BenchResult {
  std::string name;
  size_t verts;
  size_t faces;
  double t_topology_ms;
  double t_curvature_ms;
  double t_flow_ms;
  double fps;
};

BenchResult run_workload(int grid_size) {
  // A. Setup
  auto mesh = generate_benchmark_mesh(grid_size);

  // FIXED: Use num_points() accessor
  size_t n_verts = mesh.geometry.num_points();

  // Warmup (allocations, etc.)
  mesh.topology.build(data::TriangleTopology::Input{n_verts});
  ops::compute_curvature_measures(mesh);

  // Reset for actual timing
  mesh.topology.coboundary_offsets.clear();
  mesh.topology.coboundary_data.clear();

  // --- B. Benchmark: Topology (Graph Build) ---
  auto t0 = Clock::now();
  mesh.topology.build(data::TriangleTopology::Input{n_verts});
  auto t1 = Clock::now();

  // --- C. Benchmark: Geometry Kernel (Curvature) ---
  int iterations = 10;
  auto t2 = Clock::now();
  for (int i = 0; i < iterations; ++i) {
    auto [H, K] = ops::compute_curvature_measures(mesh);
    if (H.empty())
      std::cout << "Error";
  }
  auto t3 = Clock::now();

  // --- D. Benchmark: Physics Kernel (Flow Integration) ---
  auto t4 = Clock::now();
  for (int i = 0; i < iterations; ++i) {
    ops::integrate_mean_curvature_flow(mesh, 0.01);
  }
  auto t5 = Clock::now();

  // Calculate Stats
  double ms_topo = std::chrono::duration<double, std::milli>(t1 - t0).count();
  double ms_curv =
      std::chrono::duration<double, std::milli>(t3 - t2).count() / iterations;
  double ms_flow =
      std::chrono::duration<double, std::milli>(t5 - t4).count() / iterations;
  double total_frame_ms = ms_curv + ms_flow;

  return BenchResult{"Grid " + std::to_string(grid_size) + "x" +
                         std::to_string(grid_size),
                     n_verts, // FIXED
                     mesh.topology.num_faces(),
                     ms_topo,
                     ms_curv,
                     ms_flow,
                     1000.0 / total_frame_ms};
}

// ==============================================================================
// 3. MAIN ENTRY POINT
// ==============================================================================
int main() {
  std::vector<int> sizes = {100, 250, 500, 1000};

  std::cout << "\n============================================================="
               "====================\n";
  std::cout << " IGNEOUS GEOMETRY ENGINE BENCHMARK \n";
  std::cout << "==============================================================="
               "==================\n";
  std::cout << std::left << std::setw(15) << "Mesh" << std::setw(12) << "Verts"
            << std::setw(12) << "Faces" << std::setw(15) << "Topo (ms)"
            << std::setw(15) << "Curv (ms)" << std::setw(15) << "Flow (ms)"
            << std::setw(10) << "Sim FPS"
            << "\n";
  std::cout << "---------------------------------------------------------------"
               "------------------\n";

  for (int s : sizes) {
    auto res = run_workload(s);

    std::cout << std::left << std::setw(15) << res.name << std::setw(12)
              << res.verts << std::setw(12) << res.faces << std::fixed
              << std::setprecision(3) << std::setw(15) << res.t_topology_ms
              << std::setw(15) << res.t_curvature_ms << std::setw(15)
              << res.t_flow_ms << std::setw(10) << res.fps << "\n";
  }
  std::cout << "==============================================================="
               "==================\n\n";

  return 0;
}