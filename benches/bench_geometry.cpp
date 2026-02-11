#include "igneous/core/algebra.hpp"
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
using igneous::core::Multivector;
using igneous::data::Mesh;

// ==============================================================================
// 1. HELPER: Procedural Mesh Generator
// ==============================================================================
// Creates a "Wavy Grid" of size N x N.
// This ensures we have a dense mesh with curvature to process.
Mesh<Sig> generate_benchmark_mesh(int side_length) {
  Mesh<Sig> mesh;
  mesh.name = "Benchmark_Grid_" + std::to_string(side_length);

  // 1. Generate Vertices (z = sin(x) + cos(y))
  float scale = 10.0f / side_length;
  for (int y = 0; y < side_length; ++y) {
    for (int x = 0; x < side_length; ++x) {
      float px = x * scale;
      float py = y * scale;
      float pz = std::sin(px) + std::cos(py); // Add curvature

      auto mv = Multivector<float, Sig>::from_blade(0, 0);
      mv[1] = px;
      mv[2] = py;
      mv[3] = pz;
      mesh.geometry.points.push_back(mv);
    }
  }

  // 2. Generate Topology (Quads -> Triangles)
  // We manually fill faces_to_vertices to simulate a loaded file
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

  // We do NOT build coboundaries here. We want to benchmark that!
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

  // Warmup (allocations, etc.)
  mesh.topology.build_coboundaries(mesh.geometry.points.size());
  ops::compute_curvature_measures(mesh);

  // Reset for actual timing
  mesh.topology.coboundary_offsets.clear();
  mesh.topology.coboundary_data.clear();

  // --- B. Benchmark: Topology (Graph Build) ---
  auto t0 = Clock::now();
  mesh.topology.build_coboundaries(mesh.geometry.points.size());
  auto t1 = Clock::now();

  // --- C. Benchmark: Geometry Kernel (Curvature) ---
  // Run 10 times to average out noise
  int iterations = 10;
  auto t2 = Clock::now();
  for (int i = 0; i < iterations; ++i) {
    auto [H, K] = ops::compute_curvature_measures(mesh);
    // Sink to prevent optimization (simple XOR sum)
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
                     mesh.geometry.points.size(),
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
  // 100x100 = 10k verts (L1 Cache fit)
  // 500x500 = 250k verts (RAM Bandwidth bound)
  // 1000x1000 = 1M verts (Stress test)

  // Header
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