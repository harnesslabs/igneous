#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <igneous/core/algebra.hpp>
#include <igneous/data/mesh.hpp>
#include <igneous/data/topology.hpp>
#include <igneous/ops/curvature.hpp>
#include <igneous/ops/flow.hpp>

using namespace igneous;
using Clock = std::chrono::high_resolution_clock;
using Sig = core::Euclidean3D;
using Mesh = data::Mesh<Sig, data::TriangleTopology>;

static Mesh generate_benchmark_mesh(int side_length) {
  Mesh mesh;
  mesh.name = "Benchmark_Grid_" + std::to_string(side_length);
  mesh.geometry.reserve(static_cast<size_t>(side_length * side_length));

  const float scale = 10.0f / static_cast<float>(side_length);
  for (int y = 0; y < side_length; ++y) {
    for (int x = 0; x < side_length; ++x) {
      const float px = x * scale;
      const float py = y * scale;
      const float pz = std::sin(px) + std::cos(py);
      mesh.geometry.push_point({px, py, pz});
    }
  }

  mesh.topology.faces_to_vertices.reserve(
      static_cast<size_t>((side_length - 1) * (side_length - 1) * 6));

  for (int y = 0; y < side_length - 1; ++y) {
    for (int x = 0; x < side_length - 1; ++x) {
      const uint32_t i0 = static_cast<uint32_t>(y * side_length + x);
      const uint32_t i1 = static_cast<uint32_t>(y * side_length + (x + 1));
      const uint32_t i2 = static_cast<uint32_t>((y + 1) * side_length + x);
      const uint32_t i3 = static_cast<uint32_t>((y + 1) * side_length + (x + 1));

      mesh.topology.faces_to_vertices.push_back(i0);
      mesh.topology.faces_to_vertices.push_back(i1);
      mesh.topology.faces_to_vertices.push_back(i2);

      mesh.topology.faces_to_vertices.push_back(i1);
      mesh.topology.faces_to_vertices.push_back(i3);
      mesh.topology.faces_to_vertices.push_back(i2);
    }
  }

  return mesh;
}

struct BenchResult {
  std::string name;
  size_t verts = 0;
  size_t faces = 0;
  double t_topology_ms = 0.0;
  double t_curvature_ms = 0.0;
  double t_flow_ms = 0.0;
  double fps = 0.0;
};

static BenchResult run_workload(int grid_size) {
  auto mesh = generate_benchmark_mesh(grid_size);
  const size_t n_verts = mesh.geometry.num_points();

  mesh.topology.build({n_verts, true});

  std::vector<float> H;
  std::vector<float> K;
  ops::CurvatureWorkspace<Sig, data::TriangleTopology> curvature_ws;
  ops::FlowWorkspace<Sig, data::TriangleTopology> flow_ws;

  ops::compute_curvature_measures(mesh, H, K, curvature_ws);

  mesh.topology.vertex_face_offsets.clear();
  mesh.topology.vertex_face_data.clear();
  mesh.topology.vertex_neighbor_offsets.clear();
  mesh.topology.vertex_neighbor_data.clear();

  const auto t0 = Clock::now();
  mesh.topology.build({n_verts, true});
  const auto t1 = Clock::now();

  constexpr int iterations = 10;

  const auto t2 = Clock::now();
  for (int i = 0; i < iterations; ++i) {
    ops::compute_curvature_measures(mesh, H, K, curvature_ws);
  }
  const auto t3 = Clock::now();

  const auto t4 = Clock::now();
  for (int i = 0; i < iterations; ++i) {
    ops::integrate_mean_curvature_flow(mesh, 0.01f, flow_ws);
  }
  const auto t5 = Clock::now();

  const double ms_topo = std::chrono::duration<double, std::milli>(t1 - t0).count();
  const double ms_curv =
      std::chrono::duration<double, std::milli>(t3 - t2).count() / iterations;
  const double ms_flow =
      std::chrono::duration<double, std::milli>(t5 - t4).count() / iterations;

  return {
      "Grid " + std::to_string(grid_size) + "x" + std::to_string(grid_size),
      n_verts,
      mesh.topology.num_faces(),
      ms_topo,
      ms_curv,
      ms_flow,
      1000.0 / (ms_curv + ms_flow),
  };
}

int main() {
  const std::vector<int> sizes = {100, 250, 500, 1000};

  std::cout << "\n==========================================================================\n";
  std::cout << " IGNEOUS GEOMETRY ENGINE BENCHMARK\n";
  std::cout << "==========================================================================\n";
  std::cout << std::left << std::setw(15) << "Mesh" << std::setw(12) << "Verts"
            << std::setw(12) << "Faces" << std::setw(15) << "Topo (ms)"
            << std::setw(15) << "Curv (ms)" << std::setw(15) << "Flow (ms)"
            << std::setw(10) << "Sim FPS" << "\n";
  std::cout << "--------------------------------------------------------------------------\n";

  for (int size : sizes) {
    const auto res = run_workload(size);
    std::cout << std::left << std::setw(15) << res.name << std::setw(12)
              << res.verts << std::setw(12) << res.faces << std::fixed
              << std::setprecision(3) << std::setw(15) << res.t_topology_ms
              << std::setw(15) << res.t_curvature_ms << std::setw(15)
              << res.t_flow_ms << std::setw(10) << res.fps << "\n";
  }

  std::cout << "==========================================================================\n\n";
  return 0;
}
