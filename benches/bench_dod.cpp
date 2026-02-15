#include <benchmark/benchmark.h>

#include <Eigen/Dense>
#include <cmath>
#include <cstdlib>
#include <random>
#include <vector>

#include <igneous/core/algebra.hpp>
#include <igneous/data/mesh.hpp>
#include <igneous/ops/curvature.hpp>
#include <igneous/ops/flow.hpp>
#include <igneous/ops/diffusion/geometry.hpp>
#include <igneous/ops/diffusion/hodge.hpp>
#include <igneous/ops/diffusion/spectral.hpp>

using MeshSig = igneous::core::Euclidean3D;
using SurfaceMesh = igneous::data::Mesh<MeshSig, igneous::data::TriangleTopology>;
using DiffusionMesh = igneous::data::Mesh<MeshSig, igneous::data::DiffusionTopology>;

namespace {
struct BenchEnvSetup {
  BenchEnvSetup() { setenv("IGNEOUS_BENCH_MODE", "1", 1); }
} kBenchEnvSetup;
} // namespace

static SurfaceMesh make_grid_mesh(int side_length) {
  SurfaceMesh mesh;
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
      const uint32_t i1 = static_cast<uint32_t>(y * side_length + x + 1);
      const uint32_t i2 = static_cast<uint32_t>((y + 1) * side_length + x);
      const uint32_t i3 = static_cast<uint32_t>((y + 1) * side_length + x + 1);

      mesh.topology.faces_to_vertices.push_back(i0);
      mesh.topology.faces_to_vertices.push_back(i1);
      mesh.topology.faces_to_vertices.push_back(i2);

      mesh.topology.faces_to_vertices.push_back(i1);
      mesh.topology.faces_to_vertices.push_back(i3);
      mesh.topology.faces_to_vertices.push_back(i2);
    }
  }

  mesh.topology.build({mesh.geometry.num_points(), true});
  return mesh;
}

static DiffusionMesh make_diffusion_cloud(size_t n_points) {
  setenv("IGNEOUS_BENCH_MODE", "1", 1);
  DiffusionMesh mesh;
  mesh.geometry.reserve(n_points);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> angle_dist(0.0f, 6.283185f);
  std::uniform_real_distribution<float> radius_dist(0.6f, 1.0f);

  for (size_t i = 0; i < n_points; ++i) {
    const float u = angle_dist(rng);
    const float v = angle_dist(rng);
    const float r = radius_dist(rng);

    const float x = (2.0f + r * std::cos(v)) * std::cos(u);
    const float y = (2.0f + r * std::cos(v)) * std::sin(u);
    const float z = r * std::sin(v);
    mesh.geometry.push_point({x, y, z});
  }

  mesh.topology.build({mesh.geometry.x_span(), mesh.geometry.y_span(),
                       mesh.geometry.z_span(), 32});
  return mesh;
}

static void bench_mesh_topology_build(benchmark::State &state) {
  SurfaceMesh mesh = make_grid_mesh(static_cast<int>(state.range(0)));
  for (auto _ : state) {
    mesh.topology.vertex_face_offsets.clear();
    mesh.topology.vertex_face_data.clear();
    mesh.topology.vertex_neighbor_offsets.clear();
    mesh.topology.vertex_neighbor_data.clear();
    mesh.topology.build({mesh.geometry.num_points(), true});
    benchmark::DoNotOptimize(mesh.topology.vertex_neighbor_data.data());
  }
}

static void bench_curvature_kernel(benchmark::State &state) {
  SurfaceMesh mesh = make_grid_mesh(static_cast<int>(state.range(0)));
  std::vector<float> H;
  std::vector<float> K;
  igneous::ops::CurvatureWorkspace<MeshSig, igneous::data::TriangleTopology> ws;

  for (auto _ : state) {
    igneous::ops::compute_curvature_measures(mesh, H, K, ws);
    benchmark::DoNotOptimize(H.data());
    benchmark::DoNotOptimize(K.data());
  }
}

static void bench_flow_kernel(benchmark::State &state) {
  SurfaceMesh mesh = make_grid_mesh(static_cast<int>(state.range(0)));
  igneous::ops::FlowWorkspace<MeshSig, igneous::data::TriangleTopology> ws;

  for (auto _ : state) {
    igneous::ops::integrate_mean_curvature_flow(mesh, 0.01f, ws);
    benchmark::DoNotOptimize(mesh.geometry.x.data());
  }
}

static void bench_diffusion_build(benchmark::State &state) {
  DiffusionMesh mesh = make_diffusion_cloud(static_cast<size_t>(state.range(0)));

  for (auto _ : state) {
    mesh.topology.build({mesh.geometry.x_span(), mesh.geometry.y_span(),
                         mesh.geometry.z_span(), 32});
    benchmark::DoNotOptimize(mesh.topology.markov_values.size());
  }
}

static void bench_markov_step(benchmark::State &state) {
  DiffusionMesh mesh = make_diffusion_cloud(static_cast<size_t>(state.range(0)));
  Eigen::VectorXf u = Eigen::VectorXf::Ones(static_cast<int>(mesh.geometry.num_points()));
  Eigen::VectorXf u_next =
      Eigen::VectorXf::Zero(static_cast<int>(mesh.geometry.num_points()));

  for (auto _ : state) {
    igneous::ops::apply_markov_transition(mesh, u, u_next);
    u.swap(u_next);
    benchmark::DoNotOptimize(u.data());
  }
}

static void bench_markov_multi_step(benchmark::State &state) {
  const size_t n_points = static_cast<size_t>(state.range(0));
  const int steps = static_cast<int>(state.range(1));
  DiffusionMesh mesh = make_diffusion_cloud(n_points);
  Eigen::VectorXf u = Eigen::VectorXf::Ones(static_cast<int>(mesh.geometry.num_points()));
  Eigen::VectorXf u_next =
      Eigen::VectorXf::Zero(static_cast<int>(mesh.geometry.num_points()));
  igneous::ops::DiffusionWorkspace<DiffusionMesh> ws;

  for (auto _ : state) {
    igneous::ops::apply_markov_transition_steps(mesh, u, steps, u_next, ws);
    u.swap(u_next);
    benchmark::DoNotOptimize(u.data());
  }
}

static void bench_eigenbasis(benchmark::State &state) {
  DiffusionMesh mesh = make_diffusion_cloud(static_cast<size_t>(state.range(0)));
  const int basis = static_cast<int>(state.range(1));

  for (auto _ : state) {
    igneous::ops::compute_eigenbasis(mesh, basis);
    benchmark::DoNotOptimize(mesh.topology.eigen_basis.data());
  }
}

static void bench_1form_gram(benchmark::State &state) {
  DiffusionMesh mesh = make_diffusion_cloud(static_cast<size_t>(state.range(0)));
  igneous::ops::compute_eigenbasis(mesh, static_cast<int>(state.range(1)));
  igneous::ops::GeometryWorkspace<DiffusionMesh> ws;

  for (auto _ : state) {
    auto G = igneous::ops::compute_1form_gram_matrix(mesh, 0.05f, ws);
    benchmark::DoNotOptimize(G.data());
  }
}

static void bench_weak_derivative(benchmark::State &state) {
  DiffusionMesh mesh = make_diffusion_cloud(static_cast<size_t>(state.range(0)));
  igneous::ops::compute_eigenbasis(mesh, static_cast<int>(state.range(1)));
  igneous::ops::HodgeWorkspace<DiffusionMesh> ws;

  for (auto _ : state) {
    auto D = igneous::ops::compute_weak_exterior_derivative(mesh, 0.05f, ws);
    benchmark::DoNotOptimize(D.data());
  }
}

static void bench_curl_energy(benchmark::State &state) {
  DiffusionMesh mesh = make_diffusion_cloud(static_cast<size_t>(state.range(0)));
  igneous::ops::compute_eigenbasis(mesh, static_cast<int>(state.range(1)));
  igneous::ops::HodgeWorkspace<DiffusionMesh> ws;

  for (auto _ : state) {
    auto E = igneous::ops::compute_curl_energy_matrix(mesh, 0.05f, ws);
    benchmark::DoNotOptimize(E.data());
  }
}

static void bench_hodge_solve(benchmark::State &state) {
  DiffusionMesh mesh = make_diffusion_cloud(static_cast<size_t>(state.range(0)));
  igneous::ops::compute_eigenbasis(mesh, static_cast<int>(state.range(1)));

  igneous::ops::GeometryWorkspace<DiffusionMesh> geom_ws;
  igneous::ops::HodgeWorkspace<DiffusionMesh> hodge_ws;

  const auto G = igneous::ops::compute_1form_gram_matrix(mesh, 0.05f, geom_ws);
  const auto D = igneous::ops::compute_weak_exterior_derivative(mesh, 0.05f, hodge_ws);
  const auto E = igneous::ops::compute_curl_energy_matrix(mesh, 0.05f, hodge_ws);
  const auto L = igneous::ops::compute_hodge_laplacian_matrix(D, E);

  for (auto _ : state) {
    auto [evals, evecs] = igneous::ops::compute_hodge_spectrum(L, G);
    benchmark::DoNotOptimize(evals.data());
    benchmark::DoNotOptimize(evecs.data());
  }
}

BENCHMARK(bench_mesh_topology_build)->Arg(400);
BENCHMARK(bench_curvature_kernel)->Arg(400);
BENCHMARK(bench_flow_kernel)->Arg(400);
BENCHMARK(bench_diffusion_build)->Arg(2000);
BENCHMARK(bench_markov_step)->Arg(2000);
BENCHMARK(bench_markov_multi_step)->Args({2000, 20})->Args({20000, 20});
BENCHMARK(bench_eigenbasis)->Args({2000, 16});
BENCHMARK(bench_1form_gram)->Args({2000, 16});
BENCHMARK(bench_weak_derivative)->Args({2000, 16});
BENCHMARK(bench_curl_energy)->Args({2000, 16});
BENCHMARK(bench_hodge_solve)->Args({2000, 16});

BENCHMARK_MAIN();
