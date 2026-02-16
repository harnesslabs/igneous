#include <benchmark/benchmark.h>

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <igneous/core/algebra.hpp>
#include <igneous/data/space.hpp>
#include <igneous/io/importer.hpp>
#include <igneous/ops/diffusion/geometry.hpp>
#include <igneous/ops/diffusion/hodge.hpp>
#include <igneous/ops/diffusion/spectral.hpp>
#include <igneous/ops/transform.hpp>

using MeshSig = igneous::core::Euclidean3D;
using DiffusionMesh = igneous::data::Space<igneous::data::DiffusionGeometry>;

namespace {
struct BenchEnvSetup {
  BenchEnvSetup() {
    setenv("IGNEOUS_BENCH_MODE", "1", 1);
  }
} kBenchEnvSetup;

std::filesystem::path resolve_bunny_path() {
  static const std::array<const char*, 4> kCandidates = {"assets/bunny.obj", "../assets/bunny.obj",
                                                         "../../assets/bunny.obj",
                                                         "../../../assets/bunny.obj"};
  for (const char* candidate : kCandidates) {
    const std::filesystem::path path(candidate);
    if (std::filesystem::exists(path)) {
      return path;
    }
  }
  return {};
}

DiffusionMesh make_bunny_geometry() {
  DiffusionMesh mesh;
  const std::filesystem::path bunny = resolve_bunny_path();
  if (bunny.empty()) {
    return mesh;
  }
  igneous::io::load_obj(mesh, bunny.string());
  igneous::ops::normalize(mesh);
  mesh.structure.clear();
  return mesh;
}

void generate_torus(DiffusionMesh& mesh, size_t n_points, float R, float r) {
  mesh.clear();
  mesh.reserve(n_points);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(0.0f, 6.283185f);

  for (size_t i = 0; i < n_points; ++i) {
    const float u = dist(gen);
    const float v = dist(gen);

    const float x = (R + r * std::cos(v)) * std::cos(u);
    const float y = (R + r * std::cos(v)) * std::sin(u);
    const float z = r * std::sin(v);
    mesh.push_point({x, y, z});
  }
}

int compute_max_y_vertex(const DiffusionMesh& mesh) {
  float max_y = -std::numeric_limits<float>::infinity();
  int max_y_idx = 0;
  for (size_t i = 0; i < mesh.num_points(); ++i) {
    const auto p = mesh.get_vec3(i);
    if (p.y > max_y) {
      max_y = p.y;
      max_y_idx = static_cast<int>(i);
    }
  }
  return max_y_idx;
}

void build_diffusion_geometry(DiffusionMesh& mesh, float /*bandwidth*/, int k_neighbors) {
  mesh.structure.build({mesh.x_span(), mesh.y_span(), mesh.z_span(), k_neighbors});
}

void bench_pipeline_diffusion_main(benchmark::State& state) {
  static DiffusionMesh base_mesh = make_bunny_geometry();
  if (!base_mesh.is_valid()) {
    state.SkipWithError("Failed to load assets/bunny.obj");
    return;
  }

  constexpr float kBandwidth = 0.005f;
  constexpr int kNeighbors = 32;

  DiffusionMesh mesh = base_mesh;
  const int max_y_idx = compute_max_y_vertex(mesh);
  const int steps = static_cast<int>(state.range(0));
  Eigen::VectorXf u = Eigen::VectorXf::Zero(static_cast<int>(mesh.num_points()));
  Eigen::VectorXf u_next = Eigen::VectorXf::Zero(static_cast<int>(mesh.num_points()));
  igneous::ops::diffusion::DiffusionWorkspace<DiffusionMesh> diffusion_ws;

  for (auto _ : state) {
    build_diffusion_geometry(mesh, kBandwidth, kNeighbors);
    u.setZero();
    u[max_y_idx] = 1000.0f;
    igneous::ops::diffusion::apply_markov_transition_steps(mesh, u, steps, u_next, diffusion_ws);
    u.swap(u_next);

    benchmark::DoNotOptimize(u.data());
  }
}

void bench_pipeline_spectral_main(benchmark::State& state) {
  static DiffusionMesh base_mesh = make_bunny_geometry();
  if (!base_mesh.is_valid()) {
    state.SkipWithError("Failed to load assets/bunny.obj");
    return;
  }

  constexpr float kBandwidth = 0.005f;
  constexpr int kNeighbors = 32;
  constexpr int kBasis = 16;

  DiffusionMesh mesh = base_mesh;
  igneous::ops::diffusion::GeometryWorkspace<DiffusionMesh> geometry_ws;

  for (auto _ : state) {
    build_diffusion_geometry(mesh, kBandwidth, kNeighbors);
    igneous::ops::diffusion::compute_eigenbasis(mesh, kBasis);
    const Eigen::MatrixXf G =
        igneous::ops::diffusion::compute_1form_gram_matrix(mesh, kBandwidth, geometry_ws);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> solver(G);
    benchmark::DoNotOptimize(solver.eigenvalues().data());
  }
}

void bench_pipeline_hodge_main(benchmark::State& state) {
  constexpr float kBandwidth = 0.05f;
  constexpr int kNeighbors = 32;
  constexpr int kBasis = 64;

  DiffusionMesh mesh;
  generate_torus(mesh, 4000, 2.0f, 0.8f);

  igneous::ops::diffusion::GeometryWorkspace<DiffusionMesh> geom_ws;
  igneous::ops::diffusion::HodgeWorkspace<DiffusionMesh> hodge_ws;

  for (auto _ : state) {
    build_diffusion_geometry(mesh, kBandwidth, kNeighbors);
    igneous::ops::diffusion::compute_eigenbasis(mesh, kBasis);

    const Eigen::MatrixXf G =
        igneous::ops::diffusion::compute_1form_gram_matrix(mesh, kBandwidth, geom_ws);
    const Eigen::MatrixXf D =
        igneous::ops::diffusion::compute_weak_exterior_derivative(mesh, kBandwidth, hodge_ws);
    const Eigen::MatrixXf E =
        igneous::ops::diffusion::compute_curl_energy_matrix(mesh, kBandwidth, hodge_ws);
    const Eigen::MatrixXf L = igneous::ops::diffusion::compute_hodge_laplacian_matrix(D, E);
    auto [evals, evecs] = igneous::ops::diffusion::compute_hodge_spectrum(L, G);

    const Eigen::VectorXf theta_0 =
        igneous::ops::diffusion::compute_circular_coordinates(mesh, evecs.col(0), kBandwidth);
    const Eigen::VectorXf theta_1 =
        igneous::ops::diffusion::compute_circular_coordinates(mesh, evecs.col(1), kBandwidth);

    benchmark::DoNotOptimize(evals.data());
    benchmark::DoNotOptimize(theta_0.data());
    benchmark::DoNotOptimize(theta_1.data());
  }
}

void bench_hodge_phase_structure_build(benchmark::State& state) {
  constexpr float kBandwidth = 0.05f;
  constexpr int kNeighbors = 32;
  DiffusionMesh mesh;
  generate_torus(mesh, 4000, 2.0f, 0.8f);

  for (auto _ : state) {
    build_diffusion_geometry(mesh, kBandwidth, kNeighbors);
    benchmark::DoNotOptimize(mesh.structure.markov_values.size());
  }
}

void bench_hodge_phase_eigenbasis(benchmark::State& state) {
  constexpr float kBandwidth = 0.05f;
  constexpr int kNeighbors = 32;
  constexpr int kBasis = 64;
  DiffusionMesh mesh;
  generate_torus(mesh, 4000, 2.0f, 0.8f);
  build_diffusion_geometry(mesh, kBandwidth, kNeighbors);

  for (auto _ : state) {
    igneous::ops::diffusion::compute_eigenbasis(mesh, kBasis);
    benchmark::DoNotOptimize(mesh.structure.eigen_basis.data());
  }
}

void bench_hodge_phase_gram(benchmark::State& state) {
  constexpr float kBandwidth = 0.05f;
  constexpr int kNeighbors = 32;
  constexpr int kBasis = 64;
  DiffusionMesh mesh;
  generate_torus(mesh, 4000, 2.0f, 0.8f);
  build_diffusion_geometry(mesh, kBandwidth, kNeighbors);
  igneous::ops::diffusion::compute_eigenbasis(mesh, kBasis);
  igneous::ops::diffusion::GeometryWorkspace<DiffusionMesh> geom_ws;

  for (auto _ : state) {
    const Eigen::MatrixXf G =
        igneous::ops::diffusion::compute_1form_gram_matrix(mesh, kBandwidth, geom_ws);
    benchmark::DoNotOptimize(G.data());
  }
}

void bench_hodge_phase_weak_derivative(benchmark::State& state) {
  constexpr float kBandwidth = 0.05f;
  constexpr int kNeighbors = 32;
  constexpr int kBasis = 64;
  DiffusionMesh mesh;
  generate_torus(mesh, 4000, 2.0f, 0.8f);
  build_diffusion_geometry(mesh, kBandwidth, kNeighbors);
  igneous::ops::diffusion::compute_eigenbasis(mesh, kBasis);
  igneous::ops::diffusion::HodgeWorkspace<DiffusionMesh> hodge_ws;

  for (auto _ : state) {
    const Eigen::MatrixXf D =
        igneous::ops::diffusion::compute_weak_exterior_derivative(mesh, kBandwidth, hodge_ws);
    benchmark::DoNotOptimize(D.data());
  }
}

void bench_hodge_phase_curl_energy(benchmark::State& state) {
  constexpr float kBandwidth = 0.05f;
  constexpr int kNeighbors = 32;
  constexpr int kBasis = 64;
  DiffusionMesh mesh;
  generate_torus(mesh, 4000, 2.0f, 0.8f);
  build_diffusion_geometry(mesh, kBandwidth, kNeighbors);
  igneous::ops::diffusion::compute_eigenbasis(mesh, kBasis);
  igneous::ops::diffusion::HodgeWorkspace<DiffusionMesh> hodge_ws;

  for (auto _ : state) {
    const Eigen::MatrixXf E =
        igneous::ops::diffusion::compute_curl_energy_matrix(mesh, kBandwidth, hodge_ws);
    benchmark::DoNotOptimize(E.data());
  }
}

void bench_hodge_phase_solve(benchmark::State& state) {
  constexpr float kBandwidth = 0.05f;
  constexpr int kNeighbors = 32;
  constexpr int kBasis = 64;
  DiffusionMesh mesh;
  generate_torus(mesh, 4000, 2.0f, 0.8f);
  build_diffusion_geometry(mesh, kBandwidth, kNeighbors);
  igneous::ops::diffusion::compute_eigenbasis(mesh, kBasis);

  igneous::ops::diffusion::GeometryWorkspace<DiffusionMesh> geom_ws;
  igneous::ops::diffusion::HodgeWorkspace<DiffusionMesh> hodge_ws;
  const Eigen::MatrixXf G =
      igneous::ops::diffusion::compute_1form_gram_matrix(mesh, kBandwidth, geom_ws);
  const Eigen::MatrixXf D =
      igneous::ops::diffusion::compute_weak_exterior_derivative(mesh, kBandwidth, hodge_ws);
  const Eigen::MatrixXf E =
      igneous::ops::diffusion::compute_curl_energy_matrix(mesh, kBandwidth, hodge_ws);
  const Eigen::MatrixXf L = igneous::ops::diffusion::compute_hodge_laplacian_matrix(D, E);

  for (auto _ : state) {
    auto [evals, evecs] = igneous::ops::diffusion::compute_hodge_spectrum(L, G);
    benchmark::DoNotOptimize(evals.data());
    benchmark::DoNotOptimize(evecs.data());
  }
}

void bench_hodge_phase_circular(benchmark::State& state) {
  constexpr float kBandwidth = 0.05f;
  constexpr int kNeighbors = 32;
  constexpr int kBasis = 64;
  DiffusionMesh mesh;
  generate_torus(mesh, 4000, 2.0f, 0.8f);
  build_diffusion_geometry(mesh, kBandwidth, kNeighbors);
  igneous::ops::diffusion::compute_eigenbasis(mesh, kBasis);

  igneous::ops::diffusion::GeometryWorkspace<DiffusionMesh> geom_ws;
  igneous::ops::diffusion::HodgeWorkspace<DiffusionMesh> hodge_ws;
  const Eigen::MatrixXf G =
      igneous::ops::diffusion::compute_1form_gram_matrix(mesh, kBandwidth, geom_ws);
  const Eigen::MatrixXf D =
      igneous::ops::diffusion::compute_weak_exterior_derivative(mesh, kBandwidth, hodge_ws);
  const Eigen::MatrixXf E =
      igneous::ops::diffusion::compute_curl_energy_matrix(mesh, kBandwidth, hodge_ws);
  const Eigen::MatrixXf L = igneous::ops::diffusion::compute_hodge_laplacian_matrix(D, E);
  auto [evals, evecs] = igneous::ops::diffusion::compute_hodge_spectrum(L, G);

  for (auto _ : state) {
    const Eigen::VectorXf theta_0 =
        igneous::ops::diffusion::compute_circular_coordinates(mesh, evecs.col(0), kBandwidth);
    const Eigen::VectorXf theta_1 =
        igneous::ops::diffusion::compute_circular_coordinates(mesh, evecs.col(1), kBandwidth);
    benchmark::DoNotOptimize(theta_0.data());
    benchmark::DoNotOptimize(theta_1.data());
    benchmark::DoNotOptimize(evals.data());
  }
}
} // namespace

BENCHMARK(bench_pipeline_diffusion_main)->Arg(20)->Arg(100);
BENCHMARK(bench_pipeline_spectral_main);
BENCHMARK(bench_pipeline_hodge_main);

BENCHMARK(bench_hodge_phase_structure_build);
BENCHMARK(bench_hodge_phase_eigenbasis);
BENCHMARK(bench_hodge_phase_gram);
BENCHMARK(bench_hodge_phase_weak_derivative);
BENCHMARK(bench_hodge_phase_curl_energy);
BENCHMARK(bench_hodge_phase_solve);
BENCHMARK(bench_hodge_phase_circular);

BENCHMARK_MAIN();
