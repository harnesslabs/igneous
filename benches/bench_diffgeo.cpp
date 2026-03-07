#include <benchmark/benchmark.h>

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <vector>

#include <igneous/core/algebra.hpp>
#include <igneous/data/space.hpp>
#include <igneous/data/structures/diffusion_geometry.hpp>
#include <igneous/ops/diffusion/forms.hpp>
#include <igneous/ops/diffusion/hodge.hpp>
#include <igneous/ops/diffusion/products.hpp>
#include <igneous/ops/diffusion/spectral.hpp>

using DiffusionMesh = igneous::data::Space<igneous::data::DiffusionGeometry>;

namespace {
struct BenchEnvSetup {
  BenchEnvSetup() {
    setenv("IGNEOUS_BENCH_MODE", "1", 1);
  }
} kBenchEnvSetup;

struct DiffgeoConfig {
  size_t n_points = 0;
  int n_basis = 0;
  int k_neighbors = 0;
  int n_coeff = 0;
};

struct HarmonicModes {
  Eigen::VectorXf evals1;
  Eigen::MatrixXf evecs1;
  std::vector<int> harmonic1_idx;
};

DiffgeoConfig config_from_state(const benchmark::State& state) {
  return {
      static_cast<size_t>(state.range(0)),
      static_cast<int>(state.range(1)),
      static_cast<int>(state.range(2)),
      static_cast<int>(state.range(3)),
  };
}

void generate_torus(DiffusionMesh& mesh, size_t n_points, float R, float r, uint32_t seed = 0) {
  mesh.clear();
  mesh.reserve(n_points);

  std::mt19937 gen(seed);
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

DiffusionMesh make_torus_mesh(const DiffgeoConfig& cfg) {
  DiffusionMesh mesh;
  generate_torus(mesh, cfg.n_points, 2.0f, 1.0f, 0);
  return mesh;
}

void build_diffusion_geometry(DiffusionMesh& mesh, int k_neighbors) {
  mesh.structure.build({mesh.x_span(), mesh.y_span(), mesh.z_span(), k_neighbors, 8, -0.5f, 0.0f,
                        true});
}

std::array<std::array<Eigen::VectorXf, 3>, 3>
build_gamma_data_immersion(const DiffusionMesh& mesh) {
  std::array<Eigen::VectorXf, 3> data_coords;
  std::array<Eigen::VectorXf, 3> immersion_coords;
  igneous::ops::diffusion::fill_data_coordinate_vectors(mesh, data_coords);
  igneous::ops::diffusion::fill_coordinate_vectors(mesh, immersion_coords);

  std::array<std::array<Eigen::VectorXf, 3>, 3> gamma{};
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      gamma[a][b].resize(static_cast<int>(mesh.num_points()));
      igneous::ops::diffusion::carre_du_champ(mesh, data_coords[a], immersion_coords[b], 0.0f,
                                              gamma[a][b]);
    }
  }
  return gamma;
}

std::vector<igneous::core::Vec3>
reconstruct_1form_ambient(const DiffusionMesh& mesh, const Eigen::VectorXf& coeffs,
                          int n_coefficients,
                          const std::array<std::array<Eigen::VectorXf, 3>, 3>& gamma_data) {
  const Eigen::MatrixXf pointwise =
      igneous::ops::diffusion::coefficients_to_pointwise(mesh, coeffs, 1, n_coefficients);
  std::vector<igneous::core::Vec3> field(mesh.num_points(), {0.0f, 0.0f, 0.0f});
  if (pointwise.rows() == 0 || pointwise.cols() < 3) {
    return field;
  }

  for (size_t i = 0; i < mesh.num_points(); ++i) {
    const int p = static_cast<int>(i);
    field[i] = {
        gamma_data[0][0][p] * pointwise(p, 0) + gamma_data[0][1][p] * pointwise(p, 1) +
            gamma_data[0][2][p] * pointwise(p, 2),
        gamma_data[1][0][p] * pointwise(p, 0) + gamma_data[1][1][p] * pointwise(p, 1) +
            gamma_data[1][2][p] * pointwise(p, 2),
        gamma_data[2][0][p] * pointwise(p, 0) + gamma_data[2][1][p] * pointwise(p, 1) +
            gamma_data[2][2][p] * pointwise(p, 2),
    };
  }

  return field;
}

std::vector<igneous::core::Vec3>
reconstruct_2form_dual_ambient(const DiffusionMesh& mesh, const Eigen::VectorXf& coeffs,
                               int n_coefficients,
                               const std::array<std::array<Eigen::VectorXf, 3>, 3>& gamma_data) {
  const Eigen::MatrixXf pointwise =
      igneous::ops::diffusion::coefficients_to_pointwise(mesh, coeffs, 2, n_coefficients);
  std::vector<igneous::core::Vec3> field(mesh.num_points(), {0.0f, 0.0f, 0.0f});
  if (pointwise.rows() == 0 || pointwise.cols() < 3) {
    return field;
  }

  for (size_t i = 0; i < mesh.num_points(); ++i) {
    const int p = static_cast<int>(i);
    const float w01 = pointwise(p, 0);
    const float w02 = pointwise(p, 1);
    const float w12 = pointwise(p, 2);

    Eigen::Matrix3f W = Eigen::Matrix3f::Zero();
    W(0, 1) = w01;
    W(1, 0) = -w01;
    W(0, 2) = w02;
    W(2, 0) = -w02;
    W(1, 2) = w12;
    W(2, 1) = -w12;

    Eigen::Matrix3f G = Eigen::Matrix3f::Zero();
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        G(a, b) = gamma_data[a][b][p];
      }
    }

    const Eigen::Matrix3f ambient = G * W * G.transpose();
    field[i] = {ambient(1, 2), -ambient(0, 2), ambient(0, 1)};
  }

  return field;
}

Eigen::VectorXf pad_1form_coeffs(const Eigen::VectorXf& coeffs, int n_basis, int n_coefficients) {
  Eigen::VectorXf padded = Eigen::VectorXf::Zero(n_basis * 3);
  const int copy_n = std::min(n_basis, n_coefficients);
  for (int i = 0; i < copy_n; ++i) {
    for (int a = 0; a < 3; ++a) {
      padded(i * 3 + a) = coeffs(i * 3 + a);
    }
  }
  return padded;
}

HarmonicModes compute_harmonic_1modes(DiffusionMesh& mesh, int n_basis, int n_coeff,
                                      igneous::ops::diffusion::DiffusionFormWorkspace<DiffusionMesh>&
                                          forms_ws) {
  igneous::ops::diffusion::compute_eigenbasis(mesh, n_basis);
  const Eigen::MatrixXf G1 =
      igneous::ops::diffusion::compute_kform_gram_matrix(mesh, 1, n_coeff, forms_ws);
  const Eigen::MatrixXf down1 =
      igneous::ops::diffusion::compute_down_laplacian_matrix(mesh, 1, n_coeff, forms_ws);
  const Eigen::MatrixXf up1 =
      igneous::ops::diffusion::compute_up_laplacian_matrix(mesh, 1, n_coeff, forms_ws);
  const Eigen::MatrixXf L1 = igneous::ops::diffusion::assemble_hodge_laplacian_matrix(up1, down1);
  auto [evals1, evecs1] = igneous::ops::diffusion::compute_form_spectrum(L1, G1);
  auto harmonic1_idx =
      igneous::ops::diffusion::extract_harmonic_mode_indices(evals1, 1e-3f, 3);

  return {
      std::move(evals1),
      std::move(evecs1),
      std::move(harmonic1_idx),
  };
}

void bench_diffgeo_pipeline(benchmark::State& state) {
  const DiffgeoConfig cfg = config_from_state(state);
  DiffusionMesh mesh = make_torus_mesh(cfg);
  igneous::ops::diffusion::DiffusionFormWorkspace<DiffusionMesh> forms_ws;

  for (auto _ : state) {
    forms_ws.reset_cache();
    build_diffusion_geometry(mesh, cfg.k_neighbors);
    igneous::ops::diffusion::compute_eigenbasis(mesh, cfg.n_basis);

    const Eigen::MatrixXf G1 =
        igneous::ops::diffusion::compute_kform_gram_matrix(mesh, 1, cfg.n_coeff, forms_ws);
    const Eigen::MatrixXf G1_inv =
        igneous::ops::diffusion::compute_kform_gram_inverse(mesh, 1, cfg.n_coeff, forms_ws);
    const Eigen::MatrixXf D1 =
        igneous::ops::diffusion::compute_weak_exterior_derivative(mesh, 1, cfg.n_coeff, forms_ws);
    const Eigen::MatrixXf down1 =
        igneous::ops::diffusion::compute_down_laplacian_matrix(mesh, 1, cfg.n_coeff, forms_ws);
    const Eigen::MatrixXf up1 =
        igneous::ops::diffusion::compute_up_laplacian_matrix(mesh, 1, cfg.n_coeff, forms_ws);
    const Eigen::MatrixXf L1 =
        igneous::ops::diffusion::assemble_hodge_laplacian_matrix(up1, down1);
    auto [evals1, evecs1] = igneous::ops::diffusion::compute_form_spectrum(L1, G1);

    if (evals1.size() == 0 || evecs1.cols() == 0) {
      state.SkipWithError("1-form spectrum failed");
      break;
    }

    const auto harmonic1_idx =
        igneous::ops::diffusion::extract_harmonic_mode_indices(evals1, 1e-3f, 3);
    const int idx0 = harmonic1_idx.empty() ? 0 : harmonic1_idx[0];
    const int idx1 = harmonic1_idx.size() > 1 ? harmonic1_idx[1] : idx0;

    const auto gamma_data = build_gamma_data_immersion(mesh);
    const Eigen::VectorXf theta_0 = igneous::ops::diffusion::compute_circular_coordinates(
        mesh, pad_1form_coeffs(evecs1.col(idx0), mesh.structure.eigen_basis.cols(), cfg.n_coeff),
        0.0f, 1.0f, 0, nullptr);
    const Eigen::VectorXf theta_1 = igneous::ops::diffusion::compute_circular_coordinates(
        mesh, pad_1form_coeffs(evecs1.col(idx1), mesh.structure.eigen_basis.cols(), cfg.n_coeff),
        0.0f, 1.0f, 1, nullptr);

    const Eigen::MatrixXf G2 =
        igneous::ops::diffusion::compute_kform_gram_matrix(mesh, 2, cfg.n_coeff, forms_ws);
    const Eigen::MatrixXf down2 =
        igneous::ops::diffusion::assemble_down_laplacian_matrix_from_previous(D1, G1_inv);
    const Eigen::MatrixXf up2 =
        igneous::ops::diffusion::compute_up_laplacian_matrix(mesh, 2, cfg.n_coeff, forms_ws);
    const Eigen::MatrixXf L2 =
        igneous::ops::diffusion::assemble_hodge_laplacian_matrix(up2, down2);
    auto [evals2, evecs2] = igneous::ops::diffusion::compute_form_spectrum(L2, G2);

    if (evals2.size() == 0 || evecs2.cols() == 0) {
      state.SkipWithError("2-form spectrum failed");
      break;
    }

    const auto harmonic2_idx =
        igneous::ops::diffusion::extract_harmonic_mode_indices(evals2, 1e-3f, 3);
    Eigen::VectorXf wedge_coeffs = Eigen::VectorXf::Zero(cfg.n_coeff * 3);
    if (!harmonic1_idx.empty()) {
      const int rhs_col = harmonic1_idx.size() > 1 ? harmonic1_idx[1] : harmonic1_idx[0];
      wedge_coeffs = igneous::ops::diffusion::compute_wedge_product_coeffs(
          mesh, evecs1.col(harmonic1_idx[0]), 1, evecs1.col(rhs_col), 1, cfg.n_coeff, forms_ws);
    }

    Eigen::VectorXf harmonic1 = evecs1.col(idx0);
    Eigen::VectorXf harmonic2 =
        evecs2.col(harmonic2_idx.empty() ? 0 : harmonic2_idx[0]);
    auto field1 = reconstruct_1form_ambient(mesh, harmonic1, cfg.n_coeff, gamma_data);
    auto field2 = reconstruct_2form_dual_ambient(mesh, harmonic2, cfg.n_coeff, gamma_data);
    auto wedge = reconstruct_2form_dual_ambient(mesh, wedge_coeffs, cfg.n_coeff, gamma_data);

    benchmark::DoNotOptimize(theta_0.data());
    benchmark::DoNotOptimize(theta_1.data());
    benchmark::DoNotOptimize(field1.data());
    benchmark::DoNotOptimize(field2.data());
    benchmark::DoNotOptimize(wedge.data());
  }
}

void bench_diffgeo_phase_structure_build(benchmark::State& state) {
  const DiffgeoConfig cfg = config_from_state(state);
  DiffusionMesh mesh = make_torus_mesh(cfg);

  for (auto _ : state) {
    build_diffusion_geometry(mesh, cfg.k_neighbors);
    benchmark::DoNotOptimize(mesh.structure.markov_values.size());
  }
}

void bench_diffgeo_phase_eigenbasis(benchmark::State& state) {
  const DiffgeoConfig cfg = config_from_state(state);
  DiffusionMesh mesh = make_torus_mesh(cfg);
  build_diffusion_geometry(mesh, cfg.k_neighbors);

  for (auto _ : state) {
    igneous::ops::diffusion::compute_eigenbasis(mesh, cfg.n_basis);
    benchmark::DoNotOptimize(mesh.structure.eigen_basis.data());
  }
}

void bench_diffgeo_phase_k1_gram(benchmark::State& state) {
  const DiffgeoConfig cfg = config_from_state(state);
  DiffusionMesh mesh = make_torus_mesh(cfg);
  build_diffusion_geometry(mesh, cfg.k_neighbors);
  igneous::ops::diffusion::compute_eigenbasis(mesh, cfg.n_basis);
  igneous::ops::diffusion::DiffusionFormWorkspace<DiffusionMesh> forms_ws;

  for (auto _ : state) {
    const Eigen::MatrixXf G1 =
        igneous::ops::diffusion::compute_kform_gram_matrix(mesh, 1, cfg.n_coeff, forms_ws);
    benchmark::DoNotOptimize(G1.data());
  }
}

void bench_diffgeo_phase_k1_down(benchmark::State& state) {
  const DiffgeoConfig cfg = config_from_state(state);
  DiffusionMesh mesh = make_torus_mesh(cfg);
  build_diffusion_geometry(mesh, cfg.k_neighbors);
  igneous::ops::diffusion::compute_eigenbasis(mesh, cfg.n_basis);
  igneous::ops::diffusion::DiffusionFormWorkspace<DiffusionMesh> forms_ws;

  for (auto _ : state) {
    const Eigen::MatrixXf down1 =
        igneous::ops::diffusion::compute_down_laplacian_matrix(mesh, 1, cfg.n_coeff, forms_ws);
    benchmark::DoNotOptimize(down1.data());
  }
}

void bench_diffgeo_phase_k1_up(benchmark::State& state) {
  const DiffgeoConfig cfg = config_from_state(state);
  DiffusionMesh mesh = make_torus_mesh(cfg);
  build_diffusion_geometry(mesh, cfg.k_neighbors);
  igneous::ops::diffusion::compute_eigenbasis(mesh, cfg.n_basis);
  igneous::ops::diffusion::DiffusionFormWorkspace<DiffusionMesh> forms_ws;

  for (auto _ : state) {
    const Eigen::MatrixXf up1 =
        igneous::ops::diffusion::compute_up_laplacian_matrix(mesh, 1, cfg.n_coeff, forms_ws);
    benchmark::DoNotOptimize(up1.data());
  }
}

void bench_diffgeo_phase_circular(benchmark::State& state) {
  const DiffgeoConfig cfg = config_from_state(state);
  DiffusionMesh mesh = make_torus_mesh(cfg);
  build_diffusion_geometry(mesh, cfg.k_neighbors);
  igneous::ops::diffusion::DiffusionFormWorkspace<DiffusionMesh> forms_ws;
  const HarmonicModes modes = compute_harmonic_1modes(mesh, cfg.n_basis, cfg.n_coeff, forms_ws);
  if (modes.evals1.size() == 0 || modes.evecs1.cols() == 0) {
    state.SkipWithError("1-form harmonic modes unavailable");
    return;
  }

  const int idx0 = modes.harmonic1_idx.empty() ? 0 : modes.harmonic1_idx[0];
  const int idx1 = modes.harmonic1_idx.size() > 1 ? modes.harmonic1_idx[1] : idx0;
  const Eigen::VectorXf alpha0 =
      pad_1form_coeffs(modes.evecs1.col(idx0), mesh.structure.eigen_basis.cols(), cfg.n_coeff);
  const Eigen::VectorXf alpha1 =
      pad_1form_coeffs(modes.evecs1.col(idx1), mesh.structure.eigen_basis.cols(), cfg.n_coeff);

  for (auto _ : state) {
    const Eigen::VectorXf theta_0 =
        igneous::ops::diffusion::compute_circular_coordinates(mesh, alpha0, 0.0f, 1.0f, 0, nullptr);
    const Eigen::VectorXf theta_1 =
        igneous::ops::diffusion::compute_circular_coordinates(mesh, alpha1, 0.0f, 1.0f, 1, nullptr);
    benchmark::DoNotOptimize(theta_0.data());
    benchmark::DoNotOptimize(theta_1.data());
  }
}

void bench_diffgeo_phase_k2_gram(benchmark::State& state) {
  const DiffgeoConfig cfg = config_from_state(state);
  DiffusionMesh mesh = make_torus_mesh(cfg);
  build_diffusion_geometry(mesh, cfg.k_neighbors);
  igneous::ops::diffusion::compute_eigenbasis(mesh, cfg.n_basis);
  igneous::ops::diffusion::DiffusionFormWorkspace<DiffusionMesh> forms_ws;

  for (auto _ : state) {
    const Eigen::MatrixXf G2 =
        igneous::ops::diffusion::compute_kform_gram_matrix(mesh, 2, cfg.n_coeff, forms_ws);
    benchmark::DoNotOptimize(G2.data());
  }
}

void bench_diffgeo_phase_k2_down(benchmark::State& state) {
  const DiffgeoConfig cfg = config_from_state(state);
  DiffusionMesh mesh = make_torus_mesh(cfg);
  build_diffusion_geometry(mesh, cfg.k_neighbors);
  igneous::ops::diffusion::compute_eigenbasis(mesh, cfg.n_basis);
  igneous::ops::diffusion::DiffusionFormWorkspace<DiffusionMesh> forms_ws;
  const Eigen::MatrixXf G1 =
      igneous::ops::diffusion::compute_kform_gram_matrix(mesh, 1, cfg.n_coeff, forms_ws);
  benchmark::DoNotOptimize(G1.data());
  const Eigen::MatrixXf G1_inv =
      igneous::ops::diffusion::compute_kform_gram_inverse(mesh, 1, cfg.n_coeff, forms_ws);
  const Eigen::MatrixXf D1 =
      igneous::ops::diffusion::compute_weak_exterior_derivative(mesh, 1, cfg.n_coeff, forms_ws);

  for (auto _ : state) {
    const Eigen::MatrixXf down2 =
        igneous::ops::diffusion::assemble_down_laplacian_matrix_from_previous(D1, G1_inv);
    benchmark::DoNotOptimize(down2.data());
  }
}

void bench_diffgeo_phase_k2_up(benchmark::State& state) {
  const DiffgeoConfig cfg = config_from_state(state);
  DiffusionMesh mesh = make_torus_mesh(cfg);
  build_diffusion_geometry(mesh, cfg.k_neighbors);
  igneous::ops::diffusion::compute_eigenbasis(mesh, cfg.n_basis);
  igneous::ops::diffusion::DiffusionFormWorkspace<DiffusionMesh> forms_ws;
  const Eigen::MatrixXf warmup =
      igneous::ops::diffusion::compute_kform_gram_matrix(mesh, 2, cfg.n_coeff, forms_ws);
  benchmark::DoNotOptimize(warmup.data());

  for (auto _ : state) {
    const Eigen::MatrixXf up2 =
        igneous::ops::diffusion::compute_up_laplacian_matrix(mesh, 2, cfg.n_coeff, forms_ws);
    benchmark::DoNotOptimize(up2.data());
  }
}

void bench_diffgeo_phase_wedge(benchmark::State& state) {
  const DiffgeoConfig cfg = config_from_state(state);
  DiffusionMesh mesh = make_torus_mesh(cfg);
  build_diffusion_geometry(mesh, cfg.k_neighbors);
  igneous::ops::diffusion::DiffusionFormWorkspace<DiffusionMesh> forms_ws;
  const HarmonicModes modes = compute_harmonic_1modes(mesh, cfg.n_basis, cfg.n_coeff, forms_ws);
  if (modes.evals1.size() == 0 || modes.evecs1.cols() == 0) {
    state.SkipWithError("1-form harmonic modes unavailable");
    return;
  }

  const int lhs_col = modes.harmonic1_idx.empty() ? 0 : modes.harmonic1_idx[0];
  const int rhs_col = modes.harmonic1_idx.size() > 1 ? modes.harmonic1_idx[1] : lhs_col;

  for (auto _ : state) {
    const Eigen::VectorXf wedge = igneous::ops::diffusion::compute_wedge_product_coeffs(
        mesh, modes.evecs1.col(lhs_col), 1, modes.evecs1.col(rhs_col), 1, cfg.n_coeff, forms_ws);
    benchmark::DoNotOptimize(wedge.data());
  }
}
} // namespace

BENCHMARK(bench_diffgeo_pipeline)->Args({1000, 50, 32, 32})->Args({4000, 64, 32, 32});
BENCHMARK(bench_diffgeo_phase_structure_build)->Args({1000, 50, 32, 32})->Args(
    {4000, 64, 32, 32});
BENCHMARK(bench_diffgeo_phase_eigenbasis)->Args({1000, 50, 32, 32})->Args(
    {4000, 64, 32, 32});
BENCHMARK(bench_diffgeo_phase_k1_gram)->Args({1000, 50, 32, 32})->Args({4000, 64, 32, 32});
BENCHMARK(bench_diffgeo_phase_k1_down)->Args({1000, 50, 32, 32})->Args({4000, 64, 32, 32});
BENCHMARK(bench_diffgeo_phase_k1_up)->Args({1000, 50, 32, 32})->Args({4000, 64, 32, 32});
BENCHMARK(bench_diffgeo_phase_circular)->Args({1000, 50, 32, 32})->Args({4000, 64, 32, 32});
BENCHMARK(bench_diffgeo_phase_k2_gram)->Args({1000, 50, 32, 32})->Args({4000, 64, 32, 32});
BENCHMARK(bench_diffgeo_phase_k2_down)->Args({1000, 50, 32, 32})->Args({4000, 64, 32, 32});
BENCHMARK(bench_diffgeo_phase_k2_up)->Args({1000, 50, 32, 32})->Args({4000, 64, 32, 32});
BENCHMARK(bench_diffgeo_phase_wedge)->Args({1000, 50, 32, 32})->Args({4000, 64, 32, 32});

BENCHMARK_MAIN();
