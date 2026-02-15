#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cmath>

#include <igneous/core/algebra.hpp>
#include <igneous/data/mesh.hpp>
#include <igneous/ops/diffusion/forms.hpp>
#include <igneous/ops/diffusion/spectral.hpp>

static igneous::data::Mesh<igneous::core::Euclidean3D, igneous::data::DiffusionTopology>
make_torus_cloud(size_t n_points) {
  using Mesh =
      igneous::data::Mesh<igneous::core::Euclidean3D, igneous::data::DiffusionTopology>;
  Mesh mesh;
  mesh.geometry.reserve(n_points);

  for (size_t i = 0; i < n_points; ++i) {
    const float u = static_cast<float>(i % 32) / 32.0f * 6.283185f;
    const float v =
        static_cast<float>(i / 32) / std::max<size_t>(1, n_points / 32) * 6.283185f;
    const float R = 2.0f;
    const float r = 0.8f;
    mesh.geometry.push_point(
        {(R + r * std::cos(v)) * std::cos(u), (R + r * std::cos(v)) * std::sin(u),
         r * std::sin(v)});
  }

  mesh.topology.build({mesh.geometry.x_span(), mesh.geometry.y_span(),
                       mesh.geometry.z_span(), 24});
  return mesh;
}

TEST_CASE("Generic diffusion form operators are finite and shape-consistent") {
  auto mesh = make_torus_cloud(960);
  igneous::ops::compute_eigenbasis(mesh, 48);
  const int n_coeff = 24;

  igneous::ops::DiffusionFormWorkspace<decltype(mesh)> ws;

  const auto G1 = igneous::ops::compute_kform_gram_matrix(mesh, 1, n_coeff, ws);
  const auto G2 = igneous::ops::compute_kform_gram_matrix(mesh, 2, n_coeff, ws);
  const auto D1 = igneous::ops::compute_weak_exterior_derivative(mesh, 1, n_coeff, ws);
  const auto up2 = igneous::ops::compute_up_laplacian_matrix(mesh, 2, n_coeff, ws);
  const auto down2 = igneous::ops::compute_down_laplacian_matrix(mesh, 2, n_coeff, ws);

  CHECK(G1.rows() == n_coeff * 3);
  CHECK(G1.cols() == n_coeff * 3);
  CHECK(G2.rows() == n_coeff * 3);
  CHECK(G2.cols() == n_coeff * 3);
  CHECK(D1.rows() == n_coeff * 3);
  CHECK(D1.cols() == n_coeff * 3);
  CHECK(up2.rows() == n_coeff * 3);
  CHECK(up2.cols() == n_coeff * 3);
  CHECK(down2.rows() == n_coeff * 3);
  CHECK(down2.cols() == n_coeff * 3);

  for (int r = 0; r < up2.rows(); ++r) {
    for (int c = 0; c < up2.cols(); ++c) {
      CHECK(std::isfinite(up2(r, c)));
      CHECK(std::isfinite(down2(r, c)));
      CHECK(up2(r, c) == doctest::Approx(up2(c, r)).epsilon(1e-3f));
      CHECK(down2(r, c) == doctest::Approx(down2(c, r)).epsilon(1e-3f));
    }
  }

  const auto L2 = igneous::ops::assemble_hodge_laplacian_matrix(up2, down2);
  auto [evals2, evecs2] = igneous::ops::compute_form_spectrum(L2, G2);
  CHECK(evals2.size() > 0);
  CHECK(evecs2.cols() > 0);
  for (int i = 0; i < evals2.size(); ++i) {
    CHECK(std::isfinite(evals2[i]));
  }

  const auto harmonic_idx = igneous::ops::extract_harmonic_mode_indices(evals2, 1e-2f, 3);
  CHECK(!harmonic_idx.empty());
}

TEST_CASE("Up-Laplacian(2) entry matches manual Schur-determinant assembly") {
  auto mesh = make_torus_cloud(640);
  igneous::ops::compute_eigenbasis(mesh, 36);
  const int n_coeff = 16;

  igneous::ops::DiffusionFormWorkspace<decltype(mesh)> ws;
  const auto up2 = igneous::ops::compute_up_laplacian_matrix(mesh, 2, n_coeff, ws);

  const auto &U = mesh.topology.eigen_basis;
  const auto &mu = mesh.topology.mu;

  igneous::ops::ensure_gamma_coords(mesh, ws);
  igneous::ops::ensure_gamma_mixed(mesh, n_coeff, ws);
  const auto idx2 = igneous::ops::get_wedge_basis_indices(3, 2);

  // Compare entry (i=0,J=0 ; l=0,K=0).
  float manual = 0.0f;
  Eigen::VectorXf gamma_phi_phi(mesh.geometry.num_points());
  igneous::ops::carre_du_champ(mesh, U.col(0), U.col(0), 0.0f, gamma_phi_phi);

  const auto &row_combo = idx2[0];
  const auto &col_combo = idx2[0];
  for (int p = 0; p < gamma_phi_phi.size(); ++p) {
    Eigen::Matrix2f Dm;
    Dm(0, 0) = ws.gamma_coords[row_combo[0]][col_combo[0]][p];
    Dm(0, 1) = ws.gamma_coords[row_combo[0]][col_combo[1]][p];
    Dm(1, 0) = ws.gamma_coords[row_combo[1]][col_combo[0]][p];
    Dm(1, 1) = ws.gamma_coords[row_combo[1]][col_combo[1]][p];
    const float detD = Dm.determinant();

    Eigen::Vector2f c_vec;
    c_vec[0] = ws.gamma_mixed[row_combo[0]](p, 0);
    c_vec[1] = ws.gamma_mixed[row_combo[1]](p, 0);

    Eigen::Vector2f b_vec;
    b_vec[0] = ws.gamma_mixed[col_combo[0]](p, 0);
    b_vec[1] = ws.gamma_mixed[col_combo[1]](p, 0);

    const Eigen::Vector2f solved = igneous::ops::solve_stable_2x2(Dm, c_vec);
    const float bDv = b_vec.dot(solved);
    manual += mu[p] * detD * (gamma_phi_phi[p] - bDv);
  }

  CHECK(up2(0, 0) == doctest::Approx(manual).epsilon(2e-3f));
}
