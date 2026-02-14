#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <Eigen/Eigenvalues>
#include <cmath>

#include <igneous/ops/geometry.hpp>
#include <igneous/ops/spectral.hpp>

#include "support/synthetic_meshes.hpp"
#include "support/test_env.hpp"
#include "support/tolerances.hpp"

TEST_CASE("carre_du_champ is symmetric, bilinear, and near-positive on squares") {
  igneous::test_support::configure_deterministic_test_env();
  auto mesh = igneous::test_support::make_helix_cloud(320);

  const int n = static_cast<int>(mesh.geometry.num_points());
  Eigen::VectorXf f = Eigen::VectorXf::LinSpaced(n, -1.0f, 1.0f);
  Eigen::VectorXf g = f.array().square();
  Eigen::VectorXf h = (f.array() * 1.37f).sin();

  Eigen::VectorXf gamma_fh(n), gamma_hf(n), gamma_gh(n), gamma_lin(n);
  igneous::ops::carre_du_champ(mesh, f, h, 0.05f, gamma_fh);
  igneous::ops::carre_du_champ(mesh, h, f, 0.05f, gamma_hf);
  igneous::ops::carre_du_champ(mesh, g, h, 0.05f, gamma_gh);

  const float symmetry_inf =
      (gamma_fh - gamma_hf).cwiseAbs().maxCoeff();
  CHECK(symmetry_inf <= 3.0f * igneous::test_support::kTolMedium);

  constexpr float a = 1.7f;
  constexpr float b = -0.3f;
  const Eigen::VectorXf linear_combo = a * f + b * g;
  igneous::ops::carre_du_champ(mesh, linear_combo, h, 0.05f, gamma_lin);

  const Eigen::VectorXf expected = a * gamma_fh + b * gamma_gh;
  const float rel_err =
      (gamma_lin - expected).norm() / std::max(1.0f, expected.norm());
  CHECK(rel_err <= 1e-3f);

  Eigen::VectorXf gamma_ff(n);
  igneous::ops::carre_du_champ(mesh, f, f, 0.05f, gamma_ff);
  CHECK(gamma_ff.minCoeff() >= -1e-4f);
}

TEST_CASE("1-form Gram matrix is symmetric and near-PSD") {
  igneous::test_support::configure_deterministic_test_env();
  auto mesh = igneous::test_support::make_helix_cloud(400);
  igneous::ops::compute_eigenbasis(mesh, 12);

  igneous::ops::GeometryWorkspace<decltype(mesh)> ws;
  const Eigen::MatrixXf G = igneous::ops::compute_1form_gram_matrix(mesh, 0.05f, ws);

  CHECK(G.rows() == 36);
  CHECK(G.cols() == 36);

  const float asym = (G - G.transpose()).cwiseAbs().maxCoeff();
  CHECK(asym <= 5.0f * igneous::test_support::kTolMedium);

  const Eigen::MatrixXf G_sym = 0.5f * (G + G.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eig(G_sym);
  REQUIRE(eig.info() == Eigen::Success);

  const float min_eval = eig.eigenvalues().minCoeff();
  CHECK(min_eval >= -2e-3f);
}
