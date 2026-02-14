#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cmath>

#include <igneous/ops/geometry.hpp>
#include <igneous/ops/spectral.hpp>

#include "support/synthetic_meshes.hpp"
#include "support/test_env.hpp"
#include "support/tolerances.hpp"

TEST_CASE("Spectral basis is mu-orthonormal with low eigen residuals") {
  igneous::test_support::configure_deterministic_test_env();
  auto mesh = igneous::test_support::make_helix_cloud(600);

  igneous::ops::compute_eigenbasis(mesh, 16);

  const auto &U = mesh.topology.eigen_basis;
  const auto &mu = mesh.topology.mu;
  const auto &evals = mesh.topology.eigen_values;

  REQUIRE(U.cols() > 0);
  REQUIRE(evals.size() == U.cols());

  const Eigen::MatrixXf mu_weighted_u = U.array().colwise() * mu.array();
  const Eigen::MatrixXf gram = U.transpose() * mu_weighted_u;
  const Eigen::VectorXf gram_diag = gram.diagonal();
  CHECK(gram_diag.minCoeff() > 1e-6f);

  const auto &diag = mesh.topology.spectral_diagnostics;
  if (diag.used_mode == igneous::data::SpectralSolveMode::SymmetricTransform) {
    const Eigen::MatrixXf identity =
        Eigen::MatrixXf::Identity(U.cols(), U.cols());
    const float ortho_err = (gram - identity).cwiseAbs().maxCoeff();
    CHECK(ortho_err <= 5e-3f);
  }

  Eigen::VectorXf tmp = Eigen::VectorXf::Zero(U.rows());
  for (int i = 0; i < U.cols(); ++i) {
    igneous::ops::apply_markov_transition(mesh, U.col(i), tmp);
    const Eigen::VectorXf residual = tmp - evals[i] * U.col(i);
    const float rel_residual =
        residual.norm() / std::max(1.0f, U.col(i).norm());
    CHECK(rel_residual <= 5e-3f);
  }
}

TEST_CASE("Large-graph spectral auto mode avoids unsafe symmetric path") {
  igneous::test_support::configure_deterministic_test_env();
  auto mesh = igneous::test_support::make_torus_cloud(4000, 0.05f, 32);

  igneous::ops::compute_eigenbasis(mesh, 64);
  const auto &diag = mesh.topology.spectral_diagnostics;

  CHECK(diag.used_mode == igneous::data::SpectralSolveMode::GenericArnoldi);
  CHECK(diag.nconv >= 32);
}
