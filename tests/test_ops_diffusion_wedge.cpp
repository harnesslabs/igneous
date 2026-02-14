#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cmath>
#include <random>

#include <igneous/core/algebra.hpp>
#include <igneous/data/mesh.hpp>
#include <igneous/ops/diffusion/products.hpp>
#include <igneous/ops/diffusion/spectral.hpp>

static igneous::data::Mesh<igneous::core::Euclidean3D, igneous::data::DiffusionTopology>
make_cloud(size_t n_points) {
  using Mesh =
      igneous::data::Mesh<igneous::core::Euclidean3D, igneous::data::DiffusionTopology>;
  Mesh mesh;
  mesh.geometry.reserve(n_points);

  for (size_t i = 0; i < n_points; ++i) {
    const float t = static_cast<float>(i) / static_cast<float>(n_points);
    const float a = t * 6.283185f;
    const float b = (t * 7.0f) * 6.283185f;
    mesh.geometry.push_point({(2.0f + 0.7f * std::cos(b)) * std::cos(a),
                              (2.0f + 0.7f * std::cos(b)) * std::sin(a),
                              0.7f * std::sin(b)});
  }

  mesh.topology.build({mesh.geometry.x_span(), mesh.geometry.y_span(),
                       mesh.geometry.z_span(), 24});
  return mesh;
}

TEST_CASE("Wedge product for 1-forms is anti-commutative") {
  auto mesh = make_cloud(720);
  igneous::ops::compute_eigenbasis(mesh, 32);
  const int n_coeff = 16;

  std::mt19937 rng(123);
  std::normal_distribution<float> normal(0.0f, 1.0f);

  Eigen::VectorXf alpha = Eigen::VectorXf::Zero(n_coeff * 3);
  Eigen::VectorXf beta = Eigen::VectorXf::Zero(n_coeff * 3);
  for (int i = 0; i < alpha.size(); ++i) {
    alpha[i] = normal(rng);
    beta[i] = normal(rng);
  }

  igneous::ops::DiffusionFormWorkspace<decltype(mesh)> ws;
  const Eigen::VectorXf ab =
      igneous::ops::compute_wedge_product_coeffs(mesh, alpha, 1, beta, 1,
                                                 n_coeff, ws);
  const Eigen::VectorXf ba =
      igneous::ops::compute_wedge_product_coeffs(mesh, beta, 1, alpha, 1,
                                                 n_coeff, ws);

  REQUIRE(ab.size() == n_coeff * 3);
  REQUIRE(ba.size() == n_coeff * 3);

  const float denom = std::max(1e-8f, ab.norm() + ba.norm());
  const float rel = (ab + ba).norm() / denom;
  CHECK(rel < 5e-4f);
}

TEST_CASE("Linearized wedge operator matches direct wedge product") {
  auto mesh = make_cloud(840);
  igneous::ops::compute_eigenbasis(mesh, 36);
  const int n_coeff = 18;

  std::mt19937 rng(7);
  std::normal_distribution<float> normal(0.0f, 1.0f);

  Eigen::VectorXf alpha = Eigen::VectorXf::Zero(n_coeff * 3);
  Eigen::VectorXf beta = Eigen::VectorXf::Zero(n_coeff * 3);
  for (int i = 0; i < alpha.size(); ++i) {
    alpha[i] = normal(rng);
    beta[i] = normal(rng);
  }

  igneous::ops::DiffusionFormWorkspace<decltype(mesh)> ws;
  const Eigen::MatrixXf op =
      igneous::ops::compute_wedge_operator_matrix(mesh, alpha, 1, 1, n_coeff, ws);
  const Eigen::VectorXf direct =
      igneous::ops::compute_wedge_product_coeffs(mesh, alpha, 1, beta, 1,
                                                 n_coeff, ws);

  REQUIRE(op.cols() == beta.size());
  REQUIRE(op.rows() == direct.size());

  const Eigen::VectorXf via_op = op * beta;
  const float denom = std::max(1e-8f, direct.norm());
  const float rel_err = (via_op - direct).norm() / denom;
  CHECK(rel_err < 1e-4f);
}
