#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cmath>
#include <igneous/core/algebra.hpp>
#include <igneous/data/buffers.hpp>
#include <igneous/data/mesh.hpp>
#include <igneous/data/topology.hpp>
#include <igneous/ops/geometry.hpp>

TEST_CASE("DiffusionTopology produces stochastic Markov matrix and valid measure") {
  using Sig = igneous::core::Euclidean3D;

  igneous::data::GeometryBuffer<float, Sig> geometry;
  geometry.reserve(16);

  for (int i = 0; i < 16; ++i) {
    const float t = static_cast<float>(i) / 16.0f;
    geometry.push_point({std::cos(t * 6.283185f), std::sin(t * 6.283185f), t});
  }

  igneous::data::DiffusionTopology topo;
  topo.build({geometry.x_span(), geometry.y_span(), geometry.z_span(), 0.05f, 8});

  CHECK(topo.P.rows() == 16);
  CHECK(topo.P.cols() == 16);
  CHECK(topo.P.nonZeros() > 0);

  Eigen::VectorXf row_sums = Eigen::VectorXf::Zero(topo.P.rows());
  for (int outer = 0; outer < topo.P.outerSize(); ++outer) {
    for (igneous::data::DiffusionTopology::SparseMatrixT::InnerIterator it(topo.P,
                                                                            outer);
         it; ++it) {
      CHECK(it.value() >= 0.0f);
      row_sums[it.row()] += it.value();
    }
  }

  for (int i = 0; i < row_sums.size(); ++i) {
    CHECK(row_sums[i] == doctest::Approx(1.0f).epsilon(1e-3f));
  }

  CHECK(topo.mu.sum() == doctest::Approx(1.0f).epsilon(1e-3f));
  for (int i = 0; i < topo.mu.size(); ++i) {
    CHECK(topo.mu[i] >= 0.0f);
  }
}

TEST_CASE("Diffusion CSR markov step matches sparse matrix product") {
  using Sig = igneous::core::Euclidean3D;
  using DiffusionMesh =
      igneous::data::Mesh<Sig, igneous::data::DiffusionTopology>;

  DiffusionMesh mesh;
  mesh.geometry.reserve(24);
  for (int i = 0; i < 24; ++i) {
    const float t = static_cast<float>(i) / 24.0f;
    mesh.geometry.push_point(
        {std::cos(t * 6.283185f), std::sin(t * 6.283185f), 0.5f * t});
  }

  mesh.topology.build({mesh.geometry.x_span(), mesh.geometry.y_span(),
                       mesh.geometry.z_span(), 0.05f, 8});

  const int n = static_cast<int>(mesh.geometry.num_points());
  Eigen::VectorXf u = Eigen::VectorXf::LinSpaced(n, -1.0f, 1.0f);
  Eigen::VectorXf expected = mesh.topology.P * u;
  Eigen::VectorXf actual = Eigen::VectorXf::Zero(n);

  igneous::ops::apply_markov_transition(mesh, u, actual);

  for (int i = 0; i < n; ++i) {
    CHECK(actual[i] == doctest::Approx(expected[i]).epsilon(1e-5f));
  }
}

TEST_CASE("Diffusion multi-step markov matches repeated single steps") {
  using Sig = igneous::core::Euclidean3D;
  using DiffusionMesh =
      igneous::data::Mesh<Sig, igneous::data::DiffusionTopology>;

  DiffusionMesh mesh;
  mesh.geometry.reserve(24);
  for (int i = 0; i < 24; ++i) {
    const float t = static_cast<float>(i) / 24.0f;
    mesh.geometry.push_point(
        {std::cos(t * 6.283185f), std::sin(t * 6.283185f), 0.5f * t});
  }

  mesh.topology.build({mesh.geometry.x_span(), mesh.geometry.y_span(),
                       mesh.geometry.z_span(), 0.05f, 8});

  const int n = static_cast<int>(mesh.geometry.num_points());
  Eigen::VectorXf u0 = Eigen::VectorXf::LinSpaced(n, -1.0f, 1.0f);

  constexpr int kSteps = 7;
  Eigen::VectorXf expected = u0;
  Eigen::VectorXf tmp = Eigen::VectorXf::Zero(n);
  for (int step = 0; step < kSteps; ++step) {
    igneous::ops::apply_markov_transition(mesh, expected, tmp);
    expected.swap(tmp);
  }

  Eigen::VectorXf actual = Eigen::VectorXf::Zero(n);
  igneous::ops::DiffusionWorkspace<DiffusionMesh> ws;
  igneous::ops::apply_markov_transition_steps(mesh, u0, kSteps, actual, ws);

  for (int i = 0; i < n; ++i) {
    CHECK(actual[i] == doctest::Approx(expected[i]).epsilon(1e-5f));
  }
}
