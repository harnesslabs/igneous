#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cmath>
#include <igneous/core/algebra.hpp>
#include <igneous/data/buffers.hpp>
#include <igneous/data/topology.hpp>

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
    for (Eigen::SparseMatrix<float>::InnerIterator it(topo.P, outer); it; ++it) {
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
