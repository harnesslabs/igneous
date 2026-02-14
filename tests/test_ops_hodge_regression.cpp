#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cmath>
#include <tuple>

#include <igneous/ops/geometry.hpp>
#include <igneous/ops/hodge.hpp>
#include <igneous/ops/spectral.hpp>

#include "support/synthetic_meshes.hpp"
#include "support/test_env.hpp"

static std::tuple<float, float, float> stats(const Eigen::VectorXf &v) {
  const float min_v = v.minCoeff();
  const float max_v = v.maxCoeff();
  const float mean = v.mean();
  const float variance = (v.array() - mean).square().mean();
  const float stddev = std::sqrt(std::max(variance, 0.0f));
  return {min_v, max_v, stddev};
}

TEST_CASE("Hodge low modes remain near harmonic on canonical torus case") {
  igneous::test_support::configure_deterministic_test_env();
  auto mesh = igneous::test_support::make_torus_cloud(4000, 0.05f, 32);

  igneous::ops::compute_eigenbasis(mesh, 64);

  igneous::ops::GeometryWorkspace<decltype(mesh)> geom_ws;
  const auto G = igneous::ops::compute_1form_gram_matrix(mesh, 0.05f, geom_ws);
  igneous::ops::HodgeWorkspace<decltype(mesh)> hodge_ws;
  const auto D = igneous::ops::compute_weak_exterior_derivative(mesh, 0.05f, hodge_ws);
  const auto E = igneous::ops::compute_curl_energy_matrix(mesh, 0.05f, hodge_ws);
  const auto L = igneous::ops::compute_hodge_laplacian_matrix(D, E);

  const float e_asym = (E - E.transpose()).cwiseAbs().maxCoeff();
  const float l_asym = (L - L.transpose()).cwiseAbs().maxCoeff();
  CHECK(e_asym <= 3e-3f);
  CHECK(l_asym <= 3e-3f);

  auto [evals, evecs] = igneous::ops::compute_hodge_spectrum(L, G);
  REQUIRE(evals.size() >= 6);

  CHECK(evals[0] < 1e-3f);
  CHECK(evals[1] < 1e-3f);
  CHECK(evals[2] < 1e-3f);
  CHECK(evals[3] > 1e-3f);

  for (int i = 0; i < 4; ++i) {
    const Eigen::VectorXf v = evecs.col(i);
    const Eigen::VectorXf gv = G * v;
    const Eigen::VectorXf residual = L * v - evals[i] * gv;
    const float rel_res = residual.norm() / std::max(1.0f, gv.norm());
    CHECK(rel_res <= 1e-2f);
  }

  const auto theta_0 =
      igneous::ops::compute_circular_coordinates(mesh, evecs.col(0), 0.05f);
  const auto theta_1 =
      igneous::ops::compute_circular_coordinates(mesh, evecs.col(1), 0.05f);

  const auto [t0_min, t0_max, t0_std] = stats(theta_0);
  const auto [t1_min, t1_max, t1_std] = stats(theta_1);

  CHECK(t0_min < 0.1f);
  CHECK(t0_max > 0.9f);
  CHECK(t0_std > 0.1f);

  CHECK(t1_min < 0.1f);
  CHECK(t1_max > 0.9f);
  CHECK(t1_std > 0.1f);
}
