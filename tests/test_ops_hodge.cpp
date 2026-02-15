#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cmath>
#include <igneous/core/algebra.hpp>
#include <igneous/data/space.hpp>
#include <igneous/ops/diffusion/geometry.hpp>
#include <igneous/ops/diffusion/hodge.hpp>
#include <igneous/ops/diffusion/spectral.hpp>

static igneous::data::Space<igneous::data::DiffusionGeometry>
make_torus_cloud(size_t n_points) {
  using Mesh = igneous::data::Space<igneous::data::DiffusionGeometry>;
  Mesh mesh;
  mesh.reserve(n_points);

  for (size_t i = 0; i < n_points; ++i) {
    const float u = static_cast<float>(i % 40) / 40.0f * 6.283185f;
    const float v = static_cast<float>(i / 40) / std::max<size_t>(1, n_points / 40) * 6.283185f;
    const float R = 2.0f;
    const float r = 0.8f;
    const float x = (R + r * std::cos(v)) * std::cos(u);
    const float y = (R + r * std::cos(v)) * std::sin(u);
    const float z = r * std::sin(v);
    mesh.push_point({x, y, z});
  }

  mesh.structure.build({mesh.x_span(), mesh.y_span(),
                       mesh.z_span(), 24});
  return mesh;
}

TEST_CASE("Hodge operators produce finite spectrum and circular coordinates") {
  auto mesh = make_torus_cloud(1000);
  igneous::ops::diffusion::compute_eigenbasis(mesh, 50);

  igneous::ops::diffusion::GeometryWorkspace<decltype(mesh)> geom_ws;
  const auto G = igneous::ops::diffusion::compute_1form_gram_matrix(mesh, 0.05f, geom_ws);

  igneous::ops::diffusion::HodgeWorkspace<decltype(mesh)> hodge_ws;
  const auto D = igneous::ops::diffusion::compute_weak_exterior_derivative(mesh, 0.05f, hodge_ws);
  const auto E = igneous::ops::diffusion::compute_curl_energy_matrix(mesh, 0.05f, hodge_ws);
  const auto L = igneous::ops::diffusion::compute_hodge_laplacian_matrix(D, E);

  CHECK(L.rows() == D.rows());
  CHECK(L.cols() == D.rows());

  for (int r = 0; r < L.rows(); ++r) {
    for (int c = 0; c < L.cols(); ++c) {
      CHECK(std::isfinite(L(r, c)));
      CHECK(L(r, c) == doctest::Approx(L(c, r)).epsilon(1e-3f));
    }
  }

  auto [evals, evecs] = igneous::ops::diffusion::compute_hodge_spectrum(L, G);
  CHECK(evals.size() > 0);
  CHECK(evecs.cols() >= 2);
  for (int i = 0; i < evals.size(); ++i) {
    CHECK(std::isfinite(evals[i]));
  }

  const auto theta_0 = igneous::ops::diffusion::compute_circular_coordinates(
      mesh, evecs.col(0), 0.0f, 1.0f, 0);
  const auto theta_1 = igneous::ops::diffusion::compute_circular_coordinates(
      mesh, evecs.col(1), 0.0f, 1.0f, 1);

  const auto check_theta = [&](const Eigen::VectorXf &theta) {
    CHECK(theta.size() == static_cast<int>(mesh.num_points()));

    float mean = 0.0f;
    for (int i = 0; i < theta.size(); ++i) {
      CHECK(std::isfinite(theta[i]));
      CHECK(theta[i] >= -0.01f);
      CHECK(theta[i] <= 6.30f);
      mean += theta[i];
    }
    mean /= static_cast<float>(std::max<int>(1, theta.size()));

    float var = 0.0f;
    for (int i = 0; i < theta.size(); ++i) {
      const float d = theta[i] - mean;
      var += d * d;
    }
    var /= static_cast<float>(std::max<int>(1, theta.size()));
    const float stddev = std::sqrt(var);
    CHECK(stddev > 0.03f);
  };

  check_theta(theta_0);
  check_theta(theta_1);
}
