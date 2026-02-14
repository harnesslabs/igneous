#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cmath>
#include <igneous/core/algebra.hpp>
#include <igneous/data/mesh.hpp>
#include <igneous/ops/geometry.hpp>
#include <igneous/ops/hodge.hpp>
#include <igneous/ops/spectral.hpp>

static igneous::data::Mesh<igneous::core::Euclidean3D, igneous::data::DiffusionTopology>
make_torus_cloud(size_t n_points) {
  using Mesh = igneous::data::Mesh<igneous::core::Euclidean3D, igneous::data::DiffusionTopology>;
  Mesh mesh;
  mesh.geometry.reserve(n_points);

  for (size_t i = 0; i < n_points; ++i) {
    const float u = static_cast<float>(i % 40) / 40.0f * 6.283185f;
    const float v = static_cast<float>(i / 40) / std::max<size_t>(1, n_points / 40) * 6.283185f;
    const float R = 2.0f;
    const float r = 0.8f;
    const float x = (R + r * std::cos(v)) * std::cos(u);
    const float y = (R + r * std::cos(v)) * std::sin(u);
    const float z = r * std::sin(v);
    mesh.geometry.push_point({x, y, z});
  }

  mesh.topology.build({mesh.geometry.x_span(), mesh.geometry.y_span(),
                       mesh.geometry.z_span(), 24});
  return mesh;
}

TEST_CASE("Hodge operators produce finite spectrum and circular coordinates") {
  auto mesh = make_torus_cloud(320);
  igneous::ops::compute_eigenbasis(mesh, 10);

  igneous::ops::GeometryWorkspace<decltype(mesh)> geom_ws;
  const auto G = igneous::ops::compute_1form_gram_matrix(mesh, 0.05f, geom_ws);

  igneous::ops::HodgeWorkspace<decltype(mesh)> hodge_ws;
  const auto D = igneous::ops::compute_weak_exterior_derivative(mesh, 0.05f, hodge_ws);
  const auto E = igneous::ops::compute_curl_energy_matrix(mesh, 0.05f, hodge_ws);
  const auto L = igneous::ops::compute_hodge_laplacian_matrix(D, E);

  CHECK(L.rows() == D.rows());
  CHECK(L.cols() == D.rows());

  for (int r = 0; r < L.rows(); ++r) {
    for (int c = 0; c < L.cols(); ++c) {
      CHECK(std::isfinite(L(r, c)));
      CHECK(L(r, c) == doctest::Approx(L(c, r)).epsilon(1e-3f));
    }
  }

  auto [evals, evecs] = igneous::ops::compute_hodge_spectrum(L, G);
  CHECK(evals.size() > 0);
  for (int i = 0; i < evals.size(); ++i) {
    CHECK(std::isfinite(evals[i]));
  }

  const auto theta =
      igneous::ops::compute_circular_coordinates(mesh, evecs.col(0), 0.0f);
  CHECK(theta.size() == static_cast<int>(mesh.geometry.num_points()));
  for (int i = 0; i < theta.size(); ++i) {
    CHECK(std::isfinite(theta[i]));
    CHECK(theta[i] >= -0.01f);
    CHECK(theta[i] <= 6.30f);
  }
}
