#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cmath>
#include <igneous/core/algebra.hpp>
#include <igneous/data/mesh.hpp>
#include <igneous/ops/geometry.hpp>
#include <igneous/ops/spectral.hpp>

static igneous::data::Mesh<igneous::core::Euclidean3D, igneous::data::DiffusionTopology>
make_diffusion_cloud(size_t n_points) {
  using Mesh = igneous::data::Mesh<igneous::core::Euclidean3D, igneous::data::DiffusionTopology>;
  Mesh mesh;
  mesh.geometry.reserve(n_points);

  for (size_t i = 0; i < n_points; ++i) {
    const float t = static_cast<float>(i) / static_cast<float>(n_points);
    mesh.geometry.push_point({std::cos(t * 6.283185f), std::sin(t * 6.283185f), t});
  }

  mesh.topology.build({mesh.geometry.x_span(), mesh.geometry.y_span(), mesh.geometry.z_span(), 0.05f, 24});
  return mesh;
}

TEST_CASE("Spectral basis and Gram matrix are finite and shaped correctly") {
  auto mesh = make_diffusion_cloud(400);

  igneous::ops::compute_eigenbasis(mesh, 8);

  CHECK(mesh.topology.eigen_basis.rows() == static_cast<int>(mesh.geometry.num_points()));
  CHECK(mesh.topology.eigen_basis.cols() >= 1);

  igneous::ops::GeometryWorkspace<decltype(mesh)> ws;
  const Eigen::MatrixXf G = igneous::ops::compute_1form_gram_matrix(mesh, 0.05f, ws);

  CHECK(G.rows() == mesh.topology.eigen_basis.cols() * 3);
  CHECK(G.cols() == mesh.topology.eigen_basis.cols() * 3);

  for (int r = 0; r < G.rows(); ++r) {
    for (int c = 0; c < G.cols(); ++c) {
      CHECK(std::isfinite(G(r, c)));
      CHECK(G(r, c) == doctest::Approx(G(c, r)).epsilon(1e-3f));
    }
  }
}
