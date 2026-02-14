#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cmath>
#include <igneous/core/algebra.hpp>
#include <igneous/ops/curvature.hpp>
#include <igneous/ops/flow.hpp>

#include "support/synthetic_meshes.hpp"
#include "support/test_env.hpp"
#include "support/tolerances.hpp"

using TriangleMesh = igneous::test_support::TriangleMesh;

static float roughness_proxy(const TriangleMesh &mesh) {
  float sum = 0.0f;
  for (size_t i = 0; i < mesh.geometry.num_points(); ++i) {
    const auto pi = mesh.geometry.get_vec3(i);
    for (uint32_t j : mesh.topology.get_vertex_neighbors(static_cast<uint32_t>(i))) {
      if (j <= i) {
        continue;
      }
      const auto pj = mesh.geometry.get_vec3(j);
      const auto d = pi - pj;
      sum += d.dot(d);
    }
  }
  return sum;
}

TEST_CASE("Curvature and flow kernels produce finite values") {
  igneous::test_support::configure_deterministic_test_env();
  using Sig = igneous::core::Euclidean3D;
  auto mesh = igneous::test_support::make_surface_grid(20);

  std::vector<float> H;
  std::vector<float> K;
  igneous::ops::CurvatureWorkspace<Sig, igneous::data::TriangleTopology> curvature_ws;
  igneous::ops::FlowWorkspace<Sig, igneous::data::TriangleTopology> flow_ws;

  igneous::ops::compute_curvature_measures(mesh, H, K, curvature_ws);

  CHECK(H.size() == mesh.geometry.num_points());
  CHECK(K.size() == mesh.geometry.num_points());

  for (float v : H) {
    CHECK(std::isfinite(v));
  }
  for (float v : K) {
    CHECK(std::isfinite(v));
  }

  const auto p_before = mesh.geometry.get_vec3(0);
  igneous::ops::integrate_mean_curvature_flow(mesh, 0.01f, flow_ws);
  const auto p_after = mesh.geometry.get_vec3(0);

  CHECK(std::isfinite(p_after.x));
  CHECK(std::isfinite(p_after.y));
  CHECK(std::isfinite(p_after.z));
  CHECK((p_before.x != doctest::Approx(p_after.x) ||
         p_before.y != doctest::Approx(p_after.y) ||
         p_before.z != doctest::Approx(p_after.z)));
}

TEST_CASE("Curvature is translation invariant") {
  igneous::test_support::configure_deterministic_test_env();
  using Sig = igneous::core::Euclidean3D;
  auto mesh_a = igneous::test_support::make_surface_grid(18);
  auto mesh_b = mesh_a;

  const igneous::core::Vec3 offset{3.0f, -2.0f, 1.25f};
  for (size_t i = 0; i < mesh_b.geometry.num_points(); ++i) {
    mesh_b.geometry.set_vec3(i, mesh_b.geometry.get_vec3(i) + offset);
  }

  std::vector<float> H_a, K_a, H_b, K_b;
  igneous::ops::CurvatureWorkspace<Sig, igneous::data::TriangleTopology> ws_a;
  igneous::ops::CurvatureWorkspace<Sig, igneous::data::TriangleTopology> ws_b;

  igneous::ops::compute_curvature_measures(mesh_a, H_a, K_a, ws_a);
  igneous::ops::compute_curvature_measures(mesh_b, H_b, K_b, ws_b);

  REQUIRE(H_a.size() == H_b.size());
  REQUIRE(K_a.size() == K_b.size());
  for (size_t i = 0; i < H_a.size(); ++i) {
    CHECK(H_a[i] == doctest::Approx(H_b[i]).epsilon(5e-3f));
    CHECK(K_a[i] == doctest::Approx(K_b[i]).epsilon(5e-3f));
  }
}

TEST_CASE("Mean curvature flow dt=0 is a no-op and dt>0 smooths roughness proxy") {
  igneous::test_support::configure_deterministic_test_env();
  using Sig = igneous::core::Euclidean3D;
  auto mesh = igneous::test_support::make_surface_grid(24);
  auto mesh_zero = mesh;
  igneous::ops::FlowWorkspace<Sig, igneous::data::TriangleTopology> ws;

  igneous::ops::integrate_mean_curvature_flow(mesh_zero, 0.0f, ws);
  for (size_t i = 0; i < mesh.geometry.num_points(); ++i) {
    const auto p = mesh.geometry.get_vec3(i);
    const auto q = mesh_zero.geometry.get_vec3(i);
    CHECK(p.x == doctest::Approx(q.x).epsilon(igneous::test_support::kTolTight));
    CHECK(p.y == doctest::Approx(q.y).epsilon(igneous::test_support::kTolTight));
    CHECK(p.z == doctest::Approx(q.z).epsilon(igneous::test_support::kTolTight));
  }

  const float rough_before = roughness_proxy(mesh);
  igneous::ops::integrate_mean_curvature_flow(mesh, 0.01f, ws);
  const float rough_after = roughness_proxy(mesh);
  CHECK(rough_after <= rough_before * 1.001f);
}
