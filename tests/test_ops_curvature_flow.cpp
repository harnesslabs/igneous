#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cmath>
#include <igneous/core/algebra.hpp>
#include <igneous/data/mesh.hpp>
#include <igneous/ops/curvature.hpp>
#include <igneous/ops/flow.hpp>

static igneous::data::Mesh<igneous::core::Euclidean3D, igneous::data::TriangleTopology>
make_grid(int side) {
  using Mesh = igneous::data::Mesh<igneous::core::Euclidean3D, igneous::data::TriangleTopology>;
  Mesh mesh;
  mesh.geometry.reserve(static_cast<size_t>(side * side));

  for (int y = 0; y < side; ++y) {
    for (int x = 0; x < side; ++x) {
      const float fx = static_cast<float>(x) / side;
      const float fy = static_cast<float>(y) / side;
      mesh.geometry.push_point({fx, fy, std::sin(fx) * std::cos(fy)});
    }
  }

  for (int y = 0; y < side - 1; ++y) {
    for (int x = 0; x < side - 1; ++x) {
      const uint32_t i0 = static_cast<uint32_t>(y * side + x);
      const uint32_t i1 = static_cast<uint32_t>(y * side + x + 1);
      const uint32_t i2 = static_cast<uint32_t>((y + 1) * side + x);
      const uint32_t i3 = static_cast<uint32_t>((y + 1) * side + x + 1);

      mesh.topology.faces_to_vertices.insert(mesh.topology.faces_to_vertices.end(), {i0, i1, i2, i1, i3, i2});
    }
  }

  mesh.topology.build({mesh.geometry.num_points(), true});
  return mesh;
}

TEST_CASE("Curvature and flow kernels produce finite values") {
  using Sig = igneous::core::Euclidean3D;
  auto mesh = make_grid(20);

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
