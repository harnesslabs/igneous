#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <igneous/ops/transform.hpp>

#include "support/synthetic_meshes.hpp"
#include "support/test_env.hpp"
#include "support/tolerances.hpp"

TEST_CASE("normalize centers and scales point clouds deterministically") {
  igneous::test_support::configure_deterministic_test_env();
  auto mesh = igneous::test_support::make_helix_cloud(200);

  igneous::ops::normalize(mesh);

  igneous::core::Vec3 min_p = {std::numeric_limits<float>::max(),
                               std::numeric_limits<float>::max(),
                               std::numeric_limits<float>::max()};
  igneous::core::Vec3 max_p = {std::numeric_limits<float>::lowest(),
                               std::numeric_limits<float>::lowest(),
                               std::numeric_limits<float>::lowest()};

  for (size_t i = 0; i < mesh.geometry.num_points(); ++i) {
    const auto p = mesh.geometry.get_vec3(i);
    min_p.x = std::min(min_p.x, p.x);
    min_p.y = std::min(min_p.y, p.y);
    min_p.z = std::min(min_p.z, p.z);
    max_p.x = std::max(max_p.x, p.x);
    max_p.y = std::max(max_p.y, p.y);
    max_p.z = std::max(max_p.z, p.z);
  }

  const auto center = (min_p + max_p) * 0.5f;
  CHECK(center.x == doctest::Approx(0.0f).epsilon(igneous::test_support::kTolTight));
  CHECK(center.y == doctest::Approx(0.0f).epsilon(igneous::test_support::kTolTight));
  CHECK(center.z == doctest::Approx(0.0f).epsilon(igneous::test_support::kTolTight));

  const auto dim = max_p - min_p;
  const float max_dim = std::max({dim.x, dim.y, dim.z});
  CHECK(max_dim == doctest::Approx(2.0f).epsilon(igneous::test_support::kTolMedium));

  std::vector<igneous::core::Vec3> once(mesh.geometry.num_points());
  for (size_t i = 0; i < mesh.geometry.num_points(); ++i) {
    once[i] = mesh.geometry.get_vec3(i);
  }

  igneous::ops::normalize(mesh);
  for (size_t i = 0; i < mesh.geometry.num_points(); ++i) {
    const auto p = mesh.geometry.get_vec3(i);
    CHECK(p.x == doctest::Approx(once[i].x).epsilon(igneous::test_support::kTolTight));
    CHECK(p.y == doctest::Approx(once[i].y).epsilon(igneous::test_support::kTolTight));
    CHECK(p.z == doctest::Approx(once[i].z).epsilon(igneous::test_support::kTolTight));
  }
}
