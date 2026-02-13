#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <filesystem>
#include <igneous/core/algebra.hpp>
#include <igneous/data/mesh.hpp>
#include <igneous/io/importer.hpp>

static std::string bunny_path() {
  if (std::filesystem::exists("assets/bunny.obj")) {
    return "assets/bunny.obj";
  }
  if (std::filesystem::exists("../assets/bunny.obj")) {
    return "../assets/bunny.obj";
  }
  return "";
}

TEST_CASE("OBJ import works for triangle, point, and diffusion topologies") {
  const std::string path = bunny_path();
  REQUIRE(!path.empty());

  using Sig = igneous::core::Euclidean3D;

  igneous::data::Mesh<Sig, igneous::data::TriangleTopology> surface_mesh;
  igneous::io::load_obj(surface_mesh, path);
  CHECK(surface_mesh.is_valid());
  CHECK(surface_mesh.geometry.num_points() > 0);
  CHECK(surface_mesh.topology.num_faces() > 0);

  igneous::data::Mesh<Sig, igneous::data::PointTopology> point_mesh;
  igneous::io::load_obj(point_mesh, path);
  CHECK(point_mesh.is_valid());
  CHECK(point_mesh.geometry.num_points() == surface_mesh.geometry.num_points());

  igneous::data::Mesh<Sig, igneous::data::DiffusionTopology> diffusion_mesh;
  igneous::io::load_obj(diffusion_mesh, path);
  CHECK(diffusion_mesh.is_valid());
  CHECK(diffusion_mesh.topology.P.rows() == static_cast<int>(diffusion_mesh.geometry.num_points()));
  CHECK(diffusion_mesh.topology.P.nonZeros() > 0);
  CHECK(diffusion_mesh.topology.mu.sum() == doctest::Approx(1.0f).epsilon(1e-3f));
}
