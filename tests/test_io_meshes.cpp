#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <filesystem>
#include <igneous/data/space.hpp>
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

TEST_CASE("OBJ import is load-only and structures build explicitly") {
  const std::string path = bunny_path();
  REQUIRE(!path.empty());

  igneous::data::Space<igneous::data::DiscreteExteriorCalculus> surface_mesh;
  igneous::io::load_obj(surface_mesh, path);
  CHECK(surface_mesh.is_valid());
  CHECK(surface_mesh.num_points() > 0);
  CHECK(surface_mesh.structure.num_faces() > 0);
  CHECK(surface_mesh.structure.vertex_face_offsets.empty());
  surface_mesh.structure.build({surface_mesh.num_points(), true});
  CHECK(surface_mesh.structure.vertex_face_offsets.size() ==
        surface_mesh.num_points() + 1);

  igneous::data::Space<igneous::data::DiffusionGeometry> point_mesh;
  igneous::io::load_obj(point_mesh, path);
  CHECK(point_mesh.is_valid());
  CHECK(point_mesh.num_points() == surface_mesh.num_points());
  CHECK(point_mesh.structure.markov_row_offsets.empty());

  igneous::data::Space<igneous::data::DiffusionGeometry> diffusion_mesh;
  igneous::io::load_obj(diffusion_mesh, path);
  CHECK(diffusion_mesh.is_valid());
  CHECK(diffusion_mesh.structure.markov_row_offsets.empty());
  diffusion_mesh.structure.build(
      {diffusion_mesh.x_span(), diffusion_mesh.y_span(), diffusion_mesh.z_span(), 32});
  CHECK(diffusion_mesh.structure.markov_row_offsets.size() ==
        diffusion_mesh.num_points() + 1);
  CHECK(diffusion_mesh.structure.markov_values.size() > 0);
  CHECK(diffusion_mesh.structure.mu.sum() == doctest::Approx(1.0f).epsilon(1e-3f));
}
