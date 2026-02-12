#include <igneous/core/algebra.hpp>
#include <igneous/data/mesh.hpp>
#include <igneous/io/exporter.hpp>
#include <igneous/io/importer.hpp>
#include <igneous/ops/transform.hpp>
#include <iostream>
#include <vector>

using namespace igneous;

// 1. Define specific mesh types
using PointCloud = data::Mesh<core::Euclidean3D, data::PointTopology>;
using SurfaceMesh = data::Mesh<core::Euclidean3D, data::TriangleTopology>;

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cout << "Usage: ./test_points <obj_file>\n";
    return 1;
  }

  std::string input_path = argv[1];

  // =========================================================
  // TEST 1: Point Cloud Workflow
  // =========================================================
  std::cout << "--- Testing Point Cloud Workflow ---\n";

  PointCloud pc;

  // A. Import (Should ignore 'f' lines)
  io::load_obj(pc, input_path);

  // B. Process (Normalize works on geometry only, so this is safe)
  ops::normalize(pc);

  // C. Visualize (Create dummy height field since we can't do curvature)
  size_t n_verts = pc.geometry.num_points();
  std::vector<double> height_field(n_verts);
  for (size_t i = 0; i < n_verts; ++i) {
    height_field[i] = pc.geometry.get_vec3(i).y; // Color by Y-height
  }

  io::export_ply_solid(pc, height_field, "output_solid_cloud.ply", 0.01);

  // =========================================================
  // TEST 2: Surface Workflow (Control Group)
  // =========================================================
  std::cout << "\n--- Testing Surface Workflow ---\n";

  SurfaceMesh surf;
  io::load_obj(surf, input_path); // Should load faces
  ops::normalize(surf);

  // Dummy field
  std::vector<double> surf_field(surf.geometry.num_points(), 0.0);
  io::export_heatmap(surf, surf_field, "output_surface.obj");

  return 0;
}