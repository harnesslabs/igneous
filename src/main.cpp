#include <format>
#include <igneous/data/mesh.hpp>
#include <igneous/io/exporter.hpp>
#include <igneous/io/importer.hpp>
#include <igneous/ops/curvature.hpp>
#include <igneous/ops/flow.hpp>
#include <igneous/ops/transform.hpp>

using namespace igneous;
using Sig = igneous::core::Euclidean3D;
using Mesh = igneous::data::Mesh<Sig>;

int main(int argc, char **argv) {
  if (argc < 2)
    return 1;

  // 1. Setup
  Mesh mesh;

  // 2. Import System
  io::load_obj(mesh, argv[1]);

  // 3. Transform System (Pre-process)
  ops::normalize(mesh);

  std::cout << "Starting simulation on " << mesh.name << "...\n";

  // 4. The Loop
  for (int frame = 0; frame < 50; ++frame) {

    // Compute System
    // Note: You'll need to update curvature.hpp to take Mesh& input
    auto [H, K] = igneous::ops::compute_curvature_measures(mesh);

    // Export System
    std::string filename = std::format("frame_{:03}.obj", frame);
    io::export_heatmap(mesh, H, filename, 2.0);

    // Physics System
    igneous::ops::integrate_mean_curvature_flow(mesh, 0.05);
  }

  return 0;
}