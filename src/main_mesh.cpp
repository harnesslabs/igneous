#include <cstdlib>
#include <format>
#include <igneous/data/mesh.hpp>
#include <igneous/io/exporter.hpp>
#include <igneous/io/importer.hpp>
#include <igneous/ops/curvature.hpp>
#include <igneous/ops/flow.hpp>
#include <igneous/ops/transform.hpp>

using namespace igneous;
using Sig = core::Euclidean3D;
using Mesh = data::Mesh<Sig, data::TriangleTopology>;

int main(int argc, char **argv) {
  if (argc < 2) {
    return 1;
  }

  const bool bench_mode = std::getenv("IGNEOUS_BENCH_MODE") != nullptr;
  const int frame_count = bench_mode ? 10 : 1000;

  Mesh mesh;
  io::load_obj(mesh, argv[1]);
  ops::normalize(mesh);

  std::vector<float> H;
  std::vector<float> K;
  ops::CurvatureWorkspace<Sig, data::TriangleTopology> curvature_ws;
  ops::FlowWorkspace<Sig, data::TriangleTopology> flow_ws;

  for (int frame = 0; frame < frame_count; ++frame) {
    ops::compute_curvature_measures(mesh, H, K, curvature_ws);

    if (!bench_mode) {
      const std::string filename = std::format("output/frame_{:03}.obj", frame);
      io::export_heatmap(mesh, H, filename, 2.0);
    }

    ops::integrate_mean_curvature_flow(mesh, 0.5f, flow_ws);
  }

  return 0;
}
