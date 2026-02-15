#include <cstdlib>
#include <format>
#include <igneous/data/space.hpp>
#include <igneous/io/exporter.hpp>
#include <igneous/io/importer.hpp>
#include <igneous/ops/dec/curvature.hpp>
#include <igneous/ops/dec/flow.hpp>
#include <igneous/ops/transform.hpp>

using namespace igneous;
using Space = data::Space<data::DiscreteExteriorCalculus>;

int main(int argc, char **argv) {
  if (argc < 2) {
    return 1;
  }

  const bool bench_mode = std::getenv("IGNEOUS_BENCH_MODE") != nullptr;
  const int frame_count = bench_mode ? 10 : 1000;

  Space space;
  io::load_obj(space, argv[1]);
  space.structure.build({space.num_points(), true});
  ops::normalize(space);

  std::vector<float> H;
  std::vector<float> K;
  ops::dec::CurvatureWorkspace<data::DiscreteExteriorCalculus> curvature_ws;
  ops::dec::FlowWorkspace<data::DiscreteExteriorCalculus> flow_ws;

  for (int frame = 0; frame < frame_count; ++frame) {
    ops::dec::compute_curvature_measures(space, H, K, curvature_ws);

    if (!bench_mode) {
      const std::string filename = std::format("output/frame_{:03}.obj", frame);
      io::export_heatmap(space, H, filename, 2.0);
    }

    ops::dec::integrate_mean_curvature_flow(space, 0.5f, flow_ws);
  }

  return 0;
}
