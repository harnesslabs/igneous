#include <igneous/algebra.hpp>
#include <igneous/curvature.hpp> // for compute_curvature
#include <igneous/flow.hpp>      // <--- NEW HEADER
#include <igneous/geometry.hpp>
#include <igneous/mesh_loader_dod.hpp>
#include <igneous/processing.hpp> // for save_colored_obj
#include <igneous/topology.hpp>
#include <iostream>
#include <string>

using namespace igneous;
using Sig = Euclidean3D;

int main(int argc, char **argv) {
  if (argc < 2)
    return 1;

  GeometryBuffer<float, Sig> geometry;
  TopologyBuffer topology;

  // 1. Load Initial Mesh
  DODLoader<Sig>::load_obj(argv[1], geometry, topology);

  // 2. Simulation Settings
  int total_frames = 50;
  double dt =
      0.5; // "Time step" - higher is faster smoothing, but can be unstable

  std::cout << "Starting Mean Curvature Flow simulation...\n";

  for (int frame = 0; frame < total_frames; ++frame) {

    // A. Analyze Geometry (Get Curvature for Color)
    // We do this BEFORE moving, so the color represents the forces about to act
    auto [H, K] = compute_curvature_measures(geometry, topology);

    // B. Export Frame
    // Filename: frame_000.obj, frame_001.obj...
    std::string filename = "frame_" + std::to_string(frame) + ".obj";
    if (frame < 10)
      filename = "frame_00" + std::to_string(frame) + ".obj";
    else if (frame < 100)
      filename = "frame_0" + std::to_string(frame) + ".obj";

    // Save with Heatmap (Sigma Clip 2.0 to see the flow clearly)
    save_colored_obj(filename, geometry, topology, H, 2.0);

    // C. Integrate Physics
    // This morphs the geometry for the next loop
    integrate_mean_curvature_flow(geometry, topology, dt);

    std::cout << "  [Frame " << frame << "] Smoothing complete.\n";
  }

  std::cout << "Done! Drag frame_*.obj into Blender or MeshLab to see the "
               "animation.\n";
  return 0;
}