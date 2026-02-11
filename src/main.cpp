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

  DODLoader<Sig>::load_obj(argv[1], geometry, topology);

  // --- CRITICAL FIX ---
  // Bring mesh to 0,0,0 and scale to [-1, 1].
  // This fixes float precision issues and makes dt=0.01 meaningful.
  normalize_mesh(geometry);

  // Simulation Settings
  int total_frames = 1000;

  // Now that scale is normalized, dt=0.01 is conservative, dt=0.1 is fast.
  double dt = 0.5;

  std::cout << "Starting Mean Curvature Flow simulation...\n";

  for (int frame = 0; frame < total_frames; ++frame) {
    // (Rest of your loop remains the same)
    auto [H, K] = compute_curvature_measures(geometry, topology);

    std::string filename =
        std::format("output/frame_{:03}.obj", frame); // C++20 formatting
    save_colored_obj(filename, geometry, topology, H, 2.0);

    integrate_mean_curvature_flow(geometry, topology, dt);
    std::cout << "  [Frame " << frame << "] Smoothing complete.\n";
  }
  return 0;
}