#include <igneous/algebra.hpp>
#include <igneous/curvature.hpp>
#include <igneous/geometry.hpp>
#include <igneous/mesh_loader_dod.hpp>
#include <igneous/processing.hpp>
#include <igneous/topology.hpp>

using namespace igneous;

// --- CONFIGURATION ---
// We use Euclidean3D for standard curvature logic
using Sig = Euclidean3D;

int main(int argc, char **argv) {
  if (argc < 2)
    return 1;

  GeometryBuffer<float, Sig> geometry;
  TopologyBuffer topology;

  // 1. Load Data
  DODLoader<Sig>::load_obj(argv[1], geometry, topology);

  // 2. The GA Kernel
  auto [field_H, field_K] = compute_curvature_measures(geometry, topology);

  // 3. Export with Statistical Normalization
  // Sigma Clip = 2.0 means we cover 95% of the data range.
  // Outliers (ear tips) will be clamped to Red/Blue.
  save_colored_obj("out_mean.obj", geometry, topology, field_H, 2.0);
  save_colored_obj("out_gauss.obj", geometry, topology, field_K, 2.0);

  return 0;
}

// int main(int argc, char **argv) {
//   if (argc < 2) {
//     std::cout << "Usage: ./igneous_engine <input.obj>\n";
//     return 1;
//   }

//   // 1. Initialize Buffers
//   GeometryBuffer<float, Sig> geometry;
//   TopologyBuffer topology;

//   // 2. Load Data (DOD Mode)
//   //
//   DODLoader<Sig>::load_obj(argv[1], geometry, topology);

//   // 3. The Math Kernel (SIMD)
//   // Uses the new Optimized Outer Product (operator^)
//   auto curvature_field = compute_mean_curvature_flow(geometry, topology);

//   // 4. Visualize
//   save_colored_obj("output_curvature.obj", geometry, topology,
//   curvature_field);

//   return 0;
// }