#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <filesystem>
#include <format>
#include <iostream>
#include <vector>

// Core Engine Headers
#include <igneous/core/algebra.hpp>
#include <igneous/data/mesh.hpp>
#include <igneous/io/exporter.hpp>
#include <igneous/io/importer.hpp>
#include <igneous/ops/transform.hpp> // for normalize

using namespace igneous;

// 1. Define our Diffusion Mesh Type
// Uses Euclidean3D for points, but DiffusionTopology for connectivity
using DiffusionMesh = data::Mesh<core::Euclidean3D, data::DiffusionTopology>;

// Helper: Convert Eigen Vector to std::vector for the exporter
std::vector<double> to_std_vector(const Eigen::VectorXf &v) {
  std::vector<double> out(v.size());
  for (int i = 0; i < v.size(); ++i) {
    out[i] = static_cast<double>(v[i]);
  }
  return out;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cout << "Usage: ./main_diffusion <obj_file>\n";
    return 1;
  }

  std::string input_path = argv[1];
  std::string output_dir = "output_diffusion";
  std::filesystem::create_directory(output_dir);

  // ==============================================================================
  // 1. IMPORT & SETUP
  // ==============================================================================
  std::cout << "\n[1] Initializing Diffusion Engine...\n";
  DiffusionMesh mesh;

  // Load OBJ (Topology build happens automatically here via Importer)
  // This triggers k-NN search and Sparse Matrix construction.
  io::load_obj(mesh, input_path);

  // Normalize geometry to [-1, 1] so bandwidth params are consistent
  ops::normalize(mesh);

  // Re-build connectivity AFTER normalize (since distances changed!)
  // We explicitly call this to ensure kernel bandwidth (t=0.01) makes sense
  // relative to the new unit scale.
  std::cout << "[Step] Re-building connectivity on normalized geometry...\n";
  mesh.topology.build({
      std::span{mesh.geometry.packed_data}, // coords
      0.005f,                               // bandwidth (t)
      32                                    // k_neighbors
  });

  // Report Stats
  const auto &P = mesh.topology.P; // The Markov Chain
  std::cout << "      Markov Chain P: " << P.rows() << "x" << P.cols()
            << " with " << P.nonZeros() << " stored entries.\n";
  std::cout << "      Sparsity: "
            << (float)P.nonZeros() / (P.rows() * P.cols()) * 100.0f << "%\n";

  // ==============================================================================
  // 2. VISUALIZE MEASURE (DENSITY)
  // ==============================================================================
  std::cout << "\n[2] Visualizing Measure (Local Density)...\n";

  // The measure 'mu' represents the sampling density.
  // Areas with more points packed together will have different
  // weight.
  auto density_field = to_std_vector(mesh.topology.mu);

  // Export as solid PLY (points -> tetrahedrons)
  io::export_ply_solid(mesh, density_field, output_dir + "/00_density.ply",
                       0.01);

  // ==============================================================================
  // 3. HEAT DIFFUSION SIMULATION
  // ==============================================================================
  std::cout << "\n[3] Running Heat Diffusion (Markov Process)...\n";

  size_t n_verts = mesh.geometry.num_points();
  Eigen::VectorXf u = Eigen::VectorXf::Zero(n_verts);

  // A. Initial Condition: Heat Source at the "Ear" (Max Y)
  int max_y_idx = 0;
  float max_y = -1e9;
  for (size_t i = 0; i < n_verts; ++i) {
    auto p = mesh.geometry.get_vec3(i);
    if (p.y > max_y) {
      max_y = p.y;
      max_y_idx = i;
    }
  }
  u[max_y_idx] = 1000.0f; // Spike of heat
  std::cout << "      Heat source placed at vertex " << max_y_idx << "\n";

  // Export Initial State
  io::export_ply_solid(mesh, to_std_vector(u), output_dir + "/heat_000.ply",
                       0.01);

  // B. Time Integration Loop
  // The heat equation is solved by repeatedly applying the Markov Chain P.
  // u_{t+1} = P * u_t
  int steps = 100;
  for (int t = 1; t <= steps; ++t) {
    // --- THE MAGIC HAPPENS HERE ---
    // Sparse Matrix-Vector Multiplication replaces Mesh Traversal
    u = P * u;

    if (t % 5 == 0) {
      std::string fname = std::format("{}/heat_{:03}.ply", output_dir, t);
      io::export_ply_solid(mesh, to_std_vector(u), fname, 0.01);
      std::cout << "      Step " << t << " diffused.\n";
    }
  }

  std::cout << "\nDone! Output saved to " << output_dir << "/\n";
  return 0;
}