#include <filesystem>
#include <igneous/data/mesh.hpp>
#include <igneous/io/exporter.hpp>
#include <igneous/io/importer.hpp>
#include <igneous/ops/geometry.hpp> // New
#include <igneous/ops/spectral.hpp> // New
#include <igneous/ops/transform.hpp>

using namespace igneous;
using DiffusionMesh = data::Mesh<core::Euclidean3D, data::DiffusionTopology>;

int main(int argc, char **argv) {
  if (argc < 2)
    return 1;
  std::string input_path = argv[1];

  DiffusionMesh mesh;
  io::load_obj(mesh, input_path);
  ops::normalize(mesh);

  // 1. Build Topology
  // Bandwidth 0.005 is critical here. Too small = noise. Too large = blur.
  float bandwidth = 0.005f;
  mesh.topology.build({std::span{mesh.geometry.packed_data}, bandwidth, 32});

  // 2. Compute Spectral Basis
  // We want the first 16 eigenfunctions
  int n_basis = 16;
  ops::compute_eigenbasis(mesh, n_basis);

  // 3. Compute 1-Form Metric
  // This is the "Mass Matrix" G
  Eigen::MatrixXf G = ops::compute_1form_gram_matrix(mesh, bandwidth);

  std::cout << "Gram Matrix Determinant: " << G.determinant() << "\n";
  std::cout << "Condition Number approx: " << G.maxCoeff() / G.minCoeff()
            << "\n";

  // 4. Visualize the Eigenfunctions
  // Phi_0 is constant (stationary distribution).
  // Phi_1 is the "Fiedler Vector" - cuts the mesh in half.
  // Phi_2, Phi_3 ... are higher frequency harmonics.
  std::string out_dir = "output_spectral";
  std::filesystem::create_directory(out_dir);

  for (int i = 0; i < 4; ++i) {
    Eigen::VectorXf phi = mesh.topology.eigen_basis.col(i);

    // Convert to std::vector for exporter
    std::vector<double> field(phi.data(), phi.data() + phi.size());

    std::string name = std::format("{}/eigen_{}.ply", out_dir, i);
    io::export_ply_solid(mesh, field, name, 0.01);
    std::cout << "Exported " << name << "\n";
  }

  return 0;
}