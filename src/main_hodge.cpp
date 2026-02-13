#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

// Engine Headers
#include <igneous/core/algebra.hpp>
#include <igneous/data/mesh.hpp>
#include <igneous/io/exporter.hpp>
#include <igneous/io/importer.hpp>
#include <igneous/ops/geometry.hpp>
#include <igneous/ops/hodge.hpp>
#include <igneous/ops/spectral.hpp>
#include <igneous/ops/transform.hpp>

using namespace igneous;
using DiffusionMesh = data::Mesh<core::Euclidean3D, data::DiffusionTopology>;

// ==============================================================================
// 1. PROCEDURAL GENERATION (TORUS)
// ==============================================================================
void generate_torus(DiffusionMesh &mesh, size_t n_points, float R, float r) {
  mesh.geometry.clear();
  mesh.geometry.reserve(n_points, 0, 0); // Fixed: reserve takes 1 arg

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(0.0f, 6.283185f);

  for (size_t i = 0; i < n_points; ++i) {
    float u = dist(gen);
    float v = dist(gen);

    // Torus parameterization
    float x = (R + r * std::cos(v)) * std::cos(u);
    float y = (R + r * std::cos(v)) * std::sin(u);
    float z = r * std::sin(v);

    mesh.geometry.push_point({x, y, z});
  }
  std::cout << "[Gen] Generated Torus with " << n_points << " points.\n";
}

// ==============================================================================
// 2. HELPER: VECTOR FIELD RECONSTRUCTION
// ==============================================================================
std::vector<core::Vec3> reconstruct_vector_field(const DiffusionMesh &mesh,
                                                 const Eigen::VectorXf &coeffs,
                                                 float bandwidth) {
  size_t n_verts = mesh.geometry.num_points();
  int n_basis = mesh.topology.eigen_basis.cols();

  // 1. Prepare Coordinate Functions
  std::vector<Eigen::VectorXf> coords(3);
  for (int d = 0; d < 3; ++d) {
    coords[d].resize(n_verts);
    for (size_t i = 0; i < n_verts; ++i) {
      if (d == 0)
        coords[0][i] = mesh.geometry.get_vec3(i).x;
      if (d == 1)
        coords[1][i] = mesh.geometry.get_vec3(i).y;
      if (d == 2)
        coords[2][i] = mesh.geometry.get_vec3(i).z;
    }
  }

  // 2. Precompute Metric Tensor Components Gamma(x_a, x_b)
  Eigen::VectorXf gamma[3][3];
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      gamma[a][b] = ops::carre_du_champ(mesh, coords[a], coords[b], bandwidth);
    }
  }

  // 3. Accumulate Field
  std::vector<core::Vec3> field(n_verts, {0, 0, 0});
  const auto &U = mesh.topology.eigen_basis;

  for (int k = 0; k < n_basis; ++k) {
    for (int a = 0; a < 3; ++a) {
      float c = coeffs(k * 3 + a);
      if (std::abs(c) < 1e-6)
        continue;

      for (int b = 0; b < 3; ++b) {
        for (size_t i = 0; i < n_verts; ++i) {
          float val = c * U(i, k) * gamma[a][b](i);
          if (b == 0)
            field[i].x += val;
          if (b == 1)
            field[i].y += val;
          if (b == 2)
            field[i].z += val;
        }
      }
    }
  }
  return field;
}

// Simple PLY exporter for Vector Fields
void export_vector_field(const std::string &filename, const DiffusionMesh &mesh,
                         const std::vector<core::Vec3> &vectors) {
  std::ofstream file(filename);
  size_t n = mesh.geometry.num_points();
  file << "ply\nformat ascii 1.0\n";
  file << "element vertex " << n << "\n";
  file << "property float x\nproperty float y\nproperty float z\n";
  // Used 'n' properties so MeshLab renders them as Normals automatically
  file << "property float nx\nproperty float ny\nproperty float nz\n";
  file << "end_header\n";

  for (size_t i = 0; i < n; ++i) {
    auto p = mesh.geometry.get_vec3(i);
    auto v = vectors[i];
    file << p.x << " " << p.y << " " << p.z << " " << v.x << " " << v.y << " "
         << v.z << "\n";
  }
}

// ==============================================================================
// 3. MAIN
// ==============================================================================
int main() {
  std::cout << "\n[Hodge] Initializing Cohomology Solver...\n";
  std::filesystem::create_directory("output_hodge");

  // 1. Generate Data
  DiffusionMesh mesh;
  generate_torus(mesh, 4000, 2.0f, 0.8f);

  // 2. Build Topology
  float bandwidth = 0.05f;
  std::cout << "[Step] Building Connectivity (t=" << bandwidth << ")... \n";
  mesh.topology.build({
      std::span{mesh.geometry.packed_data}, bandwidth,
      32 // k-neighbors
  });

  // 3. Compute Spectral Basis (Function Space)
  // TODO: Adjust this?
  int n_basis = 64;
  ops::compute_eigenbasis(mesh, n_basis);

  // 4. Compute Operator Matrices
  // G: Mass Matrix (Metric) for 1-forms
  std::cout << "[Step] Computing 1-Form Mass Matrix...\n";
  auto G = ops::compute_1form_gram_matrix(mesh, bandwidth);

  // D_weak: Weak Exterior Derivative (Functions -> 1-Forms)
  std::cout << "[Step] Computing Weak Exterior Derivative...\n";
  auto D_weak = ops::compute_weak_exterior_derivative(mesh, bandwidth);

  // E_up: Curl Energy Matrix (1-Forms -> 2-Forms -> Energy)
  std::cout << "[Step] Computing Curl Energy Matrix...\n";
  auto E_up = ops::compute_curl_energy_matrix(mesh, bandwidth);

  // 5. Construct & Solve Hodge Laplacian
  // L = L_down + L_up = (D * D^T) + E_up
  std::cout << "[Step] Constructing Full Hodge Laplacian...\n";
  auto Laplacian = ops::compute_hodge_laplacian_matrix(D_weak, E_up);

  std::cout << "[Step] Solving Generalized Eigenproblem...\n";
  auto [evals, evecs] = ops::compute_hodge_spectrum(Laplacian, G);

  std::cout << "[Step] Generating Circular Coordinates...\n";

  // Take Mode 0 (the first loop)
  auto theta_0 = ops::compute_circular_coordinates(mesh, evecs.col(0));
  std::vector<double> field_0(theta_0.data(), theta_0.data() + theta_0.size());
  io::export_ply_solid(mesh, field_0, "output_hodge/torus_angle_0.ply", 0.01);

  // Take Mode 1 (the second loop)
  auto theta_1 = ops::compute_circular_coordinates(mesh, evecs.col(1));
  std::vector<double> field_1(theta_1.data(), theta_1.data() + theta_1.size());
  io::export_ply_solid(mesh, field_1, "output_hodge/torus_angle_1.ply", 0.01);

  // 6. Analyze Results
  std::cout << "\n-------------------------------------------------\n";
  std::cout << " HODGE SPECTRUM (First 12 Eigenvalues)\n";
  std::cout << "-------------------------------------------------\n";
  for (int i = 0; i < 12 && i < evals.size(); ++i) {
    std::cout << "Mode " << i << ": lambda = " << evals[i] << "\n";
  }

  // 7. Export Harmonic Forms
  std::cout << "\n[Step] Reconstructing and Exporting Harmonic Fields...\n";

  // We export the first 3 modes.
  for (int i = 0; i < 3; ++i) {
    std::cout << "   - Reconstructing Mode " << i << "...\n";

    // Get coefficients (eigenvector column)
    Eigen::VectorXf coeffs = evecs.col(i);

    // Convert to vectors in R3
    auto vector_field = reconstruct_vector_field(mesh, coeffs, bandwidth);

    // Export
    std::string fname = std::format("output_hodge/harmonic_form_{}.ply", i);
    export_vector_field(fname, mesh, vector_field);
  }

  std::cout << "Done! Open .ply files in MeshLab/ParaView/Python to see the "
               "vectors.\n";
  return 0;
}