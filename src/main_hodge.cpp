#include <Eigen/Dense>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include <igneous/core/algebra.hpp>
#include <igneous/data/mesh.hpp>
#include <igneous/io/exporter.hpp>
#include <igneous/ops/geometry.hpp>
#include <igneous/ops/hodge.hpp>
#include <igneous/ops/spectral.hpp>

using namespace igneous;
using DiffusionMesh = data::Mesh<core::Euclidean3D, data::DiffusionTopology>;

static void generate_torus(DiffusionMesh &mesh, size_t n_points, float R,
                           float r) {
  mesh.clear();
  mesh.geometry.reserve(n_points);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(0.0f, 6.283185f);

  for (size_t i = 0; i < n_points; ++i) {
    const float u = dist(gen);
    const float v = dist(gen);

    const float x = (R + r * std::cos(v)) * std::cos(u);
    const float y = (R + r * std::cos(v)) * std::sin(u);
    const float z = r * std::sin(v);

    mesh.geometry.push_point({x, y, z});
  }
}

static std::vector<core::Vec3>
reconstruct_vector_field(const DiffusionMesh &mesh,
                         const Eigen::VectorXf &coeffs, float bandwidth) {
  const size_t n_verts = mesh.geometry.num_points();
  const int n_basis = mesh.topology.eigen_basis.cols();

  std::array<Eigen::VectorXf, 3> coords;
  ops::fill_coordinate_vectors(mesh, coords);

  std::array<std::array<Eigen::VectorXf, 3>, 3> gamma{};
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      gamma[a][b].resize(static_cast<int>(n_verts));
      ops::carre_du_champ(mesh, coords[a], coords[b], bandwidth, gamma[a][b]);
    }
  }

  std::vector<core::Vec3> field(n_verts, {0.0f, 0.0f, 0.0f});
  const auto &U = mesh.topology.eigen_basis;

  for (int k = 0; k < n_basis; ++k) {
    for (int a = 0; a < 3; ++a) {
      const float c = coeffs(k * 3 + a);
      if (std::abs(c) < 1e-7f) {
        continue;
      }

      for (int b = 0; b < 3; ++b) {
        for (size_t i = 0; i < n_verts; ++i) {
          const float val = c * U(static_cast<int>(i), k) * gamma[a][b][static_cast<int>(i)];
          if (b == 0)
            field[i].x += val;
          else if (b == 1)
            field[i].y += val;
          else
            field[i].z += val;
        }
      }
    }
  }

  return field;
}

static void export_vector_field(const std::string &filename,
                                const DiffusionMesh &mesh,
                                const std::vector<core::Vec3> &vectors) {
  std::ofstream file(filename);
  const size_t n = mesh.geometry.num_points();

  file << "ply\nformat ascii 1.0\n";
  file << "element vertex " << n << "\n";
  file << "property float x\nproperty float y\nproperty float z\n";
  file << "property float nx\nproperty float ny\nproperty float nz\n";
  file << "end_header\n";

  for (size_t i = 0; i < n; ++i) {
    const auto p = mesh.geometry.get_vec3(i);
    const auto v = vectors[i];
    file << p.x << " " << p.y << " " << p.z << " " << v.x << " " << v.y
         << " " << v.z << "\n";
  }
}

int main() {
  const bool bench_mode = std::getenv("IGNEOUS_BENCH_MODE") != nullptr;
  if (!bench_mode) {
    std::filesystem::create_directory("output_hodge");
  }

  DiffusionMesh mesh;
  generate_torus(mesh, 4000, 2.0f, 0.8f);

  const float bandwidth = 0.05f;
  mesh.topology.build({mesh.geometry.x_span(), mesh.geometry.y_span(),
                       mesh.geometry.z_span(), bandwidth, 32});

  const int n_basis = 64;
  ops::compute_eigenbasis(mesh, n_basis);

  ops::GeometryWorkspace<DiffusionMesh> geom_ws;
  const auto G = ops::compute_1form_gram_matrix(mesh, bandwidth, geom_ws);

  ops::HodgeWorkspace<DiffusionMesh> hodge_ws;
  const auto D_weak = ops::compute_weak_exterior_derivative(mesh, bandwidth, hodge_ws);
  const auto E_up = ops::compute_curl_energy_matrix(mesh, bandwidth, hodge_ws);

  const auto laplacian = ops::compute_hodge_laplacian_matrix(D_weak, E_up);
  auto [evals, evecs] = ops::compute_hodge_spectrum(laplacian, G);

  const auto theta_0 = ops::compute_circular_coordinates(mesh, evecs.col(0), bandwidth);
  const auto theta_1 = ops::compute_circular_coordinates(mesh, evecs.col(1), bandwidth);

  if (!bench_mode) {
    std::vector<float> field_0(static_cast<size_t>(theta_0.size()));
    std::vector<float> field_1(static_cast<size_t>(theta_1.size()));
    for (int i = 0; i < theta_0.size(); ++i) {
      field_0[static_cast<size_t>(i)] = theta_0[i];
      field_1[static_cast<size_t>(i)] = theta_1[i];
    }

    io::export_ply_solid(mesh, field_0, "output_hodge/torus_angle_0.ply", 0.01);
    io::export_ply_solid(mesh, field_1, "output_hodge/torus_angle_1.ply", 0.01);
  }

  std::cout << "HODGE SPECTRUM (first 12 modes)\n";
  for (int i = 0; i < 12 && i < evals.size(); ++i) {
    std::cout << "Mode " << i << ": lambda = " << evals[i] << "\n";
  }

  if (!bench_mode) {
    for (int i = 0; i < 3; ++i) {
      const Eigen::VectorXf coeffs = evecs.col(i);
      const auto vector_field = reconstruct_vector_field(mesh, coeffs, bandwidth);
      const std::string fname = std::format("output_hodge/harmonic_form_{}.ply", i);
      export_vector_field(fname, mesh, vector_field);
    }
  }

  return 0;
}
