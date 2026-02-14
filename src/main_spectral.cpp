#include <Eigen/Dense>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <iostream>
#include <vector>

#include <igneous/core/algebra.hpp>
#include <igneous/data/mesh.hpp>
#include <igneous/io/exporter.hpp>
#include <igneous/io/importer.hpp>
#include <igneous/ops/geometry.hpp>
#include <igneous/ops/spectral.hpp>
#include <igneous/ops/transform.hpp>

using namespace igneous;
using DiffusionMesh = data::Mesh<core::Euclidean3D, data::DiffusionTopology>;

int main(int argc, char **argv) {
  if (argc < 2) {
    return 1;
  }

  const bool bench_mode = std::getenv("IGNEOUS_BENCH_MODE") != nullptr;

  DiffusionMesh mesh;
  io::load_obj(mesh, argv[1]);
  ops::normalize(mesh);

  const float bandwidth = 0.005f;
  mesh.topology.build({mesh.geometry.x_span(), mesh.geometry.y_span(),
                       mesh.geometry.z_span(), 32});

  const int n_basis = 16;
  ops::compute_eigenbasis(mesh, n_basis);

  ops::GeometryWorkspace<DiffusionMesh> geometry_ws;
  const Eigen::MatrixXf G = ops::compute_1form_gram_matrix(mesh, bandwidth, geometry_ws);

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> solver(G);
  if (solver.info() == Eigen::Success) {
    const auto eigvals = solver.eigenvalues();
    const float min_sv = std::max(1e-12f, eigvals.minCoeff());
    const float max_sv = eigvals.maxCoeff();
    std::cout << "Gram cond approx: " << (max_sv / min_sv) << "\n";
  }

  if (!bench_mode) {
    const std::string out_dir = "output_spectral";
    std::filesystem::create_directory(out_dir);

    const int to_export =
        std::min(4, static_cast<int>(mesh.topology.eigen_basis.cols()));
    for (int i = 0; i < to_export; ++i) {
      const Eigen::VectorXf phi = mesh.topology.eigen_basis.col(i);
      std::vector<float> field(static_cast<size_t>(phi.size()));
      for (int j = 0; j < phi.size(); ++j) {
        field[static_cast<size_t>(j)] = phi[j];
      }

      const std::string name = std::format("{}/eigen_{}.ply", out_dir, i);
      io::export_ply_solid(mesh, field, name, 0.01);
    }
  }

  return 0;
}
