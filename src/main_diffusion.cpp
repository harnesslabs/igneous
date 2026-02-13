#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <iostream>
#include <vector>

#include <igneous/core/algebra.hpp>
#include <igneous/data/mesh.hpp>
#include <igneous/io/exporter.hpp>
#include <igneous/io/importer.hpp>
#include <igneous/ops/transform.hpp>

using namespace igneous;
using DiffusionMesh = data::Mesh<core::Euclidean3D, data::DiffusionTopology>;

static std::vector<float> to_std_vector(const Eigen::VectorXf &v) {
  std::vector<float> out(static_cast<size_t>(v.size()));
  for (int i = 0; i < v.size(); ++i) {
    out[static_cast<size_t>(i)] = v[i];
  }
  return out;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cout << "Usage: ./igneous-diffusion <obj_file>\n";
    return 1;
  }

  const bool bench_mode = std::getenv("IGNEOUS_BENCH_MODE") != nullptr;

  const std::string input_path = argv[1];
  const std::string output_dir = "output_diffusion";
  if (!bench_mode) {
    std::filesystem::create_directory(output_dir);
  }

  DiffusionMesh mesh;
  io::load_obj(mesh, input_path);
  ops::normalize(mesh);

  mesh.topology.build({mesh.geometry.x_span(), mesh.geometry.y_span(),
                       mesh.geometry.z_span(), 0.005f, 32});

  const auto &P = mesh.topology.P;
  std::cout << "Markov Chain P: " << P.rows() << "x" << P.cols() << " ("
            << P.nonZeros() << " nnz)\n";

  auto density_field = to_std_vector(mesh.topology.mu);
  if (!bench_mode) {
    io::export_ply_solid(mesh, density_field, output_dir + "/00_density.ply", 0.01);
  }

  const size_t n_verts = mesh.geometry.num_points();
  Eigen::VectorXf u = Eigen::VectorXf::Zero(static_cast<int>(n_verts));

  int max_y_idx = 0;
  float max_y = -1e9f;
  for (size_t i = 0; i < n_verts; ++i) {
    const auto p = mesh.geometry.get_vec3(i);
    if (p.y > max_y) {
      max_y = p.y;
      max_y_idx = static_cast<int>(i);
    }
  }
  u[max_y_idx] = 1000.0f;

  if (!bench_mode) {
    io::export_ply_solid(mesh, to_std_vector(u), output_dir + "/heat_000.ply", 0.01);
  }

  const int steps = bench_mode ? 20 : 100;
  for (int t = 1; t <= steps; ++t) {
    u = P * u;

    if (!bench_mode && t % 5 == 0) {
      const std::string fname = std::format("{}/heat_{:03}.ply", output_dir, t);
      io::export_ply_solid(mesh, to_std_vector(u), fname, 0.01);
    }
  }

  return 0;
}
