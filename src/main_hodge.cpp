#include <Eigen/Dense>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <igneous/core/algebra.hpp>
#include <igneous/data/mesh.hpp>
#include <igneous/io/exporter.hpp>
#include <igneous/ops/diffusion/geometry.hpp>
#include <igneous/ops/diffusion/hodge.hpp>
#include <igneous/ops/diffusion/spectral.hpp>

using namespace igneous;
using DiffusionMesh = data::Mesh<core::Euclidean3D, data::DiffusionTopology>;

struct Config {
  std::string input_csv;
  std::string output_dir = "output_hodge";
  size_t n_points = 1000;
  float major_radius = 2.0f;
  float minor_radius = 1.0f;
  uint32_t seed = 0;

  int n_basis = 50;
  int k_neighbors = 32;
  int knn_bandwidth = 8;
  float bandwidth_variability = -0.5f;
  float c = 0.0f;

  float circular_lambda = 1.0f;
  int circular_mode_0 = 0;
  int circular_mode_1 = 1;

  bool export_ply = true;
};

static void print_usage() {
  std::cout << "Usage: ./build/igneous-hodge [options]\n"
            << "  --input-csv <path>\n"
            << "  --output-dir <path>\n"
            << "  --n-points <int>\n"
            << "  --major-radius <float>\n"
            << "  --minor-radius <float>\n"
            << "  --seed <int>\n"
            << "  --n-basis <int>\n"
            << "  --k-neighbors <int>\n"
            << "  --knn-bandwidth <int>\n"
            << "  --bandwidth-variability <float>\n"
            << "  --c <float>\n"
            << "  --circular-lambda <float>\n"
            << "  --circular-mode-0 <int>\n"
            << "  --circular-mode-1 <int>\n"
            << "  --no-ply\n";
}

static bool parse_args(int argc, char **argv, Config &cfg) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_value = [&](const char *name) -> const char * {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << name << "\n";
        std::exit(1);
      }
      ++i;
      return argv[i];
    };

    if (arg == "--help" || arg == "-h") {
      print_usage();
      return false;
    }
    if (arg == "--input-csv") {
      cfg.input_csv = require_value("--input-csv");
      continue;
    }
    if (arg == "--output-dir") {
      cfg.output_dir = require_value("--output-dir");
      continue;
    }
    if (arg == "--n-points") {
      cfg.n_points = static_cast<size_t>(std::stoul(require_value("--n-points")));
      continue;
    }
    if (arg == "--major-radius") {
      cfg.major_radius = std::stof(require_value("--major-radius"));
      continue;
    }
    if (arg == "--minor-radius") {
      cfg.minor_radius = std::stof(require_value("--minor-radius"));
      continue;
    }
    if (arg == "--seed") {
      cfg.seed = static_cast<uint32_t>(std::stoul(require_value("--seed")));
      continue;
    }
    if (arg == "--n-basis") {
      cfg.n_basis = std::stoi(require_value("--n-basis"));
      continue;
    }
    if (arg == "--k-neighbors") {
      cfg.k_neighbors = std::stoi(require_value("--k-neighbors"));
      continue;
    }
    if (arg == "--knn-bandwidth") {
      cfg.knn_bandwidth = std::stoi(require_value("--knn-bandwidth"));
      continue;
    }
    if (arg == "--bandwidth-variability") {
      cfg.bandwidth_variability = std::stof(require_value("--bandwidth-variability"));
      continue;
    }
    if (arg == "--c") {
      cfg.c = std::stof(require_value("--c"));
      continue;
    }
    if (arg == "--circular-lambda") {
      cfg.circular_lambda = std::stof(require_value("--circular-lambda"));
      continue;
    }
    if (arg == "--circular-mode-0") {
      cfg.circular_mode_0 = std::stoi(require_value("--circular-mode-0"));
      continue;
    }
    if (arg == "--circular-mode-1") {
      cfg.circular_mode_1 = std::stoi(require_value("--circular-mode-1"));
      continue;
    }
    if (arg == "--no-ply") {
      cfg.export_ply = false;
      continue;
    }

    std::cerr << "Unknown argument: " << arg << "\n";
    print_usage();
    return false;
  }

  return true;
}

static void generate_torus(DiffusionMesh &mesh, size_t n_points, float major_radius,
                           float minor_radius, uint32_t seed) {
  mesh.clear();
  mesh.geometry.reserve(n_points);

  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(0.0f, 6.28318530718f);

  for (size_t i = 0; i < n_points; ++i) {
    const float u = dist(gen);
    const float v = dist(gen);
    const float x = (major_radius + minor_radius * std::cos(v)) * std::cos(u);
    const float y = (major_radius + minor_radius * std::cos(v)) * std::sin(u);
    const float z = minor_radius * std::sin(v);
    mesh.geometry.push_point({x, y, z});
  }
}

static bool load_point_cloud_csv(const std::string &filename, DiffusionMesh &mesh) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Failed to open input CSV: " << filename << "\n";
    return false;
  }

  mesh.clear();
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty()) {
      continue;
    }

    for (char &c : line) {
      if (c == ',') {
        c = ' ';
      }
    }

    std::istringstream iss(line);
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    if (!(iss >> x >> y >> z)) {
      continue;
    }
    mesh.geometry.push_point({x, y, z});
  }

  return mesh.geometry.num_points() > 0;
}

static void write_points_csv(const std::string &filename, const DiffusionMesh &mesh) {
  std::ofstream file(filename);
  file << "x,y,z\n";
  const size_t n = mesh.geometry.num_points();
  for (size_t i = 0; i < n; ++i) {
    const auto p = mesh.geometry.get_vec3(i);
    file << p.x << "," << p.y << "," << p.z << "\n";
  }
}

static void write_spectrum_csv(const std::string &filename,
                               const Eigen::VectorXf &evals) {
  std::ofstream file(filename);
  file << "mode,lambda\n";
  for (int i = 0; i < evals.size(); ++i) {
    file << i << "," << evals[i] << "\n";
  }
}

static void write_harmonic_coeffs_csv(const std::string &filename,
                                      const Eigen::MatrixXf &evecs,
                                      int n_forms) {
  std::ofstream file(filename);
  file << "coeff_index";
  for (int form = 0; form < n_forms; ++form) {
    file << ",form" << form;
  }
  file << "\n";

  for (int i = 0; i < evecs.rows(); ++i) {
    file << i;
    for (int form = 0; form < n_forms; ++form) {
      file << "," << evecs(i, form);
    }
    file << "\n";
  }
}

static std::vector<core::Vec3>
reconstruct_harmonic_ambient(const DiffusionMesh &mesh,
                             const Eigen::VectorXf &coeffs) {
  const size_t n_verts = mesh.geometry.num_points();
  const int n_basis = mesh.topology.eigen_basis.cols();
  const auto &U = mesh.topology.eigen_basis;

  std::array<Eigen::VectorXf, 3> data_coords;
  std::array<Eigen::VectorXf, 3> immersion_coords;
  ops::fill_data_coordinate_vectors(mesh, data_coords);
  ops::fill_coordinate_vectors(mesh, immersion_coords);

  std::array<std::array<Eigen::VectorXf, 3>, 3> gamma_data_imm{};
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      gamma_data_imm[a][b].resize(static_cast<int>(n_verts));
      ops::carre_du_champ(mesh, data_coords[a], immersion_coords[b], 0.0f,
                          gamma_data_imm[a][b]);
    }
  }

  Eigen::MatrixXf alpha_mat(n_basis, 3);
  for (int k = 0; k < n_basis; ++k) {
    alpha_mat(k, 0) = coeffs(k * 3 + 0);
    alpha_mat(k, 1) = coeffs(k * 3 + 1);
    alpha_mat(k, 2) = coeffs(k * 3 + 2);
  }
  const Eigen::MatrixXf q = U * alpha_mat; // [n x 3]

  std::vector<core::Vec3> field(n_verts, {0.0f, 0.0f, 0.0f});
  for (size_t i = 0; i < n_verts; ++i) {
    const int idx = static_cast<int>(i);
    const float vx = gamma_data_imm[0][0][idx] * q(idx, 0) +
                     gamma_data_imm[0][1][idx] * q(idx, 1) +
                     gamma_data_imm[0][2][idx] * q(idx, 2);
    const float vy = gamma_data_imm[1][0][idx] * q(idx, 0) +
                     gamma_data_imm[1][1][idx] * q(idx, 1) +
                     gamma_data_imm[1][2][idx] * q(idx, 2);
    const float vz = gamma_data_imm[2][0][idx] * q(idx, 0) +
                     gamma_data_imm[2][1][idx] * q(idx, 1) +
                     gamma_data_imm[2][2][idx] * q(idx, 2);
    field[i] = {vx, vy, vz};
  }

  return field;
}

static double orientation_score(const DiffusionMesh &mesh,
                                const std::vector<core::Vec3> &field) {
  const size_t n_verts = mesh.geometry.num_points();
  if (n_verts == 0) {
    return 0.0;
  }

  double accum = 0.0;
  for (size_t i = 0; i < n_verts; ++i) {
    const auto p = mesh.geometry.get_vec3(i);
    const auto v = field[i];
    accum += static_cast<double>(p.x) * static_cast<double>(v.x) +
             static_cast<double>(p.y) * static_cast<double>(v.y) +
             static_cast<double>(p.z) * static_cast<double>(v.z);
  }

  return accum / static_cast<double>(n_verts);
}

static bool canonicalize_form_sign(const DiffusionMesh &mesh,
                                   Eigen::Ref<Eigen::VectorXf> coeffs,
                                   std::vector<core::Vec3> &ambient_field) {
  ambient_field = reconstruct_harmonic_ambient(mesh, coeffs);
  if (orientation_score(mesh, ambient_field) >= 0.0) {
    return false;
  }

  coeffs *= -1.0f;
  for (auto &v : ambient_field) {
    v.x *= -1.0f;
    v.y *= -1.0f;
    v.z *= -1.0f;
  }
  return true;
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

static void write_harmonic_ambient_csv(const std::string &filename,
                                       const DiffusionMesh &mesh,
                                       const std::vector<std::vector<core::Vec3>> &fields) {
  std::ofstream file(filename);
  file << "x,y,z";
  for (size_t form = 0; form < fields.size(); ++form) {
    file << ",form" << form << "_x"
         << ",form" << form << "_y"
         << ",form" << form << "_z";
  }
  file << "\n";

  const size_t n = mesh.geometry.num_points();
  for (size_t i = 0; i < n; ++i) {
    const auto p = mesh.geometry.get_vec3(i);
    file << p.x << "," << p.y << "," << p.z;
    for (size_t form = 0; form < fields.size(); ++form) {
      const auto v = fields[form][i];
      file << "," << v.x << "," << v.y << "," << v.z;
    }
    file << "\n";
  }
}

static void write_circular_csv(const std::string &filename,
                               const DiffusionMesh &mesh,
                               const Eigen::VectorXf &theta_0,
                               const Eigen::VectorXf &theta_1) {
  std::ofstream file(filename);
  file << "x,y,z,theta_0,theta_1\n";
  const size_t n = mesh.geometry.num_points();
  for (size_t i = 0; i < n; ++i) {
    const auto p = mesh.geometry.get_vec3(i);
    const int idx = static_cast<int>(i);
    file << p.x << "," << p.y << "," << p.z << "," << theta_0[idx] << ","
         << theta_1[idx] << "\n";
  }
}

static void write_circular_modes_csv(const std::string &filename,
                                     std::complex<float> eval_0,
                                     std::complex<float> eval_1,
                                     int mode_0, int mode_1,
                                     float circular_lambda) {
  std::ofstream file(filename);
  file << "name,mode,lambda,eigenvalue_real,eigenvalue_imag\n";
  file << "theta_0," << mode_0 << "," << circular_lambda << "," << eval_0.real()
       << "," << eval_0.imag() << "\n";
  file << "theta_1," << mode_1 << "," << circular_lambda << "," << eval_1.real()
       << "," << eval_1.imag() << "\n";
}

int main(int argc, char **argv) {
  Config cfg;
  if (!parse_args(argc, argv, cfg)) {
    return 0;
  }

  const bool bench_mode = std::getenv("IGNEOUS_BENCH_MODE") != nullptr;
  const bool export_ply = cfg.export_ply && !bench_mode;

  std::filesystem::create_directories(cfg.output_dir);

  DiffusionMesh mesh;
  if (!cfg.input_csv.empty()) {
    if (!load_point_cloud_csv(cfg.input_csv, mesh)) {
      return 1;
    }
  } else {
    generate_torus(mesh, cfg.n_points, cfg.major_radius, cfg.minor_radius, cfg.seed);
  }

  mesh.topology.build({mesh.geometry.x_span(),
                       mesh.geometry.y_span(),
                       mesh.geometry.z_span(),
                       cfg.k_neighbors,
                       cfg.knn_bandwidth,
                       cfg.bandwidth_variability,
                       cfg.c,
                       true});

  ops::compute_eigenbasis(mesh, cfg.n_basis);

  ops::GeometryWorkspace<DiffusionMesh> geom_ws;
  const auto G = ops::compute_1form_gram_matrix(mesh, 0.0f, geom_ws);

  ops::HodgeWorkspace<DiffusionMesh> hodge_ws;
  const auto D_weak = ops::compute_weak_exterior_derivative(mesh, 0.0f, hodge_ws);
  const auto E_up = ops::compute_curl_energy_matrix(mesh, 0.0f, hodge_ws);

  const auto laplacian = ops::compute_hodge_laplacian_matrix(D_weak, E_up);
  auto [evals, evecs] = ops::compute_hodge_spectrum(laplacian, G);
  if (evals.size() == 0 || evecs.cols() < 2) {
    std::cerr << "Hodge spectrum solve failed.\n";
    return 1;
  }

  const int export_forms = std::min<int>(3, static_cast<int>(evecs.cols()));
  std::vector<std::vector<core::Vec3>> harmonic_fields;
  harmonic_fields.reserve(static_cast<size_t>(export_forms));
  for (int i = 0; i < export_forms; ++i) {
    std::vector<core::Vec3> field;
    canonicalize_form_sign(mesh, evecs.col(i), field);
    harmonic_fields.push_back(std::move(field));
  }

  std::complex<float> selected_eval_0(0.0f, 0.0f);
  std::complex<float> selected_eval_1(0.0f, 0.0f);
  const auto theta_0 = ops::compute_circular_coordinates(
      mesh, evecs.col(0), 0.0f, cfg.circular_lambda, cfg.circular_mode_0,
      &selected_eval_0);
  const auto theta_1 = ops::compute_circular_coordinates(
      mesh, evecs.col(1), 0.0f, cfg.circular_lambda, cfg.circular_mode_1,
      &selected_eval_1);

  std::cout << "HODGE SPECTRUM (first 12 modes)\n";
  for (int i = 0; i < 12 && i < evals.size(); ++i) {
    std::cout << "Mode " << i << ": lambda = " << evals[i] << "\n";
  }

  write_points_csv(cfg.output_dir + "/points.csv", mesh);
  write_spectrum_csv(cfg.output_dir + "/hodge_spectrum.csv", evals);

  write_harmonic_coeffs_csv(cfg.output_dir + "/harmonic_coeffs.csv", evecs,
                            export_forms);
  write_harmonic_ambient_csv(cfg.output_dir + "/harmonic_ambient.csv", mesh,
                             harmonic_fields);

  write_circular_csv(cfg.output_dir + "/circular_coordinates.csv", mesh, theta_0,
                     theta_1);
  write_circular_modes_csv(cfg.output_dir + "/circular_modes.csv", selected_eval_0,
                           selected_eval_1, cfg.circular_mode_0,
                           cfg.circular_mode_1, cfg.circular_lambda);

  if (export_ply) {
    std::vector<float> field_0(static_cast<size_t>(theta_0.size()));
    std::vector<float> field_1(static_cast<size_t>(theta_1.size()));
    for (int i = 0; i < theta_0.size(); ++i) {
      field_0[static_cast<size_t>(i)] = theta_0[i];
      field_1[static_cast<size_t>(i)] = theta_1[i];
    }

    io::export_ply_solid(mesh, field_0,
                         cfg.output_dir + "/torus_angle_0.ply", 0.01);
    io::export_ply_solid(mesh, field_1,
                         cfg.output_dir + "/torus_angle_1.ply", 0.01);

    for (int i = 0; i < export_forms; ++i) {
      const std::string fname =
          std::format("{}/harmonic_form_{}.ply", cfg.output_dir, i);
      export_vector_field(fname, mesh, harmonic_fields[static_cast<size_t>(i)]);
    }
  }

  return 0;
}
