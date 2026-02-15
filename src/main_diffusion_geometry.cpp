#include <Eigen/Dense>
#include <algorithm>
#include <complex>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <igneous/core/algebra.hpp>
#include <igneous/data/space.hpp>
#include <igneous/io/exporter.hpp>
#include <igneous/ops/diffusion/forms.hpp>
#include <igneous/ops/diffusion/hodge.hpp>
#include <igneous/ops/diffusion/products.hpp>
#include <igneous/ops/diffusion/spectral.hpp>

using namespace igneous;
using DiffusionMesh = data::Space<data::DiffusionGeometry>;

struct Config {
  std::string input_csv;
  std::string output_dir = "output_diffusion_geometry";
  size_t n_points = 1000;
  float major_radius = 2.0f;
  float minor_radius = 1.0f;
  float sphere_radius = 1.0f;
  bool generate_sphere = false;
  uint32_t seed = 0;

  int n_basis = 50;
  int n_coefficients = 50;
  int k_neighbors = 32;
  int knn_bandwidth = 8;
  float bandwidth_variability = -0.5f;
  float c = 0.0f;

  float circular_lambda = 1.0f;
  int circular_mode_0 = 0;
  int circular_mode_1 = 1;

  float harmonic_tolerance = 1e-3f;
};

static void print_usage() {
  std::cout
      << "Usage: ./build/igneous-diffusion-geometry [options]\n"
      << "  --input-csv <path>\n"
      << "  --output-dir <path>\n"
      << "  --n-points <int>\n"
      << "  --major-radius <float>\n"
      << "  --minor-radius <float>\n"
      << "  --sphere-radius <float>\n"
      << "  --generate-sphere\n"
      << "  --seed <int>\n"
      << "  --n-basis <int>\n"
      << "  --n-coefficients <int>\n"
      << "  --k-neighbors <int>\n"
      << "  --knn-bandwidth <int>\n"
      << "  --bandwidth-variability <float>\n"
      << "  --c <float>\n"
      << "  --harmonic-tolerance <float>\n"
      << "  --circular-lambda <float>\n"
      << "  --circular-mode-0 <int>\n"
      << "  --circular-mode-1 <int>\n";
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
    if (arg == "--sphere-radius") {
      cfg.sphere_radius = std::stof(require_value("--sphere-radius"));
      continue;
    }
    if (arg == "--generate-sphere") {
      cfg.generate_sphere = true;
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
    if (arg == "--n-coefficients") {
      cfg.n_coefficients = std::stoi(require_value("--n-coefficients"));
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
    if (arg == "--harmonic-tolerance") {
      cfg.harmonic_tolerance = std::stof(require_value("--harmonic-tolerance"));
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
    std::cerr << "Unknown argument: " << arg << "\n";
    print_usage();
    return false;
  }

  return true;
}

static void generate_torus(DiffusionMesh &mesh, size_t n_points, float major_radius,
                           float minor_radius, uint32_t seed) {
  mesh.clear();
  mesh.reserve(n_points);

  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(0.0f, 6.28318530718f);

  for (size_t i = 0; i < n_points; ++i) {
    const float u = dist(gen);
    const float v = dist(gen);
    const float x = (major_radius + minor_radius * std::cos(v)) * std::cos(u);
    const float y = (major_radius + minor_radius * std::cos(v)) * std::sin(u);
    const float z = minor_radius * std::sin(v);
    mesh.push_point({x, y, z});
  }
}

static void generate_sphere(DiffusionMesh &mesh, size_t n_points, float radius,
                            uint32_t seed) {
  mesh.clear();
  mesh.reserve(n_points);

  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> u01(0.0f, 1.0f);

  for (size_t i = 0; i < n_points; ++i) {
    const float u = u01(gen);
    const float v = u01(gen);
    const float theta = 2.0f * 3.14159265358979323846f * u;
    const float phi = std::acos(2.0f * v - 1.0f);
    const float sin_phi = std::sin(phi);
    mesh.push_point({radius * sin_phi * std::cos(theta),
                              radius * sin_phi * std::sin(theta),
                              radius * std::cos(phi)});
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
    mesh.push_point({x, y, z});
  }

  return mesh.num_points() > 0;
}

static std::vector<float> to_scalar_field(const Eigen::VectorXf &values) {
  std::vector<float> out(static_cast<size_t>(values.size()));
  for (int i = 0; i < values.size(); ++i) {
    out[static_cast<size_t>(i)] = values[i];
  }
  return out;
}

static void export_vector_field(const std::string &filename,
                                const DiffusionMesh &mesh,
                                const std::vector<core::Vec3> &vectors) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    return;
  }

  const size_t n = mesh.num_points();
  file << "ply\n";
  file << "format ascii 1.0\n";
  file << "element vertex " << n << "\n";
  file << "property float x\n";
  file << "property float y\n";
  file << "property float z\n";
  file << "property float nx\n";
  file << "property float ny\n";
  file << "property float nz\n";
  file << "end_header\n";

  for (size_t i = 0; i < n; ++i) {
    const auto p = mesh.get_vec3(i);
    const auto v = vectors[i];
    file << p.x << " " << p.y << " " << p.z << " " << v.x << " " << v.y
         << " " << v.z << "\n";
  }
}

static std::array<std::array<Eigen::VectorXf, 3>, 3>
build_gamma_data_immersion(const DiffusionMesh &mesh) {
  std::array<Eigen::VectorXf, 3> data_coords;
  std::array<Eigen::VectorXf, 3> immersion_coords;
  ops::diffusion::fill_data_coordinate_vectors(mesh, data_coords);
  ops::diffusion::fill_coordinate_vectors(mesh, immersion_coords);

  std::array<std::array<Eigen::VectorXf, 3>, 3> gamma{};
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      gamma[a][b].resize(static_cast<int>(mesh.num_points()));
      ops::diffusion::carre_du_champ(mesh, data_coords[a], immersion_coords[b], 0.0f,
                          gamma[a][b]);
    }
  }
  return gamma;
}

static std::vector<core::Vec3> reconstruct_1form_ambient(
    const DiffusionMesh &mesh, const Eigen::VectorXf &coeffs, int n_coefficients,
    const std::array<std::array<Eigen::VectorXf, 3>, 3> &gamma_data_immersion) {
  const Eigen::MatrixXf pointwise =
      ops::diffusion::coefficients_to_pointwise(mesh, coeffs, 1, n_coefficients);
  std::vector<core::Vec3> field(mesh.num_points(), {0.0f, 0.0f, 0.0f});
  if (pointwise.rows() == 0 || pointwise.cols() < 3) {
    return field;
  }

  for (size_t i = 0; i < mesh.num_points(); ++i) {
    const int p = static_cast<int>(i);
    const float vx = gamma_data_immersion[0][0][p] * pointwise(p, 0) +
                     gamma_data_immersion[0][1][p] * pointwise(p, 1) +
                     gamma_data_immersion[0][2][p] * pointwise(p, 2);
    const float vy = gamma_data_immersion[1][0][p] * pointwise(p, 0) +
                     gamma_data_immersion[1][1][p] * pointwise(p, 1) +
                     gamma_data_immersion[1][2][p] * pointwise(p, 2);
    const float vz = gamma_data_immersion[2][0][p] * pointwise(p, 0) +
                     gamma_data_immersion[2][1][p] * pointwise(p, 1) +
                     gamma_data_immersion[2][2][p] * pointwise(p, 2);
    field[i] = {vx, vy, vz};
  }
  return field;
}

static std::vector<core::Vec3> reconstruct_2form_dual_ambient(
    const DiffusionMesh &mesh, const Eigen::VectorXf &coeffs, int n_coefficients,
    const std::array<std::array<Eigen::VectorXf, 3>, 3> &gamma_data_immersion) {
  const Eigen::MatrixXf pointwise =
      ops::diffusion::coefficients_to_pointwise(mesh, coeffs, 2, n_coefficients);
  std::vector<core::Vec3> field(mesh.num_points(), {0.0f, 0.0f, 0.0f});
  if (pointwise.rows() == 0 || pointwise.cols() < 3) {
    return field;
  }

  for (size_t i = 0; i < mesh.num_points(); ++i) {
    const int p = static_cast<int>(i);
    const float w01 = pointwise(p, 0);
    const float w02 = pointwise(p, 1);
    const float w12 = pointwise(p, 2);

    Eigen::Matrix3f W = Eigen::Matrix3f::Zero();
    W(0, 1) = w01;
    W(1, 0) = -w01;
    W(0, 2) = w02;
    W(2, 0) = -w02;
    W(1, 2) = w12;
    W(2, 1) = -w12;

    Eigen::Matrix3f G = Eigen::Matrix3f::Zero();
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        G(a, b) = gamma_data_immersion[a][b][p];
      }
    }

    const Eigen::Matrix3f ambient = G * W * G.transpose();
    field[i] = {ambient(1, 2), -ambient(0, 2), ambient(0, 1)};
  }
  return field;
}

static Eigen::VectorXf pad_1form_coeffs(const Eigen::VectorXf &coeffs, int n_basis,
                                        int n_coefficients) {
  Eigen::VectorXf padded = Eigen::VectorXf::Zero(n_basis * 3);
  const int copy_n = std::min(n_basis, n_coefficients);
  for (int i = 0; i < copy_n; ++i) {
    for (int a = 0; a < 3; ++a) {
      padded(i * 3 + a) = coeffs(i * 3 + a);
    }
  }
  return padded;
}

int main(int argc, char **argv) {
  Config cfg;
  if (!parse_args(argc, argv, cfg)) {
    return 0;
  }

  std::filesystem::create_directories(cfg.output_dir);

  DiffusionMesh mesh;
  if (!cfg.input_csv.empty()) {
    if (!load_point_cloud_csv(cfg.input_csv, mesh)) {
      return 1;
    }
  } else if (cfg.generate_sphere) {
    generate_sphere(mesh, cfg.n_points, cfg.sphere_radius, cfg.seed);
  } else {
    generate_torus(mesh, cfg.n_points, cfg.major_radius, cfg.minor_radius, cfg.seed);
  }

  mesh.structure.build({mesh.x_span(),
                       mesh.y_span(),
                       mesh.z_span(),
                       cfg.k_neighbors,
                       cfg.knn_bandwidth,
                       cfg.bandwidth_variability,
                       cfg.c,
                       true});

  ops::diffusion::compute_eigenbasis(mesh, cfg.n_basis);
  const int n_coeff =
      std::max(1, std::min(cfg.n_coefficients,
                           static_cast<int>(mesh.structure.eigen_basis.cols())));

  ops::diffusion::DiffusionFormWorkspace<DiffusionMesh> forms_ws;

  // 1-forms
  const Eigen::MatrixXf G1 = ops::diffusion::compute_kform_gram_matrix(mesh, 1, n_coeff, forms_ws);
  const Eigen::MatrixXf down1 =
      ops::diffusion::compute_down_laplacian_matrix(mesh, 1, n_coeff, forms_ws);
  const Eigen::MatrixXf up1 = ops::diffusion::compute_up_laplacian_matrix(mesh, 1, n_coeff, forms_ws);
  const Eigen::MatrixXf L1 = ops::diffusion::assemble_hodge_laplacian_matrix(up1, down1);
  auto [evals1, evecs1] = ops::diffusion::compute_form_spectrum(L1, G1);

  if (evals1.size() == 0 || evecs1.cols() == 0) {
    std::cerr << "Failed to compute 1-form spectrum.\n";
    return 1;
  }

  const auto harmonic1_idx =
      ops::diffusion::extract_harmonic_mode_indices(evals1, cfg.harmonic_tolerance, 3);
  Eigen::MatrixXf harmonic1_coeffs(evecs1.rows(),
                                   static_cast<int>(harmonic1_idx.size()));
  for (int c = 0; c < static_cast<int>(harmonic1_idx.size()); ++c) {
    harmonic1_coeffs.col(c) = evecs1.col(harmonic1_idx[static_cast<size_t>(c)]);
  }

  const auto gamma_data_immersion = build_gamma_data_immersion(mesh);

  std::vector<std::vector<core::Vec3>> harmonic1_fields;
  for (int c = 0; c < harmonic1_coeffs.cols(); ++c) {
    harmonic1_fields.push_back(
        reconstruct_1form_ambient(mesh, harmonic1_coeffs.col(c), n_coeff,
                                  gamma_data_immersion));
  }

  const int idx0 = harmonic1_idx.empty() ? 0 : harmonic1_idx[0];
  const int idx1 = harmonic1_idx.size() > 1 ? harmonic1_idx[1] : idx0;
  const Eigen::VectorXf theta_0 = ops::diffusion::compute_circular_coordinates(
      mesh, pad_1form_coeffs(evecs1.col(idx0), mesh.structure.eigen_basis.cols(), n_coeff),
      0.0f, cfg.circular_lambda, cfg.circular_mode_0, nullptr);
  const Eigen::VectorXf theta_1 = ops::diffusion::compute_circular_coordinates(
      mesh, pad_1form_coeffs(evecs1.col(idx1), mesh.structure.eigen_basis.cols(), n_coeff),
      0.0f, cfg.circular_lambda, cfg.circular_mode_1, nullptr);

  // 2-forms
  const Eigen::MatrixXf G2 = ops::diffusion::compute_kform_gram_matrix(mesh, 2, n_coeff, forms_ws);
  const Eigen::MatrixXf down2 =
      ops::diffusion::compute_down_laplacian_matrix(mesh, 2, n_coeff, forms_ws);
  const Eigen::MatrixXf up2 = ops::diffusion::compute_up_laplacian_matrix(mesh, 2, n_coeff, forms_ws);
  const Eigen::MatrixXf L2 = ops::diffusion::assemble_hodge_laplacian_matrix(up2, down2);
  auto [evals2, evecs2] = ops::diffusion::compute_form_spectrum(L2, G2);

  if (evals2.size() == 0 || evecs2.cols() == 0) {
    std::cerr << "Failed to compute 2-form spectrum.\n";
    return 1;
  }

  const auto harmonic2_idx =
      ops::diffusion::extract_harmonic_mode_indices(evals2, cfg.harmonic_tolerance, 3);
  Eigen::MatrixXf harmonic2_coeffs(evecs2.rows(),
                                   static_cast<int>(harmonic2_idx.size()));
  for (int c = 0; c < static_cast<int>(harmonic2_idx.size()); ++c) {
    harmonic2_coeffs.col(c) = evecs2.col(harmonic2_idx[static_cast<size_t>(c)]);
  }

  std::vector<std::vector<core::Vec3>> harmonic2_fields;
  for (int c = 0; c < harmonic2_coeffs.cols(); ++c) {
    harmonic2_fields.push_back(reconstruct_2form_dual_ambient(
        mesh, harmonic2_coeffs.col(c), n_coeff, gamma_data_immersion));
  }

  // wedge(h1_0, h1_1)
  Eigen::VectorXf wedge_coeffs;
  if (harmonic1_coeffs.cols() >= 1) {
    const int rhs_col = harmonic1_coeffs.cols() >= 2 ? 1 : 0;
    wedge_coeffs = ops::diffusion::compute_wedge_product_coeffs(
        mesh, harmonic1_coeffs.col(0), 1, harmonic1_coeffs.col(rhs_col), 1, n_coeff,
        forms_ws);
  } else {
    wedge_coeffs = Eigen::VectorXf::Zero(n_coeff * 3);
  }

  const auto wedge_field = reconstruct_2form_dual_ambient(
      mesh, wedge_coeffs, n_coeff, gamma_data_immersion);

  io::export_ply_solid(mesh, to_scalar_field(theta_0),
                       cfg.output_dir + "/circular_theta_0.ply", 0.01);
  io::export_ply_solid(mesh, to_scalar_field(theta_1),
                       cfg.output_dir + "/circular_theta_1.ply", 0.01);

  for (size_t i = 0; i < harmonic1_fields.size(); ++i) {
    export_vector_field(cfg.output_dir + "/harmonic1_form_" + std::to_string(i) +
                            ".ply",
                        mesh, harmonic1_fields[i]);
  }

  for (size_t i = 0; i < harmonic2_fields.size(); ++i) {
    export_vector_field(cfg.output_dir + "/harmonic2_form_" + std::to_string(i) +
                            ".ply",
                        mesh, harmonic2_fields[i]);
  }

  export_vector_field(cfg.output_dir + "/wedge_h1h1_dual.ply", mesh, wedge_field);

  std::cout << "Computed form spectra and wedge outputs in: " << cfg.output_dir
            << "\n";
  return 0;
}
