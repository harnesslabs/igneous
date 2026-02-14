#pragma once

#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <vector>

#include <igneous/core/parallel.hpp>
#include <igneous/data/mesh.hpp>
#include <igneous/ops/geometry.hpp>

namespace igneous::ops {

template <typename MeshT> struct HodgeWorkspace {
  std::array<Eigen::VectorXf, 3> coords;
  std::array<Eigen::MatrixXf, 3> gamma_x_phi_mat; // [3] each [n_verts x n0]
  Eigen::MatrixXf weighted_u;                     // [n_verts x n0]
  std::vector<std::vector<Eigen::VectorXf>> gamma_x_phi; // [3][n0]
  std::vector<std::vector<Eigen::VectorXf>> gamma_phi_x; // [n0][3]
  std::array<std::array<Eigen::VectorXf, 3>, 3> gamma_xx;
  Eigen::VectorXf gamma_phi_phi;
  Eigen::VectorXf weight;
};

struct CircularCoordinateOptions {
  float epsilon = 1.0f;
  float laplacian_scale = 0.02f;
  bool use_laplacian_regularization = true;
  float min_eigen_magnitude = 1e-6f;
  float min_imaginary_norm = 1e-4f;
  bool allow_real_fallback = true;
};

struct HodgeDecomposition1Options {
  float harmonic_eigenvalue_tol = 1e-4f;
  float tikhonov = 1e-6f;
  bool include_smallest_mode_when_empty = true;
};

struct HodgeDecomposition1Result {
  Eigen::VectorXf exact_component;
  Eigen::VectorXf harmonic_component;
  Eigen::VectorXf coexact_component;
  Eigen::VectorXf exact_potential;
  Eigen::VectorXf hodge_eigenvalues;
};

template <typename MeshT>
Eigen::MatrixXf compute_weak_exterior_derivative(const MeshT &mesh,
                                                 float bandwidth,
                                                 HodgeWorkspace<MeshT> &workspace) {
  const auto &U = mesh.topology.eigen_basis;
  const auto &mu = mesh.topology.mu;
  const int n0 = U.cols();
  const int n_verts = static_cast<int>(mesh.geometry.num_points());

  Eigen::MatrixXf D = Eigen::MatrixXf::Zero(3 * n0, n0);

  fill_coordinate_vectors(mesh, workspace.coords);

  for (int a = 0; a < 3; ++a) {
    if (workspace.gamma_x_phi_mat[a].rows() != n_verts ||
        workspace.gamma_x_phi_mat[a].cols() != n0) {
      workspace.gamma_x_phi_mat[a].resize(n_verts, n0);
    }
    core::parallel_for_index(
        0, n0,
        [&](int i) {
          carre_du_champ(mesh, workspace.coords[a], U.col(i), bandwidth,
                         workspace.gamma_x_phi_mat[a].col(i));
        },
        8);
  }

  if (workspace.weighted_u.rows() != n_verts ||
      workspace.weighted_u.cols() != n0) {
    workspace.weighted_u.resize(n_verts, n0);
  }
  workspace.weighted_u = U.array().colwise() * mu.array();

  for (int a = 0; a < 3; ++a) {
    const Eigen::MatrixXf coupling =
        workspace.weighted_u.transpose() * workspace.gamma_x_phi_mat[a];
    for (int k = 0; k < n0; ++k) {
      D.row(k * 3 + a) = coupling.row(k);
    }
  }

  return D;
}

template <typename MeshT>
Eigen::MatrixXf compute_weak_exterior_derivative(const MeshT &mesh,
                                                 float bandwidth) {
  HodgeWorkspace<MeshT> workspace;
  return compute_weak_exterior_derivative(mesh, bandwidth, workspace);
}

template <typename MeshT>
Eigen::MatrixXf compute_curl_energy_matrix(const MeshT &mesh, float bandwidth,
                                           HodgeWorkspace<MeshT> &workspace) {
  const auto &U = mesh.topology.eigen_basis;
  const auto &mu = mesh.topology.mu;
  const auto mu_arr = mu.array();
  const int n0 = U.cols();
  const int n_basis = 3 * n0;
  const int n_verts = static_cast<int>(mesh.geometry.num_points());

  Eigen::MatrixXf E_up = Eigen::MatrixXf::Zero(n_basis, n_basis);

  fill_coordinate_vectors(mesh, workspace.coords);

  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      workspace.gamma_xx[a][b].resize(n_verts);
      carre_du_champ(mesh, workspace.coords[a], workspace.coords[b], bandwidth,
                     workspace.gamma_xx[a][b]);
    }
  }

  workspace.gamma_phi_x.assign(n0, std::vector<Eigen::VectorXf>(3));
  core::parallel_for_index(
      0, n0,
      [&](int k) {
        for (int d = 0; d < 3; ++d) {
          workspace.gamma_phi_x[k][d].resize(n_verts);
          carre_du_champ(mesh, U.col(k), workspace.coords[d], bandwidth,
                         workspace.gamma_phi_x[k][d]);
        }
      },
      8);

  core::parallel_for_index(
      0, n0,
      [&](int k) {
        Eigen::VectorXf gamma_phi_phi_local(n_verts);

        for (int l = k; l < n0; ++l) {
          carre_du_champ(mesh, U.col(k), U.col(l), bandwidth,
                         gamma_phi_phi_local);

          for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
              const float val =
                  (((gamma_phi_phi_local.array() *
                     workspace.gamma_xx[a][b].array()) -
                    (workspace.gamma_phi_x[k][b].array() *
                     workspace.gamma_phi_x[l][a].array())) *
                   mu_arr)
                      .sum();

              const int row = k * 3 + a;
              const int col = l * 3 + b;

              E_up(row, col) = val;
              if (row != col) {
                E_up(col, row) = val;
              }
            }
          }
        }
      },
      8);

  return E_up;
}

template <typename MeshT>
Eigen::MatrixXf compute_curl_energy_matrix(const MeshT &mesh, float bandwidth) {
  HodgeWorkspace<MeshT> workspace;
  return compute_curl_energy_matrix(mesh, bandwidth, workspace);
}

inline Eigen::MatrixXf compute_hodge_laplacian_matrix(
    const Eigen::MatrixXf &D_weak, const Eigen::MatrixXf &E_up) {
  const Eigen::MatrixXf L_down = D_weak * D_weak.transpose();
  return L_down + E_up;
}

inline auto compute_hodge_spectrum(const Eigen::MatrixXf &laplacian,
                                   const Eigen::MatrixXf &mass_matrix) {
  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXf> solver(laplacian,
                                                                   mass_matrix);
  return std::make_pair(solver.eigenvalues(), solver.eigenvectors());
}

template <typename MeshT>
Eigen::VectorXf compute_circular_coordinates(const MeshT &mesh,
                                             const Eigen::VectorXf &alpha_coeffs,
                                             float bandwidth,
                                             const CircularCoordinateOptions &options) {
  const auto &U = mesh.topology.eigen_basis;
  const auto &mu = mesh.topology.mu;

  const int n0 = U.cols();
  const size_t n_verts = mesh.geometry.num_points();
  const int n_verts_i = static_cast<int>(n_verts);

  HodgeWorkspace<MeshT> workspace;
  fill_coordinate_vectors(mesh, workspace.coords);

  for (int a = 0; a < 3; ++a) {
    workspace.gamma_x_phi_mat[a].resize(n_verts_i, n0);
    core::parallel_for_index(
        0, n0,
        [&](int t) {
          carre_du_champ(mesh, workspace.coords[a], U.col(t), bandwidth,
                         workspace.gamma_x_phi_mat[a].col(t));
        },
        8);
  }

  Eigen::MatrixXf alpha_mat(n0, 3);
  for (int k = 0; k < n0; ++k) {
    alpha_mat(k, 0) = alpha_coeffs(k * 3 + 0);
    alpha_mat(k, 1) = alpha_coeffs(k * 3 + 1);
    alpha_mat(k, 2) = alpha_coeffs(k * 3 + 2);
  }

  const Eigen::MatrixXf q = U * alpha_mat;
  const Eigen::MatrixXf U_t = U.transpose();

  Eigen::MatrixXf X_op(n0, n0);
  core::parallel_for_index(
      0, n0,
      [&](int t) {
        Eigen::VectorXf weight_local(n_verts_i);
        weight_local =
            mu.array() *
            ((workspace.gamma_x_phi_mat[0].col(t).array() * q.col(0).array()) +
             (workspace.gamma_x_phi_mat[1].col(t).array() * q.col(1).array()) +
             (workspace.gamma_x_phi_mat[2].col(t).array() * q.col(2).array()));
        X_op.col(t).noalias() = U_t * weight_local;
      },
      8);

  Eigen::MatrixXf L_eps = X_op;
  if (options.use_laplacian_regularization) {
    Eigen::VectorXf lap_diag = Eigen::VectorXf::Ones(n0);
    if constexpr (requires { mesh.topology.eigen_values; }) {
      if (mesh.topology.eigen_values.size() >= n0) {
        lap_diag = (Eigen::VectorXf::Ones(n0) - mesh.topology.eigen_values.head(n0))
                       .cwiseMax(0.0f);
      }
    }
    L_eps -= (options.epsilon * options.laplacian_scale) * lap_diag.asDiagonal();
  } else {
    L_eps -= options.epsilon * Eigen::MatrixXf::Identity(n0, n0);
  }

  Eigen::EigenSolver<Eigen::MatrixXf> solver(L_eps);
  const auto evals = solver.eigenvalues();
  const auto evecs = solver.eigenvectors();

  int best_idx = -1;
  float min_mag = std::numeric_limits<float>::max();
  for (int i = 0; i < n0; ++i) {
    const float mag = std::abs(evals(i));
    if (mag <= options.min_eigen_magnitude) {
      continue;
    }
    const float imag_norm = evecs.col(i).imag().norm();
    if (imag_norm <= options.min_imaginary_norm) {
      continue;
    }
    if (mag < min_mag) {
      min_mag = mag;
      best_idx = i;
    }
  }

  if (best_idx < 0 && options.allow_real_fallback) {
    for (int i = 0; i < n0; ++i) {
      const float mag = std::abs(evals(i));
      if (mag > options.min_eigen_magnitude && mag < min_mag) {
        min_mag = mag;
        best_idx = i;
      }
    }
  }

  if (best_idx < 0) {
    best_idx = 0;
    min_mag = std::abs(evals(0));
    for (int i = 1; i < n0; ++i) {
      const float mag = std::abs(evals(i));
      if (mag < min_mag) {
        min_mag = mag;
        best_idx = i;
      }
    }
  }

  const Eigen::VectorXcf z_coeffs = evecs.col(best_idx);
  Eigen::VectorXf theta(static_cast<int>(n_verts));
  constexpr float kPi = 3.14159265358979323846f;

  for (size_t i = 0; i < n_verts; ++i) {
    std::complex<float> zi(0.0f, 0.0f);
    for (int k = 0; k < n0; ++k) {
      zi += U(static_cast<int>(i), k) * z_coeffs(k);
    }
    theta[static_cast<int>(i)] = (std::arg(zi) + kPi) / (2.0f * kPi);
  }

  return theta;
}

template <typename MeshT>
Eigen::VectorXf compute_circular_coordinates(const MeshT &mesh,
                                             const Eigen::VectorXf &alpha_coeffs,
                                             float bandwidth,
                                             float epsilon = 1.0f) {
  CircularCoordinateOptions options;
  options.epsilon = epsilon;
  return compute_circular_coordinates(mesh, alpha_coeffs, bandwidth, options);
}

template <typename MeshT>
HodgeDecomposition1Result compute_hodge_decomposition_1form(
    const MeshT &mesh, const Eigen::VectorXf &alpha_coeffs, float bandwidth,
    const HodgeDecomposition1Options &options = {}) {
  HodgeDecomposition1Result result;

  const int n0 = mesh.topology.eigen_basis.cols();
  const int n_basis = n0 * 3;
  if (n0 <= 0 || alpha_coeffs.size() != n_basis) {
    result.exact_component = Eigen::VectorXf::Zero(n_basis);
    result.harmonic_component = Eigen::VectorXf::Zero(n_basis);
    result.coexact_component = Eigen::VectorXf::Zero(n_basis);
    result.exact_potential = Eigen::VectorXf::Zero(n0);
    result.hodge_eigenvalues.resize(0);
    return result;
  }

  GeometryWorkspace<MeshT> geometry_workspace;
  const Eigen::MatrixXf G =
      compute_1form_gram_matrix(mesh, bandwidth, geometry_workspace);

  HodgeWorkspace<MeshT> hodge_workspace;
  const Eigen::MatrixXf D =
      compute_weak_exterior_derivative(mesh, bandwidth, hodge_workspace);
  const Eigen::MatrixXf E =
      compute_curl_energy_matrix(mesh, bandwidth, hodge_workspace);
  const Eigen::MatrixXf L = compute_hodge_laplacian_matrix(D, E);

  const Eigen::MatrixXf GD = G * D;
  Eigen::MatrixXf normal_matrix = D.transpose() * GD;
  normal_matrix.diagonal().array() += std::max(options.tikhonov, 0.0f);

  const Eigen::VectorXf rhs = D.transpose() * (G * alpha_coeffs);
  const Eigen::LDLT<Eigen::MatrixXf> exact_solver(normal_matrix);

  result.exact_potential = exact_solver.solve(rhs);
  if (exact_solver.info() != Eigen::Success) {
    result.exact_potential = Eigen::VectorXf::Zero(n0);
  }
  result.exact_component = D * result.exact_potential;

  const Eigen::VectorXf residual = alpha_coeffs - result.exact_component;
  auto [evals, evecs] = compute_hodge_spectrum(L, G);
  result.hodge_eigenvalues = evals;

  std::vector<int> harmonic_indices;
  harmonic_indices.reserve(static_cast<size_t>(evals.size()));
  for (int i = 0; i < evals.size(); ++i) {
    if (evals[i] <= options.harmonic_eigenvalue_tol) {
      harmonic_indices.push_back(i);
    } else {
      break;
    }
  }
  if (harmonic_indices.empty() && options.include_smallest_mode_when_empty &&
      evals.size() > 0) {
    harmonic_indices.push_back(0);
  }

  result.harmonic_component = Eigen::VectorXf::Zero(n_basis);
  if (!harmonic_indices.empty()) {
    const Eigen::VectorXf G_residual = G * residual;
    for (int idx : harmonic_indices) {
      const Eigen::VectorXf basis = evecs.col(idx);
      const float coeff = basis.dot(G_residual);
      result.harmonic_component.noalias() += coeff * basis;
    }
  }

  result.coexact_component =
      alpha_coeffs - result.exact_component - result.harmonic_component;
  return result;
}

} // namespace igneous::ops
