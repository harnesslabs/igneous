#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <utility>
#include <vector>

#include <igneous/core/parallel.hpp>
#include <igneous/data/mesh.hpp>
#include <igneous/ops/diffusion/forms.hpp>
#include <igneous/ops/diffusion/geometry.hpp>
#include <igneous/ops/diffusion/products.hpp>

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

inline std::pair<Eigen::VectorXf, Eigen::MatrixXf>
compute_hodge_spectrum(const Eigen::MatrixXf &laplacian,
                       const Eigen::MatrixXf &mass_matrix,
                       float rcond = 1e-5f) {
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> mass_solver(mass_matrix);
  if (mass_solver.info() != Eigen::Success) {
    return std::make_pair(Eigen::VectorXf(), Eigen::MatrixXf());
  }

  const auto &mass_evals = mass_solver.eigenvalues();
  const auto &mass_evecs = mass_solver.eigenvectors();
  std::vector<int> keep_indices;
  keep_indices.reserve(static_cast<size_t>(mass_evals.size()));
  for (int i = 0; i < mass_evals.size(); ++i) {
    if (mass_evals[i] > rcond) {
      keep_indices.push_back(i);
    }
  }

  if (keep_indices.empty()) {
    return std::make_pair(Eigen::VectorXf(), Eigen::MatrixXf::Zero(laplacian.rows(), 0));
  }

  const int k = static_cast<int>(keep_indices.size());
  Eigen::MatrixXf phi(laplacian.rows(), k);
  for (int col = 0; col < k; ++col) {
    const int idx = keep_indices[static_cast<size_t>(col)];
    const float inv_sqrt = 1.0f / std::sqrt(std::max(mass_evals[idx], 1e-12f));
    phi.col(col) = mass_evecs.col(idx) * inv_sqrt;
  }

  const Eigen::MatrixXf restricted = phi.transpose() * laplacian * phi;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> solver(restricted);
  if (solver.info() != Eigen::Success) {
    return std::make_pair(Eigen::VectorXf(), Eigen::MatrixXf::Zero(laplacian.rows(), 0));
  }

  const Eigen::VectorXf evals = solver.eigenvalues();
  const Eigen::MatrixXf evecs = phi * solver.eigenvectors();
  return std::make_pair(evals, evecs);
}

template <typename MeshT>
Eigen::VectorXf compute_circular_coordinates(const MeshT &mesh,
                                             const Eigen::VectorXf &alpha_coeffs,
                                             float bandwidth,
                                             float lambda = 1.0f,
                                             int positive_imag_mode = 0,
                                             std::complex<float> *selected_eval = nullptr) {
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

  Eigen::MatrixXf laplacian0_weak = Eigen::MatrixXf::Zero(n0, n0);
  Eigen::VectorXf gamma_local(n_verts_i);
  for (int i = 0; i < n0; ++i) {
    for (int j = i; j < n0; ++j) {
      carre_du_champ(mesh, U.col(i), U.col(j), bandwidth, gamma_local);
      const float val = (gamma_local.array() * mu.array()).sum();
      laplacian0_weak(i, j) = val;
      laplacian0_weak(j, i) = val;
    }
  }

  Eigen::MatrixXf function_gram = U.transpose() * (U.array().colwise() * mu.array()).matrix();
  Eigen::MatrixXf operator_weak = X_op - (lambda * laplacian0_weak);
  Eigen::GeneralizedEigenSolver<Eigen::MatrixXf> solver(operator_weak, function_gram);
  if (solver.info() != Eigen::Success) {
    return Eigen::VectorXf::Zero(static_cast<int>(n_verts));
  }

  Eigen::VectorXcf evals = solver.eigenvalues().cast<std::complex<float>>();
  Eigen::MatrixXcf evecs = solver.eigenvectors().cast<std::complex<float>>();

  std::vector<int> order(static_cast<size_t>(n0));
  for (int i = 0; i < n0; ++i) {
    order[static_cast<size_t>(i)] = i;
  }
  std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
    const std::complex<float> a = evals[lhs];
    const std::complex<float> b = evals[rhs];
    const float abs_a = std::abs(a);
    const float abs_b = std::abs(b);
    if (std::abs(abs_a - abs_b) > 1e-7f) {
      return abs_a < abs_b;
    }
    if (std::abs(a.real() - b.real()) > 1e-7f) {
      return a.real() < b.real();
    }
    return a.imag() < b.imag();
  });

  Eigen::VectorXcf sorted_evals(n0);
  Eigen::MatrixXcf sorted_evecs(n0, n0);
  for (int i = 0; i < n0; ++i) {
    const int src = order[static_cast<size_t>(i)];
    sorted_evals[i] = evals[src];
    sorted_evecs.col(i) = evecs.col(src);
  }

  std::vector<int> positive_imag_indices;
  for (int i = 0; i < n0; ++i) {
    if (sorted_evals[i].imag() > 1e-6f) {
      positive_imag_indices.push_back(i);
    }
  }
  if (positive_imag_indices.empty()) {
    return Eigen::VectorXf::Zero(static_cast<int>(n_verts));
  }

  const int mode =
      std::clamp(positive_imag_mode, 0,
                 static_cast<int>(positive_imag_indices.size()) - 1);
  const int best_idx = positive_imag_indices[static_cast<size_t>(mode)];
  if (selected_eval != nullptr) {
    *selected_eval = sorted_evals[best_idx];
  }

  const Eigen::VectorXcf z_coeffs = sorted_evecs.col(best_idx);
  Eigen::VectorXf theta(static_cast<int>(n_verts));
  constexpr float kTwoPi = 6.28318530717958647692f;

  for (size_t i = 0; i < n_verts; ++i) {
    std::complex<float> zi(0.0f, 0.0f);
    for (int k = 0; k < n0; ++k) {
      zi += U(static_cast<int>(i), k) * z_coeffs(k);
    }
    float angle = std::atan2(zi.imag(), zi.real());
    if (angle < 0.0f) {
      angle += static_cast<float>(kTwoPi);
    }
    theta[static_cast<int>(i)] = angle;
  }

  return theta;
}

} // namespace igneous::ops
