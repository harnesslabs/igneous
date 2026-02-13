#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <vector>

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
    for (int i = 0; i < n0; ++i) {
      carre_du_champ(mesh, workspace.coords[a], U.col(i), bandwidth,
                     workspace.gamma_x_phi_mat[a].col(i));
    }
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
  for (int k = 0; k < n0; ++k) {
    for (int d = 0; d < 3; ++d) {
      workspace.gamma_phi_x[k][d].resize(n_verts);
      carre_du_champ(mesh, U.col(k), workspace.coords[d], bandwidth,
                     workspace.gamma_phi_x[k][d]);
    }
  }

  workspace.gamma_phi_phi.resize(n_verts);

  for (int k = 0; k < n0; ++k) {
    for (int l = k; l < n0; ++l) {
      carre_du_champ(mesh, U.col(k), U.col(l), bandwidth,
                     workspace.gamma_phi_phi);

      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          const Eigen::VectorXf term1 =
              workspace.gamma_phi_phi.cwiseProduct(workspace.gamma_xx[a][b]);
          const Eigen::VectorXf term2 = workspace.gamma_phi_x[k][b].cwiseProduct(
              workspace.gamma_phi_x[l][a]);

          const float val = (term1 - term2).dot(mu);

          const int row = k * 3 + a;
          const int col = l * 3 + b;

          E_up(row, col) = val;
          if (row != col) {
            E_up(col, row) = val;
          }
        }
      }
    }
  }

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
                                             float epsilon = 1e-3f) {
  const auto &U = mesh.topology.eigen_basis;
  const auto &mu = mesh.topology.mu;

  const int n0 = U.cols();
  const size_t n_verts = mesh.geometry.num_points();

  HodgeWorkspace<MeshT> workspace;
  fill_coordinate_vectors(mesh, workspace.coords);

  workspace.gamma_x_phi.assign(3, std::vector<Eigen::VectorXf>(n0));
  for (int a = 0; a < 3; ++a) {
    for (int t = 0; t < n0; ++t) {
      workspace.gamma_x_phi[a][t].resize(static_cast<int>(n_verts));
      carre_du_champ(mesh, workspace.coords[a], U.col(t), bandwidth,
                     workspace.gamma_x_phi[a][t]);
    }
  }

  Eigen::MatrixXf X_op = Eigen::MatrixXf::Zero(n0, n0);

  for (int s = 0; s < n0; ++s) {
    for (int t = 0; t < n0; ++t) {
      float val = 0.0f;
      for (int k = 0; k < n0; ++k) {
        for (int a = 0; a < 3; ++a) {
          const float coeff = alpha_coeffs(k * 3 + a);
          if (std::abs(coeff) < 1e-7f) {
            continue;
          }

          const Eigen::VectorXf weight = U.col(s).cwiseProduct(U.col(k)).cwiseProduct(mu);
          val += coeff * weight.dot(workspace.gamma_x_phi[a][t]);
        }
      }
      X_op(s, t) = val;
    }
  }

  Eigen::MatrixXf L_eps = X_op - epsilon * Eigen::MatrixXf::Identity(n0, n0);

  Eigen::EigenSolver<Eigen::MatrixXf> solver(L_eps);
  const auto evals = solver.eigenvalues();
  const auto evecs = solver.eigenvectors();

  int best_idx = 0;
  float min_mag = std::numeric_limits<float>::max();
  for (int i = 0; i < n0; ++i) {
    const float mag = std::abs(evals(i));
    if (mag > 1e-6f && mag < min_mag) {
      min_mag = mag;
      best_idx = i;
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

} // namespace igneous::ops
