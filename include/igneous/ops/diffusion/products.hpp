#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <vector>

#include <igneous/ops/diffusion/basis.hpp>
#include <igneous/ops/diffusion/forms.hpp>

namespace igneous::ops {

template <typename MeshT>
Eigen::VectorXf compute_wedge_product_coeffs(
    const MeshT &mesh, const Eigen::VectorXf &alpha_coeffs, int k1,
    const Eigen::VectorXf &beta_coeffs, int k2, int n_coefficients,
    DiffusionFormWorkspace<MeshT> &workspace) {
  (void)workspace;
  const int d = ambient_dim_3d();
  const int k_total = k1 + k2;
  const int n1 =
      std::max(1, std::min(n_coefficients,
                           static_cast<int>(mesh.topology.eigen_basis.cols())));

  if (k_total > d) {
    return Eigen::VectorXf::Zero(n1);
  }

  const int C1 = std::max(1, binomial_coeff(d, k1));
  const int C2 = std::max(1, binomial_coeff(d, k2));
  const int Cout = std::max(1, binomial_coeff(d, k_total));

  if (alpha_coeffs.size() != n1 * C1 || beta_coeffs.size() != n1 * C2) {
    return Eigen::VectorXf();
  }

  const Eigen::MatrixXf alpha_pw =
      coefficients_to_pointwise(mesh, alpha_coeffs, k1, n1);
  const Eigen::MatrixXf beta_pw =
      coefficients_to_pointwise(mesh, beta_coeffs, k2, n1);
  if (alpha_pw.rows() == 0 || beta_pw.rows() == 0) {
    return Eigen::VectorXf();
  }

  Eigen::MatrixXf wedge_pw = Eigen::MatrixXf::Zero(alpha_pw.rows(), Cout);
  const auto wedge_idx = get_wedge_product_indices(d, k1, k2);

  for (size_t t = 0; t < wedge_idx.target_indices.size(); ++t) {
    const int target = wedge_idx.target_indices[t];
    const int left = wedge_idx.left_indices[t];
    const int right = wedge_idx.right_indices[t];
    const float sign = static_cast<float>(wedge_idx.signs[t]);
    wedge_pw.col(target).array() +=
        sign * alpha_pw.col(left).array() * beta_pw.col(right).array();
  }

  return project_pointwise_to_coefficients(mesh, wedge_pw, n1);
}

template <typename MeshT>
Eigen::VectorXf compute_wedge_product_coeffs(const MeshT &mesh,
                                             const Eigen::VectorXf &alpha_coeffs,
                                             int k1,
                                             const Eigen::VectorXf &beta_coeffs,
                                             int k2,
                                             int n_coefficients) {
  DiffusionFormWorkspace<MeshT> workspace;
  return compute_wedge_product_coeffs(mesh, alpha_coeffs, k1, beta_coeffs, k2,
                                      n_coefficients, workspace);
}

template <typename MeshT>
Eigen::MatrixXf compute_wedge_operator_matrix(
    const MeshT &mesh, const Eigen::VectorXf &alpha_coeffs, int k_left,
    int k_right, int n_coefficients, DiffusionFormWorkspace<MeshT> &workspace) {
  const int d = ambient_dim_3d();
  const int n1 =
      std::max(1, std::min(n_coefficients,
                           static_cast<int>(mesh.topology.eigen_basis.cols())));
  const int in_dim = n1 * std::max(1, binomial_coeff(d, k_right));
  const int out_dim = n1 * std::max(1, binomial_coeff(d, k_left + k_right));

  Eigen::MatrixXf op = Eigen::MatrixXf::Zero(out_dim, in_dim);
  for (int col = 0; col < in_dim; ++col) {
    Eigen::VectorXf beta = Eigen::VectorXf::Zero(in_dim);
    beta[col] = 1.0f;
    const Eigen::VectorXf wedge =
        compute_wedge_product_coeffs(mesh, alpha_coeffs, k_left, beta, k_right,
                                     n1, workspace);
    if (wedge.size() == out_dim) {
      op.col(col) = wedge;
    }
  }
  return op;
}

template <typename MeshT>
Eigen::MatrixXf compute_wedge_operator_matrix(const MeshT &mesh,
                                              const Eigen::VectorXf &alpha_coeffs,
                                              int k_left, int k_right,
                                              int n_coefficients) {
  DiffusionFormWorkspace<MeshT> workspace;
  return compute_wedge_operator_matrix(mesh, alpha_coeffs, k_left, k_right,
                                       n_coefficients, workspace);
}

} // namespace igneous::ops
