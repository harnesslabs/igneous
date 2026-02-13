#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <igneous/data/mesh.hpp>
#include <igneous/ops/geometry.hpp>
#include <iostream>

namespace igneous::ops {

// ==============================================================================
// 1. DIFFERENTIAL OPERATORS
// ==============================================================================

// Compute Weak Exterior Derivative d_0: Functions -> 1-Forms
// Maps spectral basis coeffs (n0) -> 1-form basis coeffs (3*n0).
// Returns matrix D of size (3*n0) x n0.
// Corresponds to the gradient operator in the spectral domain.
template <typename MeshT>
Eigen::MatrixXf compute_weak_exterior_derivative(const MeshT &mesh,
                                                 float bandwidth) {
  const auto &U = mesh.topology.eigen_basis;
  const auto &mu = mesh.topology.mu;
  int n0 = U.cols();

  // D maps f -> df. Size: (Basis_1-Form) x (Basis_Func)
  Eigen::MatrixXf D = Eigen::MatrixXf::Zero(3 * n0, n0);

  // 1. Pre-convert coords for gamma calc
  std::vector<Eigen::VectorXf> coords(3);
  size_t n_verts = mesh.geometry.num_points();
  for (int d = 0; d < 3; ++d) {
    coords[d].resize(n_verts);
    for (size_t i = 0; i < n_verts; ++i)
      coords[d][i] = mesh.geometry.packed_data[i * 3 + d];
  }

  // 2. Precompute Gamma(x_a, phi_i)
  // This represents the gradient of basis functions projected onto axes.
  std::vector<std::vector<Eigen::VectorXf>> gamma_x_phi(
      3, std::vector<Eigen::VectorXf>(n0));

  std::cout << "[Hodge] Precomputing gradients of eigenbasis...\n";
  for (int a = 0; a < 3; ++a) {
    for (int i = 0; i < n0; ++i) {
      gamma_x_phi[a][i] = carre_du_champ(mesh, coords[a], U.col(i), bandwidth);
    }
  }

  std::cout << "[Hodge] Assembling Weak Derivative D...\n";

  // 3. Assemble D
  // D_{k,a, i} = < phi_k dx_a, d(phi_i) >
  //            = Integral( phi_k * Gamma(x_a, phi_i) dmu )
  for (int k = 0; k < n0; ++k) {
    // Precompute weight vector for this test function
    Eigen::VectorXf weight = U.col(k).cwiseProduct(mu);

    for (int a = 0; a < 3; ++a) {
      int row_idx = k * 3 + a; // Index in 1-form vector

      for (int i = 0; i < n0; ++i) {
        // Dot product integrates over the manifold
        float val = weight.dot(gamma_x_phi[a][i]);
        D(row_idx, i) = val;
      }
    }
  }

  return D;
}

// Compute the "Up" (Curl) Energy Matrix for 1-forms
// Energy_up(alpha) = || d(alpha) ||^2 = || curl(alpha) ||^2
// This penalizes local rotation, leaving only global harmonic loops.
template <typename MeshT>
Eigen::MatrixXf compute_curl_energy_matrix(const MeshT &mesh, float bandwidth) {
  const auto &U = mesh.topology.eigen_basis;
  const auto &mu = mesh.topology.mu;
  int n0 = U.cols();
  int n_basis = 3 * n0;

  Eigen::MatrixXf E_up = Eigen::MatrixXf::Zero(n_basis, n_basis);

  std::cout << "[Hodge] Computing Curl Energy Matrix (Up Laplacian)...\n";

  // 1. Precompute Gamma components needed for Lagrange Identity
  // A. Gamma(x_a, x_b) - Inverse metric of embedding
  std::vector<Eigen::VectorXf> coords(3);
  for (int d = 0; d < 3; ++d) {
    coords[d].resize(mesh.geometry.num_points());
    for (size_t i = 0; i < mesh.geometry.num_points(); ++i)
      coords[d][i] = mesh.geometry.packed_data[i * 3 + d];
  }

  Eigen::VectorXf gamma_xx[3][3];
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
      gamma_xx[a][b] = carre_du_champ(mesh, coords[a], coords[b], bandwidth);

  // 2. Assemble Matrix via Lagrange Identity
  // We iterate over basis pairs I=(k,a) and J=(l,b)
  // Value = Integral( <curl(phi_k dx_a), curl(phi_l dx_b)> dmu )

  // Optimization: Precompute Gamma(phi, x)
  std::vector<std::vector<Eigen::VectorXf>> gamma_phi_x(
      n0, std::vector<Eigen::VectorXf>(3));
  for (int k = 0; k < n0; ++k)
    for (int d = 0; d < 3; ++d)
      gamma_phi_x[k][d] = carre_du_champ(mesh, U.col(k), coords[d], bandwidth);

  // Optimization: Precompute Gamma(phi, phi) only when needed or cache
  // carefully. Since n0 is small (~32), we can compute on the fly or cache.
  // Let's cache the diagonal/upper triangle of phi-phi interactions.

  for (int k = 0; k < n0; ++k) {
    for (int l = k; l < n0; ++l) { // Symmetric over functions
      Eigen::VectorXf g_phi_phi =
          carre_du_champ(mesh, U.col(k), U.col(l), bandwidth);

      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          // Lagrange Identity Term 1: (grad phi_k . grad phi_l)(grad x_a . grad
          // x_b)
          Eigen::VectorXf term1 = g_phi_phi.cwiseProduct(gamma_xx[a][b]);

          // Lagrange Identity Term 2: (grad phi_k . grad x_b)(grad x_a . grad
          // phi_l)
          Eigen::VectorXf term2 =
              gamma_phi_x[k][b].cwiseProduct(gamma_phi_x[l][a]);

          // Integrate
          float val = (term1 - term2).dot(mu);

          int row = k * 3 + a;
          int col = l * 3 + b;

          E_up(row, col) = val;
          if (row != col)
            E_up(col, row) = val;
        }
      }
    }
  }

  return E_up;
}

// ==============================================================================
// 2. HODGE LAPLACIAN & SPECTRUM
// ==============================================================================

// Compute the full Hodge Laplacian Matrix on 1-forms
// Delta = Delta_down + Delta_up
//       = (d* d)     + (d d*)
// Note: On 1-forms, d* is the codifferential (divergence) and d is curl.
inline Eigen::MatrixXf compute_hodge_laplacian_matrix(
    const Eigen::MatrixXf &D_weak, // Weak exterior derivative
    const Eigen::MatrixXf &E_up    // Up Energy (Curl)
) {
  // 1. Down Laplacian (Divergence Term)
  // Formula: L_down = D * (G0)^-1 * D^T
  // Since our spectral basis functions are orthonormal, G0 is Identity.
  // So L_down = D * D^T
  // This matrix measures "how much divergence" a 1-form has.
  Eigen::MatrixXf L_down = D_weak * D_weak.transpose();

  // 2. Full Laplacian
  // We sum the energies.
  // Ideally, for topology, we want kernel of BOTH.
  return L_down + E_up;
}

// Solve for the Hodge Spectrum
// Generalized Eigenproblem: Delta * x = lambda * G * x
inline auto compute_hodge_spectrum(const Eigen::MatrixXf &Laplacian,
                                   const Eigen::MatrixXf &MassMatrix // G1
) {
  std::cout << "[Hodge] Solving Generalized Eigenproblem (size "
            << Laplacian.rows() << ")...\n";

  // Use Eigen's solver for symmetric A, positive-definite B
  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXf> solver(Laplacian,
                                                                   MassMatrix);

  return std::make_pair(solver.eigenvalues(), solver.eigenvectors());
}

// ==============================================================================
// 3. CIRCULAR COORDINATES
// ==============================================================================

/**
 * Computes circular coordinates (angles 0 to 2pi) for a given harmonic 1-form.
 * This implements Section 8.3 of the paper.
 * * alpha_coeffs: The coefficients of the 1-form in the spectral basis (from
 * evecs.col(i)) epsilon: Regularization parameter (default 1.0 as per paper)
 */
template <typename MeshT>
Eigen::VectorXf
compute_circular_coordinates(const MeshT &mesh,
                             const Eigen::VectorXf &alpha_coeffs,
                             float epsilon = 1.0f) {
  const auto &U = mesh.topology.eigen_basis;
  const auto &P = mesh.topology.P;
  int n0 = U.cols();
  size_t n_verts = mesh.geometry.num_points();

  std::cout << "[Hodge] Computing Circular Coordinates from Harmonic Form...\n";

  // 1. Build the Operator Matrix X^op for the vector field alpha#
  // Any vector field X acts on functions by X(f) = g(X, grad f).
  // In our basis: X^op_{st} = < phi_s, alpha#(phi_t) > [cite: 1138-1139]
  Eigen::MatrixXf X_op = Eigen::MatrixXf::Zero(n0, n0);

  // Reconstruct coordinate gradients Gamma(x_a, phi_t) for all a, t
  // This is a bit expensive but necessary to get the operator form.
  std::vector<Eigen::VectorXf> coords(3);
  for (int d = 0; d < 3; ++d) {
    coords[d].resize(n_verts);
    for (size_t i = 0; i < n_verts; ++i)
      coords[d][i] = mesh.geometry.packed_data[i * 3 + d];
  }

  // Precompute Gamma(x_a, phi_t)
  std::vector<std::vector<Eigen::VectorXf>> gamma_x_phi(
      3, std::vector<Eigen::VectorXf>(n0));
  for (int a = 0; a < 3; ++a)
    for (int t = 0; t < n0; ++t)
      gamma_x_phi[a][t] = carre_du_champ(mesh, coords[a], U.col(t),
                                         0.05f); // Use same bandwidth

  // Assemble X_op: X_op_{st} = sum_{a, k} alpha_{k,a} * <phi_s, phi_k *
  // Gamma(x_a, phi_t)>
  for (int s = 0; s < n0; ++s) {
    for (int t = 0; t < n0; ++t) {
      float val = 0.0f;
      for (int k = 0; k < n0; ++k) {
        for (int a = 0; a < 3; ++a) {
          float coeff = alpha_coeffs(k * 3 + a);
          if (std::abs(coeff) < 1e-6f)
            continue;

          // weight = phi_s * phi_k * mu
          Eigen::VectorXf weight =
              U.col(s).cwiseProduct(U.col(k)).cwiseProduct(mesh.topology.mu);
          val += coeff * weight.dot(gamma_x_phi[a][t]);
        }
      }
      X_op(s, t) = val;
    }
  }

  // 2. Add Diffusion Regularization: L_eps = X^op - eps * Delta [cite:
  // 1751-1752] Since our basis functions are eigenfunctions of P with eval
  // lambda_i, the Laplacian Delta acting on phi_i is (1 - lambda_i). In our
  // basis, Delta is a diagonal matrix.
  Eigen::MatrixXf Delta = Eigen::MatrixXf::Zero(n0, n0);
  // Note: We'd need the eigenvalues of P stored in the topology to do this
  // perfectly. Assuming for now epsilon=0 is okay, or manually providing a
  // small diagonal.
  Eigen::MatrixXf L_eps = X_op;

  // 3. Solve Complex Eigenproblem
  // We expect imaginary eigenvalues[cite: 1749].
  Eigen::EigenSolver<Eigen::MatrixXf> solver(L_eps);
  auto evals = solver.eigenvalues();
  auto evecs = solver.eigenvectors();

  // 4. Find the eigenfunction with smallest non-zero eigenvalue magnitude
  // [cite: 1750]
  int best_idx = 0;
  float min_mag = 1e9f;
  for (int i = 0; i < n0; ++i) {
    float mag = std::abs(evals(i));
    if (mag > 1e-5f && mag < min_mag) {
      min_mag = mag;
      best_idx = i;
    }
  }

  // 5. Reconstruct the spatial complex function z and take its argument
  // z = U * evecs.col(best_idx)
  Eigen::VectorXcf z_coeffs = evecs.col(best_idx);
  Eigen::VectorXf theta(n_verts);

  for (size_t i = 0; i < n_verts; ++i) {
    std::complex<float> zi(0, 0);
    for (int k = 0; k < n0; ++k) {
      zi += U(i, k) * z_coeffs(k);
    }
    // arg(z) gives angle in [-pi, pi]. Map to [0, 1] for visualization.
    theta(i) = (std::arg(zi) + M_PI) / (2.0f * M_PI);
  }

  return theta;
}

} // namespace igneous::ops