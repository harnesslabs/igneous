#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include <igneous/core/parallel.hpp>
#include <igneous/ops/diffusion/forms.hpp>
#include <igneous/ops/diffusion/geometry.hpp>

namespace igneous::ops::diffusion {

/// \brief Reusable buffers for diffusion-Hodge assembly.
template <typename MeshT> struct HodgeWorkspace {
  /// \brief Coordinate vectors used in gamma evaluations.
  std::array<Eigen::VectorXf, 3> coords;
  /// \brief `Gamma(x_a, phi_i)` matrix cache for each axis.
  std::array<Eigen::MatrixXf, 3> gamma_x_phi_mat;
  /// \brief Basis weighted by stationary measure.
  Eigen::MatrixXf weighted_u;
  /// \brief `Gamma(x_a, x_b)` caches.
  std::array<std::array<Eigen::VectorXf, 3>, 3> gamma_xx;
  /// \brief Temporary `Gamma(phi_i, phi_j)` buffer.
  Eigen::VectorXf gamma_phi_phi;
  /// \brief Generic temporary weight vector.
  Eigen::VectorXf weight;
  /// \brief Cached state for `gamma_x_phi_mat` and `weighted_u`.
  int mixed_n_verts = -1;
  /// \brief Cached scalar basis count for `gamma_x_phi_mat` and `weighted_u`.
  int mixed_n0 = -1;
  /// \brief Cached bandwidth for `gamma_x_phi_mat`.
  float mixed_bandwidth = std::numeric_limits<float>::quiet_NaN();
  /// \brief Whether mixed gamma and weighted basis caches are ready.
  bool mixed_ready = false;
  /// \brief Cached state for `gamma_xx`.
  int gamma_xx_n_verts = -1;
  /// \brief Cached bandwidth for `gamma_xx`.
  float gamma_xx_bandwidth = std::numeric_limits<float>::quiet_NaN();
  /// \brief Whether coordinate gamma caches are ready.
  bool gamma_xx_ready = false;

  void reset_cache() {
    mixed_n_verts = -1;
    mixed_n0 = -1;
    mixed_bandwidth = std::numeric_limits<float>::quiet_NaN();
    mixed_ready = false;
    gamma_xx_n_verts = -1;
    gamma_xx_bandwidth = std::numeric_limits<float>::quiet_NaN();
    gamma_xx_ready = false;
  }
};

/// \brief Reusable buffers for circular-coordinate assembly on a fixed mesh/basis state.
template <typename MeshT> struct CircularCoordinateWorkspace {
  std::array<Eigen::VectorXf, 3> coords;
  std::array<Eigen::VectorXf, 3> coord_means;
  std::array<Eigen::MatrixXf, 3> gamma_x_phi_mat;
  Eigen::MatrixXf weighted_u;
  Eigen::MatrixXf function_gram;
  Eigen::MatrixXf laplacian0_weak;
  Eigen::MatrixXf markov_basis;
  Eigen::MatrixXf markov_tmp;
  Eigen::MatrixXf alpha_mat;
  Eigen::MatrixXf q;
  Eigen::MatrixXf weighted_gamma;
  Eigen::MatrixXf x_op;
  Eigen::VectorXf gamma_local;
  Eigen::VectorXf z_real;
  Eigen::VectorXf z_imag;
  int prepared_n_verts = -1;
  int prepared_n0 = -1;
  float prepared_bandwidth = std::numeric_limits<float>::quiet_NaN();

  void reset() {
    prepared_n_verts = -1;
    prepared_n0 = -1;
    prepared_bandwidth = std::numeric_limits<float>::quiet_NaN();
  }
};

template <typename MeshT>
void apply_markov_transition_matrix(const MeshT& mesh, const Eigen::MatrixXf& input,
                                    Eigen::MatrixXf& output) {
  const int n_verts = static_cast<int>(mesh.num_points());
  const int n_cols = input.cols();
  output.resize(n_verts, n_cols);

  const auto& row_offsets = mesh.structure.markov_row_offsets;
  const auto& col_indices = mesh.structure.markov_col_indices;
  const auto& weights = mesh.structure.markov_values;

  core::parallel_for_index(
      0, n_verts,
      [&](int row_idx) {
        thread_local Eigen::RowVectorXf accum;
        if (accum.size() != n_cols) {
          accum.resize(n_cols);
        }
        accum.setZero();

        const int begin = row_offsets[static_cast<size_t>(row_idx)];
        const int end = row_offsets[static_cast<size_t>(row_idx) + 1];
        for (int idx = begin; idx < end; ++idx) {
          accum.noalias() += weights[static_cast<size_t>(idx)] *
                             input.row(col_indices[static_cast<size_t>(idx)]);
        }
        output.row(row_idx) = accum;
      },
      128);
}

template <typename MeshT>
void prepare_circular_coordinate_workspace(const MeshT& mesh, float bandwidth,
                                          CircularCoordinateWorkspace<MeshT>& workspace) {
  const auto& U = mesh.structure.eigen_basis;
  const auto& mu = mesh.structure.mu;
  const int n0 = U.cols();
  const int n_verts = static_cast<int>(mesh.num_points());

  if (workspace.prepared_n_verts == n_verts && workspace.prepared_n0 == n0 &&
      workspace.prepared_bandwidth == bandwidth) {
    return;
  }

  fill_coordinate_vectors(mesh, workspace.coords);

  workspace.weighted_u.resize(n_verts, n0);
  workspace.weighted_u = U.array().colwise() * mu.array();

  workspace.function_gram.resize(n0, n0);
  workspace.function_gram.noalias() = U.transpose() * workspace.weighted_u;

  const bool use_mean_centres = [&]() {
    if constexpr (requires { mesh.structure.use_mean_centres; }) {
      return mesh.structure.use_mean_centres;
    }
    return false;
  }();

  if (use_mean_centres) {
    workspace.markov_basis.resize(n_verts, n0);
    apply_markov_transition_matrix(mesh, U, workspace.markov_basis);

    Eigen::VectorXf b(n_verts);
    Eigen::VectorXf inv_gamma_denom(n_verts);
    for (int i = 0; i < n_verts; ++i) {
      float denom = 2.0f;
      if constexpr (requires { mesh.structure.local_bandwidths; }) {
        if (mesh.structure.local_bandwidths.size() == n_verts) {
          denom =
              2.0f * std::max(mesh.structure.local_bandwidths[static_cast<size_t>(i)], 1e-8f);
        }
      }
      b[i] = mu[i] / denom;
      inv_gamma_denom[i] = 1.0f / denom;
    }

    Eigen::VectorXf ptb = Eigen::VectorXf::Zero(n_verts);
    const auto& row_offsets = mesh.structure.markov_row_offsets;
    const auto& col_indices = mesh.structure.markov_col_indices;
    const auto& weights = mesh.structure.markov_values;
    for (int i = 0; i < n_verts; ++i) {
      const int begin = row_offsets[static_cast<size_t>(i)];
      const int end = row_offsets[static_cast<size_t>(i) + 1];
      const float bi = b[i];
      for (int idx = begin; idx < end; ++idx) {
        ptb[col_indices[static_cast<size_t>(idx)]] += weights[static_cast<size_t>(idx)] * bi;
      }
    }

    for (int a = 0; a < 3; ++a) {
      workspace.coord_means[a].resize(n_verts);
      apply_markov_transition(mesh, workspace.coords[a], workspace.coord_means[a]);
      workspace.markov_tmp.resize(n_verts, n0);
      workspace.markov_tmp = U.array().colwise() * workspace.coords[a].array();
      apply_markov_transition_matrix(mesh, workspace.markov_tmp, workspace.gamma_x_phi_mat[a]);
      workspace.gamma_x_phi_mat[a].array() -=
          workspace.markov_basis.array().colwise() * workspace.coord_means[a].array();
      workspace.gamma_x_phi_mat[a].array().colwise() *= inv_gamma_denom.array();
    }

    workspace.laplacian0_weak.resize(n0, n0);
    workspace.markov_tmp = U.array().colwise() * ptb.array();
    workspace.laplacian0_weak.noalias() = U.transpose() * workspace.markov_tmp;
    workspace.markov_tmp = workspace.markov_basis.array().colwise() * b.array();
    workspace.laplacian0_weak.noalias() -= workspace.markov_basis.transpose() * workspace.markov_tmp;
    workspace.laplacian0_weak = 0.5f * (workspace.laplacian0_weak + workspace.laplacian0_weak.transpose());
  } else {
    for (int a = 0; a < 3; ++a) {
      workspace.gamma_x_phi_mat[a].resize(n_verts, n0);
      core::parallel_for_index(
          0, n0,
          [&](int t) {
            carre_du_champ(mesh, workspace.coords[a], U.col(t), bandwidth,
                           workspace.gamma_x_phi_mat[a].col(t));
          },
          8);
    }

    workspace.laplacian0_weak.setZero(n0, n0);
    workspace.gamma_local.resize(n_verts);
    for (int i = 0; i < n0; ++i) {
      for (int j = i; j < n0; ++j) {
        carre_du_champ(mesh, U.col(i), U.col(j), bandwidth, workspace.gamma_local);
        const float val = (workspace.gamma_local.array() * mu.array()).sum();
        workspace.laplacian0_weak(i, j) = val;
        workspace.laplacian0_weak(j, i) = val;
      }
    }
  }

  workspace.alpha_mat.resize(n0, 3);
  workspace.q.resize(n_verts, 3);
  workspace.weighted_gamma.resize(n_verts, n0);
  workspace.x_op.resize(n0, n0);
  workspace.z_real.resize(n_verts);
  workspace.z_imag.resize(n_verts);
  workspace.prepared_n_verts = n_verts;
  workspace.prepared_n0 = n0;
  workspace.prepared_bandwidth = bandwidth;
}

inline bool hodge_cache_matches(float lhs, float rhs) {
  return std::abs(lhs - rhs) <= 1e-12f;
}

template <typename MeshT>
void ensure_hodge_mixed_cache(const MeshT& mesh, float bandwidth,
                              HodgeWorkspace<MeshT>& workspace) {
  const auto& U = mesh.structure.eigen_basis;
  const auto& mu = mesh.structure.mu;
  const int n0 = U.cols();
  const int n_verts = static_cast<int>(mesh.num_points());

  if (workspace.mixed_ready && workspace.mixed_n_verts == n_verts &&
      workspace.mixed_n0 == n0 && hodge_cache_matches(workspace.mixed_bandwidth, bandwidth)) {
    return;
  }

  fill_coordinate_vectors(mesh, workspace.coords);

  for (int a = 0; a < 3; ++a) {
    workspace.gamma_x_phi_mat[a].resize(n_verts, n0);
    core::parallel_for_index(
        0, n0,
        [&](int i) {
          carre_du_champ(mesh, workspace.coords[a], U.col(i), bandwidth,
                         workspace.gamma_x_phi_mat[a].col(i));
        },
        8);
  }

  workspace.weighted_u.resize(n_verts, n0);
  workspace.weighted_u = U.array().colwise() * mu.array();
  workspace.mixed_n_verts = n_verts;
  workspace.mixed_n0 = n0;
  workspace.mixed_bandwidth = bandwidth;
  workspace.mixed_ready = true;
}

template <typename MeshT>
void ensure_hodge_gamma_xx(const MeshT& mesh, float bandwidth, HodgeWorkspace<MeshT>& workspace) {
  const int n_verts = static_cast<int>(mesh.num_points());
  if (workspace.gamma_xx_ready && workspace.gamma_xx_n_verts == n_verts &&
      hodge_cache_matches(workspace.gamma_xx_bandwidth, bandwidth)) {
    return;
  }

  fill_coordinate_vectors(mesh, workspace.coords);

  for (int a = 0; a < 3; ++a) {
    for (int b = a; b < 3; ++b) {
      workspace.gamma_xx[a][b].resize(n_verts);
      carre_du_champ(mesh, workspace.coords[a], workspace.coords[b], bandwidth,
                     workspace.gamma_xx[a][b]);
      if (a != b) {
        workspace.gamma_xx[b][a] = workspace.gamma_xx[a][b];
      }
    }
  }

  workspace.gamma_xx_n_verts = n_verts;
  workspace.gamma_xx_bandwidth = bandwidth;
  workspace.gamma_xx_ready = true;
}

/**
 * \brief Assemble weak exterior derivative matrix for 1-forms.
 * \param mesh Input diffusion space.
 * \param bandwidth Diffusion bandwidth parameter.
 * \param workspace Scratch buffers.
 * \return Weak derivative matrix.
 */
template <typename MeshT>
Eigen::MatrixXf compute_weak_exterior_derivative(const MeshT& mesh, float bandwidth,
                                                 HodgeWorkspace<MeshT>& workspace) {
  const auto& U = mesh.structure.eigen_basis;
  const int n0 = U.cols();

  Eigen::MatrixXf D = Eigen::MatrixXf::Zero(3 * n0, n0);
  ensure_hodge_mixed_cache(mesh, bandwidth, workspace);

  for (int a = 0; a < 3; ++a) {
    const Eigen::MatrixXf coupling =
        workspace.weighted_u.transpose() * workspace.gamma_x_phi_mat[a];
    for (int k = 0; k < n0; ++k) {
      D.row(k * 3 + a) = coupling.row(k);
    }
  }

  return D;
}

/**
 * \brief Convenience overload for weak exterior derivative assembly.
 * \param mesh Input diffusion space.
 * \param bandwidth Diffusion bandwidth parameter.
 * \return Weak derivative matrix.
 */
template <typename MeshT>
Eigen::MatrixXf compute_weak_exterior_derivative(const MeshT& mesh, float bandwidth) {
  HodgeWorkspace<MeshT> workspace;
  return compute_weak_exterior_derivative(mesh, bandwidth, workspace);
}

/**
 * \brief Assemble curl-energy term used in diffusion Hodge Laplacian.
 * \param mesh Input diffusion space.
 * \param bandwidth Diffusion bandwidth parameter.
 * \param workspace Scratch buffers.
 * \return Curl-energy matrix.
 */
template <typename MeshT>
Eigen::MatrixXf compute_curl_energy_matrix(const MeshT& mesh, float bandwidth,
                                           HodgeWorkspace<MeshT>& workspace) {
  const auto& U = mesh.structure.eigen_basis;
  const auto& mu = mesh.structure.mu;
  const auto mu_arr = mu.array();
  const int n0 = U.cols();
  const int n_basis = 3 * n0;
  const int n_verts = static_cast<int>(mesh.num_points());

  Eigen::MatrixXf E_up = Eigen::MatrixXf::Zero(n_basis, n_basis);
  ensure_hodge_gamma_xx(mesh, bandwidth, workspace);
  ensure_hodge_mixed_cache(mesh, bandwidth, workspace);

  core::parallel_for_index(
      0, n0,
      [&](int k) {
        Eigen::VectorXf gamma_phi_phi_local(n_verts);

        for (int l = k; l < n0; ++l) {
          carre_du_champ(mesh, U.col(k), U.col(l), bandwidth, gamma_phi_phi_local);

          for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
              const float val =
                  (((gamma_phi_phi_local.array() * workspace.gamma_xx[a][b].array()) -
                    (workspace.gamma_x_phi_mat[b].col(k).array() *
                     workspace.gamma_x_phi_mat[a].col(l).array())) *
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

/**
 * \brief Convenience overload for curl-energy assembly.
 * \param mesh Input diffusion space.
 * \param bandwidth Diffusion bandwidth parameter.
 * \return Curl-energy matrix.
 */
template <typename MeshT>
Eigen::MatrixXf compute_curl_energy_matrix(const MeshT& mesh, float bandwidth) {
  HodgeWorkspace<MeshT> workspace;
  return compute_curl_energy_matrix(mesh, bandwidth, workspace);
}

/**
 * \brief Combine down and up terms into a Hodge Laplacian.
 * \param D_weak Weak derivative matrix.
 * \param E_up Curl-energy matrix.
 * \return Hodge Laplacian matrix.
 */
inline Eigen::MatrixXf compute_hodge_laplacian_matrix(const Eigen::MatrixXf& D_weak,
                                                      const Eigen::MatrixXf& E_up) {
  const Eigen::MatrixXf L_down = D_weak * D_weak.transpose();
  return L_down + E_up;
}

/**
 * \brief Solve generalized Hodge eigenproblem `(L, G)` with regularization cutoff.
 * \param laplacian Hodge Laplacian matrix.
 * \param mass_matrix Gram/mass matrix.
 * \param rcond Eigenvalue threshold for regularization.
 * \return Pair `(eigenvalues, eigenvectors)`.
 */
inline std::pair<Eigen::VectorXf, Eigen::MatrixXf>
compute_hodge_spectrum(const Eigen::MatrixXf& laplacian, const Eigen::MatrixXf& mass_matrix,
                       float rcond = 1e-5f) {
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> mass_solver(mass_matrix);
  if (mass_solver.info() != Eigen::Success) {
    return std::make_pair(Eigen::VectorXf(), Eigen::MatrixXf());
  }

  const auto& mass_evals = mass_solver.eigenvalues();
  const auto& mass_evecs = mass_solver.eigenvectors();
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

/**
 * \brief Compute scalar circular coordinates from a harmonic 1-form mode.
 *
 * This solves a generalized eigenproblem for an advection-diffusion operator
 * built from `alpha_coeffs`.
 * \param mesh Input diffusion space.
 * \param alpha_coeffs Harmonic 1-form coefficients.
 * \param bandwidth Diffusion bandwidth parameter.
 * \param lambda Diffusion regularization weight.
 * \param positive_imag_mode Which positive-imaginary mode to choose.
 * \param selected_eval Optional output for selected complex eigenvalue.
 * \return Angle field in `[0, 2*pi)`.
 */
template <typename MeshT>
Eigen::VectorXf compute_circular_coordinates(const MeshT& mesh, const Eigen::VectorXf& alpha_coeffs,
                                             float bandwidth, float lambda,
                                             int positive_imag_mode,
                                             std::complex<float>* selected_eval,
                                             CircularCoordinateWorkspace<MeshT>& workspace) {
  const auto& U = mesh.structure.eigen_basis;
  const auto& mu = mesh.structure.mu;

  const int n0 = U.cols();
  const int n_verts = static_cast<int>(mesh.num_points());

  prepare_circular_coordinate_workspace(mesh, bandwidth, workspace);

  const Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>> alpha_map(
      alpha_coeffs.data(), n0, 3);
  workspace.alpha_mat = alpha_map;

  workspace.q.noalias() = U * workspace.alpha_mat;
  workspace.weighted_gamma =
      (workspace.gamma_x_phi_mat[0].array().colwise() *
       (workspace.q.col(0).array() * mu.array()))
          .matrix();
  workspace.weighted_gamma.array() +=
      workspace.gamma_x_phi_mat[1].array().colwise() *
      (workspace.q.col(1).array() * mu.array());
  workspace.weighted_gamma.array() +=
      workspace.gamma_x_phi_mat[2].array().colwise() *
      (workspace.q.col(2).array() * mu.array());

  workspace.x_op.noalias() = U.transpose() * workspace.weighted_gamma;

  const Eigen::MatrixXf operator_weak = workspace.x_op - (lambda * workspace.laplacian0_weak);
  Eigen::GeneralizedEigenSolver<Eigen::MatrixXf> solver(operator_weak, workspace.function_gram);
  if (solver.info() != Eigen::Success) {
    return Eigen::VectorXf::Zero(n_verts);
  }

  const Eigen::VectorXcf evals = solver.eigenvalues().cast<std::complex<float>>();
  const Eigen::MatrixXcf evecs = solver.eigenvectors().cast<std::complex<float>>();

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

  int positive_imag_count = 0;
  for (int i = 0; i < n0; ++i) {
    if (evals[order[static_cast<size_t>(i)]].imag() > 1e-6f) {
      ++positive_imag_count;
    }
  }
  if (positive_imag_count == 0) {
    return Eigen::VectorXf::Zero(n_verts);
  }

  const int mode = std::clamp(positive_imag_mode, 0, positive_imag_count - 1);
  int selected_src = -1;
  int positive_index = 0;
  for (int i = 0; i < n0; ++i) {
    const int src = order[static_cast<size_t>(i)];
    if (evals[src].imag() <= 1e-6f) {
      continue;
    }
    if (positive_index == mode) {
      selected_src = src;
      break;
    }
    ++positive_index;
  }
  if (selected_src < 0) {
    return Eigen::VectorXf::Zero(n_verts);
  }

  if (selected_eval != nullptr) {
    *selected_eval = evals[selected_src];
  }

  const Eigen::VectorXcf z_coeffs = evecs.col(selected_src);
  workspace.z_real.noalias() = U * z_coeffs.real();
  workspace.z_imag.noalias() = U * z_coeffs.imag();

  Eigen::VectorXf theta(n_verts);
  constexpr float kTwoPi = 6.28318530717958647692f;
  for (int i = 0; i < n_verts; ++i) {
    float angle = std::atan2(workspace.z_imag[i], workspace.z_real[i]);
    if (angle < 0.0f) {
      angle += static_cast<float>(kTwoPi);
    }
    theta[i] = angle;
  }

  return theta;
}

/**
 * \brief Compute scalar circular coordinates from a harmonic 1-form mode.
 *
 * This solves a generalized eigenproblem for an advection-diffusion operator
 * built from `alpha_coeffs`.
 * \param mesh Input diffusion space.
 * \param alpha_coeffs Harmonic 1-form coefficients.
 * \param bandwidth Diffusion bandwidth parameter.
 * \param lambda Diffusion regularization weight.
 * \param positive_imag_mode Which positive-imaginary mode to choose.
 * \param selected_eval Optional output for selected complex eigenvalue.
 * \return Angle field in `[0, 2*pi)`.
 */
template <typename MeshT>
Eigen::VectorXf compute_circular_coordinates(const MeshT& mesh, const Eigen::VectorXf& alpha_coeffs,
                                             float bandwidth, float lambda = 1.0f,
                                             int positive_imag_mode = 0,
                                             std::complex<float>* selected_eval = nullptr) {
  CircularCoordinateWorkspace<MeshT> workspace;
  return compute_circular_coordinates(mesh, alpha_coeffs, bandwidth, lambda, positive_imag_mode,
                                      selected_eval, workspace);
}

} // namespace igneous::ops::diffusion
