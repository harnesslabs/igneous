#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <igneous/core/algebra.hpp>
#include <igneous/core/parallel.hpp>
#include <igneous/ops/diffusion/basis.hpp>
#include <igneous/ops/diffusion/geometry.hpp>

namespace igneous::ops::diffusion {

/// \brief Shared scratch buffers for diffusion k-form operators.
template <typename MeshT> struct DiffusionFormWorkspace {
  /// \brief Coordinate vectors used for gamma evaluations.
  std::array<Eigen::VectorXf, 3> coords;
  /// \brief `Gamma(x_a, x_b)` caches.
  std::array<std::array<Eigen::VectorXf, 3>, 3> gamma_coords;
  /// \brief `Gamma(x_a, phi_i)` matrix cache per axis.
  std::array<Eigen::MatrixXf, 3> gamma_mixed;
  /// \brief Generic temporary gamma vector.
  Eigen::VectorXf gamma_tmp;
  /// \brief Number of columns currently available in `gamma_mixed`.
  int gamma_mixed_cols = 0;
  /// \brief Whether `gamma_coords` has been initialized.
  bool gamma_coords_ready = false;
};

/// \brief Precomputed compound-minor determinants for metric pullbacks.
struct CompoundDeterminantData {
  /// \brief Basis index combinations for the target degree.
  std::vector<std::vector<int>> indices;
  /// \brief Determinant vectors for each basis-pair minor.
  std::vector<Eigen::VectorXf> values;
  /// \brief Degree (`k`) these determinants correspond to.
  int degree = 0;
  /// \brief Number of basis components for this degree.
  int n_components = 0;
};

/**
 * \brief Ambient coordinate dimension used by current diffusion form operators.
 * \return Ambient dimension (`3`).
 */
inline int ambient_dim_3d() { return 3; }

/**
 * \brief Determinant of a tiny dense matrix packed row-major in `data`.
 * \param data Matrix values in row-major order.
 * \param k Matrix dimension.
 * \return Determinant value.
 */
inline float determinant_small(const float *data, int k) {
  if (k == 0) {
    return 1.0f;
  }
  if (k == 1) {
    return data[0];
  }
  if (k == 2) {
    return data[0] * data[3] - data[1] * data[2];
  }

  // k==3 in this workflow.
  const float a = data[0];
  const float b = data[1];
  const float c = data[2];
  const float d = data[3];
  const float e = data[4];
  const float f = data[5];
  const float g = data[6];
  const float h = data[7];
  const float i = data[8];
  return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
}

/**
 * \brief Build per-point minors of `Gamma(x,x)` for selected row/column subsets.
 * \param gamma_coords Precomputed coordinate gamma tensors.
 * \param rows Row-index subset.
 * \param cols Column-index subset.
 * \return Vector of determinant values per point.
 */
inline Eigen::VectorXf build_minor_det_vector(
    const std::array<std::array<Eigen::VectorXf, 3>, 3> &gamma_coords,
    const std::vector<int> &rows, const std::vector<int> &cols) {
  const int n = gamma_coords[0][0].size();
  const int k = static_cast<int>(rows.size());
  Eigen::VectorXf out(n);
  std::vector<float> tmp(static_cast<size_t>(k * k), 0.0f);

  for (int p = 0; p < n; ++p) {
    for (int r = 0; r < k; ++r) {
      for (int c = 0; c < k; ++c) {
        tmp[static_cast<size_t>(r * k + c)] =
            gamma_coords[rows[static_cast<size_t>(r)]][cols[static_cast<size_t>(c)]][p];
      }
    }
    out[p] = determinant_small(tmp.data(), k);
  }
  return out;
}

/**
 * \brief Ensure `workspace.gamma_coords` is initialized.
 * \param mesh Input diffusion space.
 * \param workspace Scratch workspace to populate.
 */
template <typename MeshT>
void ensure_gamma_coords(const MeshT &mesh, DiffusionFormWorkspace<MeshT> &workspace) {
  if (workspace.gamma_coords_ready) {
    return;
  }

  fill_coordinate_vectors(mesh, workspace.coords);
  const int n = static_cast<int>(mesh.num_points());
  for (int a = 0; a < 3; ++a) {
    for (int b = a; b < 3; ++b) {
      workspace.gamma_coords[a][b].resize(n);
      carre_du_champ(mesh, workspace.coords[a], workspace.coords[b], 0.0f,
                     workspace.gamma_coords[a][b]);
      if (a != b) {
        workspace.gamma_coords[b][a] = workspace.gamma_coords[a][b];
      }
    }
  }
  workspace.gamma_coords_ready = true;
}

/**
 * \brief Ensure mixed coordinate/basis gamma caches have at least `n_coefficients` columns.
 * \param mesh Input diffusion space.
 * \param n_coefficients Required basis column count.
 * \param workspace Scratch workspace to populate.
 */
template <typename MeshT>
void ensure_gamma_mixed(const MeshT &mesh, int n_coefficients,
                        DiffusionFormWorkspace<MeshT> &workspace) {
  const auto &U = mesh.structure.eigen_basis;
  const int n = static_cast<int>(mesh.num_points());
  const int n1 = std::max(1, std::min(n_coefficients, static_cast<int>(U.cols())));

  if (workspace.gamma_mixed_cols >= n1 && workspace.gamma_mixed[0].rows() == n) {
    return;
  }

  fill_coordinate_vectors(mesh, workspace.coords);
  for (int d = 0; d < 3; ++d) {
    workspace.gamma_mixed[d].resize(n, n1);
    core::parallel_for_index(
        0, n1,
        [&](int i) {
          carre_du_champ(mesh, workspace.coords[d], U.col(i), 0.0f,
                         workspace.gamma_mixed[d].col(i));
        },
        8);
  }
  workspace.gamma_mixed_cols = n1;
}

/**
 * \brief Compute all compound determinant vectors for k-form metric pullbacks.
 * \param mesh Input diffusion space.
 * \param k Exterior degree.
 * \param workspace Scratch workspace.
 * \return Precomputed compound determinant data.
 */
template <typename MeshT>
CompoundDeterminantData compute_compound_determinants(
    const MeshT &mesh, int k, DiffusionFormWorkspace<MeshT> &workspace) {
  ensure_gamma_coords(mesh, workspace);

  CompoundDeterminantData out;
  out.degree = k;
  out.indices = get_wedge_basis_indices(ambient_dim_3d(), k);
  if (out.indices.empty()) {
    return out;
  }

  out.n_components = static_cast<int>(out.indices.size());
  out.values.resize(static_cast<size_t>(out.n_components * out.n_components));

  for (int a = 0; a < out.n_components; ++a) {
    for (int b = 0; b < out.n_components; ++b) {
      out.values[static_cast<size_t>(a * out.n_components + b)] =
          build_minor_det_vector(workspace.gamma_coords,
                                 out.indices[static_cast<size_t>(a)],
                                 out.indices[static_cast<size_t>(b)]);
    }
  }

  return out;
}

/**
 * \brief Assemble Gram matrix for k-form basis coefficients.
 * \param mesh Input diffusion space.
 * \param k Exterior degree.
 * \param n_coefficients Basis truncation size.
 * \param workspace Scratch workspace.
 * \return Symmetric Gram matrix for k-forms.
 */
template <typename MeshT>
Eigen::MatrixXf compute_kform_gram_matrix(const MeshT &mesh, int k,
                                          int n_coefficients,
                                          DiffusionFormWorkspace<MeshT> &workspace) {
  const auto &U = mesh.structure.eigen_basis;
  const auto &mu = mesh.structure.mu;
  const int n1 = std::max(1, std::min(n_coefficients, static_cast<int>(U.cols())));
  const auto compounds = compute_compound_determinants(mesh, k, workspace);
  const int Ck = std::max(1, compounds.n_components);
  Eigen::MatrixXf G = Eigen::MatrixXf::Zero(n1 * Ck, n1 * Ck);

  const int n = static_cast<int>(mesh.num_points());
  Eigen::ArrayXf weights(n);

  if (k == 0) {
    for (int i = 0; i < n1; ++i) {
      for (int l = i; l < n1; ++l) {
        const float val =
            (U.col(i).array() * U.col(l).array() * mu.array()).sum();
        G(i, l) = val;
        if (i != l) {
          G(l, i) = val;
        }
      }
    }
    return G;
  }

  for (int i = 0; i < n1; ++i) {
    for (int l = i; l < n1; ++l) {
      weights = U.col(i).array() * U.col(l).array() * mu.array();
      for (int a = 0; a < Ck; ++a) {
        for (int b = 0; b < Ck; ++b) {
          const auto &compound =
              compounds.values[static_cast<size_t>(a * Ck + b)];
          const float val = (weights * compound.array()).sum();
          const int row = i * Ck + a;
          const int col = l * Ck + b;
          G(row, col) = val;
          if (row != col) {
            G(col, row) = val;
          }
        }
      }
    }
  }

  return G;
}

/**
 * \brief Convenience overload for k-form Gram assembly.
 * \param mesh Input diffusion space.
 * \param k Exterior degree.
 * \param n_coefficients Basis truncation size.
 * \return Symmetric Gram matrix for k-forms.
 */
template <typename MeshT>
Eigen::MatrixXf compute_kform_gram_matrix(const MeshT &mesh, int k,
                                          int n_coefficients) {
  DiffusionFormWorkspace<MeshT> workspace;
  return compute_kform_gram_matrix(mesh, k, n_coefficients, workspace);
}

/**
 * \brief Assemble weak exterior derivative matrix `d_k` in coefficient space.
 * \param mesh Input diffusion space.
 * \param k Exterior degree.
 * \param n_coefficients Basis truncation size.
 * \param workspace Scratch workspace.
 * \return Weak derivative matrix.
 */
template <typename MeshT>
Eigen::MatrixXf compute_weak_exterior_derivative(
    const MeshT &mesh, int k, int n_coefficients,
    DiffusionFormWorkspace<MeshT> &workspace) {
  if (k < 0 || k > 2) {
    return Eigen::MatrixXf();
  }

  const auto &U = mesh.structure.eigen_basis;
  const auto &mu = mesh.structure.mu;
  const int n1 = std::max(1, std::min(n_coefficients, static_cast<int>(U.cols())));
  ensure_gamma_mixed(mesh, n1, workspace);

  if (k == 0) {
    const int d = ambient_dim_3d();
    Eigen::MatrixXf D = Eigen::MatrixXf::Zero(n1 * d, n1);
    Eigen::MatrixXf weighted_u = U.leftCols(n1).array().colwise() * mu.array();
    for (int a = 0; a < d; ++a) {
      const Eigen::MatrixXf coupling =
          weighted_u.transpose() * workspace.gamma_mixed[a];
      for (int i = 0; i < n1; ++i) {
        D.row(i * d + a) = coupling.row(i);
      }
    }
    return D;
  }

  const auto kp1 = kp1_children_and_signs(ambient_dim_3d(), k);
  const auto compounds = compute_compound_determinants(mesh, k, workspace);
  const int Ck = static_cast<int>(kp1.idx_k.size());
  const int Ckp1 = static_cast<int>(kp1.idx_kp1.size());

  Eigen::MatrixXf D = Eigen::MatrixXf::Zero(n1 * Ckp1, n1 * Ck);
  const int n = static_cast<int>(mesh.num_points());

  for (int I = 0; I < n1; ++I) {
    for (int i = 0; i < n1; ++i) {
      for (int Jp = 0; Jp < Ckp1; ++Jp) {
        for (int J = 0; J < Ck; ++J) {
          float accum = 0.0f;
          for (int p = 0; p < n; ++p) {
            float laplace_sum = 0.0f;
            for (int r = 0; r < k + 1; ++r) {
              const int coord_dim =
                  kp1.idx_kp1[static_cast<size_t>(Jp)][static_cast<size_t>(r)];
              const int child_idx =
                  kp1.children[static_cast<size_t>(Jp)][static_cast<size_t>(r)];
              const auto &minor_det =
                  compounds.values[static_cast<size_t>(child_idx * Ck + J)];
              laplace_sum +=
                  static_cast<float>(kp1.signs[static_cast<size_t>(r)]) *
                  workspace.gamma_mixed[coord_dim](p, i) * minor_det[p];
            }
            accum += U(p, I) * laplace_sum * mu[p];
          }
          D(I * Ckp1 + Jp, i * Ck + J) = accum;
        }
      }
    }
  }

  return D;
}

/**
 * \brief Convenience overload for weak exterior derivative assembly.
 * \param mesh Input diffusion space.
 * \param k Exterior degree.
 * \param n_coefficients Basis truncation size.
 * \return Weak derivative matrix.
 */
template <typename MeshT>
Eigen::MatrixXf compute_weak_exterior_derivative(const MeshT &mesh, int k,
                                                 int n_coefficients) {
  DiffusionFormWorkspace<MeshT> workspace;
  return compute_weak_exterior_derivative(mesh, k, n_coefficients, workspace);
}

/**
 * \brief Forward declaration of symmetric pseudo-inverse helper.
 * \param matrix Symmetric matrix.
 * \param rcond Relative conditioning threshold.
 * \return Pseudo-inverse matrix.
 */
inline Eigen::MatrixXf pseudo_inverse_symmetric(const Eigen::MatrixXf &matrix,
                                                float rcond);

/**
 * \brief Assemble codifferential matrix `delta_k` from `d_{k-1}` and mass.
 * \param mesh Input diffusion space.
 * \param k Exterior degree.
 * \param n_coefficients Basis truncation size.
 * \param workspace Scratch workspace.
 * \return Codifferential matrix.
 */
template <typename MeshT>
Eigen::MatrixXf compute_codifferential_matrix(
    const MeshT &mesh, int k, int n_coefficients,
    DiffusionFormWorkspace<MeshT> &workspace) {
  if (k <= 0) {
    return Eigen::MatrixXf();
  }
  const Eigen::MatrixXf D_prev =
      compute_weak_exterior_derivative(mesh, k - 1, n_coefficients, workspace);
  const Eigen::MatrixXf G_prev =
      compute_kform_gram_matrix(mesh, k - 1, n_coefficients, workspace);
  const Eigen::MatrixXf G_prev_inv =
      pseudo_inverse_symmetric(G_prev, 1e-5f);
  return G_prev_inv * D_prev.transpose();
}

/**
 * \brief Convenience overload for codifferential assembly.
 * \param mesh Input diffusion space.
 * \param k Exterior degree.
 * \param n_coefficients Basis truncation size.
 * \return Codifferential matrix.
 */
template <typename MeshT>
Eigen::MatrixXf compute_codifferential_matrix(const MeshT &mesh, int k,
                                              int n_coefficients) {
  DiffusionFormWorkspace<MeshT> workspace;
  return compute_codifferential_matrix(mesh, k, n_coefficients, workspace);
}

/**
 * \brief Numerically stable solve for small 2x2 systems with SVD fallback.
 * \param A Input `2x2` matrix.
 * \param b Right-hand side vector.
 * \return Solution vector.
 */
inline Eigen::Vector2f solve_stable_2x2(const Eigen::Matrix2f &A,
                                        const Eigen::Vector2f &b) {
  const float det = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
  if (std::abs(det) > 1e-8f) {
    const float inv_det = 1.0f / det;
    Eigen::Vector2f x;
    x[0] = inv_det * (A(1, 1) * b[0] - A(0, 1) * b[1]);
    x[1] = inv_det * (-A(1, 0) * b[0] + A(0, 0) * b[1]);
    return x;
  }

  Eigen::JacobiSVD<Eigen::Matrix2f> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Eigen::Vector2f s = svd.singularValues();
  Eigen::Vector2f inv_s = Eigen::Vector2f::Zero();
  for (int i = 0; i < 2; ++i) {
    if (s[i] > 1e-8f) {
      inv_s[i] = 1.0f / s[i];
    }
  }
  return svd.matrixV() * inv_s.asDiagonal() * svd.matrixU().transpose() * b;
}

/**
 * \brief Pseudo-inverse for symmetric matrices using eigenvalue thresholding.
 * \param matrix Input matrix.
 * \param rcond Relative threshold for inverting eigenvalues.
 * \return Pseudo-inverse matrix.
 */
inline Eigen::MatrixXf pseudo_inverse_symmetric(const Eigen::MatrixXf &matrix,
                                                float rcond = 1e-5f) {
  if (matrix.rows() == 0 || matrix.cols() == 0) {
    return Eigen::MatrixXf(matrix.cols(), matrix.rows());
  }

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> solver(matrix);
  if (solver.info() != Eigen::Success) {
    return Eigen::MatrixXf::Zero(matrix.cols(), matrix.rows());
  }

  const auto &evals = solver.eigenvalues();
  const auto &evecs = solver.eigenvectors();
  Eigen::VectorXf inv = Eigen::VectorXf::Zero(evals.size());

  const float max_eval = evals.size() > 0 ? evals.cwiseAbs().maxCoeff() : 0.0f;
  const float tol = std::max(rcond * max_eval, rcond);
  for (int i = 0; i < evals.size(); ++i) {
    if (evals[i] > tol) {
      inv[i] = 1.0f / evals[i];
    }
  }

  return evecs * inv.asDiagonal() * evecs.transpose();
}

/**
 * \brief Assemble up-Laplacian contribution for k-forms.
 * \param mesh Input diffusion space.
 * \param k Exterior degree.
 * \param n_coefficients Basis truncation size.
 * \param workspace Scratch workspace.
 * \return Up-Laplacian matrix.
 */
template <typename MeshT>
Eigen::MatrixXf compute_up_laplacian_matrix(const MeshT &mesh, int k,
                                            int n_coefficients,
                                            DiffusionFormWorkspace<MeshT> &workspace) {
  if (k < 0 || k > 2) {
    return Eigen::MatrixXf();
  }

  const auto &U = mesh.structure.eigen_basis;
  const auto &mu = mesh.structure.mu;
  const int n1 = std::max(1, std::min(n_coefficients, static_cast<int>(U.cols())));
  const int n = static_cast<int>(mesh.num_points());
  ensure_gamma_coords(mesh, workspace);
  ensure_gamma_mixed(mesh, n1, workspace);

  if (k == 0) {
    Eigen::MatrixXf L = Eigen::MatrixXf::Zero(n1, n1);
    for (int i = 0; i < n1; ++i) {
      for (int l = i; l < n1; ++l) {
        workspace.gamma_tmp.resize(n);
        carre_du_champ(mesh, U.col(i), U.col(l), 0.0f, workspace.gamma_tmp);
        const float val = (workspace.gamma_tmp.array() * mu.array()).sum();
        L(i, l) = val;
        if (i != l) {
          L(l, i) = val;
        }
      }
    }
    return L;
  }

  const auto compounds = compute_compound_determinants(mesh, k, workspace);
  const int Ck = compounds.n_components;
  Eigen::MatrixXf L = Eigen::MatrixXf::Zero(n1 * Ck, n1 * Ck);

  for (int i = 0; i < n1; ++i) {
    for (int l = i; l < n1; ++l) {
      workspace.gamma_tmp.resize(n);
      carre_du_champ(mesh, U.col(i), U.col(l), 0.0f, workspace.gamma_tmp);

      for (int J = 0; J < Ck; ++J) {
        for (int K = 0; K < Ck; ++K) {
          float val = 0.0f;
          if (k == 1) {
            const auto &gJK = workspace.gamma_coords[J][K];
            const Eigen::ArrayXf term =
                (workspace.gamma_tmp.array() * gJK.array()) -
                (workspace.gamma_mixed[K].col(i).array() *
                 workspace.gamma_mixed[J].col(l).array());
            val = (term * mu.array()).sum();
          } else {
            const auto &row_combo = compounds.indices[static_cast<size_t>(J)];
            const auto &col_combo = compounds.indices[static_cast<size_t>(K)];
            for (int p = 0; p < n; ++p) {
              Eigen::Matrix2f Dm;
              Dm(0, 0) = workspace.gamma_coords[row_combo[0]][col_combo[0]][p];
              Dm(0, 1) = workspace.gamma_coords[row_combo[0]][col_combo[1]][p];
              Dm(1, 0) = workspace.gamma_coords[row_combo[1]][col_combo[0]][p];
              Dm(1, 1) = workspace.gamma_coords[row_combo[1]][col_combo[1]][p];

              const float detD = Dm.determinant();

              Eigen::Vector2f c_vec;
              c_vec[0] = workspace.gamma_mixed[row_combo[0]](p, l);
              c_vec[1] = workspace.gamma_mixed[row_combo[1]](p, l);

              Eigen::Vector2f b_vec;
              b_vec[0] = workspace.gamma_mixed[col_combo[0]](p, i);
              b_vec[1] = workspace.gamma_mixed[col_combo[1]](p, i);

              const Eigen::Vector2f solved = solve_stable_2x2(Dm, c_vec);
              const float bDv = b_vec.dot(solved);
              val += mu[p] * detD * (workspace.gamma_tmp[p] - bDv);
            }
          }

          const int row = i * Ck + J;
          const int col = l * Ck + K;
          L(row, col) = val;
          if (row != col) {
            L(col, row) = val;
          }
        }
      }
    }
  }

  return L;
}

/**
 * \brief Convenience overload for up-Laplacian assembly.
 * \param mesh Input diffusion space.
 * \param k Exterior degree.
 * \param n_coefficients Basis truncation size.
 * \return Up-Laplacian matrix.
 */
template <typename MeshT>
Eigen::MatrixXf compute_up_laplacian_matrix(const MeshT &mesh, int k,
                                            int n_coefficients) {
  DiffusionFormWorkspace<MeshT> workspace;
  return compute_up_laplacian_matrix(mesh, k, n_coefficients, workspace);
}

/**
 * \brief Assemble down-Laplacian contribution for k-forms.
 * \param mesh Input diffusion space.
 * \param k Exterior degree.
 * \param n_coefficients Basis truncation size.
 * \param workspace Scratch workspace.
 * \return Down-Laplacian matrix.
 */
template <typename MeshT>
Eigen::MatrixXf compute_down_laplacian_matrix(
    const MeshT &mesh, int k, int n_coefficients,
    DiffusionFormWorkspace<MeshT> &workspace) {
  if (k <= 0) {
    const int Ck = std::max(1, binomial_coeff(ambient_dim_3d(), std::max(0, k)));
    const int n1 = std::max(1, std::min(n_coefficients, static_cast<int>(mesh.structure.eigen_basis.cols())));
    return Eigen::MatrixXf::Zero(n1 * Ck, n1 * Ck);
  }

  const Eigen::MatrixXf D_prev =
      compute_weak_exterior_derivative(mesh, k - 1, n_coefficients, workspace);
  const Eigen::MatrixXf G_prev =
      compute_kform_gram_matrix(mesh, k - 1, n_coefficients, workspace);
  const Eigen::MatrixXf G_prev_inv =
      pseudo_inverse_symmetric(G_prev, 1e-5f);
  return D_prev * G_prev_inv * D_prev.transpose();
}

/**
 * \brief Convenience overload for down-Laplacian assembly.
 * \param mesh Input diffusion space.
 * \param k Exterior degree.
 * \param n_coefficients Basis truncation size.
 * \return Down-Laplacian matrix.
 */
template <typename MeshT>
Eigen::MatrixXf compute_down_laplacian_matrix(const MeshT &mesh, int k,
                                              int n_coefficients) {
  DiffusionFormWorkspace<MeshT> workspace;
  return compute_down_laplacian_matrix(mesh, k, n_coefficients, workspace);
}

/**
 * \brief Sum up and down contributions into a Hodge Laplacian.
 * \param up Up-Laplacian contribution.
 * \param down Down-Laplacian contribution.
 * \return Combined Hodge Laplacian.
 */
inline Eigen::MatrixXf
assemble_hodge_laplacian_matrix(const Eigen::MatrixXf &up,
                                const Eigen::MatrixXf &down) {
  return up + down;
}

/**
 * \brief Solve generalized eigenproblem for form Laplacian and mass matrix.
 * \param laplacian Form Laplacian matrix.
 * \param mass_matrix Form mass matrix.
 * \param rcond Relative threshold for mass-space regularization.
 * \return Pair `(eigenvalues, eigenvectors)`.
 */
inline std::pair<Eigen::VectorXf, Eigen::MatrixXf>
compute_form_spectrum(const Eigen::MatrixXf &laplacian,
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

  return std::make_pair(solver.eigenvalues(), phi * solver.eigenvectors());
}

/**
 * \brief Extract near-zero eigenvalue indices as harmonic modes.
 * \param evals Eigenvalue vector.
 * \param tolerance Absolute threshold used to classify harmonic modes.
 * \param max_modes Maximum number of modes to return.
 * \return Harmonic mode indices.
 */
inline std::vector<int> extract_harmonic_mode_indices(const Eigen::VectorXf &evals,
                                                      float tolerance = 1e-3f,
                                                      int max_modes = 3) {
  std::vector<int> out;
  for (int i = 0; i < evals.size(); ++i) {
    if (std::abs(evals[i]) <= tolerance) {
      out.push_back(i);
      if (static_cast<int>(out.size()) >= max_modes) {
        break;
      }
    }
  }
  if (out.empty() && evals.size() > 0) {
    const int fallback = std::min(max_modes, static_cast<int>(evals.size()));
    for (int i = 0; i < fallback; ++i) {
      out.push_back(i);
    }
  }
  return out;
}

/**
 * \brief Expand flattened coefficient vector to pointwise k-form components.
 * \param mesh Input diffusion space.
 * \param coeffs Flattened coefficient vector.
 * \param k Exterior degree.
 * \param n_coefficients Basis truncation size.
 * \return Pointwise matrix (`n_points x C(k)`).
 */
template <typename MeshT>
Eigen::MatrixXf coefficients_to_pointwise(const MeshT &mesh,
                                          const Eigen::VectorXf &coeffs, int k,
                                          int n_coefficients) {
  const auto &U = mesh.structure.eigen_basis;
  const int n1 = std::max(1, std::min(n_coefficients, static_cast<int>(U.cols())));
  const int Ck = std::max(1, binomial_coeff(ambient_dim_3d(), k));
  if (coeffs.size() != n1 * Ck) {
    return Eigen::MatrixXf();
  }

  Eigen::Map<const Eigen::MatrixXf> coeff_mat(coeffs.data(), Ck, n1);
  Eigen::MatrixXf pointwise = U.leftCols(n1) * coeff_mat.transpose();
  return pointwise;
}

/**
 * \brief Project pointwise components back to coefficient vectors.
 * \param mesh Input diffusion space.
 * \param pointwise Pointwise form component matrix.
 * \param n_coefficients Basis truncation size.
 * \return Flattened coefficient vector.
 */
template <typename MeshT>
Eigen::VectorXf project_pointwise_to_coefficients(const MeshT &mesh,
                                                  const Eigen::MatrixXf &pointwise,
                                                  int n_coefficients) {
  const auto &U = mesh.structure.eigen_basis;
  const auto &mu = mesh.structure.mu;
  const int n1 = std::max(1, std::min(n_coefficients, static_cast<int>(U.cols())));

  if (pointwise.rows() != U.rows()) {
    return Eigen::VectorXf();
  }

  const Eigen::MatrixXf U1 = U.leftCols(n1);
  const Eigen::MatrixXf gram =
      U1.transpose() * (U1.array().colwise() * mu.array()).matrix();
  Eigen::LDLT<Eigen::MatrixXf> solver(gram);

  Eigen::MatrixXf coeffs(pointwise.cols(), n1);
  for (int c = 0; c < pointwise.cols(); ++c) {
    const Eigen::VectorXf rhs =
        U1.transpose() * (pointwise.col(c).array() * mu.array()).matrix();
    coeffs.row(c) = solver.solve(rhs).transpose();
  }

  Eigen::VectorXf flattened(coeffs.size());
  Eigen::Map<Eigen::VectorXf>(flattened.data(), flattened.size()) =
      Eigen::Map<const Eigen::VectorXf>(coeffs.data(), coeffs.size());
  return flattened;
}

/**
 * \brief Convert ambient 2-form components `(w01,w02,w12)` to dual 3D vectors.
 * \param pointwise_2form Pointwise 2-form matrix.
 * \return Per-point dual vector field.
 */
inline std::vector<core::Vec3>
pointwise_2form_to_dual_vectors(const Eigen::MatrixXf &pointwise_2form) {
  std::vector<core::Vec3> out(static_cast<size_t>(pointwise_2form.rows()),
                              {0.0f, 0.0f, 0.0f});
  if (pointwise_2form.cols() < 3) {
    return out;
  }

  for (int p = 0; p < pointwise_2form.rows(); ++p) {
    const float w01 = pointwise_2form(p, 0);
    const float w02 = pointwise_2form(p, 1);
    const float w12 = pointwise_2form(p, 2);
    out[static_cast<size_t>(p)] = {w12, -w02, w01};
  }
  return out;
}

} // namespace igneous::ops::diffusion
