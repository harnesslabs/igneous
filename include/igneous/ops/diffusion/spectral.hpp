#pragma once
#include <Eigen/Sparse>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/SparseGenMatProd.h>
#include <Spectra/SymEigsSolver.h>
#include <cmath>
#include <cstdlib>
#include <igneous/core/parallel.hpp>
#include <igneous/data/space.hpp>
#include <iostream>
#include <type_traits>

namespace igneous::ops::diffusion {

/**
 * \brief Spectra-compatible matrix product wrapper over Markov CSR data.
 *
 * Implements the matrix-operation interface expected by Spectra.
 */
class MarkovCsrMatProd {
public:
  using Scalar = float;

  /// \brief Construct operator view over CSR arrays.
  MarkovCsrMatProd(const std::vector<int> &row_offsets,
                   const std::vector<int> &col_indices,
                   const std::vector<float> &values, int n)
      : row_offsets_(row_offsets), col_indices_(col_indices), values_(values),
        n_(n) {}

  /// \return Matrix row count.
  [[nodiscard]] int rows() const { return n_; }
  /// \return Matrix column count.
  [[nodiscard]] int cols() const { return n_; }

  /**
   * \brief Apply matrix-vector product.
   * \param x_in Input vector.
   * \param y_out Output vector.
   */
  void perform_op(const Scalar *x_in, Scalar *y_out) const {
    core::parallel_for_index(
        0, n_,
        [&](int i) {
          const int begin = row_offsets_[static_cast<size_t>(i)];
          const int end = row_offsets_[static_cast<size_t>(i) + 1];
          const int count = end - begin;
          const int *cols = col_indices_.data() + begin;
          const float *vals = values_.data() + begin;

          float acc = 0.0f;
          int k = 0;
          for (; k + 3 < count; k += 4) {
            acc += vals[k + 0] * x_in[cols[k + 0]];
            acc += vals[k + 1] * x_in[cols[k + 1]];
            acc += vals[k + 2] * x_in[cols[k + 2]];
            acc += vals[k + 3] * x_in[cols[k + 3]];
          }
          for (; k < count; ++k) {
            acc += vals[k] * x_in[cols[k]];
          }
          y_out[i] = acc;
        },
        8192);
  }

private:
  /// \brief CSR row offsets.
  const std::vector<int> &row_offsets_;
  /// \brief CSR column indices.
  const std::vector<int> &col_indices_;
  /// \brief CSR values.
  const std::vector<float> &values_;
  /// \brief Matrix dimension.
  int n_ = 0;
};

/// \brief Spectra operator for the symmetrized Markov transform.
class MarkovSymmetricCsrMatProd {
public:
  using Scalar = float;

  /// \brief Construct normalized symmetric Markov operator.
  MarkovSymmetricCsrMatProd(const std::vector<int> &row_offsets,
                            const std::vector<int> &col_indices,
                            const std::vector<float> &values,
                            const Eigen::VectorXf &sqrt_mu,
                            const Eigen::VectorXf &inv_sqrt_mu, int n)
      : row_offsets_(row_offsets), col_indices_(col_indices), values_(values),
        sqrt_mu_(sqrt_mu), inv_sqrt_mu_(inv_sqrt_mu), n_(n) {}

  /// \return Matrix row count.
  [[nodiscard]] int rows() const { return n_; }
  /// \return Matrix column count.
  [[nodiscard]] int cols() const { return n_; }

  /**
   * \brief Apply matrix-vector product.
   * \param x_in Input vector.
   * \param y_out Output vector.
   */
  void perform_op(const Scalar *x_in, Scalar *y_out) const {
    const float *sqrt_mu_data = sqrt_mu_.data();
    const float *inv_sqrt_mu_data = inv_sqrt_mu_.data();

    core::parallel_for_index(
        0, n_,
        [&](int i) {
          const int begin = row_offsets_[static_cast<size_t>(i)];
          const int end = row_offsets_[static_cast<size_t>(i) + 1];
          const int count = end - begin;
          const int *cols = col_indices_.data() + begin;
          const float *vals = values_.data() + begin;

          float acc = 0.0f;
          int k = 0;
          for (; k + 3 < count; k += 4) {
            const int j0 = cols[k + 0];
            const int j1 = cols[k + 1];
            const int j2 = cols[k + 2];
            const int j3 = cols[k + 3];
            acc += vals[k + 0] * x_in[j0] * inv_sqrt_mu_data[j0];
            acc += vals[k + 1] * x_in[j1] * inv_sqrt_mu_data[j1];
            acc += vals[k + 2] * x_in[j2] * inv_sqrt_mu_data[j2];
            acc += vals[k + 3] * x_in[j3] * inv_sqrt_mu_data[j3];
          }
          for (; k < count; ++k) {
            const int j = cols[k];
            acc += vals[k] * x_in[j] * inv_sqrt_mu_data[j];
          }
          y_out[i] = sqrt_mu_data[i] * acc;
        },
        8192);
  }

private:
  /// \brief CSR row offsets.
  const std::vector<int> &row_offsets_;
  /// \brief CSR column indices.
  const std::vector<int> &col_indices_;
  /// \brief CSR values.
  const std::vector<float> &values_;
  /// \brief `sqrt(mu)` scaling.
  const Eigen::VectorXf &sqrt_mu_;
  /// \brief `1/sqrt(mu)` scaling.
  const Eigen::VectorXf &inv_sqrt_mu_;
  /// \brief Matrix dimension.
  int n_ = 0;
};

/// \brief Spectra operator over the symmetric kernel CSR matrix.
class SymmetricKernelCsrMatProd {
public:
  using Scalar = float;

  /// \brief Construct operator view over symmetric-kernel CSR arrays.
  SymmetricKernelCsrMatProd(const std::vector<int> &row_offsets,
                            const std::vector<int> &col_indices,
                            const std::vector<float> &values, int n)
      : row_offsets_(row_offsets), col_indices_(col_indices), values_(values),
        n_(n) {}

  /// \return Matrix row count.
  [[nodiscard]] int rows() const { return n_; }
  /// \return Matrix column count.
  [[nodiscard]] int cols() const { return n_; }

  /**
   * \brief Apply matrix-vector product.
   * \param x_in Input vector.
   * \param y_out Output vector.
   */
  void perform_op(const Scalar *x_in, Scalar *y_out) const {
    core::parallel_for_index(
        0, n_,
        [&](int i) {
          const int begin = row_offsets_[static_cast<size_t>(i)];
          const int end = row_offsets_[static_cast<size_t>(i) + 1];
          const int count = end - begin;
          const int *cols = col_indices_.data() + begin;
          const float *vals = values_.data() + begin;

          float acc = 0.0f;
          int k = 0;
          for (; k + 3 < count; k += 4) {
            acc += vals[k + 0] * x_in[cols[k + 0]];
            acc += vals[k + 1] * x_in[cols[k + 1]];
            acc += vals[k + 2] * x_in[cols[k + 2]];
            acc += vals[k + 3] * x_in[cols[k + 3]];
          }
          for (; k < count; ++k) {
            acc += vals[k] * x_in[cols[k]];
          }
          y_out[i] = acc;
        },
        8192);
  }

private:
  /// \brief CSR row offsets.
  const std::vector<int> &row_offsets_;
  /// \brief CSR column indices.
  const std::vector<int> &col_indices_;
  /// \brief CSR values.
  const std::vector<float> &values_;
  /// \brief Matrix dimension.
  int n_ = 0;
};

/// \brief Spectra operator for normalized symmetric-kernel eigensolve.
class NormalizedSymmetricKernelCsrMatProd {
public:
  using Scalar = float;

  /// \brief Construct normalized symmetric-kernel operator.
  NormalizedSymmetricKernelCsrMatProd(const std::vector<int> &row_offsets,
                                      const std::vector<int> &col_indices,
                                      const std::vector<float> &values,
                                      const Eigen::VectorXf &inv_sqrt_rows,
                                      int n)
      : row_offsets_(row_offsets), col_indices_(col_indices), values_(values),
        inv_sqrt_rows_(inv_sqrt_rows), n_(n) {}

  /// \return Matrix row count.
  [[nodiscard]] int rows() const { return n_; }
  /// \return Matrix column count.
  [[nodiscard]] int cols() const { return n_; }

  /**
   * \brief Apply matrix-vector product.
   * \param x_in Input vector.
   * \param y_out Output vector.
   */
  void perform_op(const Scalar *x_in, Scalar *y_out) const {
    const float *inv_sqrt_data = inv_sqrt_rows_.data();
    core::parallel_for_index(
        0, n_,
        [&](int i) {
          const int begin = row_offsets_[static_cast<size_t>(i)];
          const int end = row_offsets_[static_cast<size_t>(i) + 1];
          const int count = end - begin;
          const int *cols = col_indices_.data() + begin;
          const float *vals = values_.data() + begin;

          float acc = 0.0f;
          int k = 0;
          for (; k + 3 < count; k += 4) {
            const int j0 = cols[k + 0];
            const int j1 = cols[k + 1];
            const int j2 = cols[k + 2];
            const int j3 = cols[k + 3];
            acc += vals[k + 0] * (x_in[j0] * inv_sqrt_data[j0]);
            acc += vals[k + 1] * (x_in[j1] * inv_sqrt_data[j1]);
            acc += vals[k + 2] * (x_in[j2] * inv_sqrt_data[j2]);
            acc += vals[k + 3] * (x_in[j3] * inv_sqrt_data[j3]);
          }
          for (; k < count; ++k) {
            const int j = cols[k];
            acc += vals[k] * (x_in[j] * inv_sqrt_data[j]);
          }
          y_out[i] = inv_sqrt_data[i] * acc;
        },
        8192);
  }

private:
  /// \brief CSR row offsets.
  const std::vector<int> &row_offsets_;
  /// \brief CSR column indices.
  const std::vector<int> &col_indices_;
  /// \brief CSR values.
  const std::vector<float> &values_;
  /// \brief Inverse square-root row scaling.
  const Eigen::VectorXf &inv_sqrt_rows_;
  /// \brief Matrix dimension.
  int n_ = 0;
};

/**
 * \brief Compute a diffusion eigenbasis from Markov CSR data.
 * \param mesh Input diffusion space. `mesh.structure.eigen_basis` is overwritten.
 * \param n_eigenvectors Number of eigenvectors requested.
 */
template <typename MeshT>
void compute_eigenbasis(MeshT &mesh, int n_eigenvectors) {
  const bool verbose = std::getenv("IGNEOUS_BENCH_MODE") == nullptr;
  if (verbose) {
    std::cout << "[Spectral] Computing top " << n_eigenvectors
              << " eigenfunctions...\n";
  }

  static_assert(requires {
                  mesh.structure.markov_row_offsets;
                  mesh.structure.markov_col_indices;
                  mesh.structure.markov_values;
                },
                "compute_eigenbasis requires markov CSR arrays.");

  const int n = mesh.structure.markov_row_offsets.empty()
                    ? 0
                    : static_cast<int>(mesh.structure.markov_row_offsets.size() - 1);

  const auto solve_with_op = [&](auto &op) {
    using OpType = std::decay_t<decltype(op)>;

    const int full_ncv = std::min(n, std::max(2 * n_eigenvectors + 1, 20));
    const bool try_compact = n_eigenvectors >= 32;
    const int compact_ncv =
        try_compact ? std::min(n, std::max(n_eigenvectors + 16, 20)) : full_ncv;

    Spectra::GenEigsSolver<OpType> eigs(op, n_eigenvectors, compact_ncv);
    eigs.init();
    int nconv = eigs.compute(Spectra::SortRule::LargestReal);

    if (try_compact &&
        (eigs.info() != Spectra::CompInfo::Successful ||
         nconv < n_eigenvectors) &&
        full_ncv > compact_ncv) {
      Spectra::GenEigsSolver<OpType> fallback(op, n_eigenvectors, full_ncv);
      fallback.init();
      nconv = fallback.compute(Spectra::SortRule::LargestReal);

      if (fallback.info() == Spectra::CompInfo::Successful) {
        mesh.structure.eigen_basis = fallback.eigenvectors(nconv).real();
        if (verbose) {
          std::cout << "[Spectral] Converged! Found " << nconv
                    << " eigenvectors. Basis shape: "
                    << mesh.structure.eigen_basis.rows() << "x"
                    << mesh.structure.eigen_basis.cols() << "\n";
        }
        return;
      }
    }

    if (eigs.info() == Spectra::CompInfo::Successful) {
      mesh.structure.eigen_basis = eigs.eigenvectors(nconv).real();

      if (verbose) {
        std::cout << "[Spectral] Converged! Found " << nconv
                  << " eigenvectors. Basis shape: "
                  << mesh.structure.eigen_basis.rows() << "x"
                  << mesh.structure.eigen_basis.cols() << "\n";
      }
      return;
    }

    if (verbose) {
      std::cerr << "[Spectral] Failed. Info: " << (int)eigs.info() << "\n";
    }
  };

  if (n <= 1) {
    mesh.structure.eigen_basis =
        Eigen::MatrixXf::Ones(std::max(1, n), std::max(1, n));
    return;
  }

  if constexpr (requires {
                  mesh.structure.symmetric_row_offsets;
                  mesh.structure.symmetric_col_indices;
                  mesh.structure.symmetric_values;
                  mesh.structure.symmetric_row_sums;
                }) {
    const auto &sym_row_offsets = mesh.structure.symmetric_row_offsets;
    const auto &sym_col_indices = mesh.structure.symmetric_col_indices;
    const auto &sym_values = mesh.structure.symmetric_values;
    const auto &row_sums = mesh.structure.symmetric_row_sums;
    if (!sym_row_offsets.empty() &&
        static_cast<int>(sym_row_offsets.size()) == n + 1 &&
        row_sums.size() == n) {
      const int k_eval = std::max(1, std::min(n_eigenvectors, std::max(1, n - 1)));
      const int full_ncv = std::min(n, std::max(2 * k_eval + 1, 20));
      const bool try_compact = k_eval >= 32;
      const int compact_ncv =
          try_compact ? std::min(n, std::max(k_eval + 16, 20)) : full_ncv;

      Eigen::VectorXf inv_sqrt_rows(n);
      for (int i = 0; i < n; ++i) {
        inv_sqrt_rows[i] = 1.0f / std::sqrt(std::max(row_sums[i], 1e-12f));
      }

      NormalizedSymmetricKernelCsrMatProd op(sym_row_offsets, sym_col_indices,
                                             sym_values, inv_sqrt_rows, n);
      const auto solve_symmetric = [&](int ncv) -> bool {
        Spectra::SymEigsSolver<NormalizedSymmetricKernelCsrMatProd> eigs(op,
                                                                          k_eval,
                                                                          ncv);
        eigs.init();
        const int nconv = eigs.compute(Spectra::SortRule::LargestMagn);
        if (eigs.info() != Spectra::CompInfo::Successful || nconv <= 0) {
          return false;
        }

        Eigen::MatrixXf basis = eigs.eigenvectors(nconv);
        basis = basis.array().colwise() * inv_sqrt_rows.array();
        if (std::abs(basis(0, 0)) > 1e-12f) {
          basis /= basis(0, 0);
        }
        mesh.structure.eigen_basis = basis;

        if (verbose) {
          std::cout << "[Spectral] Converged! Found " << nconv
                    << " eigenvectors. Basis shape: "
                    << mesh.structure.eigen_basis.rows() << "x"
                    << mesh.structure.eigen_basis.cols() << "\n";
        }
        return true;
      };

      if ((try_compact && solve_symmetric(compact_ncv)) ||
          solve_symmetric(full_ncv)) {
        return;
      }

      if (verbose) {
        std::cerr << "[Spectral] Symmetric-kernel solve failed, falling back to "
                     "generic solver.\n";
      }
    }
  }

  MarkovCsrMatProd fallback_op(mesh.structure.markov_row_offsets,
                               mesh.structure.markov_col_indices,
                               mesh.structure.markov_values, n);
  solve_with_op(fallback_op);
}

} // namespace igneous::ops::diffusion
