#pragma once
#include <Eigen/Sparse>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseGenMatProd.h>
#include <cmath>
#include <cstdlib>
#include <igneous/core/parallel.hpp>
#include <igneous/data/mesh.hpp>
#include <iostream>
#include <type_traits>

namespace igneous::ops {

class MarkovCsrMatProd {
public:
  using Scalar = float;

  MarkovCsrMatProd(const std::vector<int> &row_offsets,
                   const std::vector<int> &col_indices,
                   const std::vector<float> &values, int n)
      : row_offsets_(row_offsets), col_indices_(col_indices), values_(values),
        n_(n) {}

  [[nodiscard]] int rows() const { return n_; }
  [[nodiscard]] int cols() const { return n_; }

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
  const std::vector<int> &row_offsets_;
  const std::vector<int> &col_indices_;
  const std::vector<float> &values_;
  int n_ = 0;
};

class MarkovSymmetricCsrMatProd {
public:
  using Scalar = float;

  MarkovSymmetricCsrMatProd(const std::vector<int> &row_offsets,
                            const std::vector<int> &col_indices,
                            const std::vector<float> &values,
                            const Eigen::VectorXf &sqrt_mu,
                            const Eigen::VectorXf &inv_sqrt_mu, int n)
      : row_offsets_(row_offsets), col_indices_(col_indices), values_(values),
        sqrt_mu_(sqrt_mu), inv_sqrt_mu_(inv_sqrt_mu), n_(n) {}

  [[nodiscard]] int rows() const { return n_; }
  [[nodiscard]] int cols() const { return n_; }

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
  const std::vector<int> &row_offsets_;
  const std::vector<int> &col_indices_;
  const std::vector<float> &values_;
  const Eigen::VectorXf &sqrt_mu_;
  const Eigen::VectorXf &inv_sqrt_mu_;
  int n_ = 0;
};

// Computes the first k eigenvectors of the Markov Chain P.
template <typename MeshT>
void compute_eigenbasis(MeshT &mesh, int n_eigenvectors) {
  const bool verbose = std::getenv("IGNEOUS_BENCH_MODE") == nullptr;
  if (verbose) {
    std::cout << "[Spectral] Computing top " << n_eigenvectors
              << " eigenfunctions...\n";
  }

  const int n = [&]() {
    if constexpr (requires { mesh.topology.markov_row_offsets; }) {
      if (mesh.topology.markov_row_offsets.empty()) {
        return 0;
      }
      return static_cast<int>(mesh.topology.markov_row_offsets.size() - 1);
    } else if constexpr (requires { mesh.topology.P; }) {
      return static_cast<int>(mesh.topology.P.rows());
    } else {
      return 0;
    }
  }();

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
        mesh.topology.eigen_basis = fallback.eigenvectors(nconv).real();
        if (verbose) {
          std::cout << "[Spectral] Converged! Found " << nconv
                    << " eigenvectors. Basis shape: "
                    << mesh.topology.eigen_basis.rows() << "x"
                    << mesh.topology.eigen_basis.cols() << "\n";
        }
        return;
      }
    }

    if (eigs.info() == Spectra::CompInfo::Successful) {
      mesh.topology.eigen_basis = eigs.eigenvectors(nconv).real();

      if (verbose) {
        std::cout << "[Spectral] Converged! Found " << nconv
                  << " eigenvectors. Basis shape: "
                  << mesh.topology.eigen_basis.rows() << "x"
                  << mesh.topology.eigen_basis.cols() << "\n";
      }
      return;
    }

    if (verbose) {
      std::cerr << "[Spectral] Failed. Info: " << (int)eigs.info() << "\n";
    }
  };

  if constexpr (requires {
                  mesh.topology.markov_row_offsets;
                  mesh.topology.markov_col_indices;
                  mesh.topology.markov_values;
                }) {
    const bool use_symmetric_solver = n >= 2200;
    if (use_symmetric_solver) {
      Eigen::VectorXf sqrt_mu(n);
      Eigen::VectorXf inv_sqrt_mu(n);
      for (int i = 0; i < n; ++i) {
        const float mu_i = std::max(mesh.topology.mu[i], 1e-12f);
        const float sqrt_mu_i = std::sqrt(mu_i);
        sqrt_mu[i] = sqrt_mu_i;
        inv_sqrt_mu[i] = 1.0f / sqrt_mu_i;
      }

      MarkovSymmetricCsrMatProd sym_op(mesh.topology.markov_row_offsets,
                                       mesh.topology.markov_col_indices,
                                       mesh.topology.markov_values, sqrt_mu,
                                       inv_sqrt_mu, n);

      const int full_ncv = std::min(n, std::max(2 * n_eigenvectors + 1, 20));
      const bool try_compact = n_eigenvectors >= 32;
      const int compact_ncv =
          try_compact ? std::min(n, std::max(n_eigenvectors + 16, 20)) : full_ncv;

      const auto solve_symmetric = [&](int ncv) -> bool {
        Spectra::SymEigsSolver<MarkovSymmetricCsrMatProd> eigs(
            sym_op, n_eigenvectors, ncv);
        eigs.init();
        const int nconv = eigs.compute(Spectra::SortRule::LargestAlge);
        if (eigs.info() != Spectra::CompInfo::Successful || nconv <= 0) {
          return false;
        }

        mesh.topology.eigen_basis = eigs.eigenvectors(nconv);
        mesh.topology.eigen_basis =
            mesh.topology.eigen_basis.array().colwise() * inv_sqrt_mu.array();

        if (verbose) {
          std::cout << "[Spectral] Converged! Found " << nconv
                    << " eigenvectors. Basis shape: "
                    << mesh.topology.eigen_basis.rows() << "x"
                    << mesh.topology.eigen_basis.cols() << "\n";
        }
        return true;
      };

      if ((try_compact && solve_symmetric(compact_ncv)) ||
          solve_symmetric(full_ncv)) {
        return;
      }

      if (verbose) {
        std::cerr
            << "[Spectral] Symmetric solve failed, falling back to generic solver.\n";
      }
    }

    MarkovCsrMatProd fallback_op(mesh.topology.markov_row_offsets,
                                 mesh.topology.markov_col_indices,
                                 mesh.topology.markov_values, n);
    solve_with_op(fallback_op);
  } else if constexpr (requires { mesh.topology.P; }) {
    const auto &P = mesh.topology.P;
    Spectra::SparseGenMatProd<float> op(P);
    solve_with_op(op);
  } else {
    static_assert(
        requires { mesh.topology.P; },
        "compute_eigenbasis requires markov CSR arrays or sparse matrix P.");
  }
}

} // namespace igneous::ops
