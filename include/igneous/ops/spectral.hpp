#pragma once
#include <Eigen/Sparse>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/SparseGenMatProd.h>
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

// Computes the first k eigenvectors of the Markov Chain P.
template <typename MeshT>
void compute_eigenbasis(MeshT &mesh, int n_eigenvectors) {
  const auto &P = mesh.topology.P;

  const bool verbose = std::getenv("IGNEOUS_BENCH_MODE") == nullptr;
  if (verbose) {
    std::cout << "[Spectral] Computing top " << n_eigenvectors
              << " eigenfunctions...\n";
  }

  const int n = static_cast<int>(P.rows());

  const auto solve_with_op = [&](auto &op) {
    using OpType = std::decay_t<decltype(op)>;

    const int full_ncv = std::min(n, std::max(2 * n_eigenvectors + 1, 20));
    const bool try_compact = n_eigenvectors >= 32;
    const int compact_ncv =
        try_compact ? std::min(n, std::max(n_eigenvectors + 16, 20)) : full_ncv;

    Spectra::GenEigsSolver<OpType> eigs(op, n_eigenvectors, compact_ncv);
    eigs.init();
    int nconv = eigs.compute(Spectra::SortRule::LargestMagn);

    if (try_compact &&
        (eigs.info() != Spectra::CompInfo::Successful ||
         nconv < n_eigenvectors) &&
        full_ncv > compact_ncv) {
      Spectra::GenEigsSolver<OpType> fallback(op, n_eigenvectors, full_ncv);
      fallback.init();
      nconv = fallback.compute(Spectra::SortRule::LargestMagn);

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
    MarkovCsrMatProd op(mesh.topology.markov_row_offsets,
                        mesh.topology.markov_col_indices,
                        mesh.topology.markov_values, n);
    solve_with_op(op);
  } else {
    Spectra::SparseGenMatProd<float> op(P);
    solve_with_op(op);
  }
}

} // namespace igneous::ops
