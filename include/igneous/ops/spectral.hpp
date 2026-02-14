#pragma once

#include <Eigen/Sparse>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/SparseGenMatProd.h>
#include <Spectra/SymEigsSolver.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <igneous/core/parallel.hpp>
#include <igneous/data/mesh.hpp>
#include <iostream>
#include <limits>
#include <type_traits>
#include <unordered_map>

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

struct SpectralSolveOptions {
  data::SpectralSolveMode mode = data::SpectralSolveMode::Auto;
  int symmetric_min_rows = 2200;
  float reversibility_mean_tol = 1e-6f;
  float reversibility_max_tol = 1e-4f;
};

struct ReversibilityStats {
  float detailed_balance_mean = 0.0f;
  float detailed_balance_max = 0.0f;
  float transformed_asymmetry_mean = 0.0f;
  float transformed_asymmetry_max = 0.0f;
};

[[nodiscard]] inline const char *to_string(data::SpectralSolveMode mode) {
  switch (mode) {
  case data::SpectralSolveMode::Auto:
    return "auto";
  case data::SpectralSolveMode::GenericArnoldi:
    return "generic";
  case data::SpectralSolveMode::SymmetricTransform:
    return "symmetric";
  }
  return "unknown";
}

[[nodiscard]] inline bool reversibility_passes(const ReversibilityStats &stats,
                                               const SpectralSolveOptions &options) {
  return stats.detailed_balance_mean <= options.reversibility_mean_tol &&
         stats.detailed_balance_max <= options.reversibility_max_tol &&
         stats.transformed_asymmetry_mean <= options.reversibility_mean_tol &&
         stats.transformed_asymmetry_max <= options.reversibility_max_tol;
}

inline ReversibilityStats
compute_reversibility_stats(const std::vector<int> &row_offsets,
                            const std::vector<int> &col_indices,
                            const std::vector<float> &values,
                            const Eigen::VectorXf &mu) {
  ReversibilityStats stats;
  const int n = static_cast<int>(row_offsets.size()) - 1;
  if (n <= 0 || mu.size() != n) {
    return stats;
  }

  auto edge_key = [](int i, int j) -> uint64_t {
    return (static_cast<uint64_t>(static_cast<uint32_t>(i)) << 32) |
           static_cast<uint32_t>(j);
  };

  std::unordered_map<uint64_t, float> edge_weights;
  edge_weights.reserve(values.size() * 2);
  for (int i = 0; i < n; ++i) {
    const int begin = row_offsets[static_cast<size_t>(i)];
    const int end = row_offsets[static_cast<size_t>(i) + 1];
    for (int idx = begin; idx < end; ++idx) {
      const int j = col_indices[static_cast<size_t>(idx)];
      edge_weights.emplace(edge_key(i, j), values[static_cast<size_t>(idx)]);
    }
  }

  const Eigen::ArrayXf safe_mu = mu.array().max(1e-12f);
  const Eigen::ArrayXf sqrt_mu = safe_mu.sqrt();
  const Eigen::ArrayXf inv_sqrt_mu = sqrt_mu.inverse();

  double db_sum = 0.0;
  double sym_sum = 0.0;
  float db_max = 0.0f;
  float sym_max = 0.0f;
  int samples = 0;

  for (int i = 0; i < n; ++i) {
    const int begin = row_offsets[static_cast<size_t>(i)];
    const int end = row_offsets[static_cast<size_t>(i) + 1];
    for (int idx = begin; idx < end; ++idx) {
      const int j = col_indices[static_cast<size_t>(idx)];
      const float pij = values[static_cast<size_t>(idx)];

      const auto rev_it = edge_weights.find(edge_key(j, i));
      const float pji = (rev_it == edge_weights.end()) ? 0.0f : rev_it->second;

      const float db = std::abs(mu[i] * pij - mu[j] * pji);
      db_sum += db;
      db_max = std::max(db_max, db);

      const float sij = sqrt_mu[i] * pij * inv_sqrt_mu[j];
      const float sji = sqrt_mu[j] * pji * inv_sqrt_mu[i];
      const float sym = std::abs(sij - sji);
      sym_sum += sym;
      sym_max = std::max(sym_max, sym);
      ++samples;
    }
  }

  if (samples > 0) {
    const float inv_samples = 1.0f / static_cast<float>(samples);
    stats.detailed_balance_mean = static_cast<float>(db_sum) * inv_samples;
    stats.transformed_asymmetry_mean = static_cast<float>(sym_sum) * inv_samples;
  }
  stats.detailed_balance_max = db_max;
  stats.transformed_asymmetry_max = sym_max;
  return stats;
}

template <typename MeshT>
void compute_eigenbasis(MeshT &mesh, int n_eigenvectors,
                        const SpectralSolveOptions &options) {
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

  if (n <= 0 || n_eigenvectors <= 0) {
    if constexpr (requires { mesh.topology.eigen_basis; }) {
      mesh.topology.eigen_basis.resize(0, 0);
    }
    if constexpr (requires { mesh.topology.eigen_values; }) {
      mesh.topology.eigen_values.resize(0);
    }
    return;
  }

  const auto assign_diagnostics = [&](data::SpectralSolveDiagnostics diagnostics) {
    if constexpr (requires { mesh.topology.spectral_diagnostics; }) {
      mesh.topology.spectral_diagnostics = diagnostics;
    }
  };

  auto diagnostics = data::SpectralSolveDiagnostics{};
  diagnostics.requested_mode = options.mode;

  const auto store_success = [&](const Eigen::MatrixXf &basis,
                                 const Eigen::VectorXf &evals,
                                 data::SpectralSolveMode used_mode,
                                 int nconv) {
    if constexpr (requires { mesh.topology.eigen_basis; }) {
      mesh.topology.eigen_basis = basis;
    }
    if constexpr (requires { mesh.topology.eigen_values; }) {
      mesh.topology.eigen_values = evals;
    }
    diagnostics.used_mode = used_mode;
    diagnostics.nconv = nconv;

    if (verbose) {
      std::cout << "[Spectral] Converged! Found " << nconv
                << " eigenvectors. Basis shape: " << basis.rows() << "x"
                << basis.cols() << " using " << to_string(used_mode) << " solve\n";
    }
    assign_diagnostics(diagnostics);
  };

  const auto solve_generic = [&](auto &op,
                                 data::SpectralSolveMode used_mode) -> bool {
    using OpType = std::decay_t<decltype(op)>;

    const int full_ncv = std::min(n, std::max(2 * n_eigenvectors + 1, 20));
    const bool try_compact = n_eigenvectors >= 32;
    const int compact_ncv =
        try_compact ? std::min(n, std::max(n_eigenvectors + 16, 20)) : full_ncv;

    Spectra::GenEigsSolver<OpType> eigs(op, n_eigenvectors, compact_ncv);
    eigs.init();
    int nconv = eigs.compute(Spectra::SortRule::LargestReal);

    if (try_compact &&
        (eigs.info() != Spectra::CompInfo::Successful || nconv < n_eigenvectors) &&
        full_ncv > compact_ncv) {
      Spectra::GenEigsSolver<OpType> fallback(op, n_eigenvectors, full_ncv);
      fallback.init();
      nconv = fallback.compute(Spectra::SortRule::LargestReal);
      diagnostics.used_fallback = true;

      if (fallback.info() == Spectra::CompInfo::Successful && nconv > 0) {
        store_success(fallback.eigenvectors(nconv).real(),
                      fallback.eigenvalues().real(), used_mode, nconv);
        return true;
      }
    }

    if (eigs.info() == Spectra::CompInfo::Successful && nconv > 0) {
      store_success(eigs.eigenvectors(nconv).real(), eigs.eigenvalues().real(),
                    used_mode, nconv);
      return true;
    }

    return false;
  };

  if constexpr (requires {
                  mesh.topology.markov_row_offsets;
                  mesh.topology.markov_col_indices;
                  mesh.topology.markov_values;
                }) {
    const auto &row_offsets = mesh.topology.markov_row_offsets;
    const auto &col_indices = mesh.topology.markov_col_indices;
    const auto &weights = mesh.topology.markov_values;
    const auto &mu = mesh.topology.mu;

    ReversibilityStats stats{};
    if (options.mode != data::SpectralSolveMode::GenericArnoldi) {
      stats = compute_reversibility_stats(row_offsets, col_indices, weights, mu);
    }

    diagnostics.detailed_balance_mean = stats.detailed_balance_mean;
    diagnostics.detailed_balance_max = stats.detailed_balance_max;
    diagnostics.transformed_asymmetry_mean = stats.transformed_asymmetry_mean;
    diagnostics.transformed_asymmetry_max = stats.transformed_asymmetry_max;
    diagnostics.reversibility_pass = reversibility_passes(stats, options);

    bool try_symmetric = false;
    if (options.mode == data::SpectralSolveMode::SymmetricTransform) {
      try_symmetric = true;
    } else if (options.mode == data::SpectralSolveMode::Auto) {
      try_symmetric = n >= options.symmetric_min_rows &&
                      diagnostics.reversibility_pass;
    }

    if (verbose) {
      std::cout << "[Spectral] Requested mode: " << to_string(options.mode)
                << ", reversibility pass: "
                << (diagnostics.reversibility_pass ? "yes" : "no")
                << " (db mean/max: " << diagnostics.detailed_balance_mean << "/"
                << diagnostics.detailed_balance_max << ", sym mean/max: "
                << diagnostics.transformed_asymmetry_mean << "/"
                << diagnostics.transformed_asymmetry_max << ")\n";
    }

    if (try_symmetric) {
      Eigen::VectorXf sqrt_mu(n);
      Eigen::VectorXf inv_sqrt_mu(n);
      for (int i = 0; i < n; ++i) {
        const float mu_i = std::max(mu[i], 1e-12f);
        const float sqrt_mu_i = std::sqrt(mu_i);
        sqrt_mu[i] = sqrt_mu_i;
        inv_sqrt_mu[i] = 1.0f / sqrt_mu_i;
      }

      MarkovSymmetricCsrMatProd sym_op(row_offsets, col_indices, weights, sqrt_mu,
                                       inv_sqrt_mu, n);

      const int full_ncv = std::min(n, std::max(2 * n_eigenvectors + 1, 20));
      const bool try_compact = n_eigenvectors >= 32;
      const int compact_ncv =
          try_compact ? std::min(n, std::max(n_eigenvectors + 16, 20)) : full_ncv;

      const auto solve_symmetric = [&](int ncv) -> bool {
        Spectra::SymEigsSolver<MarkovSymmetricCsrMatProd> eigs(sym_op,
                                                               n_eigenvectors, ncv);
        eigs.init();
        const int nconv = eigs.compute(Spectra::SortRule::LargestAlge);
        if (eigs.info() != Spectra::CompInfo::Successful || nconv <= 0) {
          return false;
        }

        Eigen::MatrixXf basis = eigs.eigenvectors(nconv);
        basis = basis.array().colwise() * inv_sqrt_mu.array();
        store_success(basis, eigs.eigenvalues(),
                      data::SpectralSolveMode::SymmetricTransform, nconv);
        return true;
      };

      if ((try_compact && solve_symmetric(compact_ncv)) ||
          solve_symmetric(full_ncv)) {
        return;
      }

      diagnostics.used_fallback = true;
      if (verbose) {
        std::cerr
            << "[Spectral] Symmetric solve failed; falling back to generic solve.\n";
      }
    }

    MarkovCsrMatProd generic_op(row_offsets, col_indices, weights, n);
    if (solve_generic(generic_op, data::SpectralSolveMode::GenericArnoldi)) {
      return;
    }

    if (verbose) {
      std::cerr << "[Spectral] Failed to compute eigenbasis.\n";
    }
    assign_diagnostics(diagnostics);
  } else if constexpr (requires { mesh.topology.P; }) {
    const auto &P = mesh.topology.P;
    Spectra::SparseGenMatProd<float> op(P);
    if (solve_generic(op, data::SpectralSolveMode::GenericArnoldi)) {
      return;
    }
    if (verbose) {
      std::cerr << "[Spectral] Failed to compute eigenbasis.\n";
    }
  } else {
    static_assert(
        requires { mesh.topology.P; },
        "compute_eigenbasis requires markov CSR arrays or sparse matrix P.");
  }
}

template <typename MeshT>
void compute_eigenbasis(MeshT &mesh, int n_eigenvectors) {
  compute_eigenbasis(mesh, n_eigenvectors, SpectralSolveOptions{});
}

} // namespace igneous::ops
