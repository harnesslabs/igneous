#pragma once
#include <Eigen/Sparse>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/SparseGenMatProd.h>
#include <cstdlib>
#include <igneous/data/mesh.hpp>
#include <iostream>

namespace igneous::ops {

// Computes the first k eigenvectors of the Markov Chain P.
template <typename MeshT>
void compute_eigenbasis(MeshT &mesh, int n_eigenvectors) {
  const auto &P = mesh.topology.P;

  // 1. Define Operation
  using OpType = Spectra::SparseGenMatProd<float>;
  OpType op(P);

  const bool verbose = std::getenv("IGNEOUS_BENCH_MODE") == nullptr;
  if (verbose) {
    std::cout << "[Spectral] Computing top " << n_eigenvectors
              << " eigenfunctions...\n";
  }

  const int n = static_cast<int>(P.rows());
  const int full_ncv = std::min(n, std::max(2 * n_eigenvectors + 1, 20));
  const bool try_compact = n_eigenvectors >= 32;
  const int compact_ncv =
      try_compact ? std::min(n, std::max(n_eigenvectors + 16, 20)) : full_ncv;

  // 2. Configure Solver
  Spectra::GenEigsSolver<OpType> eigs(op, n_eigenvectors, compact_ncv);
  eigs.init();

  // Compute eigenvalues with largest magnitude (closest to 1.0 for Markov
  // Chain)
  int nconv = eigs.compute(Spectra::SortRule::LargestMagn);

  // Retry with a larger Arnoldi space if compact solve did not fully converge.
  if (try_compact &&
      (eigs.info() != Spectra::CompInfo::Successful || nconv < n_eigenvectors) &&
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

  // 3. Check Success
  if (eigs.info() == Spectra::CompInfo::Successful) {
    // FIX: Use nconv to handle partial convergence
    mesh.topology.eigen_basis = eigs.eigenvectors(nconv).real();

    if (verbose) {
      std::cout << "[Spectral] Converged! Found " << nconv
                << " eigenvectors. Basis shape: "
                << mesh.topology.eigen_basis.rows() << "x"
                << mesh.topology.eigen_basis.cols() << "\n";
    }
  } else {
    if (verbose) {
      std::cerr << "[Spectral] Failed. Info: " << (int)eigs.info() << "\n";
    }
  }
}

} // namespace igneous::ops
