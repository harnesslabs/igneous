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

  // 2. Configure Solver
  // Arnoldi vectors (ncv) must be > n_eigenvectors
  int ncv = std::max(2 * n_eigenvectors + 1, 20);
  Spectra::GenEigsSolver<OpType> eigs(op, n_eigenvectors, ncv);

  const bool verbose = std::getenv("IGNEOUS_BENCH_MODE") == nullptr;
  if (verbose) {
    std::cout << "[Spectral] Computing top " << n_eigenvectors
              << " eigenfunctions...\n";
  }

  eigs.init();

  // Compute eigenvalues with largest magnitude (closest to 1.0 for Markov
  // Chain)
  int nconv = eigs.compute(Spectra::SortRule::LargestMagn);

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
