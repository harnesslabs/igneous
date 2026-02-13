#pragma once
#include <Eigen/Dense>
#include <igneous/core/algebra.hpp>
#include <igneous/data/mesh.hpp>
#include <vector>

namespace igneous::ops {

// The Fundamental Operator: Gamma(f, h) [cite: 162]
// Computes the "inner product of gradients" at every point.
// Returns a dense vector of size N.
template <typename MeshT>
Eigen::VectorXf carre_du_champ(const MeshT &mesh, const Eigen::VectorXf &f,
                               const Eigen::VectorXf &h,
                               float bandwidth // This is rho^2 or t
) {
  // Formula: (1/2t) * sum P_ij (df)(dh)
  // We can compute this efficiently using the sparse P

  // 1. Mean-center the functions relative to neighbors (optional but robust
  // [cite: 240]) Or use the simpler graph laplacian form: L(fh) - fL(h) -
  // hL(f). Let's use the explicit sparse iteration for clarity (Eq 3 in paper).

  Eigen::VectorXf gamma = Eigen::VectorXf::Zero(mesh.geometry.num_points());
  const auto &P = mesh.topology.P;

  // Parallelize this loop in production!
  for (int k = 0; k < P.outerSize(); ++k) {
    for (typename Eigen::SparseMatrix<float>::InnerIterator it(P, k); it;
         ++it) {
      int i = it.row();
      int j = it.col();
      float P_ij = it.value();

      float df = f[j] - f[i];
      float dh = h[j] - h[i];

      gamma[i] += P_ij * df * dh;
    }
  }

  return gamma / (2.0f * bandwidth);
}

// Compute the Gram Matrix G^(1) for 1-forms
// This is the "Mass Matrix" of our basis vector fields.
// The basis is { phi_i * dx_c } where c is {0,1,2} (x,y,z).
// Size: (n_basis * 3) x (n_basis * 3)
template <typename MeshT>
Eigen::MatrixXf compute_1form_gram_matrix(const MeshT &mesh, float bandwidth) {
  int n_basis = mesh.topology.eigen_basis.cols();
  int dim = 3; // 3D ambient space
  int n_total = n_basis * dim;

  Eigen::MatrixXf G = Eigen::MatrixXf::Zero(n_total, n_total);

  // Pre-convert coordinate functions to Eigen vectors for easy math
  std::vector<Eigen::VectorXf> coords(3);
  size_t n_verts = mesh.geometry.num_points();
  for (int d = 0; d < 3; ++d)
    coords[d].resize(n_verts);

  for (size_t i = 0; i < n_verts; ++i) {
    auto p = mesh.geometry.get_vec3(i);
    coords[0][i] = p.x;
    coords[1][i] = p.y;
    coords[2][i] = p.z;
  }

  // Compute G_IJ = sum_p mu_p * (phi_i * phi_i') * Gamma(x_j, x_j')
  // This is the integral of the metric over the manifold

  std::cout << "[Geometry] Building Gram Matrix G (" << n_total << "x"
            << n_total << ")...\n";

  // 1. Precompute Gamma(x_a, x_b) for a,b in {0,1,2}
  // These are the "components of the inverse metric tensor" in ambient space.
  Eigen::VectorXf gamma_coords[3][3];
  for (int a = 0; a < 3; ++a) {
    for (int b = a; b < 3; ++b) {
      gamma_coords[a][b] =
          carre_du_champ(mesh, coords[a], coords[b], bandwidth);
      if (a != b)
        gamma_coords[b][a] = gamma_coords[a][b]; // Symmetric
    }
  }

  // 2. Assemble G
  // Nested loops: Basis functions (i, k) and Dimensions (a, b)
  const auto &U = mesh.topology.eigen_basis;
  const auto &mu = mesh.topology.mu;

  for (int i = 0; i < n_basis; ++i) {
    for (int k = i; k < n_basis; ++k) {
      // Precompute product of basis functions weighted by measure
      // weight_p = U_pi * U_pk * mu_p
      Eigen::VectorXf weights =
          U.col(i).cwiseProduct(U.col(k)).cwiseProduct(mu);

      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          // Integrate: sum(weight * Gamma(x_a, x_b))
          float val = weights.dot(gamma_coords[a][b]);

          int row = i * 3 + a;
          int col = k * 3 + b;

          G(row, col) = val;
          if (row != col)
            G(col, row) = val;
        }
      }
    }
  }

  return G;
}

} // namespace igneous::ops