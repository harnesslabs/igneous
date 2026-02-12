#pragma once
#include <Eigen/Sparse>
#include <igneous/data/mesh.hpp>
#include <igneous/data/topology.hpp>

using igneous::data::DiffusionTopology;
using igneous::data::Mesh;

template <typename Sig>
std::vector<float>
compute_carre_du_champ(const Mesh<Sig, DiffusionTopology> &mesh,
                       const std::vector<float> &f, const std::vector<float> &h,
                       float t // bandwidth
) {
  size_t n = mesh.geometry.num_points();
  std::vector<float> gamma(n);

  // Iterate over non-zero entries of Sparse Matrix P
  for (int k = 0; k < mesh.topology.P.outerSize(); ++k) {
    for (typename Eigen::SparseMatrix<float>::InnerIterator it(mesh.topology.P,
                                                               k);
         it; ++it) {
      int i = it.row(); // row index
      int j = it.col(); // col index
      float P_ij = it.value();

      float df = f[j] - f[i];
      float dh = h[j] - h[i];

      gamma[i] += P_ij * df * dh;
    }
  }

  // Scale by 1/(2t)
  for (auto &val : gamma)
    val /= (2.0f * t);

  return gamma;
}