#pragma once
#include <igneous/core/algebra.hpp>
#include <igneous/data/mesh.hpp>
#include <iostream>

namespace igneous::ops {

using igneous::core::Multivector;
using igneous::data::Mesh;

template <typename Sig> void normalize(Mesh<Sig> &mesh) {
  // UPDATED: Check num_points() instead of .points.empty()
  if (mesh.geometry.num_points() == 0)
    return;

  using Field = float; // Assuming float for now
  size_t n_verts = mesh.geometry.num_points();

  // 1. Compute Bounding Box
  // UPDATED: Use get_point(0)
  auto min_p = mesh.geometry.get_point(0);
  auto max_p = mesh.geometry.get_point(0);

  // UPDATED: Iterate using index
  for (size_t i = 0; i < n_verts; ++i) {
    auto p = mesh.geometry.get_point(i);
    for (int k = 1; k <= 3; ++k) {
      if (p[k] < min_p[k])
        min_p[k] = p[k];
      if (p[k] > max_p[k])
        max_p[k] = p[k];
    }
  }

  // 2. Center and Scale
  Multivector<Field, Sig> center;
  float max_dim = 0.0f;
  for (int k = 1; k <= 3; ++k) {
    center[k] = (min_p[k] + max_p[k]) * 0.5f;
    float dim = max_p[k] - min_p[k];
    if (dim > max_dim)
      max_dim = dim;
  }

  float scale = (max_dim > 1e-9f) ? (2.0f / max_dim) : 1.0f;

  std::cout << "[Transform] Centered and scaled (x" << scale << ")\n";

  // 3. Apply Transform
  for (size_t i = 0; i < n_verts; ++i) {
    // UPDATED: Read -> Modify -> Write pattern
    auto p = mesh.geometry.get_point(i);
    auto temp = p - center;

    // Apply scale component-wise
    for (int k = 1; k <= 3; ++k) {
      p[k] = temp[k] * scale;
    }

    // Write back to packed storage
    mesh.geometry.set_point(i, p);
  }
}

} // namespace igneous::ops