// include/igneous/ops/transform.hpp
#pragma once
#include "igneous/core/algebra.hpp"
#include <igneous/data/mesh.hpp>
#include <iostream>

namespace igneous::ops {

using igneous::core::Multivector;
using igneous::data::Mesh;

template <typename Sig> void normalize(Mesh<Sig> &mesh) {
  if (mesh.geometry.points.empty())
    return;

  using Field = float; // Assuming float for now

  // 1. Compute Bounding Box
  auto min_p = mesh.geometry.points[0];
  auto max_p = mesh.geometry.points[0];

  for (const auto &p : mesh.geometry.points) {
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

  // 3. Apply
  for (auto &p : mesh.geometry.points) {
    auto temp = p - center;
    for (int k = 1; k <= 3; ++k) {
      p[k] = temp[k] * scale;
    }
  }
}

} // namespace igneous::ops