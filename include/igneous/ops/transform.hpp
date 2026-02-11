#pragma once
#include <algorithm>
#include <igneous/core/algebra.hpp>
#include <igneous/core/blades.hpp>
#include <igneous/data/mesh.hpp>
#include <iostream>
#include <limits>

namespace igneous::ops {

using igneous::core::Vec3;
using igneous::data::Mesh;

template <typename Sig> void normalize(Mesh<Sig> &mesh) {
  size_t n_verts = mesh.geometry.num_points();
  if (n_verts == 0)
    return;

  // 1. Compute Bounds (Safely using Vec3)
  Vec3 min_p = {std::numeric_limits<float>::max(),
                std::numeric_limits<float>::max(),
                std::numeric_limits<float>::max()};
  Vec3 max_p = {std::numeric_limits<float>::lowest(),
                std::numeric_limits<float>::lowest(),
                std::numeric_limits<float>::lowest()};

  for (size_t i = 0; i < n_verts; ++i) {
    Vec3 p = mesh.geometry.get_vec3(i);

    if (p.x < min_p.x)
      min_p.x = p.x;
    if (p.x > max_p.x)
      max_p.x = p.x;
    if (p.y < min_p.y)
      min_p.y = p.y;
    if (p.y > max_p.y)
      max_p.y = p.y;
    if (p.z < min_p.z)
      min_p.z = p.z;
    if (p.z > max_p.z)
      max_p.z = p.z;
  }

  // 2. Compute Center and Scale
  Vec3 center = (min_p + max_p) * 0.5f;
  Vec3 dim = max_p - min_p;

  float max_dim = std::max({dim.x, dim.y, dim.z});
  float scale_scalar = (max_dim > 1e-9f) ? (2.0f / max_dim) : 1.0f;

  std::cout << "[Transform] Bounds Z: " << min_p.z << " to " << max_p.z << "\n";
  std::cout << "[Transform] Centered and scaled (x" << scale_scalar << ")\n";

  // 3. Apply Transform
  for (size_t i = 0; i < n_verts; ++i) {
    Vec3 p = mesh.geometry.get_vec3(i);
    Vec3 p_new = (p - center) * scale_scalar;
    mesh.geometry.set_vec3(i, p_new);
  }
}

} // namespace igneous::ops