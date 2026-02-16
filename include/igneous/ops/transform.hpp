#pragma once

#include <algorithm>
#include <cstddef>
#include <igneous/core/blades.hpp>
#include <igneous/data/space.hpp>
#include <limits>

namespace igneous::ops {

/**
 * \brief Normalize geometry to a unit box centered at the origin.
 *
 * The largest axis extent is mapped to length `2.0`.
 * \param mesh Input/output space geometry.
 */
template <typename StructureT> void normalize(data::Space<StructureT>& mesh) {
  const size_t n_verts = mesh.num_points();
  if (n_verts == 0) {
    return;
  }

  core::Vec3 min_p = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                      std::numeric_limits<float>::max()};
  core::Vec3 max_p = {std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
                      std::numeric_limits<float>::lowest()};

  for (size_t i = 0; i < n_verts; ++i) {
    const core::Vec3 p = mesh.get_vec3(i);
    min_p.x = std::min(min_p.x, p.x);
    min_p.y = std::min(min_p.y, p.y);
    min_p.z = std::min(min_p.z, p.z);
    max_p.x = std::max(max_p.x, p.x);
    max_p.y = std::max(max_p.y, p.y);
    max_p.z = std::max(max_p.z, p.z);
  }

  const core::Vec3 center = (min_p + max_p) * 0.5f;
  const core::Vec3 dim = max_p - min_p;

  const float max_dim = std::max({dim.x, dim.y, dim.z});
  const float scale_scalar = (max_dim > 1e-9f) ? (2.0f / max_dim) : 1.0f;

  for (size_t i = 0; i < n_verts; ++i) {
    const core::Vec3 p = mesh.get_vec3(i);
    mesh.set_vec3(i, (p - center) * scale_scalar);
  }
}

} // namespace igneous::ops
