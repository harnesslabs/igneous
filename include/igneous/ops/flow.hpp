#pragma once
#include <algorithm> // for std::max
#include <igneous/core/blades.hpp>
#include <igneous/data/mesh.hpp>
#include <vector>

namespace igneous::ops {

using igneous::core::IsSignature;
using igneous::core::Vec3;
using igneous::data::Mesh;

template <IsSignature Sig>
void integrate_mean_curvature_flow(Mesh<Sig> &mesh, double dt) {
  // Use Vec3 (12 bytes) instead of Multivector (32 bytes)
  // This reduces the 'displacements' buffer size by ~60%
  // and makes the math operations pure SIMD candidates.

  auto &geometry = mesh.geometry;
  const auto &topology = mesh.topology;

  size_t num_verts = geometry.num_points();

  // 1. COMPUTE FLOW
  // Allocation: 1M verts * 12 bytes = 12MB (vs 32MB before)
  // This fits much better in L3 Cache.
  std::vector<Vec3> displacements(num_verts);

  for (size_t i = 0; i < num_verts; ++i) {
    auto faces = topology.get_faces_for_vertex(i);
    if (faces.empty())
      continue; // displacements[i] is already {0,0,0}

    Vec3 sum_neighbors = {0.0f, 0.0f, 0.0f};
    double count = 0.0;

    for (uint32_t f_idx : faces) {
      uint32_t i0 = topology.get_vertex_for_face(f_idx, 0);
      uint32_t i1 = topology.get_vertex_for_face(f_idx, 1);
      uint32_t i2 = topology.get_vertex_for_face(f_idx, 2);

      // Fast Blade Access (No padding read)
      if (i0 != i)
        sum_neighbors = sum_neighbors + geometry.get_vec3(i0);
      if (i1 != i)
        sum_neighbors = sum_neighbors + geometry.get_vec3(i1);
      if (i2 != i)
        sum_neighbors = sum_neighbors + geometry.get_vec3(i2);

      count += 2.0;
    }

    float inv_c = 1.0f / std::max(1.0, count);
    Vec3 average_pos = sum_neighbors * inv_c;

    displacements[i] = average_pos - geometry.get_vec3(i);
  }

  // 2. INTEGRATE
  // This loop is perfectly linear and trivial for the compiler to vectorise.
  // It loads 12 bytes, adds 12 bytes, writes 12 bytes.
  float dt_f = (float)dt;

  for (size_t i = 0; i < num_verts; ++i) {
    Vec3 p = geometry.get_vec3(i);
    Vec3 d = displacements[i];

    // Position Update: P += D * dt
    geometry.set_vec3(i, p + d * dt_f);
  }
}

} // namespace igneous::ops