#pragma once
#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

#include <igneous/core/algebra.hpp>
#include <igneous/core/blades.hpp>
#include <igneous/data/mesh.hpp>

namespace igneous::ops {

using igneous::core::Bivec3;
using igneous::core::IsSignature;
using igneous::core::Vec3;
using igneous::data::Mesh;

template <IsSignature Sig>
std::pair<std::vector<double>, std::vector<double>>
compute_curvature_measures(const Mesh<Sig> &mesh) {
  const auto &geometry = mesh.geometry;
  const auto &topology = mesh.topology;

  size_t num_verts = geometry.num_points();
  size_t num_faces = topology.num_faces();

  // Outputs
  std::vector<double> H(num_verts, 0.0);
  std::vector<double> K(num_verts, 0.0);

  // 1. FACE NORMALS (Blade Optimized)
  // Store raw Bivectors (12 bytes) instead of full Multivectors (32 bytes).
  // This reduces this buffer from ~64MB to ~24MB for 1M faces.
  std::vector<Bivec3> face_normals(num_faces);

  // Linear pass -> Auto-Vectorization friendly
  for (size_t f = 0; f < num_faces; ++f) {
    // Fetch as Vec3 (12 bytes, no padding)
    Vec3 p0 = geometry.get_vec3(topology.get_vertex_for_face(f, 0));
    Vec3 p1 = geometry.get_vec3(topology.get_vertex_for_face(f, 1));
    Vec3 p2 = geometry.get_vec3(topology.get_vertex_for_face(f, 2));

    // Wedge Product: (p1-p0) ^ (p2-p0) -> Bivec3
    face_normals[f] = (p1 - p0) ^ (p2 - p0);
  }

  // 2. VERTEX GATHER
  for (size_t i = 0; i < num_verts; ++i) {
    auto faces = topology.get_faces_for_vertex(i);
    if (faces.empty())
      continue;

    // Accumulators (Registers)
    double angle_sum = 0.0;
    double area_sum = 0.0;

    // Normal Accumulation (Dual Components)
    float n_xy = 0.0f;
    float n_yz = 0.0f;
    float n_zx = 0.0f;

    // Centroid Accumulation
    Vec3 sum_pos = {0.0f, 0.0f, 0.0f};
    double neighbor_count = 0.0;

    const Vec3 P = geometry.get_vec3(i);

    for (uint32_t f_idx : faces) {
      // A. Accumulate Normal
      Bivec3 fn = face_normals[f_idx];
      n_xy += fn.xy;
      n_yz += fn.yz;
      n_zx += fn.zx;

      // B. Topology Lookup
      uint32_t i0 = topology.get_vertex_for_face(f_idx, 0);
      uint32_t i1 = topology.get_vertex_for_face(f_idx, 1);
      uint32_t i2 = topology.get_vertex_for_face(f_idx, 2);

      // Identify neighbors relative to P
      Vec3 p_a, p_b;
      if (i0 == i) {
        p_a = geometry.get_vec3(i1);
        p_b = geometry.get_vec3(i2);
      } else if (i1 == i) {
        p_a = geometry.get_vec3(i2);
        p_b = geometry.get_vec3(i0);
      } else {
        p_a = geometry.get_vec3(i0);
        p_b = geometry.get_vec3(i1);
      }

      // C. Angle Deficit
      Vec3 u = p_a - P;
      Vec3 v = p_b - P;

      float dot = u.dot(v);
      Bivec3 wedge = u ^ v;
      float wedge_mag = wedge.norm();

      angle_sum += std::atan2(wedge_mag, dot);
      area_sum += 0.5 * wedge_mag;

      // D. Centroid
      sum_pos = sum_pos + p_a + p_b;
      neighbor_count += 2.0;
    }

    // 3. COMPUTE METRICS

    // Gaussian K
    if (area_sum > 1e-12) {
      K[i] = (2.0 * std::numbers::pi - angle_sum) / (area_sum / 3.0);
    }

    // Mean H
    // Dual Map: xy->z, yz->x, zx->y
    float n_mag_sq = n_xy * n_xy + n_yz * n_yz + n_zx * n_zx;
    float n_inv = (n_mag_sq > 1e-12f) ? 1.0f / std::sqrt(n_mag_sq) : 0.0f;

    Vec3 normal = {
        n_yz * n_inv, // x
        n_zx * n_inv, // y
        n_xy * n_inv  // z
    };

    float inv_c = 1.0f / std::max(1.0, neighbor_count);
    Vec3 centroid = sum_pos * inv_c;
    Vec3 laplacian = centroid - P;

    H[i] = laplacian.dot(normal);
  }

  return {H, K};
}

} // namespace igneous::ops