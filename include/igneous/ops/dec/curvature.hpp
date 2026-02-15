#pragma once

#include <algorithm>
#include <cmath>
#include <numbers>
#include <type_traits>
#include <vector>

#include <igneous/core/blades.hpp>
#include <igneous/core/parallel.hpp>
#include <igneous/data/space.hpp>
#include <igneous/data/structure.hpp>

namespace igneous::ops::dec {

/// \brief Scratch storage reused by curvature evaluation.
template <data::SurfaceStructure StructureT> struct CurvatureWorkspace {
  /// \brief Unnormalized per-face bivector normals.
  std::vector<core::Bivec3> face_normals;
};

/**
 * \brief Compute mean (`H`) and Gaussian (`K`) curvature approximations.
 *
 * Uses angle-defect and neighborhood Laplacian estimates on triangle surfaces.
 * \param space Input surface space.
 * \param H Output mean-curvature-like scalar values per vertex.
 * \param K Output Gaussian-curvature-like scalar values per vertex.
 * \param workspace Reused temporary buffers.
 */
template <data::SurfaceStructure StructureT>
void compute_curvature_measures(const data::Space<StructureT> &space,
                                std::vector<float> &H,
                                std::vector<float> &K,
                                CurvatureWorkspace<StructureT> &workspace) {
  const auto &structure = space.structure;

  const size_t num_verts = space.num_points();
  const size_t num_faces = structure.num_faces();

  H.assign(num_verts, 0.0f);
  K.assign(num_verts, 0.0f);

  workspace.face_normals.resize(num_faces);

  const auto get_face_vertex = [&](size_t face_idx, int corner) -> uint32_t {
    if constexpr (std::is_same_v<StructureT, data::DiscreteExteriorCalculus>) {
      if (corner == 0)
        return structure.face_v0[face_idx];
      if (corner == 1)
        return structure.face_v1[face_idx];
      return structure.face_v2[face_idx];
    }
    return structure.get_vertex_for_face(face_idx, corner);
  };

  core::parallel_for_index(
      0, static_cast<int>(num_faces),
      [&](int face_idx) {
        const size_t f = static_cast<size_t>(face_idx);
        const uint32_t i0 = get_face_vertex(f, 0);
        const uint32_t i1 = get_face_vertex(f, 1);
        const uint32_t i2 = get_face_vertex(f, 2);

        const core::Vec3 p0 = space.get_vec3(i0);
        const core::Vec3 p1 = space.get_vec3(i1);
        const core::Vec3 p2 = space.get_vec3(i2);

        workspace.face_normals[f] = (p1 - p0) ^ (p2 - p0);
      },
      256);

  core::parallel_for_index(
      0, static_cast<int>(num_verts),
      [&](int vertex_idx) {
        const size_t i = static_cast<size_t>(vertex_idx);
        const auto faces = structure.get_faces_for_vertex(static_cast<uint32_t>(i));
        if (faces.empty()) {
          return;
        }

        float angle_sum = 0.0f;
        float area_sum = 0.0f;

        float n_xy = 0.0f;
        float n_yz = 0.0f;
        float n_zx = 0.0f;

        core::Vec3 sum_pos{0.0f, 0.0f, 0.0f};
        float neighbor_count = 0.0f;

        const core::Vec3 p = space.get_vec3(i);

        for (uint32_t f_idx : faces) {
          const core::Bivec3 fn = workspace.face_normals[f_idx];
          n_xy += fn.xy;
          n_yz += fn.yz;
          n_zx += fn.zx;

          const uint32_t i0 = get_face_vertex(f_idx, 0);
          const uint32_t i1 = get_face_vertex(f_idx, 1);
          const uint32_t i2 = get_face_vertex(f_idx, 2);

          core::Vec3 p_a;
          core::Vec3 p_b;
          if (i0 == i) {
            p_a = space.get_vec3(i1);
            p_b = space.get_vec3(i2);
          } else if (i1 == i) {
            p_a = space.get_vec3(i2);
            p_b = space.get_vec3(i0);
          } else {
            p_a = space.get_vec3(i0);
            p_b = space.get_vec3(i1);
          }

          const core::Vec3 u = p_a - p;
          const core::Vec3 v = p_b - p;

          const float dot = u.dot(v);
          const core::Bivec3 wedge = u ^ v;
          const float wedge_mag = wedge.norm();

          angle_sum += std::atan2(wedge_mag, dot);
          area_sum += 0.5f * wedge_mag;

          sum_pos = sum_pos + p_a + p_b;
          neighbor_count += 2.0f;
        }

        if (area_sum > 1e-12f) {
          K[i] = static_cast<float>(
              (2.0 * std::numbers::pi_v<double> - angle_sum) /
              (static_cast<double>(area_sum) / 3.0));
        }

        const float n_mag_sq = n_xy * n_xy + n_yz * n_yz + n_zx * n_zx;
        const float n_inv = (n_mag_sq > 1e-12f) ? 1.0f / std::sqrt(n_mag_sq) : 0.0f;

        const core::Vec3 normal = {n_yz * n_inv, n_zx * n_inv, n_xy * n_inv};

        const float inv_c = 1.0f / std::max(1.0f, neighbor_count);
        const core::Vec3 centroid = sum_pos * inv_c;
        const core::Vec3 laplacian = centroid - p;

        H[i] = laplacian.dot(normal);
      },
      128);
}

} // namespace igneous::ops::dec
