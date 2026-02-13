#pragma once

#include <igneous/core/blades.hpp>
#include <igneous/core/parallel.hpp>
#include <igneous/data/mesh.hpp>
#include <igneous/data/topology.hpp>
#include <type_traits>
#include <vector>

namespace igneous::ops {

template <core::IsSignature Sig, data::SurfaceTopology Topo>
struct FlowWorkspace {
  std::vector<core::Vec3> displacements;
};

template <core::IsSignature Sig, data::SurfaceTopology Topo>
void integrate_mean_curvature_flow(data::Mesh<Sig, Topo> &mesh, float dt,
                                   FlowWorkspace<Sig, Topo> &workspace) {
  auto &geometry = mesh.geometry;
  const auto &topology = mesh.topology;

  const size_t num_verts = geometry.num_points();

  if (workspace.displacements.size() != num_verts) {
    workspace.displacements.resize(num_verts);
  }

  if constexpr (std::is_same_v<Topo, data::TriangleTopology>) {
    const auto &x = geometry.x;
    const auto &y = geometry.y;
    const auto &z = geometry.z;
    const auto &neighbor_offsets = topology.vertex_neighbor_offsets;
    const auto &neighbor_data = topology.vertex_neighbor_data;

    core::parallel_for_index(
        0, static_cast<int>(num_verts),
        [&](int vertex_idx) {
          const size_t i = static_cast<size_t>(vertex_idx);
          const uint32_t begin = neighbor_offsets[i];
          const uint32_t end = neighbor_offsets[i + 1];
          if (begin == end) {
            workspace.displacements[i] = {0.0f, 0.0f, 0.0f};
            return;
          }

          float sx = 0.0f;
          float sy = 0.0f;
          float sz = 0.0f;
          for (uint32_t idx = begin; idx < end; ++idx) {
            const uint32_t n_idx = neighbor_data[idx];
            sx += x[n_idx];
            sy += y[n_idx];
            sz += z[n_idx];
          }

          const float inv_count = 1.0f / static_cast<float>(end - begin);
          workspace.displacements[i] = {sx * inv_count - x[i], sy * inv_count - y[i],
                                        sz * inv_count - z[i]};
        },
        131072);

    core::parallel_for_index(
        0, static_cast<int>(num_verts),
        [&](int vertex_idx) {
          const size_t i = static_cast<size_t>(vertex_idx);
          const core::Vec3 &d = workspace.displacements[i];
          geometry.x[i] += d.x * dt;
          geometry.y[i] += d.y * dt;
          geometry.z[i] += d.z * dt;
        },
        131072);
    return;
  }

  core::parallel_for_index(
      0, static_cast<int>(num_verts),
      [&](int vertex_idx) {
        const size_t i = static_cast<size_t>(vertex_idx);
        const auto neighbors = topology.get_vertex_neighbors(static_cast<uint32_t>(i));
        if (neighbors.empty()) {
          workspace.displacements[i] = {0.0f, 0.0f, 0.0f};
          return;
        }

        core::Vec3 sum_neighbors{0.0f, 0.0f, 0.0f};
        for (uint32_t n_idx : neighbors) {
          sum_neighbors = sum_neighbors + geometry.get_vec3(n_idx);
        }

        const float inv_count = 1.0f / static_cast<float>(neighbors.size());
        const core::Vec3 average_pos = sum_neighbors * inv_count;
        workspace.displacements[i] = average_pos - geometry.get_vec3(i);
      },
      131072);

  core::parallel_for_index(
      0, static_cast<int>(num_verts),
      [&](int vertex_idx) {
        const size_t i = static_cast<size_t>(vertex_idx);
        const core::Vec3 p = geometry.get_vec3(i);
        const core::Vec3 d = workspace.displacements[i];
        geometry.set_vec3(i, p + d * dt);
      },
      131072);
}

} // namespace igneous::ops
