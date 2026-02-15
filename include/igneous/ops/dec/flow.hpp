#pragma once

#include <type_traits>
#include <vector>

#include <igneous/core/blades.hpp>
#include <igneous/core/parallel.hpp>
#include <igneous/data/space.hpp>
#include <igneous/data/structure.hpp>

namespace igneous::ops::dec {

template <data::SurfaceStructure StructureT> struct FlowWorkspace {
  std::vector<core::Vec3> displacements;
};

template <data::SurfaceStructure StructureT>
void integrate_mean_curvature_flow(data::Space<StructureT> &space, float dt,
                                   FlowWorkspace<StructureT> &workspace) {
  const auto &structure = space.structure;

  const size_t num_verts = space.num_points();

  if (workspace.displacements.size() != num_verts) {
    workspace.displacements.resize(num_verts);
  }

  if constexpr (std::is_same_v<StructureT, data::DiscreteExteriorCalculus>) {
    const auto &neighbor_offsets = structure.vertex_neighbor_offsets;
    const auto &neighbor_data = structure.vertex_neighbor_data;

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
            sx += space.x[n_idx];
            sy += space.y[n_idx];
            sz += space.z[n_idx];
          }

          const float inv_count = 1.0f / static_cast<float>(end - begin);
          workspace.displacements[i] = {sx * inv_count - space.x[i],
                                        sy * inv_count - space.y[i],
                                        sz * inv_count - space.z[i]};
        },
        131072);

    core::parallel_for_index(
        0, static_cast<int>(num_verts),
        [&](int vertex_idx) {
          const size_t i = static_cast<size_t>(vertex_idx);
          const core::Vec3 &d = workspace.displacements[i];
          space.x[i] += d.x * dt;
          space.y[i] += d.y * dt;
          space.z[i] += d.z * dt;
        },
        131072);
    return;
  }

  core::parallel_for_index(
      0, static_cast<int>(num_verts),
      [&](int vertex_idx) {
        const size_t i = static_cast<size_t>(vertex_idx);
        const auto neighbors = structure.get_vertex_neighbors(static_cast<uint32_t>(i));
        if (neighbors.empty()) {
          workspace.displacements[i] = {0.0f, 0.0f, 0.0f};
          return;
        }

        core::Vec3 sum_neighbors{0.0f, 0.0f, 0.0f};
        for (uint32_t n_idx : neighbors) {
          sum_neighbors = sum_neighbors + space.get_vec3(n_idx);
        }

        const float inv_count = 1.0f / static_cast<float>(neighbors.size());
        const core::Vec3 average_pos = sum_neighbors * inv_count;
        workspace.displacements[i] = average_pos - space.get_vec3(i);
      },
      131072);

  core::parallel_for_index(
      0, static_cast<int>(num_verts),
      [&](int vertex_idx) {
        const size_t i = static_cast<size_t>(vertex_idx);
        const core::Vec3 p = space.get_vec3(i);
        const core::Vec3 d = workspace.displacements[i];
        space.set_vec3(i, p + d * dt);
      },
      131072);
}

} // namespace igneous::ops::dec
