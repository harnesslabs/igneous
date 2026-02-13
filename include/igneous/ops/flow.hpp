#pragma once

#include <igneous/core/blades.hpp>
#include <igneous/data/mesh.hpp>
#include <igneous/data/topology.hpp>
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

  workspace.displacements.assign(num_verts, {0.0f, 0.0f, 0.0f});

  for (size_t i = 0; i < num_verts; ++i) {
    const auto neighbors = topology.get_vertex_neighbors(static_cast<uint32_t>(i));
    if (neighbors.empty()) {
      continue;
    }

    core::Vec3 sum_neighbors{0.0f, 0.0f, 0.0f};
    for (uint32_t n_idx : neighbors) {
      sum_neighbors = sum_neighbors + geometry.get_vec3(n_idx);
    }

    const float inv_count = 1.0f / static_cast<float>(neighbors.size());
    const core::Vec3 average_pos = sum_neighbors * inv_count;
    workspace.displacements[i] = average_pos - geometry.get_vec3(i);
  }

  for (size_t i = 0; i < num_verts; ++i) {
    const core::Vec3 p = geometry.get_vec3(i);
    const core::Vec3 d = workspace.displacements[i];
    geometry.set_vec3(i, p + d * dt);
  }
}

} // namespace igneous::ops
