// include/igneous/ops/flow.hpp
#pragma once
#include <algorithm> // for std::max
#include <igneous/data/mesh.hpp>
#include <vector>

namespace igneous::ops {

using igneous::core::IsSignature;
using igneous::core::Multivector;
using igneous::data::Mesh;

template <IsSignature Sig>
void integrate_mean_curvature_flow(Mesh<Sig> &mesh, double dt) {
  // Define Field to match the Mesh's storage type (float)
  using Field = float;

  auto &geometry = mesh.geometry;
  const auto &topology = mesh.topology; // Topology is read-only here

  // UPDATED: Use accessor .num_points()
  size_t num_verts = geometry.num_points();

  // 1. Compute Update Vectors (The "Flow")
  // We store these in a temporary buffer so we don't read "future" data
  // while calculating "current" step.
  std::vector<Multivector<Field, Sig>> displacements(num_verts);

  for (size_t i = 0; i < num_verts; ++i) {
    auto faces = topology.get_faces_for_vertex(i);
    if (faces.empty())
      continue;

    Multivector<Field, Sig> sum_neighbors; // Zero init
    double count = 0.0;

    // Iterate Ring
    for (uint32_t f_idx : faces) {
      uint32_t i0 = topology.get_vertex_for_face(f_idx, 0);
      uint32_t i1 = topology.get_vertex_for_face(f_idx, 1);
      uint32_t i2 = topology.get_vertex_for_face(f_idx, 2);

      // UPDATED: Use get_point instead of direct array access
      if (i0 != i)
        sum_neighbors = sum_neighbors + geometry.get_point(i0);
      if (i1 != i)
        sum_neighbors = sum_neighbors + geometry.get_point(i1);
      if (i2 != i)
        sum_neighbors = sum_neighbors + geometry.get_point(i2);

      count += 2.0;
    }

    // Calculate Laplacian Vector
    Multivector<Field, Sig> average_pos = sum_neighbors;
    double inv_c = 1.0 / std::max(1.0, count);
    for (size_t k = 0; k < Sig::size; ++k)
      average_pos[k] *= (float)inv_c;

    // UPDATED: Use get_point
    displacements[i] = average_pos - geometry.get_point(i);
  }

  // 2. Apply Updates (Integration Step)
  for (size_t i = 0; i < num_verts; ++i) {
    for (size_t k = 0; k < Sig::size; ++k)
      displacements[i][k] *= (float)dt;

    // UPDATED: Read, Add, Write back using set_point
    auto current_p = geometry.get_point(i);
    geometry.set_point(i, current_p + displacements[i]);
  }
}

} // namespace igneous::ops