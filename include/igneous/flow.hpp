// flow.hpp
#pragma once
#include <igneous/geometry.hpp>
#include <igneous/topology.hpp>
#include <vector>

namespace igneous {

template <typename Field, IsSignature Sig>
void integrate_mean_curvature_flow(GeometryBuffer<Field, Sig> &geometry,
                                   const TopologyBuffer &topology, double dt) {
  size_t num_verts = geometry.points.size();

  // 1. Compute Update Vectors (The "Flow")
  // We store these in a temporary buffer so we don't read "future" data
  // while calculating "current" step.
  std::vector<Multivector<Field, Sig>> displacements(num_verts);

  for (size_t i = 0; i < num_verts; ++i) {
    auto faces = topology.get_faces_for_vertex(i);
    if (faces.empty())
      continue;

    // Uniform Laplacian (Umbrella Operator)
    // L(p) = sum(p_neighbor - p)
    // This vector points in the direction of minimizing surface area.

    Multivector<Field, Sig> sum_neighbors; // Zero init
    double count = 0.0;

    // Iterate Ring
    for (uint32_t f_idx : faces) {
      uint32_t i0 = topology.get_vertex_for_face(f_idx, 0);
      uint32_t i1 = topology.get_vertex_for_face(f_idx, 1);
      uint32_t i2 = topology.get_vertex_for_face(f_idx, 2);

      // Add vertices that are NOT center 'i'
      // (Note: This adds neighbors multiple times if they share multiple faces,
      // which effectively weights them by valence. This is acceptable for
      // simple smoothing.)
      if (i0 != i)
        sum_neighbors = sum_neighbors + geometry.points[i0];
      if (i1 != i)
        sum_neighbors = sum_neighbors + geometry.points[i1];
      if (i2 != i)
        sum_neighbors = sum_neighbors + geometry.points[i2];

      // Each face contributes 2 neighbors.
      count += 2.0;
    }

    // Calculate Laplacian Vector
    // V_flow = (Average_Center - P)
    Multivector<Field, Sig> average_pos = sum_neighbors;
    // Scalar division
    double inv_c = 1.0 / std::max(1.0, count);
    for (size_t k = 0; k < Sig::size; ++k)
      average_pos[k] *= inv_c;

    // The vector from P to the average of its neighbors
    displacements[i] = average_pos - geometry.points[i];
  }

  // 2. Apply Updates (Integration Step)
  // P_new = P_old + dt * Flow_Vector
  for (size_t i = 0; i < num_verts; ++i) {
    // Scalar multiplication of the displacement vector
    for (size_t k = 0; k < Sig::size; ++k)
      displacements[i][k] *= dt;

    geometry.points[i] = geometry.points[i] + displacements[i];
  }
}

} // namespace igneous