// curvature.hpp
#pragma once
#include <cmath>
#include <igneous/core/algebra.hpp>
#include <igneous/data/mesh.hpp>
#include <numbers>

namespace igneous::ops {

using igneous::core::IsSignature;
using igneous::core::Multivector;
using igneous::data::Mesh;

template <IsSignature Sig>
std::pair<std::vector<double>, std::vector<double>>
compute_curvature_measures(const Mesh<Sig> &mesh) {

  const auto &geometry = mesh.geometry;
  const auto &topology = mesh.topology;

  // Use 'float' for storage
  using Field = float;

  size_t num_verts = geometry.points.size();
  std::vector<double> H(num_verts, 0.0);
  std::vector<double> K(num_verts, 0.0);

  // Pre-cache face normals (area-weighted)
  std::vector<Multivector<Field, Sig>> face_normals(topology.num_faces());
  for (size_t f = 0; f < topology.num_faces(); ++f) {
    uint32_t i0 = topology.get_vertex_for_face(f, 0);
    uint32_t i1 = topology.get_vertex_for_face(f, 1);
    uint32_t i2 = topology.get_vertex_for_face(f, 2);
    auto u = geometry.points[i1] - geometry.points[i0];
    auto v = geometry.points[i2] - geometry.points[i0];
    face_normals[f] = u ^ v;
  }

  // Parallel Gather
  for (size_t i = 0; i < num_verts; ++i) {
    auto faces = topology.get_faces_for_vertex(i);
    if (faces.empty())
      continue;

    double angle_sum = 0.0;
    double area_sum = 0.0;

    Multivector<Field, Sig> sum_normals;
    Multivector<Field, Sig> sum_neighbor_pos;
    double neighbor_count = 0.0;

    const auto &P = geometry.points[i];

    for (uint32_t f_idx : faces) {
      // A. Accumulate Normal
      sum_normals = sum_normals + face_normals[f_idx];

      // B. Angle Deficit Logic
      uint32_t i0 = topology.get_vertex_for_face(f_idx, 0);
      uint32_t i1 = topology.get_vertex_for_face(f_idx, 1);
      uint32_t i2 = topology.get_vertex_for_face(f_idx, 2);

      Multivector<Field, Sig> u, v;
      // Get vectors radiating FROM the current vertex i
      if (i0 == i) {
        u = geometry.points[i1] - P;
        v = geometry.points[i2] - P;
      } else if (i1 == i) {
        u = geometry.points[i2] - P;
        v = geometry.points[i0] - P;
      } else {
        u = geometry.points[i0] - P;
        v = geometry.points[i1] - P;
      }

      // GA Angle: atan2(|u^v|, u.v)
      double dot = 0.0;
      for (int k = 1; k <= 3; ++k)
        dot += u[k] * v[k];

      double wedge_mag_sq = 0.0;
      auto wedge = u ^ v;
      for (size_t k = 0; k < Sig::size; ++k)
        wedge_mag_sq += wedge[k] * wedge[k];
      double wedge_mag = std::sqrt(wedge_mag_sq);

      angle_sum += std::atan2(wedge_mag, dot);
      area_sum += 0.5 * wedge_mag; // Barycentric area

      // C. Mean Curvature Helpers
      if (i0 == i) {
        sum_neighbor_pos =
            sum_neighbor_pos + geometry.points[i1] + geometry.points[i2];
      } else if (i1 == i) {
        sum_neighbor_pos =
            sum_neighbor_pos + geometry.points[i2] + geometry.points[i0];
      } else {
        sum_neighbor_pos =
            sum_neighbor_pos + geometry.points[i0] + geometry.points[i1];
      }
      neighbor_count += 2.0;
    }

    // 1. Gaussian (K) = (2PI - sum_angles) / Area
    // This detects "cone-ness"
    if (area_sum > 1e-12) {
      // Note: 2*PI for internal vertices. PI for boundary.
      // We assume closed mesh for simplicity here.
      K[i] = (2.0 * std::numbers::pi - angle_sum) / (area_sum / 3.0);
    }

    // 2. Mean (H) = Signed Distance to Centroid
    double n_mag_sq = 0.0;
    for (size_t k = 0; k < Sig::size; ++k)
      n_mag_sq += sum_normals[k] * sum_normals[k];
    double n_inv = (n_mag_sq > 1e-12) ? 1.0 / std::sqrt(n_mag_sq) : 0.0;

    auto centroid = sum_neighbor_pos;
    double inv_count = 1.0 / std::max(1.0, neighbor_count);
    for (size_t k = 0; k < Sig::size; ++k)
      centroid[k] *= inv_count;

    auto laplacian = centroid - P; // Vector from Vertex to Average Neighbor

    // Project Laplacian onto Vertex Normal
    double signed_H = 0.0;
    for (int k = 1; k <= 3; ++k)
      signed_H += laplacian[k] * (sum_normals[k] * n_inv);

    // Remove manual scaling. Let the exporter handle it.
    H[i] = signed_H;
  }

  return {H, K};
}

} // namespace igneous::ops