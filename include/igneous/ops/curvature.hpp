// include/igneous/ops/curvature.hpp
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

  using Field = float;

  // UPDATED: Use accessor .num_points()
  size_t num_verts = geometry.num_points();
  std::vector<double> H(num_verts, 0.0);
  std::vector<double> K(num_verts, 0.0);

  // Pre-cache face normals (area-weighted)
  std::vector<Multivector<Field, Sig>> face_normals(topology.num_faces());
  for (size_t f = 0; f < topology.num_faces(); ++f) {
    uint32_t i0 = topology.get_vertex_for_face(f, 0);
    uint32_t i1 = topology.get_vertex_for_face(f, 1);
    uint32_t i2 = topology.get_vertex_for_face(f, 2);

    // UPDATED: Use get_point
    auto p0 = geometry.get_point(i0);
    auto p1 = geometry.get_point(i1);
    auto p2 = geometry.get_point(i2);

    auto u = p1 - p0;
    auto v = p2 - p0;
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

    // UPDATED: Use get_point for the center vertex
    const auto P = geometry.get_point(i);

    for (uint32_t f_idx : faces) {
      sum_normals = sum_normals + face_normals[f_idx];

      uint32_t i0 = topology.get_vertex_for_face(f_idx, 0);
      uint32_t i1 = topology.get_vertex_for_face(f_idx, 1);
      uint32_t i2 = topology.get_vertex_for_face(f_idx, 2);

      Multivector<Field, Sig> u, v;
      // UPDATED: Use get_point inside neighbor logic
      if (i0 == i) {
        u = geometry.get_point(i1) - P;
        v = geometry.get_point(i2) - P;
      } else if (i1 == i) {
        u = geometry.get_point(i2) - P;
        v = geometry.get_point(i0) - P;
      } else {
        u = geometry.get_point(i0) - P;
        v = geometry.get_point(i1) - P;
      }

      double dot = 0.0;
      for (int k = 1; k <= 3; ++k)
        dot += u[k] * v[k];

      double wedge_mag_sq = 0.0;
      auto wedge = u ^ v;
      for (size_t k = 0; k < Sig::size; ++k)
        wedge_mag_sq += wedge[k] * wedge[k];
      double wedge_mag = std::sqrt(wedge_mag_sq);

      angle_sum += std::atan2(wedge_mag, dot);
      area_sum += 0.5 * wedge_mag;

      // UPDATED: Use get_point for neighbor position sums
      if (i0 == i) {
        sum_neighbor_pos =
            sum_neighbor_pos + geometry.get_point(i1) + geometry.get_point(i2);
      } else if (i1 == i) {
        sum_neighbor_pos =
            sum_neighbor_pos + geometry.get_point(i2) + geometry.get_point(i0);
      } else {
        sum_neighbor_pos =
            sum_neighbor_pos + geometry.get_point(i0) + geometry.get_point(i1);
      }
      neighbor_count += 2.0;
    }

    if (area_sum > 1e-12) {
      K[i] = (2.0 * std::numbers::pi - angle_sum) / (area_sum / 3.0);
    }

    double n_mag_sq = 0.0;
    for (size_t k = 0; k < Sig::size; ++k)
      n_mag_sq += sum_normals[k] * sum_normals[k];
    double n_inv = (n_mag_sq > 1e-12) ? 1.0 / std::sqrt(n_mag_sq) : 0.0;

    auto centroid = sum_neighbor_pos;
    double inv_count = 1.0 / std::max(1.0, neighbor_count);
    for (size_t k = 0; k < Sig::size; ++k)
      centroid[k] *= inv_count;

    auto laplacian = centroid - P;

    double signed_H = 0.0;
    for (int k = 1; k <= 3; ++k)
      signed_H += laplacian[k] * (sum_normals[k] * n_inv);

    H[i] = signed_H;
  }

  return {H, K};
}

} // namespace igneous::ops