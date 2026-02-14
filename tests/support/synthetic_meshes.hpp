#pragma once

#include <cmath>
#include <random>

#include <igneous/core/algebra.hpp>
#include <igneous/data/mesh.hpp>

namespace igneous::test_support {

using DiffusionMesh =
    igneous::data::Mesh<igneous::core::Euclidean3D,
                        igneous::data::DiffusionTopology>;
using TriangleMesh =
    igneous::data::Mesh<igneous::core::Euclidean3D,
                        igneous::data::TriangleTopology>;

inline DiffusionMesh make_torus_cloud(size_t n_points, float bandwidth = 0.05f,
                                      int k_neighbors = 24) {
  DiffusionMesh mesh;
  mesh.geometry.reserve(n_points);

  constexpr float kR = 2.0f;
  constexpr float kr = 0.8f;
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(0.0f, 6.283185f);

  for (size_t i = 0; i < n_points; ++i) {
    const float u = dist(gen);
    const float v = dist(gen);
    const float x = (kR + kr * std::cos(v)) * std::cos(u);
    const float y = (kR + kr * std::cos(v)) * std::sin(u);
    const float z = kr * std::sin(v);
    mesh.geometry.push_point({x, y, z});
  }

  mesh.topology.build({mesh.geometry.x_span(), mesh.geometry.y_span(),
                       mesh.geometry.z_span(), bandwidth, k_neighbors});
  return mesh;
}

inline DiffusionMesh make_helix_cloud(size_t n_points, float bandwidth = 0.05f,
                                      int k_neighbors = 24) {
  DiffusionMesh mesh;
  mesh.geometry.reserve(n_points);
  for (size_t i = 0; i < n_points; ++i) {
    const float t = static_cast<float>(i) / std::max<size_t>(n_points - 1, 1);
    mesh.geometry.push_point(
        {std::cos(t * 6.283185f), std::sin(t * 6.283185f), 2.0f * t - 1.0f});
  }

  mesh.topology.build({mesh.geometry.x_span(), mesh.geometry.y_span(),
                       mesh.geometry.z_span(), bandwidth, k_neighbors});
  return mesh;
}

inline TriangleMesh make_surface_grid(int side) {
  TriangleMesh mesh;
  mesh.geometry.reserve(static_cast<size_t>(side * side));

  for (int y = 0; y < side; ++y) {
    for (int x = 0; x < side; ++x) {
      const float fx = static_cast<float>(x) / static_cast<float>(side);
      const float fy = static_cast<float>(y) / static_cast<float>(side);
      mesh.geometry.push_point({fx, fy, std::sin(fx) * std::cos(fy)});
    }
  }

  for (int y = 0; y < side - 1; ++y) {
    for (int x = 0; x < side - 1; ++x) {
      const uint32_t i0 = static_cast<uint32_t>(y * side + x);
      const uint32_t i1 = static_cast<uint32_t>(y * side + x + 1);
      const uint32_t i2 = static_cast<uint32_t>((y + 1) * side + x);
      const uint32_t i3 = static_cast<uint32_t>((y + 1) * side + x + 1);

      mesh.topology.faces_to_vertices.insert(mesh.topology.faces_to_vertices.end(),
                                             {i0, i1, i2, i1, i3, i2});
    }
  }

  mesh.topology.build({mesh.geometry.num_points(), true});
  return mesh;
}

} // namespace igneous::test_support
