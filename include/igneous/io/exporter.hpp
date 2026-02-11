#pragma once
#include <algorithm>
#include <cmath>
#include <fstream>
#include <igneous/data/mesh.hpp>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

namespace igneous::io {

using igneous::data::Mesh;

// Internal helper (keep hidden or static)
inline std::tuple<float, float, float> get_heatmap_color(double t) {
  t = std::max(0.0, std::min(1.0, t));
  float r, g, b;
  if (t < 0.5) {
    float local_t = t * 2.0;
    r = local_t;
    g = local_t;
    b = 1.0f;
  } else {
    float local_t = (t - 0.5) * 2.0;
    r = 1.0f;
    g = 1.0f - local_t;
    b = 1.0f - local_t;
  }
  return {r, g, b};
}

template <typename Sig>
void export_heatmap(const Mesh<Sig> &mesh, const std::vector<double> &field,
                    const std::string &filename, double sigma_clip = 2.0) {
  // 1. Stats
  double sum = 0, sum_sq = 0;
  int count = 0;
  for (double val : field) {
    if (std::isfinite(val)) {
      sum += val;
      sum_sq += val * val;
      count++;
    }
  }
  double mean = (count > 0) ? sum / count : 0.0;
  double var = (count > 0) ? (sum_sq / count) - (mean * mean) : 0.0;
  double std_dev = std::sqrt(var);

  double range = sigma_clip * std_dev;
  double min_v = -range; // Centered on 0 for curvature
  double max_v = range;

  std::ofstream file(filename);

  // Vertices
  // UPDATED: Use num_points() accessor
  size_t n_verts = mesh.geometry.num_points();

  for (size_t i = 0; i < n_verts; ++i) {
    // FIX: Use get_vec3 instead of get_point
    // This returns a struct {x, y, z} that maps correctly to storage
    auto p = mesh.geometry.get_vec3(i);

    double val = (i < field.size()) ? field[i] : 0.0;
    double t = (val - min_v) / (max_v - min_v);
    auto [r, g, b] = get_heatmap_color(t);

    // FIX: Access .x .y .z directly
    file << "v " << p.x << " " << p.y << " " << p.z << " " << r << " " << g
         << " " << b << "\n";
  }

  // Faces
  for (size_t i = 0; i < mesh.topology.num_faces(); ++i) {
    file << "f " << mesh.topology.get_vertex_for_face(i, 0) + 1 << " "
         << mesh.topology.get_vertex_for_face(i, 1) + 1 << " "
         << mesh.topology.get_vertex_for_face(i, 2) + 1 << "\n";
  }

  std::cout << "[IO] Exported " << filename << "\n";
}

} // namespace igneous::io