#pragma once
#include <algorithm>
#include <cmath>
#include <fstream>
#include <igneous/core/topology.hpp>
#include <igneous/data/mesh.hpp>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

namespace igneous::io {

using igneous::data::Mesh;

// Internal helper
inline std::tuple<uint8_t, uint8_t, uint8_t> get_heatmap_color_bytes(double t) {
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
  return {static_cast<uint8_t>(r * 255), static_cast<uint8_t>(g * 255),
          static_cast<uint8_t>(b * 255)};
}

// -----------------------------------------------------------------------------
// 1. PLY EXPORTER (Best for Point Clouds)
// -----------------------------------------------------------------------------
template <typename Sig, typename Topo>
void export_ply(const Mesh<Sig, Topo> &mesh, const std::vector<double> &field,
                const std::string &filename, double sigma_clip = 2.0) {
  // Stats
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
  double std_dev =
      std::sqrt((count > 0) ? (sum_sq / count) - (mean * mean) : 0.0);
  double min_v = -(sigma_clip * std_dev);
  double max_v = (sigma_clip * std_dev);

  std::ofstream file(filename);
  if (!file.is_open())
    return;

  size_t n_verts = mesh.geometry.num_points();

  // Header
  file << "ply\n";
  file << "format ascii 1.0\n";
  file << "element vertex " << n_verts << "\n";
  file << "property float x\n";
  file << "property float y\n";
  file << "property float z\n";
  file << "property uchar red\n";
  file << "property uchar green\n";
  file << "property uchar blue\n";
  file << "end_header\n";

  // Data
  for (size_t i = 0; i < n_verts; ++i) {
    auto p = mesh.geometry.get_vec3(i);
    double val = (i < field.size()) ? field[i] : 0.0;
    double t = (val - min_v) / (max_v - min_v);
    auto [r, g, b] = get_heatmap_color_bytes(t);

    file << p.x << " " << p.y << " " << p.z << " " << (int)r << " " << (int)g
         << " " << (int)b << "\n";
  }

  std::cout << "[IO] Exported PLY " << filename << "\n";
}

// -----------------------------------------------------------------------------
// 2. OBJ EXPORTER (Legacy / Surface)
// -----------------------------------------------------------------------------
template <typename Sig, typename Topo>
void export_heatmap(const Mesh<Sig, Topo> &mesh,
                    const std::vector<double> &field,
                    const std::string &filename, double sigma_clip = 2.0) {
  // (Stats logic omitted for brevity, same as above)...
  // ... copy stats logic ...
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
  double std_dev =
      std::sqrt((count > 0) ? (sum_sq / count) - (mean * mean) : 0.0);
  double min_v = -(sigma_clip * std_dev);
  double max_v = (sigma_clip * std_dev);

  std::ofstream file(filename);
  size_t n_verts = mesh.geometry.num_points();

  for (size_t i = 0; i < n_verts; ++i) {
    auto p = mesh.geometry.get_vec3(i);
    double val = (i < field.size()) ? field[i] : 0.0;
    double t = (val - min_v) / (max_v - min_v);
    auto [r, g, b] = get_heatmap_color_bytes(t);

    // OBJ Vertex Colors (Non-standard but common extension: r g b as float 0-1)
    file << "v " << p.x << " " << p.y << " " << p.z << " " << (r / 255.0f)
         << " " << (g / 255.0f) << " " << (b / 255.0f) << "\n";
  }

  // OPTION A: Write Faces (Surfaces)
  if constexpr (igneous::data::SurfaceTopology<Topo>) {
    size_t n_faces = mesh.topology.num_faces();
    for (size_t i = 0; i < n_faces; ++i) {
      file << "f " << mesh.topology.get_vertex_for_face(i, 0) + 1 << " "
           << mesh.topology.get_vertex_for_face(i, 1) + 1 << " "
           << mesh.topology.get_vertex_for_face(i, 2) + 1 << "\n";
    }
  }
  // OPTION B: Write Points (Point Clouds)
  else {
    // OBJ 'p' tag for point lists.
    // Note: macOS Preview might still ignore this, but it's the valid OBJ way.
    file << "p";
    for (size_t i = 0; i < n_verts; ++i) {
      file << " " << (i + 1);
      // Break lines occasionally to be nice to editors
      if (i % 1000 == 999)
        file << "\np";
    }
    file << "\n";
  }

  std::cout << "[IO] Exported OBJ " << filename << "\n";
}

// -----------------------------------------------------------------------------
// 3. SOLID PLY EXPORTER (Points -> Tetrahedrons)
// -----------------------------------------------------------------------------
template <typename Sig, typename Topo>
void export_ply_solid(const Mesh<Sig, Topo> &mesh,
                      const std::vector<double> &field,
                      const std::string &filename, double radius = 0.01,
                      double sigma_clip = 2.0) {

  // 1. Setup Stats & File
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
  double std_dev =
      std::sqrt((count > 0) ? (sum_sq / count) - (mean * mean) : 0.0);
  double min_v = -(sigma_clip * std_dev);
  double max_v = (sigma_clip * std_dev);

  std::ofstream file(filename);
  if (!file.is_open())
    return;

  size_t n_points = mesh.geometry.num_points();
  size_t n_verts = n_points * 4; // 4 verts per tetrahedron
  size_t n_faces = n_points * 4; // 4 faces per tetrahedron

  // 2. Write Header
  file << "ply\n";
  file << "format ascii 1.0\n";
  file << "element vertex " << n_verts << "\n";
  file << "property float x\n";
  file << "property float y\n";
  file << "property float z\n";
  file << "property uchar red\n";
  file << "property uchar green\n";
  file << "property uchar blue\n";
  file << "element face " << n_faces << "\n";
  file << "property list uchar int vertex_index\n";
  file << "end_header\n";

  // 3. Write Vertices
  // Offsets for a simple tetrahedron
  // (Top, Front-Left, Front-Right, Back)
  const float s = static_cast<float>(radius);
  const float h = s * 1.5f; // Height

  for (size_t i = 0; i < n_points; ++i) {
    auto p = mesh.geometry.get_vec3(i);

    // Color
    double val = (i < field.size()) ? field[i] : 0.0;
    double t = (val - min_v) / (max_v - min_v);
    auto [r, g, b] = get_heatmap_color_bytes(t);
    int ir = r, ig = g, ib = b;

    // v0: Top
    file << (p.x) << " " << (p.y + h) << " " << (p.z) << " " << ir << " " << ig
         << " " << ib << "\n";
    // v1: Front Left
    file << (p.x - s) << " " << (p.y - s) << " " << (p.z + s) << " " << ir
         << " " << ig << " " << ib << "\n";
    // v2: Front Right
    file << (p.x + s) << " " << (p.y - s) << " " << (p.z + s) << " " << ir
         << " " << ig << " " << ib << "\n";
    // v3: Back
    file << (p.x) << " " << (p.y - s) << " " << (p.z - s) << " " << ir << " "
         << ig << " " << ib << "\n";
  }

  // 4. Write Faces
  for (size_t i = 0; i < n_points; ++i) {
    size_t base = i * 4;
    // Face 1: 0-1-2
    file << "3 " << base << " " << base + 1 << " " << base + 2 << "\n";
    // Face 2: 0-2-3
    file << "3 " << base << " " << base + 2 << " " << base + 3 << "\n";
    // Face 3: 0-3-1
    file << "3 " << base << " " << base + 3 << " " << base + 1 << "\n";
    // Face 4: 1-3-2 (Bottom)
    file << "3 " << base + 1 << " " << base + 3 << " " << base + 2 << "\n";
  }

  std::cout << "[IO] Exported Solid PLY " << filename << "\n";
}

} // namespace igneous::io