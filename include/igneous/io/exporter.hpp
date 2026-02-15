#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <tuple>
#include <vector>

#include <igneous/data/space.hpp>
#include <igneous/data/structure.hpp>

namespace igneous::io {

using igneous::data::Space;

inline std::tuple<uint8_t, uint8_t, uint8_t> get_heatmap_color_bytes(double t) {
  t = std::clamp(t, 0.0, 1.0);
  float r = 0.0f;
  float g = 0.0f;
  float b = 0.0f;

  if (t < 0.5) {
    const float local_t = static_cast<float>(t * 2.0);
    r = local_t;
    g = local_t;
    b = 1.0f;
  } else {
    const float local_t = static_cast<float>((t - 0.5) * 2.0);
    r = 1.0f;
    g = 1.0f - local_t;
    b = 1.0f - local_t;
  }

  return {
      static_cast<uint8_t>(r * 255.0f),
      static_cast<uint8_t>(g * 255.0f),
      static_cast<uint8_t>(b * 255.0f),
  };
}

template <typename Field>
inline std::pair<double, double> compute_field_bounds(std::span<const Field> field, double sigma_clip) {
  double sum = 0.0;
  double sum_sq = 0.0;
  int count = 0;

  for (const Field raw_val : field) {
    const double val = static_cast<double>(raw_val);
    if (!std::isfinite(val)) {
      continue;
    }
    sum += val;
    sum_sq += val * val;
    count++;
  }

  const double mean = (count > 0) ? (sum / count) : 0.0;
  const double variance = (count > 0) ? std::max(0.0, (sum_sq / count) - (mean * mean)) : 0.0;
  const double std_dev = std::sqrt(variance);

  return {mean - (sigma_clip * std_dev), mean + (sigma_clip * std_dev)};
}

template <typename StructureT, typename Field>
void export_ply(const Space<StructureT> &mesh, std::span<const Field> field,
                const std::string &filename, double sigma_clip = 2.0) {
  const auto [min_v, max_v] = compute_field_bounds(field, sigma_clip);

  std::ofstream file(filename);
  if (!file.is_open()) {
    return;
  }

  const size_t n_verts = mesh.num_points();

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

  const double denom = std::max(1e-12, max_v - min_v);
  for (size_t i = 0; i < n_verts; ++i) {
    const auto p = mesh.get_vec3(i);
    const double val = (i < field.size()) ? static_cast<double>(field[i]) : 0.0;
    const double t = (val - min_v) / denom;
    const auto [r, g, b] = get_heatmap_color_bytes(t);

    file << p.x << " " << p.y << " " << p.z << " " << static_cast<int>(r) << " " << static_cast<int>(g)
         << " " << static_cast<int>(b) << "\n";
  }

  std::cout << "[IO] Exported PLY " << filename << "\n";
}

template <typename StructureT, typename Field>
void export_ply(const Space<StructureT> &mesh, const std::vector<Field> &field,
                const std::string &filename, double sigma_clip = 2.0) {
  export_ply(mesh, std::span<const Field>(field.data(), field.size()), filename,
             sigma_clip);
}

template <typename StructureT, typename Field>
void export_heatmap(const Space<StructureT> &mesh, std::span<const Field> field,
                    const std::string &filename, double sigma_clip = 2.0) {
  const auto [min_v, max_v] = compute_field_bounds(field, sigma_clip);

  std::ofstream file(filename);
  if (!file.is_open()) {
    return;
  }

  const size_t n_verts = mesh.num_points();
  const double denom = std::max(1e-12, max_v - min_v);

  for (size_t i = 0; i < n_verts; ++i) {
    const auto p = mesh.get_vec3(i);
    const double val = (i < field.size()) ? static_cast<double>(field[i]) : 0.0;
    const double t = (val - min_v) / denom;
    const auto [r, g, b] = get_heatmap_color_bytes(t);

    file << "v " << p.x << " " << p.y << " " << p.z << " " << (r / 255.0f) << " " << (g / 255.0f) << " "
         << (b / 255.0f) << "\n";
  }

  if constexpr (igneous::data::SurfaceStructure<StructureT>) {
    const size_t n_faces = mesh.structure.num_faces();
    for (size_t i = 0; i < n_faces; ++i) {
      file << "f " << mesh.structure.get_vertex_for_face(i, 0) + 1 << " "
           << mesh.structure.get_vertex_for_face(i, 1) + 1 << " "
           << mesh.structure.get_vertex_for_face(i, 2) + 1 << "\n";
    }
  } else {
    file << "p";
    for (size_t i = 0; i < n_verts; ++i) {
      file << " " << (i + 1);
      if (i % 1000 == 999) {
        file << "\np";
      }
    }
    file << "\n";
  }

  std::cout << "[IO] Exported OBJ " << filename << "\n";
}

template <typename StructureT, typename Field>
void export_heatmap(const Space<StructureT> &mesh,
                    const std::vector<Field> &field,
                    const std::string &filename, double sigma_clip = 2.0) {
  export_heatmap(mesh, std::span<const Field>(field.data(), field.size()),
                 filename, sigma_clip);
}

template <typename StructureT, typename Field>
void export_ply_solid(const Space<StructureT> &mesh, std::span<const Field> field,
                      const std::string &filename, double radius = 0.01,
                      double sigma_clip = 2.0) {
  const auto [min_v, max_v] = compute_field_bounds(field, sigma_clip);

  std::ofstream file(filename);
  if (!file.is_open()) {
    return;
  }

  const size_t n_points = mesh.num_points();
  const size_t n_verts = n_points * 4;
  const size_t n_faces = n_points * 4;

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

  const float s = static_cast<float>(radius);
  const float h = s * 1.5f;
  const double denom = std::max(1e-12, max_v - min_v);

  for (size_t i = 0; i < n_points; ++i) {
    const auto p = mesh.get_vec3(i);
    const double val = (i < field.size()) ? static_cast<double>(field[i]) : 0.0;
    const double t = (val - min_v) / denom;
    const auto [r, g, b] = get_heatmap_color_bytes(t);

    const int ir = r;
    const int ig = g;
    const int ib = b;

    file << p.x << " " << (p.y + h) << " " << p.z << " " << ir << " " << ig << " " << ib << "\n";
    file << (p.x - s) << " " << (p.y - s) << " " << (p.z + s) << " " << ir << " " << ig << " " << ib << "\n";
    file << (p.x + s) << " " << (p.y - s) << " " << (p.z + s) << " " << ir << " " << ig << " " << ib << "\n";
    file << p.x << " " << (p.y - s) << " " << (p.z - s) << " " << ir << " " << ig << " " << ib << "\n";
  }

  for (size_t i = 0; i < n_points; ++i) {
    const size_t base = i * 4;
    file << "3 " << base << " " << (base + 1) << " " << (base + 2) << "\n";
    file << "3 " << base << " " << (base + 2) << " " << (base + 3) << "\n";
    file << "3 " << base << " " << (base + 3) << " " << (base + 1) << "\n";
    file << "3 " << (base + 1) << " " << (base + 3) << " " << (base + 2) << "\n";
  }

  std::cout << "[IO] Exported Solid PLY " << filename << "\n";
}

template <typename StructureT, typename Field>
void export_ply_solid(const Space<StructureT> &mesh,
                      const std::vector<Field> &field,
                      const std::string &filename, double radius = 0.01,
                      double sigma_clip = 2.0) {
  export_ply_solid(mesh, std::span<const Field>(field.data(), field.size()),
                   filename, radius, sigma_clip);
}

} // namespace igneous::io
