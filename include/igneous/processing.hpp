#pragma once
#include <algorithm>
#include <cmath>
#include <fstream>
#include <igneous/geometry.hpp>
#include <igneous/topology.hpp>
#include <iostream>
#include <string>
#include <vector>

namespace igneous {

// The Kernel: Computes the circumcircle (or plane) for every triangle.
// In CGA, A ^ B ^ C represents the circle passing through the 3 points.
template <typename Field, IsSignature Sig>
void compute_face_trivectors(const GeometryBuffer<Field, Sig> &geometry,
                             const TopologyBuffer &topology,
                             GeometryBuffer<Field, Sig> &output) {
  size_t n_faces = topology.num_faces();

  // Ensure output buffer has space
  if (output.faces.size() < n_faces) {
    output.faces.resize(n_faces);
  }

  // This loop is extremely friendly to the CPU Prefetcher
  for (size_t f = 0; f < n_faces; ++f) {

    // 1. Topology Lookup (Integer math, fast)
    uint32_t i0 = topology.get_vertex_for_face(f, 0);
    uint32_t i1 = topology.get_vertex_for_face(f, 1);
    uint32_t i2 = topology.get_vertex_for_face(f, 2);

    // 2. Geometry Fetch (Linear memory access if vertices are sorted nicely)
    // Note: We use const references to avoid copying the Multivector stack
    const auto &A = geometry.points[i0];
    const auto &B = geometry.points[i1];
    const auto &C = geometry.points[i2];

    // 3. The CGA Magic
    // This computes the blade representing the face.
    // If Sig is PGA3D, this is the plane equation.
    // If Sig is Conformal (5D), this is the Circle/Point-Pair.
    output.faces[f] = A ^ B ^ C;
  }
}

template <typename Field, IsSignature Sig>
std::vector<double>
compute_mean_curvature_flow(const GeometryBuffer<Field, Sig> &geometry,
                            const TopologyBuffer &topology) {
  size_t num_faces = topology.num_faces();
  size_t num_verts = geometry.points.size();

  // Temp Buffer: Face Normals (Bivectors/Trivectors depending on algebra)
  std::vector<Multivector<Field, Sig>> face_normals(num_faces);

  // Temp Buffer: Vertex Normals (Accumulator)
  std::vector<Multivector<Field, Sig>> vertex_normals(num_verts); // Init to 0

  // PASS 1: Face Normals (Pure Stream)
  // "The normal of a triangle is the outer product of its edges"
  // In PGA/CGA: A ^ B ^ C gives the plane/circle.
  for (size_t f = 0; f < num_faces; ++f) {
    uint32_t i0 = topology.get_vertex_for_face(f, 0);
    uint32_t i1 = topology.get_vertex_for_face(f, 1);
    uint32_t i2 = topology.get_vertex_for_face(f, 2);

    const auto &A = geometry.points[i0];
    const auto &B = geometry.points[i1];
    const auto &C = geometry.points[i2];

    // The "Blade" of the face.
    // In Euclidean3D (3,0): This is a pseudoscalar (volume), need edges for
    // normal. In Euclidean3D, Normal = (B-A) ^ (C-A). Let's assume input is
    // Euclidean vectors for this specific kernel logic.
    auto edge1 = B - A;
    auto edge2 = C - A;

    // This calculates the Bivector Area (Normal)
    face_normals[f] = edge1 ^ edge2;
  }

  // PASS 2: Vertex Accumulation (Scatter)
  // We iterate faces and add their normal to their constituent vertices.
  // (This is often faster than iterating vertices and gathering if cache is
  // managed)
  for (size_t f = 0; f < num_faces; ++f) {
    uint32_t idx[3] = {topology.get_vertex_for_face(f, 0),
                       topology.get_vertex_for_face(f, 1),
                       topology.get_vertex_for_face(f, 2)};

    // Simply add the bivector.
    // Larger faces (larger area) contribute more weight automatically.
    vertex_normals[idx[0]] = vertex_normals[idx[0]] + face_normals[f];
    vertex_normals[idx[1]] = vertex_normals[idx[1]] + face_normals[f];
    vertex_normals[idx[2]] = vertex_normals[idx[2]] + face_normals[f];
  }

  // PASS 3: Compute Sign/Magnitude (Stream)
  std::vector<double> curvature(num_verts);

  for (size_t v = 0; v < num_verts; ++v) {
    auto &N = vertex_normals[v];

    // Normalize the bivector N (Simplified Euclidean normalize)
    // Magnitude formula for 3D bivector: sqrt(e12*e12 + e23*e23 + e31*e31)
    double mag_sq = 0.0;
    // In Sig<3,0>, bivectors are at indices 3, 5, 6 (binary 011, 101, 110)
    // Or simply sum all components:
    for (size_t k = 0; k < Sig::size; ++k)
      mag_sq += N[k] * N[k];

    double mag = std::sqrt(mag_sq);

    if (mag > 1e-9) {
      // Rough approximation of Mean Curvature:
      // 1.0 / magnitude of accumulated area?
      // Actually, let's just output the convexity for coloring.
      // We need a sign.

      // To get Sign (Concave/Convex), we check a neighbor.
      // Get first neighbor from topology (expensive lookup without
      // coboundaries) For visualization, let's return the Magnitude of the
      // Normal scaled by how "sharp" it is (inverse of average area).

      curvature[v] = mag; // Placeholder for visualization
    }
  }
  return curvature;
}

// Diverging Color Map (Blue -> White -> Red)
// t is in range [0, 1]. 0.5 is White.
std::tuple<float, float, float> get_heatmap_color(double t) {
  t = std::max(0.0, std::min(1.0, t));

  // Blue (0.0) -> Cyan (0.25) -> White (0.5) -> Yellow (0.75) -> Red (1.0)
  // This matches the "Cool-Warm" aesthetic of your reference.
  float r, g, b;

  if (t < 0.5) {
    // Blue to White
    float local_t = t * 2.0; // 0 to 1
    r = local_t;
    g = local_t;
    b = 1.0f;
  } else {
    // White to Red
    float local_t = (t - 0.5) * 2.0; // 0 to 1
    r = 1.0f;
    g = 1.0f - local_t;
    b = 1.0f - local_t;
  }
  return {r, g, b};
}

template <typename Sig>
void save_colored_obj(const std::string &filename,
                      const GeometryBuffer<float, Sig> &geo,
                      const TopologyBuffer &topo,
                      const std::vector<double> &raw_field,
                      double sigma_clip = 2.0) {
  // 1. Calculate Statistics (Robust Auto-Ranging)
  double sum = 0, sum_sq = 0;
  int count = 0;
  for (double val : raw_field) {
    if (std::isfinite(val)) {
      sum += val;
      sum_sq += val * val;
      count++;
    }
  }

  double mean = (count > 0) ? sum / count : 0.0;
  double variance = (count > 0) ? (sum_sq / count) - (mean * mean) : 0.0;
  double std_dev = std::sqrt(variance);

  // 2. Define Range based on Sigma (Standard Deviations)
  // This ignores the extreme "tips" of the ears so the body gets color.
  double min_v = mean - (sigma_clip * std_dev);
  double max_v = mean + (sigma_clip * std_dev);

  // Center the range on 0 for curvature (Green/White = Flat)
  // If the mesh is generally convex, mean might be shifted,
  // but usually we want 0 to be the "neutral" color.
  double range_max = std::max(std::abs(min_v), std::abs(max_v));
  min_v = -range_max;
  max_v = range_max;

  std::cout << "Exporting " << filename << "\n";
  std::cout << "  - Range: [" << min_v << ", " << max_v << "]\n";
  std::cout << "  - Mean: " << mean << ", StdDev: " << std_dev << "\n";

  std::ofstream file(filename);

  // Write Vertices with Colors
  for (size_t i = 0; i < geo.points.size(); ++i) {
    float x = geo.points[i][1];
    float y = geo.points[i][2];
    float z = geo.points[i][3];

    double val = raw_field[i];

    // Normalize to [0, 1]
    double t = (val - min_v) / (max_v - min_v);

    auto [r, g, b] = get_heatmap_color(t);

    file << "v " << x << " " << y << " " << z << " " << r << " " << g << " "
         << b << "\n";
  }

  // Write Faces (1-based indices)
  for (size_t i = 0; i < topo.num_faces(); ++i) {
    uint32_t v0 = topo.get_vertex_for_face(i, 0) + 1;
    uint32_t v1 = topo.get_vertex_for_face(i, 1) + 1;
    uint32_t v2 = topo.get_vertex_for_face(i, 2) + 1;
    file << "f " << v0 << " " << v1 << " " << v2 << "\n";
  }
}

template <typename Field, IsSignature Sig>
void normalize_mesh(GeometryBuffer<Field, Sig> &geometry) {
  if (geometry.points.empty())
    return;

  // 1. Compute Bounding Box
  // Initialize with first point
  Multivector<Field, Sig> min_p = geometry.points[0];
  Multivector<Field, Sig> max_p = geometry.points[0];

  for (const auto &p : geometry.points) {
    // Check X, Y, Z (indices 1, 2, 3)
    for (int k = 1; k <= 3; ++k) {
      if (p[k] < min_p[k])
        min_p[k] = p[k];
      if (p[k] > max_p[k])
        max_p[k] = p[k];
    }
  }

  // 2. Calculate Center and Scale
  Multivector<Field, Sig> center;
  float max_dim = 0.0f;

  for (int k = 1; k <= 3; ++k) {
    center[k] = (min_p[k] + max_p[k]) * 0.5f;
    float dim = max_p[k] - min_p[k];
    if (dim > max_dim)
      max_dim = dim;
  }

  // If scale is 0 (single point), avoid div/0
  float scale_factor = (max_dim > 1e-9f) ? (2.0f / max_dim) : 1.0f;

  std::cout << "[Normalization] Shift: " << center[1] << " " << center[2] << " "
            << center[3] << "\n";
  std::cout << "[Normalization] Scale: " << scale_factor << "x\n";

  // 3. Apply Transform
  for (auto &p : geometry.points) {
    // p_new = (p - center) * scale
    auto temp = p - center;
    for (int k = 1; k <= 3; ++k) {
      p[k] = temp[k] * scale_factor;
    }
  }
}

} // namespace igneous