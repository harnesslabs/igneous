#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <numbers>
#include <string>
#include <vector>

#include <fmt/core.h>
#include <igneous/algebra.hpp>
#include <igneous/mesh_loader.hpp>
#include <igneous/simplex.hpp>

using namespace igneous;
using Alg = Multivector<double, Euclidean3D>;
using Mesh = SimplexMesh<Alg>;

// ==================================================================================
// MATH KERNEL
// ==================================================================================

// Robust Angle Calculation
double angle_between(const Alg &a, const Alg &b, const Alg &origin) {
  auto u = a - origin;
  auto v = b - origin;
  double dot = u[1] * v[1] + u[2] * v[2] + u[3] * v[3];
  double len_u_sq = u[1] * u[1] + u[2] * u[2] + u[3] * u[3];
  double len_v_sq = v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
  if (len_u_sq < 1e-12 || len_v_sq < 1e-12)
    return 0.0;
  return std::acos(
      std::max(-1.0, std::min(1.0, dot / std::sqrt(len_u_sq * len_v_sq))));
}

// Triangle Area
double triangle_area(const Alg &a, const Alg &b, const Alg &c) {
  double u1 = b[1] - a[1], u2 = b[2] - a[2], u3 = b[3] - a[3];
  double v1 = c[1] - a[1], v2 = c[2] - a[2], v3 = c[3] - a[3];
  double cx = u2 * v3 - u3 * v2, cy = u3 * v1 - u1 * v3, cz = u1 * v2 - u2 * v1;
  return 0.5 * std::sqrt(cx * cx + cy * cy + cz * cz);
}

// Compute Face Normal (Unnormalized)
Alg face_normal(const Alg &a, const Alg &b, const Alg &c) {
  double u1 = b[1] - a[1], u2 = b[2] - a[2], u3 = b[3] - a[3];
  double v1 = c[1] - a[1], v2 = c[2] - a[2], v3 = c[3] - a[3];
  // Cross Product result stored in vector part
  auto n = Alg::from_blade(0, 0);
  n[1] = u2 * v3 - u3 * v2;
  n[2] = u3 * v1 - u1 * v3;
  n[3] = u1 * v2 - u2 * v1;
  return n;
}

// Simple Laplacian Smoothing
void smooth_field(const Mesh &mesh, std::vector<double> &field,
                  int iterations) {
  std::vector<double> temp = field;
  for (int iter = 0; iter < iterations; ++iter) {
    for (size_t i = 0; i < mesh.size(); ++i) {
      SimplexHandle v_h = {(uint32_t)i};
      if (mesh.get(v_h).dimension != 0)
        continue;

      double sum = 0.0, count = 0.0;
      for (SimplexHandle e_h : mesh.get(v_h).coboundary) {
        auto &edge = mesh.get(e_h);
        auto n1 = edge.boundary[0];
        auto n2 = edge.boundary[1];
        SimplexHandle neighbor = (n1.index == v_h.index) ? n2 : n1;
        sum += field[neighbor.index];
        count += 1.0;
      }
      if (count > 0.0)
        temp[i] = 0.5 * field[i] + 0.5 * (sum / count);
    }
    field = temp;
  }
}

// Export Helper with Range Print
void export_heatmap(SimplexMesh<Alg> &mesh, const std::vector<double> &field,
                    const std::string &filename, double sigma_clip = 2.0) {
  double sum = 0, sum_sq = 0, count = 0;
  for (double k : field) {
    if (std::abs(k) < 1000.0) {
      sum += k;
      sum_sq += k * k;
      count++;
    }
  }
  double mean = (count > 0) ? sum / count : 0.0;
  double std_dev =
      (count > 0) ? std::sqrt((sum_sq / count) - (mean * mean)) : 0.0;

  // Center on 0 for curvature to ensure Green = Flat
  // (Unless mean is very shifted, but usually 0 is the physical reference)
  double c_min = -sigma_clip * std_dev;
  double c_max = sigma_clip * std_dev;

  // Shift slightly if mean is significant (e.g. general convexity)
  if (std::abs(mean) > std_dev) {
    c_min += mean;
    c_max += mean;
  }

  fmt::print("  -> Exporting '{}' (Sigma Range: {:.4f} to {:.4f})\n", filename,
             c_min, c_max);
  MeshIO<Alg>::save_obj_colored(mesh, filename, field, c_min, c_max);
}

// ==================================================================================
// GEOMETRY KERNEL
// ==================================================================================

// Compute Vertex Normals (Average of Face Normals)
std::vector<Alg> compute_vertex_normals(Mesh &mesh) {
  std::vector<Alg> normals(mesh.size(), Alg::from_blade(0, 0));

  // Iterate Faces and accumulate normals to vertices
  for (size_t i = 0; i < mesh.size(); ++i) {
    if (mesh.get({(uint32_t)i}).dimension != 2)
      continue;
    auto &face = mesh.get({(uint32_t)i});

    auto e0 = face.boundary[0];
    auto e1 = face.boundary[1];
    auto vA = mesh.get(e0).boundary[0];
    auto vB = mesh.get(e0).boundary[1];
    auto vC = (mesh.get(e1).boundary[0] == vA || mesh.get(e1).boundary[0] == vB)
                  ? mesh.get(e1).boundary[1]
                  : mesh.get(e1).boundary[0];

    Alg n = face_normal(mesh.get(vA).geometry, mesh.get(vB).geometry,
                        mesh.get(vC).geometry);

    // Accumulate (Area weighted implicitly by cross product magnitude)
    normals[vA.index] = normals[vA.index] + n;
    normals[vB.index] = normals[vB.index] + n;
    normals[vC.index] = normals[vC.index] + n;
  }

  // Normalize
  for (size_t i = 0; i < mesh.size(); ++i) {
    if (mesh.get({(uint32_t)i}).dimension == 0) {
      auto &n = normals[i];
      double mag = std::sqrt(n[1] * n[1] + n[2] * n[2] + n[3] * n[3]);
      if (mag > 1e-9) {
        double inv = 1.0 / mag;
        n[1] *= inv;
        n[2] *= inv;
        n[3] *= inv;
      }
    }
  }
  return normals;
}

// ==================================================================================
// CURVATURE
// ==================================================================================

std::vector<double> compute_gaussian(Mesh &mesh) {
  fmt::print("Computing Gaussian Curvature (K)...\n");
  std::vector<double> K(mesh.size(), 0.0);
  std::vector<double> angle_sum(mesh.size(), 0.0);
  std::vector<double> area_sum(mesh.size(), 0.0);

  for (size_t i = 0; i < mesh.size(); ++i) {
    if (mesh.get({(uint32_t)i}).dimension != 2)
      continue;
    // ... (Same vertex retrieval as before) ...
    auto &face = mesh.get({(uint32_t)i});
    auto e0 = face.boundary[0];
    auto e1 = face.boundary[1];
    auto vA = mesh.get(e0).boundary[0];
    auto vB = mesh.get(e0).boundary[1];
    auto vC = (mesh.get(e1).boundary[0] == vA || mesh.get(e1).boundary[0] == vB)
                  ? mesh.get(e1).boundary[1]
                  : mesh.get(e1).boundary[0];

    const auto &pA = mesh.get(vA).geometry;
    const auto &pB = mesh.get(vB).geometry;
    const auto &pC = mesh.get(vC).geometry;

    angle_sum[vA.index] += angle_between(pB, pC, pA);
    angle_sum[vB.index] += angle_between(pA, pC, pB);
    angle_sum[vC.index] += angle_between(pA, pB, pC);

    double area = triangle_area(pA, pB, pC) / 3.0;
    area_sum[vA.index] += area;
    area_sum[vB.index] += area;
    area_sum[vC.index] += area;
  }

  for (size_t i = 0; i < mesh.size(); ++i) {
    if (mesh.get({(uint32_t)i}).dimension == 0 && area_sum[i] > 1e-9) {
      K[i] = ((2.0 * std::numbers::pi) - angle_sum[i]) / area_sum[i];
    }
  }
  return K;
}

std::vector<double> compute_mean_signed(Mesh &mesh) {
  fmt::print("Computing Signed Mean Curvature (H)...\n");
  std::vector<double> H(mesh.size(), 0.0);
  std::vector<Alg> neighbor_sum(mesh.size(), Alg::from_blade(0, 0));
  std::vector<double> neighbor_count(mesh.size(), 0.0);

  // 1. Compute Geometric Laplacian
  for (size_t i = 0; i < mesh.size(); ++i) {
    if (mesh.get({(uint32_t)i}).dimension != 1)
      continue;
    auto &edge = mesh.get({(uint32_t)i});
    auto v1 = edge.boundary[0];
    auto v2 = edge.boundary[1];
    neighbor_sum[v1.index] = neighbor_sum[v1.index] + mesh.get(v2).geometry;
    neighbor_count[v1.index] += 1.0;
    neighbor_sum[v2.index] = neighbor_sum[v2.index] + mesh.get(v1).geometry;
    neighbor_count[v2.index] += 1.0;
  }

  // 2. Get Vertex Normals for Sign
  auto normals = compute_vertex_normals(mesh);

  // 3. Compute Signed H
  for (size_t i = 0; i < mesh.size(); ++i) {
    if (mesh.get({(uint32_t)i}).dimension == 0 && neighbor_count[i] > 0) {
      Alg centroid = neighbor_sum[i];
      double inv_n = 1.0 / neighbor_count[i];
      centroid[1] *= inv_n;
      centroid[2] *= inv_n;
      centroid[3] *= inv_n;

      // Vector from Centroid to Vertex
      // For a convex bump, Vertex is "outside" centroid. Vector points Out.
      // For a concave dent, Vertex is "inside" centroid. Vector points In.
      Alg diff = mesh.get({(uint32_t)i}).geometry - centroid;

      // Project onto Normal to get Sign
      // H > 0 (Convex/Red), H < 0 (Concave/Blue)
      const auto &n = normals[i];
      double signed_mag = diff[1] * n[1] + diff[2] * n[2] + diff[3] * n[3];

      H[i] = signed_mag * 100.0; // Scale up slightly for visualization units
    }
  }
  return H;
}

int main(int argc, char **argv) {
  if (argc < 2)
    return 1;

  Mesh mesh(512 * 1024 * 1024);
  try {
    MeshIO<Alg>::load_obj(mesh, argv[1]);
  } catch (...) {
    return 1;
  }

  // 1. Gaussian: High Smoothing needed to remove "Static"
  auto K = compute_gaussian(mesh);
  smooth_field(mesh, K, 10); // Aggressive smoothing for Gaussian
  export_heatmap(mesh, K, "igneous_gaussian.obj", 1.0);

  // 2. Mean: Signed!
  auto H = compute_mean_signed(mesh);
  smooth_field(mesh, H, 5); // Moderate smoothing for Mean
  export_heatmap(mesh, H, "igneous_mean.obj", 2.0);

  return 0;
}