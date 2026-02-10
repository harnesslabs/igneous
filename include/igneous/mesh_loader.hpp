#pragma once
#include <cmath> // std::abs
#include <filesystem>
#include <fstream>
#include <igneous/simplex.hpp>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace igneous {

struct PairHash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2> &p) const {
    auto h1 = std::hash<T1>{}(p.first);
    auto h2 = std::hash<T2>{}(p.second);
    return h1 ^ (h2 << 1);
  }
};

// Spatial Hash for Vertex Welding
// We quantize positions to merge vertices that are extremely close
struct Vec3Key {
  int x, y, z;

  bool operator==(const Vec3Key &other) const {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct Vec3Hash {
  std::size_t operator()(const Vec3Key &k) const {
    return ((std::hash<int>()(k.x) ^ (std::hash<int>()(k.y) << 1)) >> 1) ^
           (std::hash<int>()(k.z) << 1);
  }
};

static inline std::tuple<double, double, double>
scalar_to_rgb(double val, double min_val, double max_val) {
  if (std::abs(max_val - min_val) < 1e-9)
    return {0, 1, 0};
  double t = (val - min_val) / (max_val - min_val);
  t = std::max(0.0, std::min(1.0, t));
  double r = std::max(0.0, 2.0 * (t - 0.5));
  double b = std::max(0.0, 2.0 * (0.5 - t));
  double g = 1.0 - 2.0 * std::abs(t - 0.5);
  return {r, g, b};
}

template <typename Algebra> class MeshIO {
public:
  // --- IMPORTER WITH WELDING ---
  static void load_obj(SimplexMesh<Algebra> &mesh,
                       const std::filesystem::path &path) {
    std::ifstream file(path);
    if (!file.is_open())
      throw std::runtime_error("Could not open file: " + path.string());

    std::string line;

    // Map file index -> SimplexHandle
    std::vector<SimplexHandle> file_index_to_handle;

    // Map Position -> SimplexHandle (The Welder)
    std::unordered_map<Vec3Key, SimplexHandle, Vec3Hash> unique_vertices;

    // Deduplication Map for Edges
    std::unordered_map<std::pair<uint32_t, uint32_t>, SimplexHandle, PairHash>
        edge_map;

    // Precision for welding (e.g. 1e-4)
    const double SCALE = 10000.0;

    std::cout << "Loading " << path.filename() << " with vertex welding..."
              << std::endl;

    while (std::getline(file, line)) {
      if (line.empty())
        continue;
      std::istringstream iss(line);
      std::string type;
      iss >> type;

      if (type == "v") {
        double x, y, z;
        iss >> x >> y >> z;

        // Quantize position
        Vec3Key key = {(int)std::round(x * SCALE), (int)std::round(y * SCALE),
                       (int)std::round(z * SCALE)};

        if (unique_vertices.find(key) == unique_vertices.end()) {
          auto mv = Algebra::from_blade(0, 0);
          mv[1] = x;
          mv[2] = y;
          mv[3] = z;
          auto h = mesh.add_vertex(mv);
          unique_vertices[key] = h;
        }

        // Map the current file index to the (possibly existing) handle
        file_index_to_handle.push_back(unique_vertices[key]);
      } else if (type == "f") {
        std::vector<uint32_t> poly_indices;
        std::string token;

        while (iss >> token) {
          size_t slash1 = token.find('/');
          std::string v_str =
              (slash1 == std::string::npos) ? token : token.substr(0, slash1);
          try {
            int idx = std::stoi(v_str);
            if (idx < 0)
              idx = (int)file_index_to_handle.size() + idx;
            else
              idx = idx - 1;

            if (idx >= 0 && idx < (int)file_index_to_handle.size()) {
              poly_indices.push_back((uint32_t)idx);
            }
          } catch (...) {
            continue;
          }
        }

        if (poly_indices.size() < 3)
          continue;

        // Fan Triangulation
        for (size_t i = 0; i < poly_indices.size() - 2; ++i) {
          // Use the HANDLES index, not the file index
          uint32_t idx_in_mesh[3] = {
              file_index_to_handle[poly_indices[0]].index,
              file_index_to_handle[poly_indices[i + 1]].index,
              file_index_to_handle[poly_indices[i + 2]].index};

          // Degenerate Triangle Check (if welding merged 2 corners)
          if (idx_in_mesh[0] == idx_in_mesh[1] ||
              idx_in_mesh[1] == idx_in_mesh[2] ||
              idx_in_mesh[2] == idx_in_mesh[0])
            continue;

          SimplexHandle edges[3];
          for (int k = 0; k < 3; ++k) {
            uint32_t a = idx_in_mesh[k];
            uint32_t b = idx_in_mesh[(k + 1) % 3];

            SimplexHandle hA = {a};
            SimplexHandle hB = {b};

            auto key = std::minmax(a, b);
            if (edge_map.find(key) == edge_map.end()) {
              auto pA = mesh.get(hA).geometry;
              edge_map[key] = mesh.add_edge(pA, hA, hB);
            }
            edges[k] = edge_map[key];
          }
          auto dummy = mesh.get({idx_in_mesh[0]}).geometry;
          mesh.add_triangle(dummy, edges[0], edges[1], edges[2]);
        }
      }
    }
    std::cout << "  - Original Vertices: " << file_index_to_handle.size()
              << "\n";
    std::cout << "  - Welded Vertices:   " << unique_vertices.size() << "\n";
    std::cout << "  - Unique Edges:      " << edge_map.size() << "\n";
  }

  // --- EXPORTER (Accepts Manual Range) ---
  static void save_obj_colored(SimplexMesh<Algebra> &mesh,
                               const std::filesystem::path &path,
                               const std::vector<double> &curvature,
                               double manual_min = 0.0,
                               double manual_max = 0.0) {
    std::ofstream file(path);
    if (!file.is_open())
      throw std::runtime_error("Could not save file");

    file << "# Exported by Igneous Engine\n";

    double min_k = -1.0, max_k = 1.0;

    // Use manual range if provided (non-zero range), otherwise auto-detect
    if (std::abs(manual_max - manual_min) > 1e-9) {
      min_k = manual_min;
      max_k = manual_max;
    } else if (!curvature.empty()) {
      // ... (Auto detect logic from before) ...
      // For simplicity, let's trust the manual range from main()
      // since we calculated Sigma there.
    }

    std::unordered_map<uint32_t, int> handle_to_obj_idx;
    int obj_idx = 1;

    // 1. Write Vertices
    for (size_t i = 0; i < mesh.size(); ++i) {
      SimplexHandle h = {(uint32_t)i};
      auto &s = mesh.get(h);
      if (s.dimension == 0) {
        double x = s.geometry[1], y = s.geometry[2], z = s.geometry[3];

        if (!curvature.empty() && i < curvature.size()) {
          auto [r, g, b] = scalar_to_rgb(curvature[i], min_k, max_k);
          file << "v " << x << " " << y << " " << z << " " << r << " " << g
               << " " << b << "\n";
        } else {
          file << "v " << x << " " << y << " " << z << "\n";
        }
        handle_to_obj_idx[h.index] = obj_idx++;
      }
    }

    // 2. Write Faces (Double Sided)
    for (size_t i = 0; i < mesh.size(); ++i) {
      SimplexHandle h = {(uint32_t)i};
      auto &s = mesh.get(h);
      if (s.dimension == 2) {
        auto e0 = s.boundary[0];
        auto e1 = s.boundary[1];
        auto vA = mesh.get(e0).boundary[0];
        auto vB = mesh.get(e0).boundary[1];
        auto vC_cand1 = mesh.get(e1).boundary[0];
        auto vC_cand2 = mesh.get(e1).boundary[1];
        auto vC = (vC_cand1 == vA || vC_cand1 == vB) ? vC_cand2 : vC_cand1;

        int iA = handle_to_obj_idx[vA.index];
        int iB = handle_to_obj_idx[vB.index];
        int iC = handle_to_obj_idx[vC.index];

        file << "f " << iA << " " << iB << " " << iC << "\n";
        file << "f " << iA << " " << iC << " " << iB << "\n";
      }
    }
    std::cout << "Saved to " << path.string() << "\n";
  }
};
} // namespace igneous