#pragma once
#include <filesystem>
#include <fstream>
#include <igneous/data/mesh.hpp>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace igneous::io {

using igneous::core::IsSignature;
using igneous::core::Multivector;
using igneous::data::GeometryBuffer;
using igneous::data::Mesh;
using igneous::data::TopologyBuffer;

// ==============================================================================
// INTERNAL HELPERS (Spatial Hashing & Parsing)
// ==============================================================================

struct Vec3Key {
  int x, y, z;
  bool operator==(const Vec3Key &o) const {
    return x == o.x && y == o.y && z == o.z;
  }
};

struct Vec3Hash {
  std::size_t operator()(const Vec3Key &k) const {
    return ((std::hash<int>()(k.x) ^ (std::hash<int>()(k.y) << 1)) >> 1) ^
           (std::hash<int>()(k.z) << 1);
  }
};

// We keep DODLoader as an internal implementation detail.
template <typename Sig> struct DODLoader {
  using Field = float;
  using MV = Multivector<Field, Sig>;

  static void load_internal(const std::filesystem::path &path,
                            GeometryBuffer<Field, Sig> &geo_out,
                            TopologyBuffer &topo_out) {
    std::ifstream file(path);
    if (!file.is_open())
      throw std::runtime_error("File not found: " + path.string());

    geo_out.clear();
    topo_out.clear();

    std::vector<uint32_t> face_indices;
    std::unordered_map<Vec3Key, uint32_t, Vec3Hash> unique_map;
    const double SCALE = 10000.0;

    std::string line;
    std::cout << "[IO] Loading " << path.filename() << "...\n";

    while (std::getline(file, line)) {
      if (line.length() < 2)
        continue;

      if (line[0] == 'v' && line[1] == ' ') {
        std::istringstream iss(line.substr(2));
        float x, y, z;
        iss >> x >> y >> z;

        Vec3Key key = {(int)std::round(x * SCALE), (int)std::round(y * SCALE),
                       (int)std::round(z * SCALE)};

        if (unique_map.find(key) == unique_map.end()) {
          // UPDATED: Use num_points()
          uint32_t idx = (uint32_t)geo_out.num_points();
          unique_map[key] = idx;
          auto mv = MV::from_blade(0, 0);
          mv[1] = x;
          mv[2] = y;
          mv[3] = z;

          // UPDATED: Use push_point()
          geo_out.push_point(mv);
        }
      } else if (line[0] == 'f' && line[1] == ' ') {
        std::istringstream iss(line.substr(2));
        std::string token;
        std::vector<uint32_t> poly;

        while (iss >> token) {
          size_t slash = token.find('/');
          std::string v_str =
              (slash == std::string::npos) ? token : token.substr(0, slash);
          int idx = std::stoi(v_str);
          if (idx < 0)
            idx = (int)unique_map.size() + idx + 1;
          poly.push_back(idx - 1);
        }

        for (size_t i = 0; i < poly.size() - 2; ++i) {
          face_indices.push_back(poly[0]);
          face_indices.push_back(poly[i + 1]);
          face_indices.push_back(poly[i + 2]);
        }
      }
    }

    topo_out.faces_to_vertices = std::move(face_indices);

    // UPDATED: Use num_points()
    topo_out.build_coboundaries(geo_out.num_points());

    std::cout << "  - Vertices: " << geo_out.num_points() << "\n";
    std::cout << "  - Faces:    " << topo_out.num_faces() << "\n";
  }
};

// ==============================================================================
// PUBLIC API (The Bridge)
// ==============================================================================

template <typename Sig>
void load_obj(Mesh<Sig> &mesh, const std::filesystem::path &path) {
  DODLoader<Sig>::load_internal(path, mesh.geometry, mesh.topology);
  mesh.name = path.stem().string();
}

} // namespace igneous::io