// include/igneous/mesh_loader_dod.hpp
#pragma once
#include <filesystem>
#include <fstream>
#include <igneous/geometry.hpp>
#include <igneous/topology.hpp>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace igneous {

// --- Spatial Hashing ---
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

// Template on Signature, not "Algebra" (which usually implies the Multivector)
template <typename Sig> struct DODLoader {
  // Define the specific Multivector type we are constructing
  using Field = float;
  using MV = Multivector<Field, Sig>;

  static void load_obj(const std::filesystem::path &path,
                       GeometryBuffer<Field, Sig> &geo_out,
                       TopologyBuffer &topo_out) {
    std::ifstream file(path);
    if (!file.is_open())
      throw std::runtime_error("File not found: " + path.string());

    geo_out.clear();
    topo_out.clear();

    // Temporary storage for face indices before moving to buffer
    std::vector<uint32_t> face_indices;

    std::unordered_map<Vec3Key, uint32_t, Vec3Hash> unique_map;
    const double SCALE = 10000.0;

    std::string line;
    std::cout << "[DOD] Streaming " << path.filename() << " to buffers..."
              << std::endl;

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
          uint32_t idx = (uint32_t)geo_out.points.size();
          unique_map[key] = idx;

          // FIXED: Use MV::from_blade instead of Sig::from_blade
          auto mv = MV::from_blade(0, 0);
          mv[1] = x;
          mv[2] = y;
          mv[3] = z;
          geo_out.points.push_back(mv);
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

          // OBJ is 1-based, we need 0-based
          poly.push_back(idx - 1);
        }

        // Fan Triangulation
        for (size_t i = 0; i < poly.size() - 2; ++i) {
          face_indices.push_back(poly[0]);
          face_indices.push_back(poly[i + 1]);
          face_indices.push_back(poly[i + 2]);
        }
      }
    }

    // Move data into the TopologyBuffer
    topo_out.faces_to_vertices = std::move(face_indices);

    // FIXED: build_coboundaries only takes num_vertices.
    // It reads faces_to_vertices directly from itself.
    topo_out.build_coboundaries(geo_out.points.size());

    std::cout << "[DOD] Loaded " << geo_out.points.size()
              << " unique vertices.\n";
    std::cout << "[DOD] Loaded " << topo_out.num_faces() << " faces.\n";
  }
};

} // namespace igneous