#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <igneous/data/space.hpp>
#include <igneous/data/structure.hpp>

namespace igneous::io {

using igneous::data::Space;

/**
 * \brief Load geometry/connectivity from an OBJ file into a `Space`.
 *
 * This function is intentionally load-only:
 * - point positions are parsed from `v` records
 * - surface faces are parsed from `f` records when `StructureT` is a surface structure
 * - no `structure.build(...)` call is performed
 *
 * \param mesh Destination space to overwrite.
 * \param filename OBJ file path.
 */
template <typename StructureT> void load_obj(Space<StructureT>& mesh, const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Failed to open " << filename << "\n";
    return;
  }

  mesh.clear();

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty()) {
      continue;
    }

    std::istringstream iss(line);
    std::string type;
    iss >> type;

    if (type == "v") {
      float x = 0.0f;
      float y = 0.0f;
      float z = 0.0f;
      iss >> x >> y >> z;
      mesh.push_point({x, y, z});
      continue;
    }

    if (type == "f") {
      if constexpr (igneous::data::SurfaceStructure<StructureT>) {
        std::string v_str;
        while (iss >> v_str) {
          const size_t slash = v_str.find('/');
          const int idx = std::stoi(v_str.substr(0, slash)) - 1;
          mesh.structure.faces_to_vertices.push_back(static_cast<uint32_t>(idx));
        }
      }
    }
  }
  std::cout << "[IO] Loaded " << filename << " (" << mesh.num_points() << " verts)\n";
}

} // namespace igneous::io
