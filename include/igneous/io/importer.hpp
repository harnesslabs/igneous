#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>

#include <igneous/data/mesh.hpp>
#include <igneous/data/topology.hpp>

namespace igneous::io {

using igneous::data::Mesh;

template <typename Sig, typename Topo>
void load_obj(Mesh<Sig, Topo> &mesh, const std::string &filename) {
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
      mesh.geometry.push_point({x, y, z});
      continue;
    }

    if (type == "f") {
      if constexpr (igneous::data::SurfaceTopology<Topo>) {
        std::string v_str;
        while (iss >> v_str) {
          const size_t slash = v_str.find('/');
          const int idx = std::stoi(v_str.substr(0, slash)) - 1;
          mesh.topology.faces_to_vertices.push_back(static_cast<uint32_t>(idx));
        }
      }
    }
  }

  const size_t n_verts = mesh.geometry.num_points();

  if constexpr (std::is_same_v<Topo, igneous::data::TriangleTopology>) {
    mesh.topology.build({n_verts, true});
  } else if constexpr (std::is_same_v<Topo, igneous::data::DiffusionTopology>) {
    mesh.topology.build({mesh.geometry.x_span(), mesh.geometry.y_span(), mesh.geometry.z_span()});
  } else {
    mesh.topology.build({});
  }

  std::cout << "[IO] Loaded " << filename << " (" << n_verts << " verts)\n";
}

} // namespace igneous::io
