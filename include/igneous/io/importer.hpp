#pragma once
#include <fstream>
#include <igneous/core/topology.hpp>
#include <igneous/data/mesh.hpp>
#include <iostream>
#include <sstream>
#include <string>

namespace igneous::io {

using igneous::data::Mesh;

// We use 'typename Topo' to let the compiler deduce the specific topology type
template <typename Sig, typename Topo>
void load_obj(Mesh<Sig, Topo> &mesh, const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Failed to open " << filename << "\n";
    return;
  }

  std::string line;

  // Clear existing data
  mesh.geometry.clear();
  mesh.topology.clear();

  while (std::getline(file, line)) {
    if (line.empty())
      continue;
    std::istringstream iss(line);
    std::string type;
    iss >> type;

    if (type == "v") {
      float x, y, z;
      iss >> x >> y >> z;

      // Direct write to geometry buffer
      mesh.geometry.packed_data.push_back(x);
      mesh.geometry.packed_data.push_back(y);
      mesh.geometry.packed_data.push_back(z);

    } else if (type == "f") {
      // COMPILE-TIME CHECK: Only parse faces if the Topology supports them!
      if constexpr (igneous::data::SurfaceTopology<Topo>) {
        std::string v_str;
        while (iss >> v_str) {
          // OBJ is 1-based, convert to 0-based
          size_t slash = v_str.find('/');
          int idx = std::stoi(v_str.substr(0, slash)) - 1;

          // Direct push to the topology's vector
          mesh.topology.faces_to_vertices.push_back(static_cast<uint32_t>(idx));
        }
      }
      // If Topo is PointTopology, this block is discarded by the compiler.
    }
  }

  // Rebuild Topology (Generic Interface)
  // Renamed from 'build_coboundaries' to 'build_connectivity' to match Concept
  size_t n_verts = mesh.geometry.num_points();
  mesh.topology.build_connectivity(n_verts);

  std::cout << "[IO] Loaded " << filename << " (" << n_verts << " verts)\n";
}

} // namespace igneous::io