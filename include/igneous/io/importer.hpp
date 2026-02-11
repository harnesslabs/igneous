#pragma once
#include <fstream>
#include <igneous/data/mesh.hpp>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace igneous::io {

using igneous::data::Mesh;

template <typename Sig>
void load_obj(Mesh<Sig> &mesh, const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Failed to open " << filename << "\n";
    return;
  }

  std::string line;
  std::vector<uint32_t> vertex_indices;

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

      // DIRECT WRITE FIX:
      // We write straight to the packed buffer.
      // This guarantees [x, y, z] layout for get_vec3()
      mesh.geometry.packed_data.push_back(x);
      mesh.geometry.packed_data.push_back(y);
      mesh.geometry.packed_data.push_back(z);

      // For PGA (4D), pad with w=1
      if constexpr (Sig::dim == 4) {
        mesh.geometry.packed_data.push_back(1.0f);
      }
    } else if (type == "f") {
      std::string v_str;
      while (iss >> v_str) {
        size_t slash = v_str.find('/');
        int idx = std::stoi(v_str.substr(0, slash)) - 1;
        vertex_indices.push_back(static_cast<uint32_t>(idx));
      }
    }
  }

  mesh.topology.faces_to_vertices = std::move(vertex_indices);

  // Rebuild Topology
  size_t n_verts = mesh.geometry.num_points();
  mesh.topology.build_coboundaries(n_verts);

  std::cout << "[IO] Loaded " << filename << " (" << n_verts << " verts)\n";
}

} // namespace igneous::io