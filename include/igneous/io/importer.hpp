#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

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

  std::string line;

  // Clear existing data
  mesh.geometry.clear();
  mesh.topology.clear();

  // 1. READ LOOP (Parse Data Only)
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
      // FIX: Only parse face indices here. DO NOT BUILD.
      // We use 'if constexpr' to check if the topology even supports faces.
      if constexpr (igneous::data::SurfaceTopology<Topo>) {
        std::string v_str;
        while (iss >> v_str) {
          size_t slash = v_str.find('/');
          int idx = std::stoi(v_str.substr(0, slash)) - 1;
          mesh.topology.faces_to_vertices.push_back(static_cast<uint32_t>(idx));
        }
      }
      // If it's DiffusionTopology or PointTopology, we just ignore 'f' lines.
    }
  }

  // 2. BUILD STEP (Once, after loading all data)
  size_t n_verts = mesh.geometry.num_points();

  if constexpr (std::is_same_v<Topo, igneous::data::TriangleTopology>) {
    // Triangle topology needs the count to resize arrays
    mesh.topology.build({n_verts});
  } else if constexpr (std::is_same_v<Topo, igneous::data::DiffusionTopology>) {
    // Diffusion topology needs the raw vertex data for the KD-Tree.
    // { vector } uses aggregate initialization for the Input struct.
    mesh.topology.build({mesh.geometry.packed_data});
  } else {
    // Fallback (PointTopology)
    mesh.topology.build({});
  }

  std::cout << "[IO] Loaded " << filename << " (" << n_verts << " verts)\n";
}

} // namespace igneous::io