#pragma once
#include "igneous/data/topology.hpp"
#include <igneous/data/buffers.hpp>
#include <string>

namespace igneous::data {

template <typename Sig, Topology Topo = TriangleTopology> struct Mesh {
  using Signature = Sig;
  using Topology = Topo;

  data::GeometryBuffer<float, Sig> geometry;

  Topo topology;

  // Metadata
  std::string name;

  bool is_valid() const {
    return !geometry.points.empty() && topology.num_faces() > 0;
  }

  void clear() {
    geometry.clear();
    topology.clear();
  }
};

} // namespace igneous::data