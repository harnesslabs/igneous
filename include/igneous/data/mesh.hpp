#pragma once

#include <string>

#include <igneous/data/buffers.hpp>
#include <igneous/data/topology.hpp>

namespace igneous::data {

template <typename Sig, Topology Topo = TriangleTopology> struct Mesh {
  using Signature = Sig;
  using TopologyType = Topo;

  data::GeometryBuffer<float, Sig> geometry;
  Topo topology;
  std::string name;

  [[nodiscard]] bool is_valid() const {
    if (geometry.num_points() == 0) {
      return false;
    }

    if constexpr (SurfaceTopology<Topo>) {
      return topology.num_faces() > 0;
    }

    return true;
  }

  void clear() {
    geometry.clear();
    topology.clear();
    name.clear();
  }
};

} // namespace igneous::data
