#pragma once
#include <igneous/data/buffers.hpp>
#include <string>

namespace igneous::data {

template <typename Sig> struct Mesh {
  // The raw data lives here
  data::GeometryBuffer<float, Sig> geometry;
  data::TopologyBuffer topology;

  // Metadata
  std::string name;

  // Helper: Are we ready to compute?
  bool is_valid() const {
    return !geometry.points.empty() && topology.num_faces() > 0;
  }

  // Helper: Clear everything
  void clear() {
    geometry.clear();
    topology.clear();
  }
};

} // namespace igneous::data