#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <algorithm>
#include <igneous/data/topology.hpp>

TEST_CASE("TriangleTopology builds face and neighbor CSR") {
  igneous::data::TriangleTopology topo;
  topo.faces_to_vertices = {
      0, 1, 2,
      0, 2, 3,
  };

  topo.build({4, true});

  CHECK(topo.num_faces() == 2);
  CHECK(topo.face_v0.size() == 2);
  CHECK(topo.face_v1.size() == 2);
  CHECK(topo.face_v2.size() == 2);

  for (size_t i = 0; i + 1 < topo.vertex_face_offsets.size(); ++i) {
    CHECK(topo.vertex_face_offsets[i] <= topo.vertex_face_offsets[i + 1]);
  }

  const auto v0_faces = topo.get_faces_for_vertex(0);
  CHECK(v0_faces.size() == 2);

  const auto v0_neighbors = topo.get_vertex_neighbors(0);
  CHECK(v0_neighbors.size() == 3);
  CHECK(std::find(v0_neighbors.begin(), v0_neighbors.end(), 1) != v0_neighbors.end());
  CHECK(std::find(v0_neighbors.begin(), v0_neighbors.end(), 2) != v0_neighbors.end());
  CHECK(std::find(v0_neighbors.begin(), v0_neighbors.end(), 3) != v0_neighbors.end());
}
