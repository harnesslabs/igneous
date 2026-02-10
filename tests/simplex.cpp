#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <fstream>
#include <igneous/algebra.hpp>
#include <igneous/mesh_loader.hpp>
#include <igneous/simplex.hpp>

using namespace igneous;
using Alg = Multivector<double, Euclidean3D>;
using Mesh = SimplexMesh<Alg>;

TEST_CASE("Hasse Diagram Construction") {
  Mesh mesh(1024 * 1024);

  // 1. Create 3 Vertices (0-Simplices)
  // Geometry is dummy data for now
  auto v1 = mesh.add_vertex(Alg::from_blade(0, 0.0));
  auto v2 = mesh.add_vertex(Alg::from_blade(0, 1.0));
  auto v3 = mesh.add_vertex(Alg::from_blade(0, 2.0));

  // Check: Vertices should have empty boundaries
  CHECK(mesh.get(v1).boundary.empty());
  CHECK(mesh.get(v1).coboundary.empty());

  // 2. Create 3 Edges (1-Simplices)
  // Edge 1 connects v1-v2
  auto e1 = mesh.add_edge(Alg::from_blade(1, 1.0), v1, v2);
  // Edge 2 connects v2-v3
  auto e2 = mesh.add_edge(Alg::from_blade(1, 1.0), v2, v3);
  // Edge 3 connects v3-v1 (Closing the loop)
  auto e3 = mesh.add_edge(Alg::from_blade(1, 1.0), v3, v1);

  SUBCASE("Level 1 Connectivity (Vertex <-> Edge)") {
    // The Edge e1 should know it comes from v1 and v2
    CHECK(mesh.get(e1).boundary.size() == 2);
    CHECK(mesh.get(e1).boundary[0] == v1);
    CHECK(mesh.get(e1).boundary[1] == v2);

    // Crucial: The Vertex v1 should know it is used by e1 and e3
    CHECK(mesh.get(v1).coboundary.size() == 2);
    // (Order depends on insertion, but they must be present)
    bool has_e1 = (mesh.get(v1).coboundary[0] == e1) ||
                  (mesh.get(v1).coboundary[1] == e1);
    bool has_e3 = (mesh.get(v1).coboundary[0] == e3) ||
                  (mesh.get(v1).coboundary[1] == e3);
    CHECK(has_e1);
    CHECK(has_e3);
  }

  // 3. Create 1 Face (2-Simplex)
  auto f1 = mesh.add_triangle(Alg::from_blade(2, 1.0), e1, e2, e3);

  SUBCASE("Level 2 Connectivity (Edge <-> Face)") {
    // Face should consist of 3 edges
    CHECK(mesh.get(f1).boundary.size() == 3);

    // Edge e1 should now have a coboundary (the face)
    CHECK(mesh.get(e1).coboundary.size() == 1);
    CHECK(mesh.get(e1).coboundary[0] == f1);
  }
}

TEST_CASE("Topology Traversal (The 'Star' Operation)") {
  // In topology, the 'Star' of a vertex is the set of all simplices connected
  // to it. This is fundamental for calculating curvature around a point.
  Mesh mesh(1024 * 1024);

  auto v_center = mesh.add_vertex(Alg::from_blade(0, 0.0));
  auto v_north = mesh.add_vertex(Alg::from_blade(0, 1.0));
  auto v_east = mesh.add_vertex(Alg::from_blade(0, 1.0));

  auto e1 = mesh.add_edge(Alg::from_blade(1, 1.0), v_center, v_north);
  auto e2 = mesh.add_edge(Alg::from_blade(1, 1.0), v_center, v_east);

  // If we ask v_center "Who connects to you?", we should get e1 and e2.
  auto &co = mesh.get(v_center).coboundary;

  CHECK(co.size() == 2);
  CHECK(co[0] == e1);
  CHECK(co[1] == e2);
  // This proves we can traverse UP the diagram
}

TEST_CASE("OBJ Mesh Loader & Topology Check") {
  // 1. Create a dummy .obj file (A single square made of 2 triangles)
  // Vertices:
  // 1 -- 2
  // |  / |
  // 0 -- 3
  //
  // Triangles: (0, 1, 2) and (0, 2, 3)
  // Shared Edge: (0, 2)

  std::ofstream tmp("test_square.obj");
  tmp << "v 0 0 0\n"; // 0
  tmp << "v 0 1 0\n"; // 1
  tmp << "v 1 1 0\n"; // 2
  tmp << "v 1 0 0\n"; // 3
  tmp << "f 1 2 3\n"; // Tri 1 (Indices 1-based)
  tmp << "f 1 3 4\n"; // Tri 2
  tmp.close();

  Mesh mesh(1024 * 1024);
  MeshIO<Alg>::load_obj(mesh, "test_square.obj");

  // 2. Verify Counts
  // Vertices: 4
  // Faces: 2
  // Edges: 5 (Outer 4 + Diagonal 1)
  // If we didn't dedup edges, we would have 6 edges (3 per triangle).

  // We need to iterate to count by dimension
  int v_count = 0, e_count = 0, f_count = 0;

  // Simple loop over all elements (handle 0 to size-1)
  for (size_t i = 0; i < mesh.size(); ++i) {
    SimplexHandle h = {(uint32_t)i};
    auto &s = mesh.get(h);
    if (s.dimension == 0)
      v_count++;
    if (s.dimension == 1)
      e_count++;
    if (s.dimension == 2)
      f_count++;
  }

  CHECK(v_count == 4);
  CHECK(f_count == 2);

  // The moment of truth: Did the diagonal edge get shared?
  // 4 outer edges + 1 shared diagonal = 5 edges.
  // If duplicate: 3 + 3 = 6 edges.
  CHECK(e_count == 5);

  std::filesystem::remove("test_square.obj");
}