#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <igneous/algebra.hpp>
#include <igneous/simplex.hpp>

using namespace igneous;

// Use Euclidean 3D for standard geometry testing
using Alg = Multivector<double, Euclidean3D>;
using Mesh = SimplexMesh<Alg>;

TEST_CASE("Simplex Mesh Operations") {
  Mesh mesh(1024 * 1024); // 1MB Arena

  // 1. Create a Triangle (3 Points, 3 Lines, 1 Face)

  // Define Points (0-Simplices)
  // Origin
  auto p0_geom =
      Alg::from_blade(0, 1.0); // Scalar 1 for point? (Just dummy data for VGA)
  auto h0 = mesh.add(p0_geom, 0);

  // X-Axis
  auto p1_geom = Alg::from_blade(1, 1.0); // e1
  auto h1 = mesh.add(p1_geom, 0);

  // Y-Axis
  auto p2_geom = Alg::from_blade(2, 1.0); // e2
  auto h2 = mesh.add(p2_geom, 0);

  CHECK(mesh.size() == 3);

  // Check Data Persistence
  CHECK(mesh.get(h1).geometry[1] == doctest::Approx(1.0));
  CHECK(mesh.get(h2).geometry[2] == doctest::Approx(1.0));

  SUBCASE("Memory Contiguity Check") {
    // In a linear arena, these should be adjacent in memory.
    auto *ptr0 = &mesh.get(h0);
    auto *ptr1 = &mesh.get(h1);

    // Calculate byte difference
    size_t diff =
        reinterpret_cast<uintptr_t>(ptr1) - reinterpret_cast<uintptr_t>(ptr0);

    // Should be exactly sizeof(Simplex<Alg>) (aligned)
    // This confirms they are packed tight!
    CHECK(diff >= sizeof(Simplex<Alg>));
    CHECK(diff < sizeof(Simplex<Alg>) + 16); // Allow small alignment padding
  }
}