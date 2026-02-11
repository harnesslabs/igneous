#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <igneous/core/algebra.hpp>

using namespace igneous;
using igneous::core::Multivector;
using igneous::core::Signature;

// Helper to define "Approx" for our multivectors if needed,
// but for now we check components individually using doctest::Approx.

// ============================================================================
// TEST SUITE 1: Euclidean 3D (Cl(3, 0, 0))
// Standard 3D space. Basis: e1, e2, e3. All square to +1.
// ============================================================================
TEST_CASE("Euclidean 3D Geometric Algebra") {
  // Basis Indices: 0, 1, 2
  using VGA = Multivector<double, Signature<3, 0, 0>>;

  auto e1 = VGA::from_blade(1 << 0, 1); // Bit 0 (1)
  auto e2 = VGA::from_blade(1 << 1, 1); // Bit 1 (2)
  auto e3 = VGA::from_blade(1 << 2, 1); // Bit 2 (4)

  SUBCASE("Generators Square to +1") {
    CHECK((e1 * e1)[0] == doctest::Approx(1.0));
    CHECK((e2 * e2)[0] == doctest::Approx(1.0));
    CHECK((e3 * e3)[0] == doctest::Approx(1.0));
  }

  SUBCASE("Anti-Commutativity (e1 e2 = -e2 e1)") {
    auto e12 = e1 * e2;
    auto e21 = e2 * e1;

    // Check it is a bivector (bit 0^1 = 3)
    CHECK(e12[3] == doctest::Approx(1.0));

    // Check sign flip
    CHECK(e21[3] == doctest::Approx(-1.0));

    // Check addition cancels out
    auto sum = e12 + e21;
    CHECK(sum[3] == doctest::Approx(0.0));
  }

  SUBCASE("The Pseudoscalar I (e1 e2 e3)") {
    auto I = e1 * e2 * e3;

    // Should be bit 0^1^2 = 7 (111 binary)
    CHECK(I[7] == doctest::Approx(1.0));

    // In 3D Euclidean, I^2 = -1
    auto I_sq = I * I;
    CHECK(I_sq[0] == doctest::Approx(-1.0));
  }

  SUBCASE("Bivector Squaring (Rotors)") {
    // (e1 e2)^2 = e1 e2 e1 e2 = -e1 e1 e2 e2 = -1 * 1 * 1 = -1
    auto e12 = e1 * e2;
    CHECK((e12 * e12)[0] == doctest::Approx(-1.0));
  }
}

// ============================================================================
// TEST SUITE 2: Minkowski Spacetime (Cl(1, 3, 0))
// Used in Special Relativity. Basis: e0 (time, +), e1,e2,e3 (space, -)
// ============================================================================
TEST_CASE("Minkowski Spacetime Algebra (STA)") {
  // Signature: 1 Positive (Time), 3 Negative (Space)
  // Index 0 -> Time (+), Indices 1,2,3 -> Space (-)
  using STA = Multivector<double, Signature<1, 3, 0>>;

  auto gamma0 = STA::from_blade(1 << 0, 1); // Time basis
  auto gamma1 = STA::from_blade(1 << 1, 1); // Space x
  auto gamma2 = STA::from_blade(1 << 2, 1); // Space y

  SUBCASE("Metric Signature (+, -, -, -)") {
    // Time squares to +1
    CHECK((gamma0 * gamma0)[0] == doctest::Approx(1.0));

    // Space squares to -1
    CHECK((gamma1 * gamma1)[0] == doctest::Approx(-1.0));
    CHECK((gamma2 * gamma2)[0] == doctest::Approx(-1.0));
  }

  SUBCASE("Spacetime Bivectors (Electric Fields)") {
    // Time * Space (e.g. gamma0 * gamma1) squares to +1
    // (g0 g1) (g0 g1) = -g0 g0 g1 g1 = -1 * 1 * -1 = +1
    auto E1 = gamma0 * gamma1;
    CHECK((E1 * E1)[0] == doctest::Approx(1.0));
  }

  SUBCASE("Spatial Bivectors (Magnetic Fields)") {
    // Space * Space (e.g. gamma1 * gamma2) squares to -1
    // (g1 g2) (g1 g2) = -g1 g1 g2 g2 = -1 * -1 * -1 = -1
    auto B3 = gamma1 * gamma2;
    CHECK((B3 * B3)[0] == doctest::Approx(-1.0));
  }
}

// ============================================================================
// TEST SUITE 3: Projective Geometric Algebra (Cl(3, 0, 1))
// The "modern" way to do 3D Euclidean geometry (points, lines, planes).
// Basis e0 squares to 0 (Degenerate metric).
// ============================================================================
TEST_CASE("Projective Geometric Algebra (PGA 3D)") {
  // 3 Positive (Space), 0 Negative, 1 Null (Origin/Infinity)
  using PGA = Multivector<double, Signature<3, 0, 1>>;

  // In PGA, basis vectors are usually PLANES
  auto e1 = PGA::from_blade(1 << 0, 1); // Plane x=0
  auto e2 = PGA::from_blade(1 << 1, 1); // Plane y=0
  auto e3 = PGA::from_blade(1 << 2, 1); // Plane z=0
  auto e0 = PGA::from_blade(1 << 3, 1); // Plane at infinity

  SUBCASE("Metric Properties") {
    // Spatial planes normalize to 1
    CHECK((e1 * e1)[0] == doctest::Approx(1.0));
    CHECK((e2 * e2)[0] == doctest::Approx(1.0));
    CHECK((e3 * e3)[0] == doctest::Approx(1.0));

    // The plane at infinity is null
    CHECK((e0 * e0)[0] == doctest::Approx(0.0));
  }

  SUBCASE("Orthogonality of Principal Planes") {
    // e1 and e2 are orthogonal, so their dot product is 0.
    // In GA, a.b = 0.5 * (ab + ba).
    // Since e1 e2 = -e2 e1 (anticommutativity), this sum is 0.
    auto e12 = e1 * e2;
    auto e21 = e2 * e1;

    // Check they are opposite
    CHECK(e12[3] == doctest::Approx(-e21[3])); // bit 3 is e1^e2 (0011)

    // Their sum (dot product) should be zero
    auto dot = e12 + e21;
    CHECK(dot[3] == doctest::Approx(0.0));
  }

  SUBCASE("Pseudoscalar") {
    // The volume element e1 e2 e3 e0
    auto I = e1 * e2 * e3 * e0;

    // Should be bit 1111 = 15
    CHECK(I[15] == doctest::Approx(1.0));
  }
}

// ============================================================================
// TEST SUITE 4: Axiomatic Properties
// Ensuring the code behaves like a Ring/Algebra
// ============================================================================
TEST_CASE("Algebraic Axioms") {
  using Alg = Multivector<double, Signature<2, 0, 0>>; // 2D Plane

  auto a = Alg::from_blade(1, 2.0) + Alg::from_blade(2, 3.0); // 2e1 + 3e2
  auto b = Alg::from_blade(1, 0.5) + Alg::from_blade(2, 4.0); // 0.5e1 + 4e2
  auto c = Alg::from_blade(3, 1.0);                           // 1e12

  SUBCASE("Distributivity: A(B + C) = AB + AC") {
    auto left = a * (b + c);
    auto right = (a * b) + (a * c);

    for (size_t i = 0; i < Alg::Size; ++i) {
      CHECK(left[i] == doctest::Approx(right[i]));
    }
  }

  SUBCASE("Associativity: (AB)C = A(BC)") {
    auto left = (a * b) * c;
    auto right = a * (b * c);

    for (size_t i = 0; i < Alg::Size; ++i) {
      CHECK(left[i] == doctest::Approx(right[i]));
    }
  }

  SUBCASE("Naive vs Unrolled Implementation Match") {
    // This confirms our optimization didn't break the math
    auto fast = a * b;
    auto slow = a.multiply_naive(b);

    for (size_t i = 0; i < Alg::Size; ++i) {
      CHECK(fast[i] == doctest::Approx(slow[i]));
    }
  }
}