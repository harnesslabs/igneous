#pragma once
#include <cmath>

namespace igneous::core {

// A pure 3-component Bivector (xy, yz, zx)
// Used for Cross Products / Areas
struct Bivec3 {
  float xy, yz,
      zx; // e12, e23, e31 (Order varies by convention, sticking to xy, yz, zx)

  // Magnitude Squared: |B|^2 = -B*B (technically).
  // For Euclidean, it's just sum of squares.
  float norm_sq() const { return xy * xy + yz * yz + zx * zx; }

  float norm() const { return std::sqrt(norm_sq()); }
};

// A pure 3-component Vector (x, y, z)
struct Vec3 {
  float x, y, z;

  // Addition
  Vec3 operator+(const Vec3 &o) const { return {x + o.x, y + o.y, z + o.z}; }
  Vec3 operator-(const Vec3 &o) const { return {x - o.x, y - o.y, z - o.z}; }

  // Scalar Mult
  Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }

  // Dot Product (Inner Product) -> Scalar
  // Note: In GA, a|b is the scalar part of ab
  float dot(const Vec3 &o) const { return x * o.x + y * o.y + z * o.z; }

  // Wedge Product (Outer Product) -> Bivector
  // This is the "Cross Product" logic
  Bivec3 operator^(const Vec3 &o) const {
    return {
        x * o.y - y * o.x, // xy (e12)
        y * o.z - z * o.y, // yz (e23)
        z * o.x - x * o.z  // zx (e31/e13 sign depends on basis)
        // Note: Standard Euclidean cross product result corresponds to dual of
        // wedge. e1^e2 = e12.
    };
  }
};

} // namespace igneous::core