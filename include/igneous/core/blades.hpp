#pragma once
#include <cmath>

namespace igneous::core {

/// \brief 3D bivector with `(xy, yz, zx)` components.
struct Bivec3 {
  /// \brief XY plane component (`e12`).
  float xy;
  /// \brief YZ plane component (`e23`).
  float yz;
  /// \brief ZX plane component (`e31`).
  float zx;

  /**
   * \brief Squared Euclidean magnitude.
   * \return Squared norm.
   */
  float norm_sq() const {
    return xy * xy + yz * yz + zx * zx;
  }

  /**
   * \brief Euclidean magnitude.
   * \return Norm.
   */
  float norm() const {
    return std::sqrt(norm_sq());
  }
};

/// \brief Lightweight 3D vector utility for geometry kernels.
struct Vec3 {
  /// \brief X coordinate.
  float x;
  /// \brief Y coordinate.
  float y;
  /// \brief Z coordinate.
  float z;

  /**
   * \brief Vector addition.
   * \param o Right-hand operand.
   * \return Sum vector.
   */
  Vec3 operator+(const Vec3& o) const {
    return {x + o.x, y + o.y, z + o.z};
  }
  /**
   * \brief Vector subtraction.
   * \param o Right-hand operand.
   * \return Difference vector.
   */
  Vec3 operator-(const Vec3& o) const {
    return {x - o.x, y - o.y, z - o.z};
  }

  /**
   * \brief Scalar multiplication.
   * \param s Scalar factor.
   * \return Scaled vector.
   */
  Vec3 operator*(float s) const {
    return {x * s, y * s, z * s};
  }

  /**
   * \brief Dot product.
   * \param o Right-hand operand.
   * \return Dot product scalar.
   */
  float dot(const Vec3& o) const {
    return x * o.x + y * o.y + z * o.z;
  }

  /**
   * \brief Exterior product producing a bivector.
   * \param o Right-hand operand.
   * \return Bivector wedge product.
   */
  Bivec3 operator^(const Vec3& o) const {
    return {x * o.y - y * o.x, y * o.z - z * o.y, z * o.x - x * o.z};
  }
};

} // namespace igneous::core
