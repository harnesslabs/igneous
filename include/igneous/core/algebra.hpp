#pragma once
#include <array>
#include <bit>
#include <concepts>
#include <utility>
#include <xsimd/xsimd.hpp>

namespace igneous::core {

/// \brief Clifford signature descriptor (`p`,`q`,`r`).
template <int P, int Q, int R = 0>
  requires(P >= 0) && (Q >= 0) && (R >= 0)
struct Signature {
  static constexpr int p = P;
  static constexpr int q = Q;
  static constexpr int r = R;
  static constexpr int dim = P + Q + R;
  static constexpr size_t size = 1ULL << dim;
};

using Euclidean3D = Signature<3, 0>;
using PGA = Signature<3, 0, 1>;
using CGA = Signature<4, 1>;

/**
 * \brief Metric coefficient for basis direction `index`.
 * \param index Basis vector index.
 * \return `+1`, `-1`, or `0` depending on signature axis type.
 */
template <typename Sig> constexpr int get_basis_metric(int index) {
  if (index < Sig::p)
    return 1;
  if (index < Sig::p + Sig::q)
    return -1;
  return 0;
}

/**
 * \brief Sign and metric factor for geometric product of basis blades.
 * \param a Left blade bitmap.
 * \param b Right blade bitmap.
 * \return Multiplicative sign/metric coefficient.
 */
template <typename Sig> constexpr int geometric_product_sign(unsigned int a, unsigned int b) {
  int sign = 1;
  unsigned int a_temp = a >> 1;
  int swaps = 0;
  while (a_temp != 0) {
    swaps += std::popcount(a_temp & b);
    a_temp >>= 1;
  }
  if ((swaps % 2) != 0)
    sign = -sign;

  unsigned int intersection = a & b;
  while (intersection != 0) {
    int i = std::countr_zero(intersection);
    sign *= get_basis_metric<Sig>(i);
    intersection &= intersection - 1;
  }
  return sign;
}

/// \brief Concept for valid signature-like types.
template <typename T>
concept IsSignature = requires {
  { T::p } -> std::convertible_to<int>;
  { T::dim } -> std::convertible_to<int>;
};

/// \brief Forward declaration of multivector type.
template <typename Field, IsSignature Sig> struct Multivector;

/// \brief Product-kernel policy object used by `Multivector`.
template <typename Field, IsSignature Sig> struct AlgebraKernels {
  using MV = Multivector<Field, Sig>;

  // --- GENERIC IMPLEMENTATION (Fallback) ---

  // Helper: Single component accumulation
  template <bool IsWedge, size_t TargetK, size_t I>
  static constexpr void accumulate(Field& acc, const MV& a, const MV& b) {
    constexpr size_t J = I ^ TargetK;

    // Wedge Constraint: No shared factors allowed
    if constexpr (IsWedge && (I & J) != 0)
      return;

    constexpr int sign = geometric_product_sign<Sig>(I, J);
    if constexpr (sign != 0) {
      if constexpr (sign == 1)
        acc += a[I] * b[J];
      else
        acc -= a[I] * b[J];
    }
  }

  // Helper: Unroll sum for one target component
  template <bool IsWedge, size_t TargetK, size_t... Is>
  static constexpr void compute_component(MV& result, const MV& a, const MV& b,
                                          std::index_sequence<Is...>) {
    Field sum = Field(0);
    (accumulate<IsWedge, TargetK, Is>(sum, a, b), ...);
    result[TargetK] = sum;
  }

  // The Generic Product
  template <bool IsWedge, size_t... Ks>
  static constexpr MV product_generic(const MV& a, const MV& b, std::index_sequence<Ks...>) {
    MV result;
    (compute_component<IsWedge, Ks>(result, a, b, std::make_index_sequence<Sig::size>{}), ...);
    return result;
  }

  /**
   * \brief Geometric product.
   * \param a Left operand.
   * \param b Right operand.
   * \return Product multivector.
   */
  static constexpr MV geometric_product(const MV& a, const MV& b) {
    return product_generic<false>(a, b, std::make_index_sequence<Sig::size>{});
  }

  /**
   * \brief Exterior/wedge product.
   * \param a Left operand.
   * \param b Right operand.
   * \return Wedge-product multivector.
   */
  static constexpr MV wedge_product(const MV& a, const MV& b) {
    return product_generic<true>(a, b, std::make_index_sequence<Sig::size>{});
  }
};

/// \brief Hand-unrolled `Cl(3,0)` kernel specialization.
template <typename Field> struct AlgebraKernels<Field, Euclidean3D> {
  using MV = Multivector<Field, Euclidean3D>;

  // Basis Indices:
  // 0: Scalar
  // 1: e1, 2: e2, 3: e12
  // 4: e3, 5: e13, 6: e23, 7: e123

  static constexpr MV wedge_product(const MV& a, const MV& b) {
    MV res;
    // 1. Scalar part (Grade 0 ^ Grade 0)
    // In wedge, scalar * anything is just scaling.
    // But typically wedge is defined for vectors/bivectors.
    // Let's implement full distributivity: (S+V+B+T) ^ (S+V+B+T)

    // Grade 0 (Scalar) - Index 0
    res[0] = a[0] * b[0];

    // Grade 1 (Vectors) - Indices 1, 2, 4
    res[1] = a[0] * b[1] + a[1] * b[0];
    res[2] = a[0] * b[2] + a[2] * b[0];
    res[4] = a[0] * b[4] + a[4] * b[0];

    // Grade 2 (Bivectors) - Indices 3(e12), 5(e13), 6(e23)
    // Components:
    // e1^e2 = e12.
    // e2^e1 = -e12.
    // Scalar interactions included.
    res[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]; // e12
    res[5] = a[0] * b[5] + a[1] * b[4] - a[4] * b[1] + a[5] * b[0]; // e13 (Wait. e1^e3 = e13)
    res[6] = a[0] * b[6] + a[2] * b[4] - a[4] * b[2] + a[6] * b[0]; // e23 (e2^e3 = e23)

    // Grade 3 (Trivector) - Index 7 (e123)
    // e1^e23 = e123
    // e2^e13 = -e123 (swap 1,2)
    // e3^e12 = e123 (swap 1,3 then 2,3... wait. e3 e1 e2 -> -e1 e3 e2 -> e1 e2
    // e3. Yes.) Plus scalar terms.
    res[7] = a[0] * b[7] + a[1] * b[6] - a[2] * b[5] + a[3] * b[4] + a[4] * b[3] - a[5] * b[2] +
             a[6] * b[1] + a[7] * b[0];

    return res;
  }

  static constexpr MV geometric_product(const MV& a, const MV& b) {
    MV res;
    // This is the full Cl(3,0) multiplication table unrolled.
    // Generated from standard cayley table logic.

    // Scalar (0)
    res[0] = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] - a[3] * b[3] + a[4] * b[4] - a[5] * b[5] -
             a[6] * b[6] - a[7] * b[7];

    // Vector (e1, e2, e3)
    // e1: 1, 0, 3(e12*e2=-e1), 5(e13*e3=-e1) ...
    res[1] = a[0] * b[1] + a[1] * b[0] - a[2] * b[3] + a[3] * b[2] - a[4] * b[5] + a[5] * b[4] -
             a[6] * b[7] - a[7] * b[6];

    res[2] = a[0] * b[2] + a[1] * b[3] + a[2] * b[0] - a[3] * b[1] - a[4] * b[6] - a[5] * b[7] +
             a[6] * b[4] + a[7] * b[5];

    res[4] = a[0] * b[4] + a[1] * b[5] + a[2] * b[6] + a[3] * b[7] + a[4] * b[0] - a[5] * b[1] -
             a[6] * b[2] + a[7] * b[3];

    // Bivector (e12, e13, e23)
    res[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0] + a[4] * b[7] + a[5] * b[6] -
             a[6] * b[5] + a[7] * b[4];

    res[5] = a[0] * b[5] + a[1] * b[4] - a[2] * b[7] - a[3] * b[6] + a[4] * b[1] + a[5] * b[0] +
             a[6] * b[3] + a[7] * b[2];

    res[6] = a[0] * b[6] + a[1] * b[7] + a[2] * b[4] + a[3] * b[5] - a[4] * b[2] - a[5] * b[3] +
             a[6] * b[0] + a[7] * b[1];

    // Trivector (e123)
    res[7] = a[0] * b[7] + a[1] * b[6] - a[2] * b[5] + a[3] * b[4] + a[4] * b[3] - a[5] * b[2] +
             a[6] * b[1] + a[7] * b[0];

    return res;
  }
};

/// \brief Fixed-size multivector value type.
template <typename Field, IsSignature Sig> struct Multivector {
  static constexpr size_t Size = Sig::size;
  alignas(xsimd::default_arch::alignment()) std::array<Field, Sig::size> data;

  /// \brief Zero-initialized multivector.
  constexpr Multivector() : data{0} {}

  /**
   * \brief Build a multivector with one basis blade set to `scale`.
   * \param bitmap Blade bitmap index.
   * \param scale Coefficient value.
   * \return Newly constructed multivector.
   */
  static constexpr Multivector from_blade(unsigned int bitmap, Field scale) {
    Multivector mv;
    if (bitmap < Sig::size)
      mv.data[bitmap] = scale;
    return mv;
  }

  /**
   * \brief Immutable component access.
   * \param i Component index.
   * \return Component value.
   */
  constexpr Field operator[](size_t i) const {
    return data[i];
  }
  /**
   * \brief Mutable component access.
   * \param i Component index.
   * \return Mutable reference to component.
   */
  constexpr Field& operator[](size_t i) {
    return data[i];
  }

  /**
   * \brief Geometric product.
   * \param other Right-hand operand.
   * \return Product multivector.
   */
  constexpr Multivector operator*(const Multivector& other) const {
    return AlgebraKernels<Field, Sig>::geometric_product(*this, other);
  }

  /**
   * \brief Wedge product.
   * \param other Right-hand operand.
   * \return Wedge-product multivector.
   */
  constexpr Multivector operator^(const Multivector& other) const {
    return AlgebraKernels<Field, Sig>::wedge_product(*this, other);
  }

  /**
   * \brief Component-wise addition.
   * \param other Right-hand operand.
   * \return Sum multivector.
   */
  constexpr Multivector operator+(const Multivector& other) const {
    Multivector res;
    for (size_t i = 0; i < Size; ++i)
      res[i] = data[i] + other[i];
    return res;
  }

  /**
   * \brief Component-wise subtraction.
   * \param other Right-hand operand.
   * \return Difference multivector.
   */
  constexpr Multivector operator-(const Multivector& other) const {
    Multivector res;
    for (size_t i = 0; i < Size; ++i)
      res[i] = data[i] - other[i];
    return res;
  }
};

/// \brief SIMD packet scalar used by wide multivectors.
using Packet = xsimd::batch<float>;
/// \brief Multivector whose scalar components are SIMD packets.
template <typename Sig> using WideMultivector = Multivector<Packet, Sig>;

} // namespace igneous::core
