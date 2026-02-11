#pragma once
#include <array>
#include <bit>
#include <concepts>
#include <utility>
#include <xsimd/xsimd.hpp>

namespace igneous::core {

// ========================================================================
// 1. SIGNATURE & METRIC
// ========================================================================
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

// Metric Helper
template <typename Sig> constexpr int get_basis_metric(int index) {
  if (index < Sig::p)
    return 1;
  if (index < Sig::p + Sig::q)
    return -1;
  return 0;
}

// Computes the sign/metric for basis blade multiplication a * b
template <typename Sig>
constexpr int geometric_product_sign(unsigned int a, unsigned int b) {
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

// Concept for valid signatures
template <typename T>
concept IsSignature = requires {
  { T::p } -> std::convertible_to<int>;
  { T::dim } -> std::convertible_to<int>;
};

// Forward declaration
template <typename Field, IsSignature Sig> struct Multivector;

// ========================================================================
// 2. KERNEL INTERFACE ( The "Engine" )
// ========================================================================
// This struct defines HOW we multiply.
// The default implementation uses the generic compile-time unroll.
// We will specialize this for Euclidean3D, PGA, CGA.

template <typename Field, IsSignature Sig> struct AlgebraKernels {
  using MV = Multivector<Field, Sig>;

  // --- GENERIC IMPLEMENTATION (Fallback) ---

  // Helper: Single component accumulation
  template <bool IsWedge, size_t TargetK, size_t I>
  static constexpr void accumulate(Field &acc, const MV &a, const MV &b) {
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
  static constexpr void compute_component(MV &result, const MV &a, const MV &b,
                                          std::index_sequence<Is...>) {
    Field sum = Field(0);
    (accumulate<IsWedge, TargetK, Is>(sum, a, b), ...);
    result[TargetK] = sum;
  }

  // The Generic Product
  template <bool IsWedge, size_t... Ks>
  static constexpr MV product_generic(const MV &a, const MV &b,
                                      std::index_sequence<Ks...>) {
    MV result;
    (compute_component<IsWedge, Ks>(result, a, b,
                                    std::make_index_sequence<Sig::size>{}),
     ...);
    return result;
  }

  // --- PUBLIC API ---
  static constexpr MV geometric_product(const MV &a, const MV &b) {
    return product_generic<false>(a, b, std::make_index_sequence<Sig::size>{});
  }

  static constexpr MV wedge_product(const MV &a, const MV &b) {
    return product_generic<true>(a, b, std::make_index_sequence<Sig::size>{});
  }
};

// ========================================================================
// 3. SPECIALIZATION: EUCLIDEAN 3D (Cl(3,0))
// ========================================================================
// Hand-unrolled kernel for maximum performance.
// Eliminates all loops and complex template instantiation depth.

template <typename Field> struct AlgebraKernels<Field, Euclidean3D> {
  using MV = Multivector<Field, Euclidean3D>;

  // Basis Indices:
  // 0: Scalar
  // 1: e1, 2: e2, 3: e12
  // 4: e3, 5: e13, 6: e23, 7: e123

  static constexpr MV wedge_product(const MV &a, const MV &b) {
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
    res[5] = a[0] * b[5] + a[1] * b[4] - a[4] * b[1] +
             a[5] * b[0]; // e13 (Wait. e1^e3 = e13)
    res[6] = a[0] * b[6] + a[2] * b[4] - a[4] * b[2] +
             a[6] * b[0]; // e23 (e2^e3 = e23)

    // Grade 3 (Trivector) - Index 7 (e123)
    // e1^e23 = e123
    // e2^e13 = -e123 (swap 1,2)
    // e3^e12 = e123 (swap 1,3 then 2,3... wait. e3 e1 e2 -> -e1 e3 e2 -> e1 e2
    // e3. Yes.) Plus scalar terms.
    res[7] = a[0] * b[7] + a[1] * b[6] - a[2] * b[5] + a[3] * b[4] +
             a[4] * b[3] - a[5] * b[2] + a[6] * b[1] + a[7] * b[0];

    return res;
  }

  static constexpr MV geometric_product(const MV &a, const MV &b) {
    MV res;
    // This is the full Cl(3,0) multiplication table unrolled.
    // Generated from standard cayley table logic.

    // Scalar (0)
    res[0] = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] - a[3] * b[3] +
             a[4] * b[4] - a[5] * b[5] - a[6] * b[6] - a[7] * b[7];

    // Vector (e1, e2, e3)
    // e1: 1, 0, 3(e12*e2=-e1), 5(e13*e3=-e1) ...
    res[1] = a[0] * b[1] + a[1] * b[0] - a[2] * b[3] + a[3] * b[2] -
             a[4] * b[5] + a[5] * b[4] - a[6] * b[7] - a[7] * b[6];

    res[2] = a[0] * b[2] + a[1] * b[3] + a[2] * b[0] - a[3] * b[1] -
             a[4] * b[6] - a[5] * b[7] + a[6] * b[4] + a[7] * b[5];

    res[4] = a[0] * b[4] + a[1] * b[5] + a[2] * b[6] + a[3] * b[7] +
             a[4] * b[0] - a[5] * b[1] - a[6] * b[2] + a[7] * b[3];

    // Bivector (e12, e13, e23)
    res[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0] +
             a[4] * b[7] + a[5] * b[6] - a[6] * b[5] + a[7] * b[4];

    res[5] = a[0] * b[5] + a[1] * b[4] - a[2] * b[7] - a[3] * b[6] +
             a[4] * b[1] + a[5] * b[0] + a[6] * b[3] + a[7] * b[2];

    res[6] = a[0] * b[6] + a[1] * b[7] + a[2] * b[4] + a[3] * b[5] -
             a[4] * b[2] - a[5] * b[3] + a[6] * b[0] + a[7] * b[1];

    // Trivector (e123)
    res[7] = a[0] * b[7] + a[1] * b[6] - a[2] * b[5] + a[3] * b[4] +
             a[4] * b[3] - a[5] * b[2] + a[6] * b[1] + a[7] * b[0];

    return res;
  }
};

// ========================================================================
// 4. MULTIVECTOR
// ========================================================================
template <typename Field, IsSignature Sig> struct Multivector {
  static constexpr size_t Size = Sig::size;
  alignas(xsimd::default_arch::alignment()) std::array<Field, Sig::size> data;

  // Constructors
  constexpr Multivector() : data{0} {}

  // Helpers
  static constexpr Multivector from_blade(unsigned int bitmap, Field scale) {
    Multivector mv;
    if (bitmap < Sig::size)
      mv.data[bitmap] = scale;
    return mv;
  }

  constexpr Field operator[](size_t i) const { return data[i]; }
  constexpr Field &operator[](size_t i) { return data[i]; }

  // --- ARITHMETIC OPERATORS (Delegating to Kernel) ---

  constexpr Multivector operator*(const Multivector &other) const {
    return AlgebraKernels<Field, Sig>::geometric_product(*this, other);
  }

  constexpr Multivector operator^(const Multivector &other) const {
    return AlgebraKernels<Field, Sig>::wedge_product(*this, other);
  }

  constexpr Multivector operator+(const Multivector &other) const {
    Multivector res;
    for (size_t i = 0; i < Size; ++i)
      res[i] = data[i] + other[i];
    return res;
  }

  constexpr Multivector operator-(const Multivector &other) const {
    Multivector res;
    for (size_t i = 0; i < Size; ++i)
      res[i] = data[i] - other[i];
    return res;
  }
};

// ========================================================================
// 5. WIDE TYPES
// ========================================================================
using Packet = xsimd::batch<float>;
template <typename Sig> using WideMultivector = Multivector<Packet, Sig>;

} // namespace igneous::core