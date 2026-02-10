#pragma once
#include <array>
#include <bit>
#include <concepts>

namespace igneous {

// ========================================================================
// SIGNATURE
// ========================================================================

// Represents Cl(p, q, r)
// P = squares to +1, Q = squares to -1, R = squares to 0
template <int P, int Q, int R = 0>
  requires(P >= 0) && (Q >= 0) && (R >= 0)
struct Signature {
  static constexpr int p = P;
  static constexpr int q = Q;
  static constexpr int r = R;
  static constexpr int dim = P + Q + R;
  static constexpr size_t size = 1ULL << dim;

  // Safety check: Prevent making algebras too big for the stack
  // 2^10 = 1024 doubles is fine. 2^20 is unsafe.
  static_assert(dim <= 16, "Algebra dimension too large for stack allocation.");
};

using Euclidean2D = Signature<2, 0>; // Cl(2,0,0) - 2D Euclidean space
using Euclidean3D = Signature<3, 0>; // Cl(3,0,0) - 3D Euclidean space
using Minkowski =
    Signature<1, 3>; // Cl(1,3,0) - 4D Minkowski space (1 time-like dimension, 3
                     // space-like dimensions)
using PGA3D =
    Signature<3, 0, 1>; // Cl(3,0,1) - 3D Projective Geometric Algebra (PGA)

// ========================================================================
// CONCEPTS
// ========================================================================
template <typename T>
concept IsSignature = requires {
  { T::p } -> std::convertible_to<int>;
  { T::q } -> std::convertible_to<int>;
  { T::r } -> std::convertible_to<int>;
};

// ========================================================================
// Math Kernel (Compile-Time Helpers)
// =========================================================================
template <typename Sig> constexpr int get_basis_metric(int index) {
  if (index < Sig::p) {
    return 1; // Positive square
  } else if (index < Sig::p + Sig::q) {
    return -1; // Negative square
  } else {
    return 0; // Null square
  }
}

// TODO: Understand this function better
template <typename Sig>
constexpr int geometric_product_sign(unsigned int a, unsigned int b) {
  int sign = 1;
  // 1. Swap Count
  unsigned int a_temp = a >> 1;
  int swaps = 0;
  while (a_temp != 0) {
    swaps += std::popcount(a_temp & b);
    a_temp >>= 1;
  }
  if ((swaps % 2) != 0)
    sign = -sign;

  // 2. Metric Contraction
  unsigned int intersection = a & b;
  while (intersection != 0) {
    int i = std::countr_zero(intersection);
    sign *= get_basis_metric<Sig>(i);
    intersection &= intersection - 1;
  }
  return sign;
}

// ========================================================================
// Multivector Class Template
// ========================================================================
template <typename Field, IsSignature Sig> struct Multivector {
  std::array<Field, Sig::size> data;

  constexpr Multivector() : data{0} {}

  // Variadic Constructor
  template <std::same_as<Field>... Args>
    requires(sizeof...(Args) == Sig::size)
  constexpr Multivector(Args... args) : data{args...} {}

  // Factory
  static constexpr Multivector from_blade(unsigned int bitmap,
                                          Field scale = 1.0) {
    Multivector mv;
    if (bitmap < Sig::size)
      mv.data[bitmap] = scale;
    return mv;
  }

  // Accessors
  constexpr Field operator[](size_t i) const { return data[i]; }
  constexpr Field &operator[](size_t i) { return data[i]; }

  // Properties
  static constexpr size_t Size = Sig::size;

  // =========================================================
  // IMPLEMENTATION A: THE NAIVE LOOP (Runtime)
  // =========================================================
  constexpr Multivector multiply_naive(const Multivector &other) const {
    Multivector result;
    for (size_t i = 0; i < Sig::size; ++i) {
      // Runtime Branching! (The "Sparsity Check")
      if (abs(data[i]) < 1e-9)
        continue;

      for (size_t j = 0; j < Sig::size; ++j) {
        // Runtime Branching!
        if (abs(other.data[j]) < 1e-9)
          continue;

        unsigned int target_bit = i ^ j;

        // Note: 'i' and 'j' are runtime variables here,
        // so the compiler might inline the sign logic but still run it.
        int sign = geometric_product_sign<Sig>(i, j);

        if (sign != 0) {
          result.data[target_bit] +=
              data[i] * other.data[j] * static_cast<Field>(sign);
        }
      }
    }
    return result;
  }

  // =========================================================
  // IMPLEMENTATION B: TEMPLATE UNROLLING (Compile-Time)
  // =========================================================
private:
  // Core Term Calculation
  template <size_t I, size_t J>
  constexpr void add_term(Multivector &result, const Multivector &other) const {
    constexpr int sign = geometric_product_sign<Sig>(I, J);
    constexpr unsigned int target_bit = I ^ J;

    if constexpr (sign != 0) {
      // Branchless math.
      // The compiler merges this into: result[k] += this[i] * other[j] * sign
      result.data[target_bit] +=
          data[I] * other.data[J] * static_cast<Field>(sign);
    }
  }

  // Inner Loop Unroller
  template <size_t I, size_t... Js>
  constexpr void unroll_inner(Multivector &result, const Multivector &other,
                              std::index_sequence<Js...>) const {
    (add_term<I, Js>(result, other), ...);
  }

  // Outer Loop Unroller
  template <size_t... Is>
  constexpr Multivector unroll_outer(const Multivector &other,
                                     std::index_sequence<Is...>) const {
    Multivector result;
    // For every I, expand the inner loop
    (unroll_inner<Is>(result, other, std::make_index_sequence<Sig::size>{}),
     ...);
    return result;
  }

public:
  constexpr Multivector operator*(const Multivector &other) const {
    return unroll_outer(other, std::make_index_sequence<Sig::size>{});
  }

  constexpr Multivector operator+(const Multivector &other) const {
    Multivector result;
    for (size_t i = 0; i < Sig::size; ++i) {
      result.data[i] = data[i] + other.data[i];
    }
    return result;
  }
};

} // namespace igneous