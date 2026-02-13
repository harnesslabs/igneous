#include <igneous/core/gpu.hpp>

namespace igneous::core::gpu {

bool available() { return false; }

void invalidate_markov_cache(const void *) {}

bool apply_markov_transition(const void *, std::span<const int>,
                             std::span<const int>, std::span<const float>,
                             std::span<const float>, std::span<float>) {
  return false;
}

bool carre_du_champ(const void *, std::span<const int>, std::span<const int>,
                    std::span<const float>, std::span<const float>,
                    std::span<const float>, float, std::span<float>) {
  return false;
}

} // namespace igneous::core::gpu
