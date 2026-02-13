#pragma once

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <span>
#include <string>

namespace igneous::core::gpu {

inline bool gpu_force_enabled() {
  const char *raw = std::getenv("IGNEOUS_GPU_FORCE");
  if (raw == nullptr) {
    return false;
  }

  std::string value(raw);
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value == "1" || value == "true" || value == "yes" || value == "on";
}

inline int gpu_min_rows() {
  const char *raw = std::getenv("IGNEOUS_GPU_MIN_ROWS");
  if (raw != nullptr) {
    const int parsed = std::atoi(raw);
    if (parsed > 0) {
      return parsed;
    }
  }
  return 8192;
}

[[nodiscard]] bool available();

void invalidate_markov_cache(const void *cache_key);

[[nodiscard]] bool apply_markov_transition(const void *cache_key,
                                           std::span<const int> row_offsets,
                                           std::span<const int> col_indices,
                                           std::span<const float> weights,
                                           std::span<const float> input,
                                           std::span<float> output);

[[nodiscard]] bool apply_markov_transition_steps(
    const void *cache_key, std::span<const int> row_offsets,
    std::span<const int> col_indices, std::span<const float> weights,
    std::span<const float> input, int steps, std::span<float> output);

[[nodiscard]] bool carre_du_champ(const void *cache_key,
                                  std::span<const int> row_offsets,
                                  std::span<const int> col_indices,
                                  std::span<const float> weights,
                                  std::span<const float> f,
                                  std::span<const float> h,
                                  float inv_2t,
                                  std::span<float> output);

} // namespace igneous::core::gpu
