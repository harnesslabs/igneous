#pragma once

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <span>
#include <string>

namespace igneous::core::gpu {

/**
 * \brief Whether `IGNEOUS_GPU_FORCE` requests unconditional GPU attempts.
 * \return `true` when force-offload is enabled.
 */
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

/**
 * \brief Minimum row count for GPU offload (`IGNEOUS_GPU_MIN_ROWS`).
 * \return Row threshold.
 */
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

/**
 * \brief Minimum `rows * steps` for multi-step GPU offload.
 * \return Work threshold.
 */
inline long long gpu_min_row_steps() {
  const char *raw = std::getenv("IGNEOUS_GPU_MIN_ROW_STEPS");
  if (raw != nullptr) {
    const long long parsed = std::atoll(raw);
    if (parsed > 0) {
      return parsed;
    }
  }
  return 200000;
}

/// \brief Report whether a GPU backend is available at runtime.
[[nodiscard]] bool available();

/**
 * \brief Invalidate cached GPU resources associated with `cache_key`.
 * \param cache_key Cache identifier (typically structure address).
 */
void invalidate_markov_cache(const void *cache_key);

/**
 * \brief GPU Markov single-step transition.
 * \param cache_key Cache identifier.
 * \param row_offsets CSR row offsets.
 * \param col_indices CSR column indices.
 * \param weights CSR values.
 * \param input Input scalar field.
 * \param output Output scalar field.
 * \return `true` when GPU path executed successfully.
 */
[[nodiscard]] bool apply_markov_transition(const void *cache_key,
                                           std::span<const int> row_offsets,
                                           std::span<const int> col_indices,
                                           std::span<const float> weights,
                                           std::span<const float> input,
                                           std::span<float> output);

/**
 * \brief GPU Markov multi-step transition.
 * \param cache_key Cache identifier.
 * \param row_offsets CSR row offsets.
 * \param col_indices CSR column indices.
 * \param weights CSR values.
 * \param input Input scalar field.
 * \param steps Number of repeated transitions.
 * \param output Output scalar field.
 * \return `true` when GPU path executed successfully.
 */
[[nodiscard]] bool apply_markov_transition_steps(
    const void *cache_key, std::span<const int> row_offsets,
    std::span<const int> col_indices, std::span<const float> weights,
    std::span<const float> input, int steps, std::span<float> output);

/**
 * \brief GPU carré du champ evaluation over CSR diffusion graph.
 * \param cache_key Cache identifier.
 * \param row_offsets CSR row offsets.
 * \param col_indices CSR column indices.
 * \param weights CSR values.
 * \param f First scalar field.
 * \param h Second scalar field.
 * \param inv_2t Scaling factor.
 * \param output Output gamma field.
 * \return `true` when GPU path executed successfully.
 */
[[nodiscard]] bool carre_du_champ(const void *cache_key,
                                  std::span<const int> row_offsets,
                                  std::span<const int> col_indices,
                                  std::span<const float> weights,
                                  std::span<const float> f,
                                  std::span<const float> h,
                                  float inv_2t,
                                  std::span<float> output);

} // namespace igneous::core::gpu
