#pragma once

#include <cstdlib>

namespace igneous::test_support {

inline void configure_deterministic_test_env() {
  setenv("IGNEOUS_BACKEND", "cpu", 1);
  setenv("IGNEOUS_NUM_THREADS", "1", 1);
  setenv("IGNEOUS_BENCH_MODE", "1", 1);
}

} // namespace igneous::test_support
