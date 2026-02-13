#pragma once

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

namespace igneous::core {

enum class ComputeBackend {
  Cpu,
  CpuParallel,
  Gpu
};

inline ComputeBackend compute_backend_from_env() {
  const char *raw = std::getenv("IGNEOUS_BACKEND");
  if (raw == nullptr) {
    return ComputeBackend::CpuParallel;
  }

  std::string value(raw);
  for (char &c : value) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }

  if (value == "cpu") {
    return ComputeBackend::Cpu;
  }
  if (value == "gpu") {
    return ComputeBackend::Gpu;
  }
  return ComputeBackend::CpuParallel;
}

inline int compute_thread_count() {
  const char *raw = std::getenv("IGNEOUS_NUM_THREADS");
  if (raw != nullptr) {
    const int requested = std::atoi(raw);
    if (requested > 0) {
      return requested;
    }
  }

  const unsigned hw = std::thread::hardware_concurrency();
  if (hw == 0) {
    return 1;
  }
  return static_cast<int>(hw);
}

template <typename Fn>
void parallel_for_index(int begin, int end, Fn &&fn, int min_parallel_range = 32) {
  if (end <= begin) {
    return;
  }

  const ComputeBackend backend = compute_backend_from_env();
  const bool allow_parallel = backend != ComputeBackend::Cpu;

  int workers = allow_parallel ? compute_thread_count() : 1;
  const int total = end - begin;
  if (workers <= 1 || total < min_parallel_range) {
    for (int i = begin; i < end; ++i) {
      fn(i);
    }
    return;
  }

  workers = std::max(1, std::min(workers, total));
  const int grain = std::max(1, total / (workers * 8));
  std::atomic<int> next(begin);
  std::vector<std::thread> threads;
  threads.reserve(static_cast<size_t>(workers - 1));

  auto run_worker = [&]() {
    while (true) {
      const int chunk_begin = next.fetch_add(grain, std::memory_order_relaxed);
      if (chunk_begin >= end) {
        break;
      }
      const int chunk_end = std::min(end, chunk_begin + grain);
      for (int i = chunk_begin; i < chunk_end; ++i) {
        fn(i);
      }
    }
  };

  for (int t = 1; t < workers; ++t) {
    threads.emplace_back(run_worker);
  }

  run_worker();

  for (auto &thread : threads) {
    thread.join();
  }
}

} // namespace igneous::core
