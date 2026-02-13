#pragma once

#include <algorithm>
#include <atomic>
#include <cctype>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <mutex>
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

inline int hardware_thread_count() {
  const unsigned hw = std::thread::hardware_concurrency();
  if (hw == 0) {
    return 1;
  }
  return static_cast<int>(hw);
}

class ParallelWorkerPool {
public:
  explicit ParallelWorkerPool(int max_workers)
      : max_workers_(std::max(0, max_workers - 1)) {
    workers_.reserve(static_cast<size_t>(max_workers_));
    for (int worker_idx = 0; worker_idx < max_workers_; ++worker_idx) {
      workers_.emplace_back([this, worker_idx]() { worker_loop(worker_idx); });
    }
  }

  ~ParallelWorkerPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
      ++generation_;
    }
    cv_work_.notify_all();
    for (auto &worker : workers_) {
      worker.join();
    }
  }

  ParallelWorkerPool(const ParallelWorkerPool &) = delete;
  ParallelWorkerPool &operator=(const ParallelWorkerPool &) = delete;

  template <typename Fn>
  void run(int begin, int end, int requested_workers, Fn &&fn) {
    if (end <= begin) {
      return;
    }

    const int total = end - begin;
    const int participants =
        std::max(1, std::min({requested_workers, total, max_workers_ + 1}));
    if (participants <= 1) {
      for (int i = begin; i < end; ++i) {
        fn(i);
      }
      return;
    }

    std::function<void(int)> job = [&](int i) { fn(i); };
    {
      std::lock_guard<std::mutex> lock(mutex_);
      begin_ = begin;
      end_ = end;
      grain_ = std::max(1, total / (participants * 8));
      next_.store(begin_, std::memory_order_relaxed);
      active_workers_ = participants - 1;
      remaining_.store(participants, std::memory_order_relaxed);
      job_ = std::move(job);
      ++generation_;
    }
    cv_work_.notify_all();

    run_chunks();
    wait_for_task_completion();
  }

private:
  void wait_for_task_completion() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_done_.wait(lock, [this]() { return remaining_.load(std::memory_order_acquire) == 0; });
    job_ = nullptr;
  }

  void worker_loop(int worker_idx) {
    uint64_t seen_generation = 0;
    while (true) {
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_work_.wait(lock, [&]() { return stop_ || generation_ != seen_generation; });
        if (stop_) {
          return;
        }
        seen_generation = generation_;
        if (worker_idx >= active_workers_) {
          continue;
        }
      }
      run_chunks();
    }
  }

  void run_chunks() {
    while (true) {
      const int chunk_begin = next_.fetch_add(grain_, std::memory_order_relaxed);
      if (chunk_begin >= end_) {
        break;
      }
      const int chunk_end = std::min(end_, chunk_begin + grain_);
      for (int i = chunk_begin; i < chunk_end; ++i) {
        job_(i);
      }
    }

    if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      std::lock_guard<std::mutex> lock(mutex_);
      cv_done_.notify_one();
    }
  }

  int max_workers_ = 0;
  std::vector<std::thread> workers_;

  std::mutex mutex_;
  std::condition_variable cv_work_;
  std::condition_variable cv_done_;

  bool stop_ = false;
  uint64_t generation_ = 0;
  int active_workers_ = 0;

  int begin_ = 0;
  int end_ = 0;
  int grain_ = 1;
  std::atomic<int> next_{0};
  std::atomic<int> remaining_{0};
  std::function<void(int)> job_;
};

inline ParallelWorkerPool &parallel_worker_pool() {
  static ParallelWorkerPool pool(hardware_thread_count());
  return pool;
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

  parallel_worker_pool().run(begin, end, workers, std::forward<Fn>(fn));
}

} // namespace igneous::core
