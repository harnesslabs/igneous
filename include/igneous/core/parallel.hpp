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

/// \brief Runtime compute backend choice.
enum class ComputeBackend {
  Cpu,
  CpuParallel,
  Gpu
};

/**
 * \brief Parse compute backend from `IGNEOUS_BACKEND`.
 * \return Selected backend enum value.
 */
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

/**
 * \brief Desired worker count from env or hardware fallback.
 * \return Worker count used by parallel loops.
 */
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

/**
 * \brief Hardware concurrency with safe fallback to `1`.
 * \return Hardware thread count.
 */
inline int hardware_thread_count() {
  const unsigned hw = std::thread::hardware_concurrency();
  if (hw == 0) {
    return 1;
  }
  return static_cast<int>(hw);
}

/**
 * \brief Reusable worker pool for index-based parallel loops.
 *
 * The caller thread participates in work execution; worker threads consume
 * chunks from a shared atomic index.
 */
class ParallelWorkerPool {
public:
  /**
   * \brief Construct pool with up to `max_workers - 1` background threads.
   * \param max_workers Total participants including caller thread.
   */
  explicit ParallelWorkerPool(int max_workers)
      : max_workers_(std::max(0, max_workers - 1)) {
    workers_.reserve(static_cast<size_t>(max_workers_));
    for (int worker_idx = 0; worker_idx < max_workers_; ++worker_idx) {
      workers_.emplace_back([this, worker_idx]() { worker_loop(worker_idx); });
    }
  }

  /// \brief Stop workers and release pool resources.
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

  /**
   * \brief Execute `[begin, end)` with up to `requested_workers` participants.
   * \param begin Inclusive loop start.
   * \param end Exclusive loop end.
   * \param requested_workers Requested total participants.
   * \param fn Loop body receiving index `i`.
   */
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
  /// \brief Block until all participants complete the current generation.
  void wait_for_task_completion() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_done_.wait(lock, [this]() { return remaining_.load(std::memory_order_acquire) == 0; });
    job_ = nullptr;
  }

  /// \brief Worker thread body waiting on generation changes.
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

  /// \brief Consume chunks for the active generation.
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

  /// \brief Maximum number of background worker threads.
  int max_workers_ = 0;
  /// \brief Background worker threads.
  std::vector<std::thread> workers_;

  /// \brief Synchronizes control-plane state.
  std::mutex mutex_;
  /// \brief Signals new work generation.
  std::condition_variable cv_work_;
  /// \brief Signals generation completion.
  std::condition_variable cv_done_;

  /// \brief Requests worker shutdown.
  bool stop_ = false;
  /// \brief Monotonic work-generation counter.
  uint64_t generation_ = 0;
  /// \brief Number of worker threads participating in current run.
  int active_workers_ = 0;

  /// \brief Current loop begin index.
  int begin_ = 0;
  /// \brief Current loop end index.
  int end_ = 0;
  /// \brief Chunk size used by workers.
  int grain_ = 1;
  /// \brief Next unclaimed index for dynamic chunking.
  std::atomic<int> next_{0};
  /// \brief Remaining participants for completion signaling.
  std::atomic<int> remaining_{0};
  /// \brief Active work function.
  std::function<void(int)> job_;
};

/**
 * \brief Singleton worker pool used by `parallel_for_index`.
 * \return Process-wide worker pool instance.
 */
inline ParallelWorkerPool &parallel_worker_pool() {
  static ParallelWorkerPool pool(hardware_thread_count());
  return pool;
}

/**
 * \brief Execute an integer index loop potentially in parallel.
 *
 * Parallel execution is gated by backend configuration and loop size.
 * \param begin Inclusive loop start.
 * \param end Exclusive loop end.
 * \param fn Loop body receiving index `i`.
 * \param min_parallel_range Minimum range length to enable parallel execution.
 */
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
