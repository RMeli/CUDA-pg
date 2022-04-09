#ifndef TIMING_H
#define TIMING_H

#include <chrono>
#include <exception>

#include "err.h"

using duration = std::chrono::milliseconds;

class Timer {
  public:
    /**
     * @brief Start timer
     *
     */
    void start() {
        ticking = true;

        // Get current time
        ti = std::chrono::high_resolution_clock::now();
    }

    /**
     * @brief Stop timer and compute time interval from start
     *
     * @return double Elapsed time from start (in milliseconds)
     */
    double stop() {
        // Get current time
        tf = std::chrono::high_resolution_clock::now();

        // Check if clock was started
        if (!ticking) {
            throw std::runtime_error("Timer not started.");
        }

        ticking = false;

        // Compute elapsed time between start and stop (in milliseconds)
        auto time_ms = std::chrono::duration_cast<duration>(tf - ti);

        return time_ms.count();
    }

  private:
    /**
     * @brief Initial and final time points
     *
     */
    std::chrono::time_point<std::chrono::high_resolution_clock> ti, tf;

    bool ticking;
};

class CUDATimer {
  public:
    CUDATimer() {
      // Create start event
      auto status = cudaEventCreate(&ti);
      cuda_check_status(status);

      // Create stop event
      status = cudaEventCreate(&tf);
      cuda_check_status(status);
    }

    ~CUDATimer() {
      // Destroy start event
      auto status = cudaEventDestroy(ti);
      cuda_check_status(status);

      // Destroy stop event
      status = cudaEventDestroy(tf);
      cuda_check_status(status);
    }

    void start() {
        ticking = true;

        // Get current time
        auto status = cudaEventRecord(ti, 0);
        cuda_check_status(status);
    }

    double stop() {
        // Get current time
        auto status = cudaEventRecord(tf, 0);
        cuda_check_status(status);

        // Syncronize event
        status = cudaEventSynchronize(tf);
        cuda_check_status(status);

        // Check if clock was started
        if (!ticking) {
            throw std::runtime_error("Timer not started.");
        }
        ticking = false;

        // Compute elapsed time between start and stop (in milliseconds)
        float time_ms;
        status = cudaEventElapsedTime(&time_ms, ti, tf);
        cuda_check_status(status);

        return time_ms;
    }

  private:
    cudaEvent_t ti, tf;
    bool ticking;
};

#endif // TIMING_H