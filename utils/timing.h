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
    void start();

    /**
     * @brief Stop timer and compute time interval from start
     *
     * @return double Elapsed time from start (in milliseconds)
     */
    double stop();

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
    CUDATimer();

    ~CUDATimer();

    void start();
    double stop();

  private:
    cudaEvent_t ti, tf;
    bool ticking;
};

#endif // TIMING_H