#ifndef TIMING_H
#define TIMING_H

#include <chrono>
#include <exception>

using duration = std::chrono::milliseconds;

class Timer{
  public:

    /**
     * @brief Start timer
     * 
     */
    void start(){
        ticking = true;

        // Get current time
        ti = std::chrono::high_resolution_clock::now();
    }

    /**
     * @brief Stop timer and compute time interval from start
     * 
     * @return double Elapsed time from start (in milliseconds)
     */
    double stop(){
        // Get current time
        tf = std::chrono::high_resolution_clock::now();

        // Check if clock was started
        if(!ticking){
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

#endif // TIMING_H