/**
 * @file timer.h
 * @brief High-Resolution CPU & GPU Timers for Benchmarking.
 * @author Ruichen Han
 * @version 2.0
 * @date 2026-01-06
 */

#pragma once

#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "common/config.h" // For CHECK_CUDA

namespace SSB_Utils {

    /**
     * @brief CPU Timer using std::chrono.
     * Use for measuring Host-side logic (Data loading, Plan parsing).
     */
    class CpuTimer {
    private:
        std::chrono::high_resolution_clock::time_point start_time;
        std::string name;

    public:
        explicit CpuTimer(const std::string& timer_name = "CPU Timer") : name(timer_name) {
            start();
        }

        void start() {
            start_time = std::chrono::high_resolution_clock::now();
        }

        double stop() {
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed = end_time - start_time;
            return elapsed.count();
        }

        // Print directly to stdout
        void print() {
            printf("[Time] %-20s: %.4f ms\n", name.c_str(), stop());
        }
    };

    /**
     * @brief GPU Timer using CUDA Events.
     * Essential for measuring Kernel execution time accurately without CPU overhead.
     * Ensures implicit synchronization via cudaEventRecord.
     */
    class GpuTimer {
    private:
        cudaEvent_t start_ev, stop_ev;
        std::string name;
        bool running;

    public:
        explicit GpuTimer(const std::string& timer_name = "GPU Timer") 
            : name(timer_name), running(false) 
        {
            CHECK_CUDA(cudaEventCreate(&start_ev));
            CHECK_CUDA(cudaEventCreate(&stop_ev));
        }

        ~GpuTimer() {
            CHECK_CUDA(cudaEventDestroy(start_ev));
            CHECK_CUDA(cudaEventDestroy(stop_ev));
        }

        void start() {
            // Record start event on the default stream (0)
            CHECK_CUDA(cudaEventRecord(start_ev, 0));
            running = true;
        }

        float stop() {
            if (!running) return 0.0f;
            
            // Record stop event
            CHECK_CUDA(cudaEventRecord(stop_ev, 0));
            // Wait for the stop event to complete (Sync)
            CHECK_CUDA(cudaEventSynchronize(stop_ev));
            
            float millis = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&millis, start_ev, stop_ev));
            running = false;
            return millis;
        }

        void print() {
            float ms = stop();
            printf("[Time] %-20s: %.4f ms\n", name.c_str(), ms);
        }
    };

    /**
     * @brief Scoped Timer (RAII).
     * Automatically prints time when the object goes out of scope.
     * Useful for block-level profiling.
     */
    class ScopedGpuTimer {
    private:
        GpuTimer timer;
    public:
        explicit ScopedGpuTimer(const std::string& name) : timer(name) {
            timer.start();
        }
        ~ScopedGpuTimer() {
            timer.print();
        }
    };

} // namespace SSB_Utils