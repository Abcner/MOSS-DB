/**
 * @file config.h
 * @brief Global System Configuration & Constants.
 * @author Ruichen Han
 * @version 2.0
 * @date 2026-01-06
 * * @note This header serves as the "Single Source of Truth" for system parameters.
 * It handles compile-time macros (from CMake) and defines hardware constants.
 */

#pragma once

#include <iostream>
#include <string>
#include <cuda_runtime.h>

// =========================================================
// 1. Data Scale Configuration
// =========================================================

// Scale Factor: Controls the dataset size.
// Can be overridden by compiler flags: cmake -DSCALE_FACTOR=100 ..
#ifndef SCALE_FACTOR
    #define SCALE_FACTOR 100
#endif

// Base path for SSB raw data (CSV/Binary).
// Can be overridden by compiler flags or environment variables in a real system.
#ifndef BASE_DATA_PATH
    #define BASE_DATA_PATH "/home/hrc2/test/MOSS-DB/data/ssb+/"
#endif

// =========================================================
// 2. GPU Hardware Constants
// =========================================================

namespace SSBConfig {

    // Fundamental GPU Architecture Constants
    constexpr int WARP_SIZE = 32;
    
    // Default Kernel Launch Configurations
    // Optimized for NVIDIA Ampere/Turing architectures (High Occupancy)
    constexpr int DEFAULT_BLOCK_SIZE = 128;
    constexpr int ITEMS_PER_THREAD = 64;
    
    // Derived Tile Size (Elements processed per Block)
    constexpr int TILE_SIZE = DEFAULT_BLOCK_SIZE * ITEMS_PER_THREAD;

    // Memory Alignment (for vectorized loads)
    constexpr int MEM_ALIGNMENT = 16; // 128-bit alignment (float4/int4)

} // namespace SSBConfig

// =========================================================
// 3. Error Handling Macros
// =========================================================
/**
 * @brief Macro to check for Kernel launch errors and async execution errors.
 * Should be called after kernel launches in debug builds.
 */
#define CHECK_KERNEL()                                                        \
    do {                                                                      \
        CHECK_CUDA(cudaGetLastError());                                       \
        CHECK_CUDA(cudaDeviceSynchronize());                                  \
    } while (0)
/**
 * @brief Macro to check CUDA API call results.
 * Wraps the call in a do-while loop to catch errors and print file/line info.
 */
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "[CUDA Error] %s (Code: %d) at %s:%d\n",          \
                    cudaGetErrorString(err), static_cast<int>(err),           \
                    __FILE__, __LINE__);                                      \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

