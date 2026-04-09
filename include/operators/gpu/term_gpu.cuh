/**
 * @file term.cuh
 * @brief Thread Termination Check Utilities.
 * @author Assistant Engineer (Refactored for TPDS Standard)
 * @version 2.0
 * @date 2025-12-30
 * * Optimization Highlights:
 * 1. Early Exit: Replaces full accumulation with short-circuit logic for efficiency.
 * 2. Type Safety: Uses array reference instead of raw pointers.
 * 3. Register Optimization: Fully unrolled loop friendly for register-resident arrays.
 */

#pragma once
#include <cuda_runtime.h>

namespace SSB_GPU {

/**
 * @brief Checks if the current thread has no active work items left.
 * * Scans the thread-local vector indices. If all indices are -1 (indicating invalid/processed),
 * the thread is considered terminated/inactive.
 * * @tparam ITEMS_PER_THREAD Number of items held by each thread (Register Tiling factor).
 * @param vecindex_groupid [in] Reference to the thread-local index array (usually in registers).
 * @return true If all items are -1 (Thread is idle).
 * @return false If at least one valid item exists (Thread is active).
 */
template <int ITEMS_PER_THREAD>
__device__ __forceinline__ bool IsThreadTerminated(
    const int (&vecindex_groupid)[ITEMS_PER_THREAD]
) {
    // Optimization: Loop Unrolling
    // Since ITEMS_PER_THREAD is a compile-time constant, this pragma ensures
    // the loop is fully unrolled, eliminating branch instructions and loop overhead.
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        // Optimization: Early Exit
        // Instead of counting all valid items, we return false immediately
        // upon finding the first valid item. This significantly reduces 
        // instruction count in active threads.
        if (vecindex_groupid[i] != -1) {
            return false;
        }
    }

    // If we reach here, no valid items were found.
    return true;
}

} // namespace SSB_GPU