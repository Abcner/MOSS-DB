/**
 * @file reduce_gpu.cuh
 * @brief Optimized GPU Reduction Primitives.
 * @author Assistant Engineer
 * @date 2026-01-21
 */

#pragma once
#include <cuda_runtime.h>

namespace SSB_GPU {

constexpr unsigned int FULL_MASK = 0xFFFFFFFF;
constexpr int WARP_SIZE = 32;

template<typename T>
__device__ __forceinline__ T WarpReduceSum(T val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD = 1>
__device__ __forceinline__ T BlockReduceSum(
    T thread_val,
    T* shared_mem) 
{
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;

    T warp_sum = WarpReduceSum(thread_val);

    // [Fix] Removed 'constexpr' to ensure compatibility with C++11/14.
    // The compiler will still optimize this branch away since BLOCK_THREADS is a template arg.
    if (BLOCK_THREADS <= WARP_SIZE) {
        return __shfl_sync(FULL_MASK, warp_sum, 0);
    } 
    else {
        if (lane_id == 0) {
            shared_mem[warp_id] = warp_sum;
        }
        __syncthreads();

        T block_sum = 0;
        if (warp_id == 0) {
            constexpr int NUM_WARPS = (BLOCK_THREADS + WARP_SIZE - 1) / WARP_SIZE;
            if (lane_id < NUM_WARPS) {
                block_sum = shared_mem[lane_id];
            }
            block_sum = WarpReduceSum(block_sum);
            if (lane_id == 0) {
                shared_mem[0] = block_sum;
            }
        }
        __syncthreads();
        return shared_mem[0];
    }
}

} // namespace SSB_GPU