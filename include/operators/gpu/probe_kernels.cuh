/**
 * @file probe_kernels.cuh
 * @brief Optimized GPU Kernels for Multi-Way Join Probe & Aggregation.
 * @author Assistant Engineer (Refactored for TPDS Standard)
 * @version 4.1 (Indirect Access Support)
 * @date 2026-01-21
 */

#pragma once

#include <cuda_runtime.h>
#include "operators/gpu/load_gpu.cuh"
#include "operators/gpu/join_gpu.cuh"
#include "operators/gpu/reduce_gpu.cuh"
#include "operators/gpu/term_gpu.cuh"

namespace SSB_GPU
{

    constexpr int SHARED_ACCUM_SIZE = 512;

// =========================================================
    // 1. Aggregation Operations
    // =========================================================
    enum class AggOp {
        SUM,      // SUM(A)
        PRODUCT,  // SUM(A * B)
        SUBTRACT  // SUM(A - B) -> For Q4 (Profit)
    };

    template <AggOp OP, typename T>
    __device__ __forceinline__ long long ApplyAggregation(T a, T b)
    {
        // Evaluated at compile-time by NVCC dead-code elimination
        if (OP == AggOp::PRODUCT) {
            return (long long)a * (long long)b;
        } else if (OP == AggOp::SUBTRACT) {
            return (long long)a - (long long)b;
        } else {
            return (long long)a;
        }
    }

    // =========================================================
    // 1.5 Warp-level Termination Helper
    // =========================================================
    /**
     * @brief Checks if all threads in a Warp have been completely filtered out.
     * This avoids harmful thread-level divergence and safely skips memory operations.
     */
    template <int ITEMS>
    __device__ __forceinline__ bool IsWarpTerminated(const int* group_ids) {
        bool is_dead = true;
        #pragma unroll
        for (int i = 0; i < ITEMS; ++i) {
            if (group_ids[i] != -1) {
                is_dead = false;
            }
        }
        // __all_sync returns non-zero only if 'is_dead' is true for ALL threads in the warp
        return __all_sync(0xFFFFFFFF, is_dead);
    }

    // =========================================================
    // 2. Probe Kernels
    // =========================================================

    /**
     * @brief Optimized Probe Kernel for Multi-Way Join & Aggregation.
     * Supports configurable Aggregation Mode (Product/Subtract) via AggOp template.
     */
    template <
        int BLOCK_THREADS, int ITEMS_PER_THREAD,
        AggOp AGG_MODE, 
        typename FK_T1, typename FK_T2, typename FK_T3, typename FK_T4,
        typename Data1_T, typename Data2_T,
        bool HAS_COL2 = true>
    __global__ void ProbeDenseKernel(
        // Join Columns & Bitmaps (Up to 4 Dimensions)
        const FK_T1 *__restrict__ fk1, const int *__restrict__ map1, int stride1,
        const FK_T2 *__restrict__ fk2, const int *__restrict__ map2, int stride2,
        const FK_T3 *__restrict__ fk3, const int *__restrict__ map3, int stride3,
        const FK_T4 *__restrict__ fk4, const int *__restrict__ map4, int stride4,
        // Aggregation Columns
        const void *__restrict__ data_col1, const void *__restrict__ lookup_col1,
        const void *__restrict__ data_col2, const void *__restrict__ lookup_col2,
        // Metadata
        int num_tuples,
        int total_groups, 
        unsigned long long *__restrict__ global_res)
    {
        // Registers for Data Locality
        int items[ITEMS_PER_THREAD];
        int group_ids[ITEMS_PER_THREAD];
        int items1[ITEMS_PER_THREAD]; // For Col 2

        const int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
        int tile_offset = blockIdx.x * TILE_SIZE;
        int num_tile_items = (blockIdx.x == gridDim.x - 1) ? (num_tuples - tile_offset) : TILE_SIZE;

        // 1. Initialize Group IDs to 0
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; i++) group_ids[i] = 0;

        // =========================================================
        // Phase A: Join Probing with Warp-level Exit
        // =========================================================
        
        // Stage 1
        if (fk1 && map1) {
            BlockLoad<FK_T1, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
                fk1 + tile_offset, items, num_tile_items);
            BlockProbe<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
                items, group_ids, map1, num_tile_items, stride1);
            // [Optimization] Safely retire the entire warp if all items are filtered
            if(IsWarpTerminated<ITEMS_PER_THREAD>(group_ids)) return;
        }
        // Stage 2
        if (fk2 && map2) {
            BlockLoadPredicated<FK_T2, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
                fk2 + tile_offset, items, num_tile_items, group_ids);
            BlockProbe<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
                items, group_ids, map2, num_tile_items, stride2);
            if(IsWarpTerminated<ITEMS_PER_THREAD>(group_ids)) return;
        }
        // Stage 3
        if (fk3 && map3) {
            BlockLoadPredicated<FK_T3, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
                fk3 + tile_offset, items, num_tile_items, group_ids);
            BlockProbe<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
                items, group_ids, map3, num_tile_items, stride3);
            if(IsWarpTerminated<ITEMS_PER_THREAD>(group_ids)) return;
        }
        // Stage 4
        if (fk4 && map4) {
            BlockLoadPredicated<FK_T4, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
                fk4 + tile_offset, items, num_tile_items, group_ids);
            BlockProbe<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
                items, group_ids, map4, num_tile_items, stride4);
            if(IsWarpTerminated<ITEMS_PER_THREAD>(group_ids)) return;
        }

        // =========================================================
        // Phase B: Predicated Data Loading
        // =========================================================

        if (!lookup_col1) {
            BlockLoadPredicated<Data1_T, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
                (const Data1_T*)data_col1 + tile_offset, items, num_tile_items, group_ids);
        } else {
            DispatchGatherLoadPredicated<int, int, Data1_T, BLOCK_THREADS, ITEMS_PER_THREAD>(
                (const int *)lookup_col1, items, num_tile_items, group_ids, (const Data1_T*)data_col1 + tile_offset);
        }

        if (HAS_COL2) {
            if (!lookup_col2) {
                BlockLoadPredicated<Data2_T, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
                    (const Data2_T *)data_col2 + tile_offset, items1, num_tile_items, group_ids);
            } else {
                DispatchGatherLoadPredicated<int, int, Data2_T, BLOCK_THREADS, ITEMS_PER_THREAD>(
                    (const int *)lookup_col2, items1, num_tile_items, group_ids, (const Data2_T *)data_col2 + tile_offset);
            }
        }

        // =========================================================
        // Phase C: Aggregation (Optimized Branch Hoisting)
        // =========================================================
        long long thread_sum = 0; 

        // [Optimization] Hoist the runtime 'total_groups' check outside the unrolled loop
        // This ensures perfectly packed instructions for the atomicAdd path (Q3.3 heavily relies on this)
        if (total_groups == 1) 
        {
            // Scalar Aggregation Path
            #pragma unroll
            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                if ((threadIdx.x + i * BLOCK_THREADS) < num_tile_items) {
                    if (group_ids[i] != -1) {
                        long long res = HAS_COL2 ? ApplyAggregation<AGG_MODE>(items[i], items1[i]) : (long long)items[i];
                        thread_sum += res;
                    }
                }
            }
        } 
        else 
        {
            // Vector Aggregation Path (Group By)
            #pragma unroll
            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                if ((threadIdx.x + i * BLOCK_THREADS) < num_tile_items) {
                    int gid = group_ids[i];
                    if (gid != -1) {
                        long long res = HAS_COL2 ? ApplyAggregation<AGG_MODE>(items[i], items1[i]) : (long long)items[i];
                        // Pure Atomic Add, no runtime branching inside
                        atomicAdd(&global_res[gid], (unsigned long long)res);
                    }
                }
            }
        }

        // =========================================================
        // Phase D: Block Reduction (Scalar Mode Only)
        // =========================================================
        if (total_groups == 1)
        {
            __shared__ long long shared_mem[32];
            long long block_total = BlockReduceSum<long long, BLOCK_THREADS>(thread_sum, shared_mem);
            if (threadIdx.x == 0) {
                atomicAdd(&global_res[0], (unsigned long long)block_total);
            }
        }
    }

//     /**
//      * @brief Sparse/Compact Probe Kernel.
//      * Same logic updates applied.
//      */
//     template <int BLOCK_THREADS, int ITEMS_PER_THREAD, typename DataT>
//     __global__ void ProbeSparseKernel(
//         const int *__restrict__ fk1, const int *__restrict__ map1,
//         const int *__restrict__ fk2, const int *__restrict__ map2,
//         const int *__restrict__ fk3, const int *__restrict__ map3,
//         const int *__restrict__ data_col1, const int *__restrict__ lookup_col1,
//         const int *__restrict__ data_col2, const int *__restrict__ lookup_col2,
//         int num_tuples,
//         unsigned long long *__restrict__ global_res)
//     {
//         int items[ITEMS_PER_THREAD];
//         int row_offsets[ITEMS_PER_THREAD];
//         int group_ids[ITEMS_PER_THREAD];
//         int valid_count = 0;

//         const int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
//         int tile_offset = blockIdx.x * TILE_SIZE;
//         int num_tile_items = (blockIdx.x == gridDim.x - 1) ? (num_tuples - tile_offset) : TILE_SIZE;

// #pragma unroll
//         for (int i = 0; i < ITEMS_PER_THREAD; i++)
//         {
//             row_offsets[i] = i;
//             group_ids[i] = 0;
//         }

//         // Stage 1
//         if (fk1 && map1)
//         {
//             BlockLoadCounted<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
//                 fk1 + tile_offset, items, num_tile_items, &valid_count);
//             BlockProbeAndCompact<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
//                 items, row_offsets, group_ids, map1, &valid_count, 1);
//             if (valid_count == 0)
//                 return;
//         }

//         // Stage 2
//         if (fk2 && map2)
//         {
//             BlockLoadIndirect<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
//                 threadIdx.x, fk2 + tile_offset, items, row_offsets, valid_count);
//             BlockProbeAndCompact<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
//                 items, row_offsets, group_ids, map2, &valid_count, 1);
//             if (valid_count == 0)
//                 return;
//         }

//         // Stage 3
//         if (fk3 && map3)
//         {
//             BlockLoadIndirect<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
//                 threadIdx.x, fk3 + tile_offset, items, row_offsets, valid_count);
//             BlockProbeAndCompact<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
//                 items, row_offsets, group_ids, map3, &valid_count, 1);
//             if (valid_count == 0)
//                 return;
//         }

//         // Data Load
//         int raw1[ITEMS_PER_THREAD];
//         int raw2[ITEMS_PER_THREAD];

//         BlockLoadIndirect<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
//             threadIdx.x, data_col1 + tile_offset, raw1, row_offsets, valid_count);

//         if (data_col2)
//         {
//             BlockLoadIndirect<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
//                 threadIdx.x, data_col2 + tile_offset, raw2, row_offsets, valid_count);
//         }

//         // Aggregate
//         long long thread_sum = 0;
// #pragma unroll
//         for (int i = 0; i < ITEMS_PER_THREAD; ++i)
//         {
//             if (i < valid_count)
//             {
//                 int val1 = raw1[i];
//                 if (lookup_col1)
//                     val1 = __ldg(&lookup_col1[val1]);

//                 int val2 = 0;
//                 if (data_col2)
//                 {
//                     val2 = raw2[i];
//                     if (lookup_col2)
//                         val2 = __ldg(&lookup_col2[val2]);
//                 }

//                 thread_sum += (data_col2) ? ApplyAggregation(val1, val2) : (long long)val1;
//             }
//         }

//         // Reduction
//         __shared__ long long shared_mem[32];
//         long long block_total = BlockReduceSum<long long, BLOCK_THREADS>(thread_sum, shared_mem);

//         if (threadIdx.x == 0)
//             atomicAdd(&global_res[1], (unsigned long long)block_total);
//     }

} // namespace SSB_GPU