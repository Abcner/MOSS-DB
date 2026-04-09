/**
 * @file build_kernels.cuh
 * @brief GPU Kernels for Building Linearized Group ID Vectors.
 * @author Assistant Engineer (Refactored for TPDS Standard)
 * @version 2.0
 * @date 2025-12-31
 * * Optimization Highlights:
 * 1. Two-Mode Construction: Separated Dense (Coalesced Write) and Sparse (Scatter Write) kernels.
 * 2. LDG Cache Optimization: Enabled via 'const __restrict__' qualifiers.
 * 3. Modular Design: Integrated with optimized Load and Probe primitives.
 */

#pragma once

#include <cuda_runtime.h>
#include "operators/gpu/load_gpu.cuh"    // Optimized BlockLoad
#include "operators/gpu/join_gpu.cuh"    // Optimized BlockProbe

namespace SSB_GPU {

/**
 * @brief Dense Construction Kernel.
 * * Scans a Foreign Key column, probes a lookup table, and writes Group IDs for ALL tuples.
 * Best used when selectivity is high (most tuples participate in the join).
 * * @tparam BLOCK_THREADS Number of threads per block.
 * @tparam ITEMS_PER_THREAD Number of items processed per thread (Register Tiling).
 * * @param fk_column [in] Foreign Key column to probe.
 * @param num_tuples [in] Total number of tuples.
 * @param lookup_table [in] Dimension table index (Perfect Hash Table).
 * @param output_indices [out] Resulting Group ID vector (Dense Write).
 * @param stride_factor [in] Factor to linearize the multi-dimensional group ID.
 */
template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void BuildDenseIndicesKernel(
    const int* __restrict__ fk_column,
    int num_tuples,
    const int* __restrict__ lookup_table,
    int* __restrict__ output_indices,
    int stride_factor)
{
    // Register arrays for data caching
    int items[ITEMS_PER_THREAD];
    int group_ids[ITEMS_PER_THREAD];

    // Calculate tile parameters
    const int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
    int tile_offset = blockIdx.x * TILE_SIZE;
    int num_tile_items = TILE_SIZE;

    // Handle the last tile boundary
    if (blockIdx.x == gridDim.x - 1) {
        num_tile_items = num_tuples - tile_offset;
    }

    // 1. Initialize Accumulators
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        group_ids[i] = 0;
    }

    // 2. Coalesced Load
    BlockLoad<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
        fk_column + tile_offset, items, num_tile_items);

    // 3. Dense Probe (Vectorized)
    // Updates group_ids based on lookup_table[items[i]]
    BlockProbe<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
        items, group_ids, lookup_table, num_tile_items, stride_factor);

    // 4. Coalesced Write Back
    // Since this is a dense kernel, we write back sequentially.
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if ((tile_offset + i * BLOCK_THREADS + threadIdx.x) < num_tuples) {
            int global_index = tile_offset + i * BLOCK_THREADS + threadIdx.x;
            output_indices[global_index] = group_ids[i];
        }
    }
}

/**
 * @brief Sparse/Scatter Construction Kernel.
 * * Scans a Foreign Key column, probes a lookup table, and writes Group IDs ONLY for valid matches.
 * Best used when selectivity is low (filtering during join).
 * * @tparam BLOCK_THREADS Number of threads per block.
 * @tparam ITEMS_PER_THREAD Number of items processed per thread.
 * * @param fk_column [in] Foreign Key column.
 * @param num_tuples [in] Total number of tuples.
 * @param lookup_table [in] Dimension table index.
 * @param output_indices [out] Resulting Group ID vector (Scatter Write).
 * @param stride_factor [in] Factor to linearize the group ID.
 */
template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void BuildSparseIndicesKernel(
    const int* __restrict__ fk_column,
    int num_tuples,
    const int* __restrict__ lookup_table,
    int* __restrict__ output_indices,
    int stride_factor)
{
    int items[ITEMS_PER_THREAD];
    int group_ids[ITEMS_PER_THREAD];
    int original_offsets[ITEMS_PER_THREAD]; // Stores relative offsets (0..ITEMS-1) for scatter

    const int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
    int tile_offset = blockIdx.x * TILE_SIZE;
    int num_tile_items = TILE_SIZE;

    if (blockIdx.x == gridDim.x - 1) {
        num_tile_items = num_tuples - tile_offset;
    }

    // 1. Initialize
    // 'original_offsets' keeps track of the original slot index before compaction
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        group_ids[i] = 0;
        original_offsets[i] = i; 
    }

    // 2. Counted Load (Track valid items if needed, though here we load everything first)
    // Reusing standard BlockLoad as we filter later
    int valid_items_count = 0; 
    BlockLoadCounted<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
        fk_column + tile_offset, items, num_tile_items, &valid_items_count);
    
    // Note: valid_items_count here actually reflects loaded items count (boundary check), 
    // but BlockProbeAndCompact expects it to serve as the "current valid count".
    // Since BlockLoadCounted increments it for every loaded item, it effectively holds 'num_tile_items_local'.

    // 3. Probe and Compact (Filter)
    // This function will move valid matches to the front of the registers
    // and update 'valid_items_count' to the number of hits.
    BlockProbeAndCompact<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
        items, original_offsets, group_ids, lookup_table, &valid_items_count, stride_factor);

    // 4. Scatter Write Back
    // Only write back the items that survived the probe filter.
    // 'original_offsets[i]' allows us to reconstruct the exact global memory location.
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (i < valid_items_count) {
            // Reconstruct global index: TileBase + StripeOffset + ThreadID
            int row_idx = original_offsets[i];
            int global_index = tile_offset + row_idx * BLOCK_THREADS + threadIdx.x;
            
            output_indices[global_index] = group_ids[i];
        }
    }
}

} // namespace SSB_GPU