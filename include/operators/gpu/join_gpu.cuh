/**
 * @file join.cuh
 * @brief Optimized Hash Join Probe Operators (Direct Access / Perfect Hash).
 * @author Assistant Engineer (Refactored for TPDS Standard)
 * @version 2.0
 * @date 2025-12-31
 * * Optimization Highlights:
 * 1. Read-Only Cache: Enabled LDG via 'const __restrict__' qualifiers.
 * 2. Register Tiling: Optimized loop structures to ensure proper unrolling for register arrays.
 * 3. Generic Types: Templated Group ID types to support various dimensions.
 */

#pragma once
#include <cuda_runtime.h>

namespace SSB_GPU {

/**
 * @brief Performs a Direct/Perfect Hash Probe for a strip of items.
 * Updates the Group ID by linearizing the multidimensional coordinate.
 * Logic: GroupID_New = GroupID_Old + (LookupValue * StrideFactor)
 * * @tparam KeyT Type of the keys (e.g., int).
 * @tparam ValueT Type of the group IDs (e.g., int or long long).
 * @tparam BLOCK_THREADS Threads per block.
 * @tparam ITEMS_PER_THREAD Items processed per thread.
 * * @param tid [in] Thread ID.
 * @param keys [in] Array of keys to probe.
 * @param group_ids [in/out] Array of group IDs to update.
 * @param lookup_table [in] The direct access table (Perfect Hash Table).
 * @param stride_factor [in] Factor for linearizing the group ID (e.g., cardinality of previous dimension).
 */
template <typename KeyT, typename ValueT, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeDirect(
    const int tid,
    const KeyT (&keys)[ITEMS_PER_THREAD],
    ValueT (&group_ids)[ITEMS_PER_THREAD],
    const ValueT* __restrict__ lookup_table, // Optimized for Read-Only Cache
    int stride_factor)
{
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++)
    {
        // Skip invalid slots
        if (group_ids[i] != -1)
        {
            // SSB specific: Keys are 1-based, convert to 0-based for lookup
            ValueT val = lookup_table[keys[i] - 1];
            
            if (val != -1)
            {
                // Update Group ID (Linearization)
                group_ids[i] += val * stride_factor;
            }
            else
            {
                // Join Miss: Mark as invalid
                group_ids[i] = -1;
            }
        }
    }
}

/**
 * @brief BlockProbeDirect with Boundary Checking.
 * Safe version for when total items is not a multiple of block size.
 */
template <typename KeyT, typename ValueT, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeDirect(
    const int tid,
    const KeyT (&keys)[ITEMS_PER_THREAD],
    ValueT (&group_ids)[ITEMS_PER_THREAD],
    const ValueT* __restrict__ lookup_table,
    int num_items,
    int stride_factor)
{
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++)
    {
        if (tid + (i * BLOCK_THREADS) < num_items)
        {
            if (group_ids[i] != -1)
            {
                ValueT val = lookup_table[keys[i] - 1];
                
                if (val != -1)
                {
                    group_ids[i] += val * stride_factor;
                }
                else
                {
                    group_ids[i] = -1;
                }
            }
        }
    }
}

/**
 * @brief Main Dispatcher for Join Probe.
 * Replaces: BlockProbeAndPHT_1_Q
 */
template <typename KeyT, typename ValueT, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbe(
    KeyT (&keys)[ITEMS_PER_THREAD],
    ValueT (&group_ids)[ITEMS_PER_THREAD],
    const ValueT* __restrict__ lookup_table,
    int num_items,
    int stride_factor)
{
    if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items)
    {
        BlockProbeDirect<KeyT, ValueT, BLOCK_THREADS, ITEMS_PER_THREAD>(
            threadIdx.x, keys, group_ids, lookup_table, stride_factor);
    }
    else
    {
        BlockProbeDirect<KeyT, ValueT, BLOCK_THREADS, ITEMS_PER_THREAD>(
            threadIdx.x, keys, group_ids, lookup_table, num_items, stride_factor);
    }
}

/**
 * @brief Probes and Compacts valid results in-place (register array).
 * Performs a "Filter" and "Map" operation simultaneously.
 * Valid items are moved to the front of the register array.
 * * Replaces: BlockProbeAndPHT2_no_group_Q
 * * @param keys [in] Keys to probe (Note: keys array itself is NOT compacted/modified).
 * @param row_ids [in/out] Row IDs to compact (vecindex_oid).
 * @param group_ids [in/out] Group IDs to update and compact (vecindex_groupid).
 * @param active_count [in/out] Pointer to the number of valid items (updated on return).
 */
template <typename KeyT, typename ValueT, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeAndCompact(
    const KeyT (&keys)[ITEMS_PER_THREAD],
    int (&row_ids)[ITEMS_PER_THREAD],      
    ValueT (&group_ids)[ITEMS_PER_THREAD], 
    const ValueT* __restrict__ lookup_table,
    int *active_count,                     
    int stride_factor)
{
    int max_len = *active_count;
    int valid_count = 0;

    // Optimization: Unroll over constant compile-time bound (ITEMS_PER_THREAD)
    // instead of runtime variable (max_len). This allows the compiler to
    // keep arrays in registers. We use an 'if' check for logic correctness.
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++)
    {
        if (i < max_len)
        {
            ValueT val = lookup_table[keys[i] - 1];
            
            if (val != -1)
            {
                // Compaction Step:
                // Move data from slot 'i' to slot 'valid_count'.
                // Since valid_count <= i, this in-place update is safe.
                // Note: Writing to dynamic index row_ids[valid_count] may still cause 
                // some local memory traffic, but unrolling minimizes overhead.
                row_ids[valid_count] = row_ids[i];
                group_ids[valid_count] = group_ids[i] + stride_factor * val;
                
                valid_count++;
            }
        }
    }
    
    // Update the count of valid items for the next operator
    *active_count = valid_count;
}

} // namespace SSB_GPU