#pragma once
#include <cuda_runtime.h>

namespace SSB_GPU {

/**
 * @brief Direct Striped Load (Coalesced).
 * Loads a strip of items from global memory into thread-local registers using a strided pattern.
 * * @tparam InputT Type of data in global memory.
 * @tparam OutputT Type of data in registers (allows implicit casting).
 * @tparam BLOCK_THREADS Number of threads in the CUDA block.
 * @tparam ITEMS_PER_THREAD Number of items per thread.
 * @param tid [in] Thread ID (usually threadIdx.x).
 * @param block_itr [in] Pointer to the start of the data block in global memory.
 * @param items [out] Reference to the register array to store loaded items.
 */
template<typename InputT, typename OutputT = InputT, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockLoadDirect(
    const int tid,
    const InputT* __restrict__ block_itr,
    OutputT (&items)[ITEMS_PER_THREAD]
) {
    const InputT* thread_itr = block_itr + tid;

    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        // Coalesced access: Global memory stride is BLOCK_THREADS
        items[ITEM] = static_cast<OutputT>(thread_itr[ITEM * BLOCK_THREADS]);
    }
}

/**
 * @brief Direct Striped Load with Bounds Checking.
 * Safe version of BlockLoadDirect for cases where data length is not a multiple of block size.
 */
template<typename InputT, typename OutputT = InputT, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockLoadDirect(
    const int tid,
    const InputT* __restrict__ block_itr,
    OutputT (&items)[ITEMS_PER_THREAD],
    int num_items
) {
    const InputT* thread_itr = block_itr + tid;

    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        if (tid + (ITEM * BLOCK_THREADS) < num_items) {
            items[ITEM] = static_cast<OutputT>(thread_itr[ITEM * BLOCK_THREADS]);
        }
    }
}

/**
 * @brief Direct Striped Load with Valid Item Counting.
 * Loads items and increments a counter for each valid item loaded.
 * Used when the number of valid items per thread needs to be tracked.
 * Replaces: BlockLoadDirect_1
 */
template<typename InputT, typename OutputT = InputT, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockLoadDirectCounted(
    const int tid,
    const InputT* __restrict__ block_itr,
    OutputT (&items)[ITEMS_PER_THREAD],
    int* max_length
) {
    const InputT* thread_itr = block_itr + tid;

    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        items[ITEM] = static_cast<OutputT>(thread_itr[ITEM * BLOCK_THREADS]);
        (*max_length)++; // Increment thread-local validity counter
    }
}

/**
 * @brief Direct Striped Load with Bounds Checking and Counting.
 * Replaces: BlockLoadDirect_1 (Guarded version)
 */
template<typename InputT, typename OutputT = InputT, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockLoadDirectCounted(
    const int tid,
    const InputT* __restrict__ block_itr,
    OutputT (&items)[ITEMS_PER_THREAD],
    int num_items,
    int* max_length
) {
    const InputT* thread_itr = block_itr + tid;

    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        if (tid + (ITEM * BLOCK_THREADS) < num_items) {
            items[ITEM] = static_cast<OutputT>(thread_itr[ITEM * BLOCK_THREADS]);
            (*max_length)++;
        }
    }
}

/**
 * @brief Predicated Striped Load.
 * Loads items only if the corresponding predicate mask (e.g., group_id) is valid (!= -1).
 * Replaces: BlockPredLoadDirect_Q1
 */
template <typename InputT, typename OutputT = InputT, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockLoadDirectPredicated(
    const int tid, 
    const InputT* __restrict__ block_itr,
    OutputT (&items)[ITEMS_PER_THREAD],
    const int (&predicate_mask)[ITEMS_PER_THREAD]
) {
    const InputT* thread_itr = block_itr + tid;

    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        if (predicate_mask[ITEM] != -1) {
            items[ITEM] = static_cast<OutputT>(thread_itr[ITEM * BLOCK_THREADS]);
        }
    }
}

/**
 * @brief Predicated Striped Load with Bounds Checking.
 * Replaces: BlockPredLoadDirect_Q1 (Guarded version)
 */
template <typename InputT, typename OutputT = InputT, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockLoadDirectPredicated(
    const int tid, 
    const InputT* __restrict__ block_itr,
    OutputT (&items)[ITEMS_PER_THREAD], 
    int num_items,
    const int (&predicate_mask)[ITEMS_PER_THREAD]
) {
    const InputT* thread_itr = block_itr + tid;

    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        if (predicate_mask[ITEM] != -1) {
            if (tid + (ITEM * BLOCK_THREADS) < num_items) {
                items[ITEM] = static_cast<OutputT>(thread_itr[ITEM * BLOCK_THREADS]);
            }
        }
    }
}

/**
 * @brief Indirect (Gather) Load.
 * Loads items based on an index array (indirect addressing).
 * Replaces: BlockLoad_Compress
 */
template<typename InputT, typename OutputT = InputT, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockLoadIndirect(
    const int tid,
    const InputT* __restrict__ block_itr,
    OutputT (&items)[ITEMS_PER_THREAD],
    const int (&indices)[ITEMS_PER_THREAD],
    int active_items_count
) {
    // Note: thread_itr base includes tid. offset = tid + vec_index * BLOCK_THREADS
    // This assumes 'indices' contains relative row offsets.
    const InputT* thread_itr = block_itr + tid; 
    
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        // Safe check for indirect access
        if (ITEM < active_items_count) {
            int vec_index = indices[ITEM];
            items[ITEM] = static_cast<OutputT>(thread_itr[vec_index * BLOCK_THREADS]);
        }
    }
}
/**
 * @brief Predicated Gather Load (Indirect Access).
 * Loads items from a base array using indices from an index array, guarded by a predicate mask.
 * * Logic: item = base_ptr[ indices[stride] - 1 ] (relative to thread position)
 * Note: The '-1' adjustment implies 1-based indexing in the source data.
 * * @tparam DataT        Type of data to load.
 * @tparam RegT         Type of register to store data (default: DataT).
 * @tparam IndexT       Type of the index array (default: int8_t).
 * @tparam BLOCK_THREADS Number of threads in the block.
 * @tparam ITEMS_PER_THREAD Items processed per thread.
 * * @param tid           [in] Thread ID.
 * @param base_ptr      [in] Pointer to the data array (e.g., Lookup Table).
 * @param items         [out] Register array to store loaded items.
 * @param mask          [in] Predicate mask (-1 indicates invalid/skip).
 * @param indices       [in] Pointer to the index array (Foreign Keys).
 */
template <
    typename DataT, 
    typename RegT = DataT, 
    typename IndexT = int8_t, 
    int BLOCK_THREADS, 
    int ITEMS_PER_THREAD
>
__device__ __forceinline__ void BlockGatherLoadPredicated(
    const int tid, 
    const DataT* __restrict__ base_ptr,
    RegT (&items)[ITEMS_PER_THREAD],
    const int (&mask)[ITEMS_PER_THREAD],
    const IndexT* __restrict__ indices
) {
    // Determine the thread-specific base address
    const IndexT* thread_ptr = indices + tid;

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        if (mask[i] != -1) {
            // Logic: Load from (ThreadBase + Index[Stride] - 1)
            // Note: indices accessed with stride (i * BLOCK_THREADS)
            IndexT idx = thread_ptr[i * BLOCK_THREADS];
            items[i] = static_cast<RegT>(base_ptr[idx - 1]);
        }
    }
}

/**
 * @brief Predicated Gather Load with Bounds Checking.
 * Safe version of BlockGatherLoadPredicated for boundary conditions.
 * * @param num_items [in] Total valid items in the input buffer (boundary limit).
 */
template <
    typename DataT, 
    typename RegT = DataT, 
    typename IndexT = int8_t, 
    int BLOCK_THREADS, 
    int ITEMS_PER_THREAD
>
__device__ __forceinline__ void BlockGatherLoadPredicated(
    const int tid, 
    const DataT* __restrict__ base_ptr,
    RegT (&items)[ITEMS_PER_THREAD], 
    int num_items,
    const int (&mask)[ITEMS_PER_THREAD],
    const IndexT* __restrict__ indices
) {
    const IndexT* thread_ptr = indices + tid;

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        if (mask[i] != -1) {
            // Boundary Check: Ensure the load index is within valid range
            if ((tid + (i * BLOCK_THREADS)) < num_items) {
                IndexT idx = thread_ptr[i * BLOCK_THREADS];
                items[i] = static_cast<RegT>(base_ptr[idx - 1]);
            }
        }
    }
}
// ============================================================================
// Wrapper Functions (Dispatchers)
// ============================================================================

/**
 * @brief Main Dispatcher for Direct Loading.
 * Automatically chooses between Safe (Guarded) and Unsafe (Unguarded) versions.
 */
template<typename InputT, typename OutputT = InputT, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockLoad(
    const InputT* __restrict__ inp,
    OutputT (&items)[ITEMS_PER_THREAD],
    int num_items
) {
    if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
        BlockLoadDirect<InputT, OutputT, BLOCK_THREADS, ITEMS_PER_THREAD>(
            threadIdx.x, inp, items);
    } else {
        BlockLoadDirect<InputT, OutputT, BLOCK_THREADS, ITEMS_PER_THREAD>(
            threadIdx.x, inp, items, num_items);
    }
}

/**
 * @brief Main Dispatcher for Counted Loading.
 * Replaces: BlockLoad_1
 */
template<typename InputT, typename OutputT = InputT, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockLoadCounted(
    const InputT* __restrict__ inp,
    OutputT (&items)[ITEMS_PER_THREAD],
    int num_items,
    int* max_length
) {
    if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
        BlockLoadDirectCounted<InputT, OutputT, BLOCK_THREADS, ITEMS_PER_THREAD>(
            threadIdx.x, inp, items, max_length);
    } else {
        BlockLoadDirectCounted<InputT, OutputT, BLOCK_THREADS, ITEMS_PER_THREAD>(
            threadIdx.x, inp, items, num_items, max_length);
    }
}

/**
 * @brief Main Dispatcher for Predicated Loading.
 * Replaces: BlockPredLoad_Q1
 */
template <typename InputT, typename OutputT = InputT, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockLoadPredicated(
    const InputT* __restrict__ inp, 
    OutputT (&items)[ITEMS_PER_THREAD], 
    int num_items,
    const int (&predicate_mask)[ITEMS_PER_THREAD]
) {
    if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
        BlockLoadDirectPredicated<InputT, OutputT, BLOCK_THREADS, ITEMS_PER_THREAD>(
            threadIdx.x, inp, items, predicate_mask);
    } else {
        BlockLoadDirectPredicated<InputT, OutputT, BLOCK_THREADS, ITEMS_PER_THREAD>(
            threadIdx.x, inp, items, num_items, predicate_mask);
    }
}
/**
 * @brief Dispatcher for Predicated Gather Load.
 * Automatically selects between Guarded (Bounds-Checked) and Unguarded implementations.
 * * @param base_ptr [in] Pointer to the data source.
 * @param items    [out] Output registers.
 * @param num_items [in] Total number of items to process.
 * @param mask     [in] Validity mask.
 * @param indices  [in] Index array (FKs).
 */
template <
    typename DataT, 
    typename RegT = DataT, 
    typename IndexT = int8_t, 
    int BLOCK_THREADS, 
    int ITEMS_PER_THREAD
>
__device__ __forceinline__ void DispatchGatherLoadPredicated(
    const DataT* __restrict__ base_ptr, 
    RegT (&items)[ITEMS_PER_THREAD], 
    int num_items,
    const int (&mask)[ITEMS_PER_THREAD],
    const IndexT* __restrict__ indices
) {
    if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
        BlockGatherLoadPredicated<DataT, RegT, IndexT, BLOCK_THREADS, ITEMS_PER_THREAD>(
            threadIdx.x, base_ptr, items, mask, indices);
    } else {
        BlockGatherLoadPredicated<DataT, RegT, IndexT, BLOCK_THREADS, ITEMS_PER_THREAD>(
            threadIdx.x, base_ptr, items, num_items, mask, indices);
    }
}
} // namespace SSB_GPU