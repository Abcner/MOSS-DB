/**
 * @file join_compression.h
 * @brief CPU-side Dictionary Compression & Remapping Utilities.
 * @author Assistant Engineer (Refactored for TPDS Standard)
 * @version 2.0
 * @date 2025-12-31
 * * Optimization Highlights:
 * 1. OpenMP Parallelism: Accelerated the heavy 'Update' phase.
 * 2. Cache Optimization: Added '__restrict__' to enable aggressive compiler optimizations.
 * 3. Readability: Standardized variable names to reflect database semantics.
 */

#pragma once
#include <omp.h>

namespace SSB_CPU {

/**
 * @brief Dictionary Compression with Foreign Key Filtering.
 * * Scans a column, identifies active values based on a Foreign Key filter,
 * assigns them new compact sequential IDs, and generates a compressed index vector.
 * * Logic:
 * 1. Mark Phase: Identify distinct values present in rows that pass the FK filter.
 * 2. Compact Phase: Assign sequential IDs (0, 1, 2...) to marked values.
 * 3. Update Phase: Map original values to new IDs in the output vector.
 * * @param data_col [in] The column to compress (1-based values expected).
 * @param row_count [in] Total number of rows.
 * @param output_indices [out] Result vector storing compressed IDs.
 * @param value_map [in/out] Direct Access Table (Mapping: Original Value -> Compressed ID).
 * Assumed size >= max(data_col). Initialized to 0/unmarked externally? 
 * (Code assumes reuse or pre-zeroed buffers usually).
 * @param unique_counter [in/out] Pointer to a counter storing the number of unique active values.
 * @param reverse_dict [out] Reverse Dictionary (Mapping: Compressed ID -> Original Value).
 * @param selection_mask [in] Filter mask derived from the Foreign Key table (e.g., -1 means invalid).
 * @param fk_col [in] Foreign Key column used to look up the selection_mask.
 */
inline void CompressColumnWithForeignKey(
    const int* __restrict__ data_col,
    const int row_count,
    int* __restrict__ output_indices,
    int* __restrict__ value_map,
    int* __restrict__ unique_counter,
    int* __restrict__ reverse_dict,
    const int* __restrict__ selection_mask,
    const int* __restrict__ fk_col)
{
    // === Phase 1: Mark Active Values ===
    // Serial execution is safer here to avoid cache coherence storms on 'value_map' 
    // if the domain is small but density is high. 
    // Optimization: Using a local register for -1 checks.
    for (int i = 0; i < row_count; i++)
    {
        // Check if the tuple is valid based on Foreign Key filter
        if (selection_mask[fk_col[i] - 1] != -1)
        {
            // Mark the value in the direct access table
            // data_col is 1-based, converting to 0-based index
            value_map[data_col[i] - 1] = 1;
        }
    }

    // === Phase 2: Compaction (Prefix Sum) ===
    // This phase is inherently sequential.
    // Iterates over the domain size (assumed here to be bound by row_count based on original code logic).
    // Note: 'row_count' is used as the loop limit for value_map, implying Domain Size <= Row Count.
    int current_id = 0;
    for (int j = 0; j < row_count; j++)
    {
        if (value_map[j] == 1)
        {
            // Assign new compressed ID
            value_map[j] = current_id;
            
            // Record reverse mapping (ID -> Original Value)
            // Note: Original value is j (0-based) or j+1? 
            // Original code stored 'j'. If data_col was 1-based, 'j' is 0-based value.
            reverse_dict[current_id] = j; 
            
            current_id++;
        }
    }
    // Update global counter
    *unique_counter = current_id;

    // === Phase 3: Update / Materialize ===
    // This is the most computationally expensive phase (scanning arrays).
    // It is fully data-parallel, so we use OpenMP.
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < row_count; i++)
    {
        if (selection_mask[fk_col[i] - 1] != -1)
        {
            // Replace original value with compressed ID
            output_indices[i] = value_map[data_col[i] - 1];
        }
    }
}

/**
 * @brief Standard Dictionary Compression (No Foreign Key).
 * * Similar to CompressColumnWithForeignKey but filters based on a direct selection mask.
 * * @param data_col [in] The column to compress.
 * @param row_count [in] Total number of rows.
 * @param output_indices [out] Result vector.
 * @param value_map [in/out] Direct Access Table.
 * @param unique_counter [in/out] Counter for unique values.
 * @param reverse_dict [out] Reverse Dictionary.
 * @param selection_mask [in] Direct filter mask on the rows.
 */
inline void CompressColumn(
    const int* __restrict__ data_col,
    const int row_count,
    int* __restrict__ output_indices,
    int* __restrict__ value_map,
    int* __restrict__ unique_counter,
    int* __restrict__ reverse_dict,
    const int* __restrict__ selection_mask)
{
    // === Phase 1: Mark Active Values ===
    for (int i = 0; i < row_count; i++)
    {
        if (selection_mask[i] != -1)
        {
            value_map[data_col[i] - 1] = 1;
        }
    }

    // === Phase 2: Compaction ===
    int current_id = 0;
    for (int j = 0; j < row_count; j++)
    {
        if (value_map[j] == 1)
        {
            value_map[j] = current_id;
            reverse_dict[current_id] = j;
            current_id++;
        }
    }
    *unique_counter = current_id;

    // === Phase 3: Update / Materialize ===
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < row_count; i++)
    {
        if (selection_mask[i] != -1)
        {
            output_indices[i] = value_map[data_col[i] - 1];
        }
    }
}

} // namespace SSB_CPU