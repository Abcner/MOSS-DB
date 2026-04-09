/**
 * @file pred.h
 * @brief CPU-side Predicate Evaluation & Selection Vector Generation.
 * @author Assistant Engineer (Refactored for TPDS Standard)
 * @version 2.0
 * @date 2025-12-31
 * * Optimization Highlights:
 * 1. Parallelism: Enabled OpenMP for multi-threaded filtering.
 * 2. Vectorization: Used 'const __restrict__' hints to enable SIMD optimizations.
 * 3. Robustness: Added 'inline' to prevent ODR (One Definition Rule) violations.
 */

#pragma once
#include <omp.h> // Ensure OpenMP support is available

namespace SSB_CPU
{

    /**
     * @brief Range Filter: Selects tuples where val >= min && val <= max.
     * Maps to SQL: WHERE col BETWEEN min AND max
     * * @param input_col [in] The column data to filter.
     * @param row_count [in] Total number of rows.
     * @param selection_vec [out] Result vector. Sets matched indices to 0.
     * (Assumes vector is pre-initialized, e.g., to -1).
     * @param min_val [in] Inclusive lower bound.
     * @param max_val [in] Inclusive upper bound.
     */
    inline void FilterRange(
        const int *__restrict__ input_col,
        const int row_count,
        int *__restrict__ selection_vec,
        const int min_val,
        const int max_val)
    {
// Enable multi-threading for large arrays
#pragma omp parallel for schedule(static)
        for (int i = 0; i < row_count; i++)
        {
            int val = input_col[i];
            // Branchless optimization logic usually applied by compiler with -O3
            if (val >= min_val && val <= max_val)
            {
                selection_vec[i] = 0;
            }
        }
    }

    /**
     * @brief Dual Equality Filter (OR): Selects tuples where val == v1 || val == v2.
     * Maps to SQL: WHERE col IN (v1, v2)
     * * @param input_col [in] The column data.
     * @param row_count [in] Total number of rows.
     * @param selection_vec [out] Result vector.
     * @param val1 [in] First match value.
     * @param val2 [in] Second match value.
     */
    inline void FilterInTwo(
        const int *__restrict__ input_col,
        const int row_count,
        int *__restrict__ selection_vec,
        const int val1,
        const int val2)
    {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < row_count; i++)
        {
            int val = input_col[i];
            if (val == val1 || val == val2)
            {
                selection_vec[i] = 0;
            }
        }
    }

    /**
     * @brief Equality Filter: Selects tuples where val == target.
     * Maps to SQL: WHERE col = target
     * * @param input_col [in] The column data.
     * @param row_count [in] Total number of rows.
     * @param selection_vec [out] Result vector.
     * @param target_val [in] The value to match.
     */
    inline void FilterEqual(
        const int *__restrict__ input_col,
        const int row_count,
        int *__restrict__ selection_vec,
        const int target_val)
    {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < row_count; i++)
        {
            if (input_col[i] == target_val)
            {
                selection_vec[i] = 0;
            }
        }
    }
/**
     * @brief Filters a Child Column and marks the corresponding Parent Rows as valid (Roll-up).
     * * Logic:
     * 1. Scans the Child Column (e.g., YEARMONTH.D_YEARMONTH).
     * 2. If a value matches 'target_val' (e.g., 'Dec1997'), looks up the Foreign Key to the Parent.
     * 3. Marks the Parent Bitmap (e.g., YEAR table) as valid (0).
     * * This allows filtering at a lower level to activate grouping keys at a higher level.
     * * @param child_col      [in]  Pointer to the child column data (e.g., D_YEARMONTH).
     * @param row_count      [in]  Number of rows in the child table.
     * @param parent_bitmap  [out] Output bitmap for the Parent table. Must be initialized to -1.
     * @param target_val     [in]  The value to filter on (e.g., ID for 'Dec1997').
     * @param fk_to_parent   [in]  Foreign Key column mapping Child -> Parent (e.g., Month -> Year).
     */
    inline void FilterChildAndMarkParent(
        const int* __restrict__ child_col,
        const size_t row_count,
        int* __restrict__ parent_bitmap,
        const int target_val,
        const int* __restrict__ fk_to_parent)
    {
        // OpenMP for parallelism. 
        // schedule(static) is used for minimal overhead as workload is uniform (scanning).
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < row_count; i++)
        {
            // Check condition on Child Column
            if (child_col[i] == target_val)
            {
                // Resolve Parent Index via Foreign Key
                // Assumption: SSB FKs are 1-based, arrays are 0-based.
                int parent_idx = fk_to_parent[i] - 1;

                // [Benign Race Condition Note]
                // Multiple threads may attempt to write '0' to the same 'parent_idx' simultaneously 
                // (e.g., if Jan1997 and Feb1997 both map to 1997).
                // Since the value written (0) is identical and scalar (int), this race is benign 
                // on modern CPU architectures and significantly faster than using atomics.
                parent_bitmap[parent_idx] = 0;
            }
        }
    }
    /**
     * @brief Masked Equality Filter (Chained Filter).
     * * Applies an equality predicate (col == target) only to rows that are marked as valid
     * in the input mask (pre-filter result). This allows cascading multiple filters
     * (e.g., WHERE Date.Year = 1993 AND Date.Month = 'Jan').
     * * @note The 'result_bitmap' must be initialized to -1 by the caller before calling this function.
     * Only matching rows will be set to 0.
     * * @param col_data      [in]  Pointer to the column data to be filtered.
     * @param num_rows      [in]  Total number of rows in the column.
     * @param result_bitmap [out] Output result vector. Matched indices are set to 0.
     * @param input_mask    [in]  Input mask from the previous filter (-1 indicates invalid).
     * @param target        [in]  The target value to match against.
     */
inline void FilterEqualWithMask(
        const int *__restrict__ col_data,
        const size_t num_rows,
        int *__restrict__ result_bitmap,
        const int *__restrict__ input_mask,
        const int target)
    {
        // Use OpenMP for multi-threaded parallel execution.
        // schedule(static) minimizes overhead for uniform array processing.
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < num_rows; i++)
        {
            // Short-Circuit Optimization:
            // 1. Check Input Mask first. If row is already invalid (-1), skip comparison.
            // 2. If Valid, check if Value matches Target.
            // Note: We only write '0' (Valid). We rely on result_bitmap being pre-filled with -1.
            if (input_mask[i] != -1 && col_data[i] == target)
            {
                result_bitmap[i] = 0;
            }
            
        }
    }
    /**
     * @brief Less Than Filter: Selects tuples where val < threshold.
     * Maps to SQL: WHERE col < threshold
     * * @param input_col [in] The column data.
     * @param row_count [in] Total number of rows.
     * @param selection_vec [out] Result vector.
     * @param threshold [in] The exclusive upper bound.
     */
    inline void FilterLessThan(
        const int *__restrict__ input_col,
        const int row_count,
        int *__restrict__ selection_vec,
        const int threshold)
    {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < row_count; i++)
        {
            if (input_col[i] < threshold)
            {
                selection_vec[i] = 0;
            }
        }
    }

} // namespace SSB_CPU