/**
 * @file join_cpu.h
 * @brief CPU-side Vectorized Join and Aggregation Operators.
 * @author MOSS-GDB System Engineering Team
 * @version 3.0 (Optimized for TPDS Standard - Pipeline Vectorization)
 * @date 2026-03-XX
 * * * Optimization Highlights:
 * 1. Vectorized Pipelining: Uses local OID buffers to keep hot data in L1/L2 cache.
 * 2. Static Polymorphism: Replaced function pointers with C++ templates for Bitmaps and Vectors.
 * 3. Dynamic Parallelism: Configurable thread counts via OpenMP.
 * 4. Late Materialization: Aggregation is only performed on fully qualified tuples.
 */

#pragma once

#include <omp.h>
#include <vector>
#include <cstdint>
#include <iostream>
#include <functional>

namespace MOSS_DB {
namespace CPU {

    // ============================================================================
    // 1. Core Lookup Primitives (Inlined for Zero-Overhead)
    // ============================================================================

    /**
     * @brief Looks up a status in a dense Compressed Vector.
     * @return The Group ID if valid, or -1 if filtered.
     */
    __attribute__((always_inline)) inline int ProbeVector(const int* __restrict__ vec, int fk_val) {
        // Assume 1-based Foreign Keys as per SSB/TPC-H standard
        return vec[fk_val - 1]; 
    }

    /**
     * @brief Looks up a status in a packed Boolean Bitmap (1 bit per row).
     * @return 1 if the bit is set (valid), -1 otherwise.
     */
    __attribute__((always_inline)) inline int ProbeBitmap(const uint8_t* __restrict__ bitmap, int fk_val) {
        int idx = fk_val - 1;
        // Fast bitwise extraction: Byte index = idx / 8, Bit offset = idx % 8
        return ((bitmap[idx >> 3] >> (idx & 7)) & 1) ? 1 : -1;
    }

    // ============================================================================
    // 2. Multi-Way Join Operators for Dimension Tables (No Aggregation)
    // ============================================================================

    /**
     * @brief Multi-threaded Multi-way Join using Compressed Vectors.
     * Used to propagate Group IDs down to intermediate tables or generate masks.
     * * @param num_tuples Total rows in the target table.
     * @param join_col_num Number of dimensions to join.
     * @param fks Array of pointers to Foreign Key columns [join_col_num].
     * @param vectors Array of pointers to Dimension Lookup Vectors [join_col_num].
     * @param factors Array of strides for Multi-dimensional grouping [join_col_num].
     * @param output_group_ids Resulting combined Group IDs.
     * @param num_threads Number of OpenMP threads to use.
     */
    inline void MultiJoin_Vector_CPU(
        size_t num_tuples, int join_col_num,
        const int** __restrict__ fks, 
        const int** __restrict__ vectors, 
        const int* __restrict__ factors,
        int* __restrict__ output_group_ids,
        int num_threads)
    {
        omp_set_num_threads(num_threads);

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < num_tuples; ++i) {
            int current_group = 0;
            bool valid = true;

            // Pipeline all joins for a single row to maximize register reuse
            for (int j = 0; j < join_col_num; ++j) {
                int status = ProbeVector(vectors[j], fks[j][i]);
                if (status == -1) {
                    valid = false;
                    break; // Short-circuit evaluation
                }
                current_group += status * factors[j];
            }
            output_group_ids[i] = valid ? current_group : -1;
        }
    }

    /**
     * @brief Multi-threaded Multi-way Join using Bitmaps.
     * Used primarily for high-selectivity filtering where grouping is not required.
     */
    inline void MultiJoin_Bitmap_CPU(
        size_t num_tuples, int join_col_num,
        const int** __restrict__ fks, 
        const uint8_t** __restrict__ bitmaps, 
        int* __restrict__ output_mask,
        int num_threads)
    {
        omp_set_num_threads(num_threads);

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < num_tuples; ++i) {
            bool valid = true;
            for (int j = 0; j < join_col_num; ++j) {
                if (ProbeBitmap(bitmaps[j], fks[j][i]) == -1) {
                    valid = false;
                    break;
                }
            }
            output_mask[i] = valid ? 1 : -1;
        }
    }

enum class AggOp { SUM, PRODUCT, SUBTRACT };

    template <AggOp OP, typename T1, typename T2>
    __attribute__((always_inline)) inline unsigned long long ApplyAggregation(T1 a, T2 b) {
        if constexpr (OP == AggOp::PRODUCT) {
            return static_cast<unsigned long long>(a) * static_cast<unsigned long long>(b);
        } else if constexpr (OP == AggOp::SUBTRACT) {
            return static_cast<unsigned long long>(a) - static_cast<unsigned long long>(b);
        } else {
            return static_cast<unsigned long long>(a);
        }
    }

    template <AggOp OP, typename T1, typename T2, bool HAS_COL2>
    __attribute__((always_inline)) inline void JoinAgg_Vector_Pipeline_CPU(
        const size_t num_tuples, 
        const int join_col_num,
        const int** __restrict__ fks, 
        const int** __restrict__ vectors, 
        const int* __restrict__ factors,
        const T1* __restrict__ agg_col1,   
        const T2* __restrict__ agg_col2,   
        unsigned long long* __restrict__ res_vec,
        const int num_threads,
        const int total_groups) 
    {
        omp_set_num_threads(num_threads);
        const int VEC_LEN = 1024; 

        // =========================================================
        // [HARDWARE FIX 1]: Anti False-Sharing Matrix
        // =========================================================
        const size_t PADDED_GROUPS = ((total_groups + 63) / 64) * 64 + 64; 

        unsigned long long* global_thread_results = nullptr;
        if (posix_memalign((void**)&global_thread_results, 4096, num_threads * PADDED_GROUPS * sizeof(unsigned long long)) != 0) {
            std::cerr << "[Fatal Error] posix_memalign failed to allocate reduction matrix." << std::endl;
            return;
        }

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            // =========================================================
            // [HARDWARE FIX 2]: Hard CPU Affinity
            // =========================================================
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(tid % std::thread::hardware_concurrency(), &cpuset);
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

            // =========================================================
            // [HARDWARE FIX 3]: NUMA First-Touch Allocation
            // =========================================================
            unsigned long long* __restrict__ local_res_ptr = global_thread_results + tid * PADDED_GROUPS;
            for (int g = 0; g < total_groups; ++g) {
                local_res_ptr[g] = 0;
            }

            size_t chunk_size = num_tuples / num_threads;
            size_t start_idx = tid * chunk_size;
            size_t thread_tuples = (tid == num_threads - 1) ? (num_tuples - start_idx) : chunk_size;
            int nblock = thread_tuples / VEC_LEN;

            uint32_t local_OID[VEC_LEN];     
            uint32_t local_GroupID[VEC_LEN];

            // Pipelined Block Processing
            for (int iter = 0; iter <= nblock; ++iter) 
            {
                size_t length = (iter == nblock) ? (thread_tuples % VEC_LEN) : VEC_LEN;
                if (length == 0) continue;
                size_t active_count = 0;

                // ---------------------------------------------------------
                // Stage 1: Branchless Sequential Probe
                // Eliminates branch misprediction penalty. CPU executes 
                // unconditionally and leverages ALU to update pointer.
                // ---------------------------------------------------------
                const int* __restrict__ fk0 = fks[0];
                const int* __restrict__ vec0 = vectors[0];
                const int factor0 = factors[0];

                for (size_t i = 0; i < length; ++i) {
                    size_t global_idx = start_idx + i + iter * VEC_LEN;
                    int status = vec0[fk0[global_idx] - 1]; 
                    
                    local_OID[active_count] = global_idx;
                    local_GroupID[active_count] = status * factor0;
                    
                    // Branchless increment: resolves to a fast ADD instruction
                    active_count += (status != -1);
                }

                // ---------------------------------------------------------
                // Stage 2: Branchless Random Gathers
                // ---------------------------------------------------------
                for (int j = 1; j < join_col_num; ++j) {
                    size_t new_active_count = 0;
                    const int* __restrict__ fkj = fks[j];
                    const int* __restrict__ vecj = vectors[j];
                    const int factorj = factors[j];

                    for (size_t i = 0; i < active_count; ++i) {
                        size_t oid = local_OID[i];
                        int status = vecj[fkj[oid] - 1];
                        
                        local_OID[new_active_count] = oid;
                        local_GroupID[new_active_count] = local_GroupID[i] + status * factorj;
                        
                        new_active_count += (status != -1);
                    }
                    active_count = new_active_count;
                }

                // ---------------------------------------------------------
                // Stage 3: Correct & Extremely Fast Software Pipelining
                // Uses Loop Peeling to remove 'if' branching.
                // Explicit Unrolling ensures Multiple Gathers, while keeping
                // Scalar Scatters to guarantee 100% correct aggregation.
                // ---------------------------------------------------------
                const size_t UNROLL_FACTOR = 4;
                const size_t PREFETCH_DIST = 16; 

                size_t prefetch_bound = 0;
                if (active_count > PREFETCH_DIST + UNROLL_FACTOR) {
                    prefetch_bound = ((active_count - PREFETCH_DIST) / UNROLL_FACTOR) * UNROLL_FACTOR;
                }
                size_t unroll_bound = (active_count / UNROLL_FACTOR) * UNROLL_FACTOR;

                if constexpr (HAS_COL2) 
                {
                    size_t i = 0;
                    
                    // Phase 3A: Pipelined Engine WITH Branchless Prefetching
                    for (; i < prefetch_bound; i += UNROLL_FACTOR) 
                    {
                        // Safely prefetch memory that will be needed soon
                        __builtin_prefetch(&agg_col1[local_OID[i + PREFETCH_DIST]], 0, 1);
                        __builtin_prefetch(&agg_col2[local_OID[i + PREFETCH_DIST]], 0, 1);
                        __builtin_prefetch(&agg_col1[local_OID[i + PREFETCH_DIST + 1]], 0, 1);
                        __builtin_prefetch(&agg_col2[local_OID[i + PREFETCH_DIST + 1]], 0, 1);

                        size_t oid0 = local_OID[i],     oid1 = local_OID[i + 1];
                        size_t oid2 = local_OID[i + 2], oid3 = local_OID[i + 3];

                        int g0 = local_GroupID[i],     g1 = local_GroupID[i + 1];
                        int g2 = local_GroupID[i + 2], g3 = local_GroupID[i + 3];

                        // Independent parallel Loads
                        T1 v1_0 = agg_col1[oid0], v1_1 = agg_col1[oid1], v1_2 = agg_col1[oid2], v1_3 = agg_col1[oid3];
                        T2 v2_0 = agg_col2[oid0], v2_1 = agg_col2[oid1], v2_2 = agg_col2[oid2], v2_3 = agg_col2[oid3];

                        // Sequential safe accumulation for RAW hazard protection
                        local_res_ptr[g0] += ApplyAggregation<OP>(v1_0, v2_0);
                        local_res_ptr[g1] += ApplyAggregation<OP>(v1_1, v2_1);
                        local_res_ptr[g2] += ApplyAggregation<OP>(v1_2, v2_2);
                        local_res_ptr[g3] += ApplyAggregation<OP>(v1_3, v2_3);
                    }
                    
                    // Phase 3B: Unrolled Cleanup Engine WITHOUT Prefetching
                    for (; i < unroll_bound; i += UNROLL_FACTOR) 
                    {
                        size_t oid0 = local_OID[i],     oid1 = local_OID[i + 1];
                        size_t oid2 = local_OID[i + 2], oid3 = local_OID[i + 3];

                        int g0 = local_GroupID[i],     g1 = local_GroupID[i + 1];
                        int g2 = local_GroupID[i + 2], g3 = local_GroupID[i + 3];

                        T1 v1_0 = agg_col1[oid0], v1_1 = agg_col1[oid1], v1_2 = agg_col1[oid2], v1_3 = agg_col1[oid3];
                        T2 v2_0 = agg_col2[oid0], v2_1 = agg_col2[oid1], v2_2 = agg_col2[oid2], v2_3 = agg_col2[oid3];

                        local_res_ptr[g0] += ApplyAggregation<OP>(v1_0, v2_0);
                        local_res_ptr[g1] += ApplyAggregation<OP>(v1_1, v2_1);
                        local_res_ptr[g2] += ApplyAggregation<OP>(v1_2, v2_2);
                        local_res_ptr[g3] += ApplyAggregation<OP>(v1_3, v2_3);
                    }

                    // Phase 3C: Scalar Tail
                    for (; i < active_count; ++i) {
                        local_res_ptr[local_GroupID[i]] += ApplyAggregation<OP>(agg_col1[local_OID[i]], agg_col2[local_OID[i]]);
                    }
                } 
                else 
                {
                    // Single-Column Fast Path
                    size_t i = 0;
                    for (; i < prefetch_bound; i += UNROLL_FACTOR) 
                    {
                        __builtin_prefetch(&agg_col1[local_OID[i + PREFETCH_DIST]], 0, 1);
                        __builtin_prefetch(&agg_col1[local_OID[i + PREFETCH_DIST + 1]], 0, 1);

                        size_t oid0 = local_OID[i],     oid1 = local_OID[i + 1];
                        size_t oid2 = local_OID[i + 2], oid3 = local_OID[i + 3];

                        int g0 = local_GroupID[i],     g1 = local_GroupID[i + 1];
                        int g2 = local_GroupID[i + 2], g3 = local_GroupID[i + 3];

                        T1 v1_0 = agg_col1[oid0], v1_1 = agg_col1[oid1], v1_2 = agg_col1[oid2], v1_3 = agg_col1[oid3];

                        local_res_ptr[g0] += ApplyAggregation<OP>(v1_0, (T2)0);
                        local_res_ptr[g1] += ApplyAggregation<OP>(v1_1, (T2)0);
                        local_res_ptr[g2] += ApplyAggregation<OP>(v1_2, (T2)0);
                        local_res_ptr[g3] += ApplyAggregation<OP>(v1_3, (T2)0);
                    }
                    
                    for (; i < unroll_bound; i += UNROLL_FACTOR) 
                    {
                        size_t oid0 = local_OID[i],     oid1 = local_OID[i + 1];
                        size_t oid2 = local_OID[i + 2], oid3 = local_OID[i + 3];

                        int g0 = local_GroupID[i],     g1 = local_GroupID[i + 1];
                        int g2 = local_GroupID[i + 2], g3 = local_GroupID[i + 3];

                        local_res_ptr[g0] += ApplyAggregation<OP>(agg_col1[oid0], (T2)0);
                        local_res_ptr[g1] += ApplyAggregation<OP>(agg_col1[oid1], (T2)0);
                        local_res_ptr[g2] += ApplyAggregation<OP>(agg_col1[oid2], (T2)0);
                        local_res_ptr[g3] += ApplyAggregation<OP>(agg_col1[oid3], (T2)0);
                    }
                    
                    for (; i < active_count; ++i) {
                        local_res_ptr[local_GroupID[i]] += ApplyAggregation<OP>(agg_col1[local_OID[i]], (T2)0);
                    }
                }
            }
        } // Implicit OpenMP Barrier Syncs All Threads

        // =========================================================
        // Stage 4: High-Speed Sequential Matrix Reduction
        // =========================================================
        for (int t = 0; t < num_threads; ++t) {
            const unsigned long long* __restrict__ local_ptr = global_thread_results + t * PADDED_GROUPS;
            for (int g = 0; g < total_groups; ++g) {
                res_vec[g] += local_ptr[g];
            }
        }

        free(global_thread_results);
    }

    // =========================================================
    // [CRITICAL FIX]: Generic FK Types via Template Parameters
    // We explicitly define up to 4 FK types to correctly issue 1-byte 
    // or 4-byte memory load instructions, preventing garbage aliasing reads.
    // =========================================================
    template <AggOp OP, typename T1, typename T2, bool HAS_COL2, 
              bool INDIRECT1 = false, bool INDIRECT2 = false,
              typename L1 = int, typename L2 = int,
              typename FK0 = int, typename FK1 = int, typename FK2 = int, typename FK3 = int>
    __attribute__((always_inline)) inline void JoinAgg_Vector_Pipeline_CPU(
        const size_t num_tuples, 
        const int join_col_num,
        const void** __restrict__ fks, 
        const int** __restrict__ vectors, 
        const int* __restrict__ factors,
        const T1* __restrict__ agg_col1,   
        const L1* __restrict__ lookup1,    
        const T2* __restrict__ agg_col2,   
        const L2* __restrict__ lookup2,    
        unsigned long long* __restrict__ res_vec,
        const int num_threads,
        const int total_groups) 
    {
        omp_set_num_threads(num_threads);
        const int VEC_LEN = 1024; 

        // =========================================================
        // [HARDWARE FIX 1]: Anti False-Sharing Matrix
        // =========================================================
        const size_t PADDED_GROUPS = ((total_groups + 63) / 64) * 64 + 64; 

        unsigned long long* global_thread_results = nullptr;
        if (posix_memalign((void**)&global_thread_results, 4096, num_threads * PADDED_GROUPS * sizeof(unsigned long long)) != 0) {
            std::cerr << "[Fatal Error] posix_memalign failed to allocate reduction matrix." << std::endl;
            return;
        }

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();

            // =========================================================
            // [HARDWARE FIX 2]: Hard CPU Affinity
            // =========================================================
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(tid % std::thread::hardware_concurrency(), &cpuset);
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

            // =========================================================
            // [HARDWARE FIX 3]: NUMA First-Touch Allocation
            // =========================================================
            unsigned long long* __restrict__ local_res_ptr = global_thread_results + tid * PADDED_GROUPS;
            for (int g = 0; g < total_groups; ++g) {
                local_res_ptr[g] = 0;
            }

            size_t chunk_size = num_tuples / num_threads;
            size_t start_idx = tid * chunk_size;
            size_t end_idx = (tid == num_threads - 1) ? num_tuples : start_idx + chunk_size;

            uint32_t local_OID[VEC_LEN];     
            uint32_t local_GroupID[VEC_LEN];

            for (size_t block_start = start_idx; block_start < end_idx; block_start += VEC_LEN) 
            {
                size_t length = (block_start + VEC_LEN > end_idx) ? (end_idx - block_start) : VEC_LEN;
                size_t active_count = 0;

                // ---------------------------------------------------------
                // Stage 1: Explicit Typed Probe for Dimension 0
                // ---------------------------------------------------------
                const FK0* __restrict__ fk0 = static_cast<const FK0*>(fks[0]);
                const int* __restrict__ vec0 = vectors[0];
                const int factor0 = factors[0];

                for (size_t i = 0; i < length; ++i) {
                    size_t global_idx = block_start + i;
                    int status = vec0[fk0[global_idx] - 1]; 
                    local_OID[active_count] = global_idx;
                    local_GroupID[active_count] = status * factor0;
                    active_count += (status != -1);
                }

                // ---------------------------------------------------------
                // Stage 2: Compile-Time Unrolled Dimensional Probes
                // ---------------------------------------------------------
                if (join_col_num > 1) {
                    size_t new_active_count = 0;
                    const FK1* __restrict__ fk1 = static_cast<const FK1*>(fks[1]);
                    const int* __restrict__ vec1 = vectors[1];
                    const int factor1 = factors[1];

                    for (size_t i = 0; i < active_count; ++i) {
                        size_t oid = local_OID[i];
                        int status = vec1[fk1[oid] - 1];
                        local_OID[new_active_count] = oid;
                        local_GroupID[new_active_count] = local_GroupID[i] + status * factor1;
                        new_active_count += (status != -1);
                    }
                    active_count = new_active_count;
                }

                if (join_col_num > 2) {
                    size_t new_active_count = 0;
                    const FK2* __restrict__ fk2 = static_cast<const FK2*>(fks[2]);
                    const int* __restrict__ vec2 = vectors[2];
                    const int factor2 = factors[2];

                    for (size_t i = 0; i < active_count; ++i) {
                        size_t oid = local_OID[i];
                        int status = vec2[fk2[oid] - 1];
                        local_OID[new_active_count] = oid;
                        local_GroupID[new_active_count] = local_GroupID[i] + status * factor2;
                        new_active_count += (status != -1);
                    }
                    active_count = new_active_count;
                }

                if (join_col_num > 3) {
                    size_t new_active_count = 0;
                    const FK3* __restrict__ fk3 = static_cast<const FK3*>(fks[3]);
                    const int* __restrict__ vec3 = vectors[3];
                    const int factor3 = factors[3];

                    for (size_t i = 0; i < active_count; ++i) {
                        size_t oid = local_OID[i];
                        int status = vec3[fk3[oid] - 1];
                        local_OID[new_active_count] = oid;
                        local_GroupID[new_active_count] = local_GroupID[i] + status * factor3;
                        new_active_count += (status != -1);
                    }
                    active_count = new_active_count;
                }

                // ---------------------------------------------------------
                // Stage 3: Extreme Pipelined Aggregation with 4x Unrolling
                // [CRITICAL FIX]: Hand-unrolled processing extracts maximum MLP 
                // (Memory-Level Parallelism) without sacrificing WAW correctness.
                // ---------------------------------------------------------
                const size_t UNROLL_FACTOR = 4;
                const size_t PREFETCH_DIST = 16; 

                // Strictly safe loop peeling boundaries
                size_t prefetch_bound = 0;
                if (active_count > PREFETCH_DIST + UNROLL_FACTOR) {
                    prefetch_bound = ((active_count - PREFETCH_DIST) / UNROLL_FACTOR) * UNROLL_FACTOR;
                }
                size_t unroll_bound = (active_count / UNROLL_FACTOR) * UNROLL_FACTOR;

                if constexpr (HAS_COL2) 
                {
                    size_t i = 0;
                    
                    // Phase 3A: Pipelined Engine WITH Branchless Prefetching
                    for (; i < prefetch_bound; i += UNROLL_FACTOR) 
                    {
                        __builtin_prefetch(&agg_col1[local_OID[i + PREFETCH_DIST]], 0, 1);
                        __builtin_prefetch(&agg_col2[local_OID[i + PREFETCH_DIST]], 0, 1);
                        __builtin_prefetch(&agg_col1[local_OID[i + PREFETCH_DIST + 1]], 0, 1);
                        __builtin_prefetch(&agg_col2[local_OID[i + PREFETCH_DIST + 1]], 0, 1);

                        size_t o0 = local_OID[i],     o1 = local_OID[i + 1];
                        size_t o2 = local_OID[i + 2], o3 = local_OID[i + 3];

                        int g0 = local_GroupID[i],     g1 = local_GroupID[i + 1];
                        int g2 = local_GroupID[i + 2], g3 = local_GroupID[i + 3];

                        // Concurrent Gathers triggered by CPU execution ports
                        L1 v1_0; if constexpr(INDIRECT1) v1_0 = lookup1[agg_col1[o0]-1]; else v1_0 = agg_col1[o0];
                        L1 v1_1; if constexpr(INDIRECT1) v1_1 = lookup1[agg_col1[o1]-1]; else v1_1 = agg_col1[o1];
                        L1 v1_2; if constexpr(INDIRECT1) v1_2 = lookup1[agg_col1[o2]-1]; else v1_2 = agg_col1[o2];
                        L1 v1_3; if constexpr(INDIRECT1) v1_3 = lookup1[agg_col1[o3]-1]; else v1_3 = agg_col1[o3];

                        L2 v2_0; if constexpr(INDIRECT2) v2_0 = lookup2[agg_col2[o0]-1]; else v2_0 = agg_col2[o0];
                        L2 v2_1; if constexpr(INDIRECT2) v2_1 = lookup2[agg_col2[o1]-1]; else v2_1 = agg_col2[o1];
                        L2 v2_2; if constexpr(INDIRECT2) v2_2 = lookup2[agg_col2[o2]-1]; else v2_2 = agg_col2[o2];
                        L2 v2_3; if constexpr(INDIRECT2) v2_3 = lookup2[agg_col2[o3]-1]; else v2_3 = agg_col2[o3];

                        // Sequential Safe Scatters (HW Store Buffer guarantees correctness on collisions)
                        local_res_ptr[g0] += ApplyAggregation<OP>(v1_0, v2_0);
                        local_res_ptr[g1] += ApplyAggregation<OP>(v1_1, v2_1);
                        local_res_ptr[g2] += ApplyAggregation<OP>(v1_2, v2_2);
                        local_res_ptr[g3] += ApplyAggregation<OP>(v1_3, v2_3);
                    }
                    
                    // Phase 3B: Unrolled Cleanup Engine WITHOUT Prefetching
                    for (; i < unroll_bound; i += UNROLL_FACTOR) 
                    {
                        size_t o0 = local_OID[i],     o1 = local_OID[i + 1];
                        size_t o2 = local_OID[i + 2], o3 = local_OID[i + 3];

                        int g0 = local_GroupID[i],     g1 = local_GroupID[i + 1];
                        int g2 = local_GroupID[i + 2], g3 = local_GroupID[i + 3];

                        L1 v1_0; if constexpr(INDIRECT1) v1_0 = lookup1[agg_col1[o0]-1]; else v1_0 = agg_col1[o0];
                        L1 v1_1; if constexpr(INDIRECT1) v1_1 = lookup1[agg_col1[o1]-1]; else v1_1 = agg_col1[o1];
                        L1 v1_2; if constexpr(INDIRECT1) v1_2 = lookup1[agg_col1[o2]-1]; else v1_2 = agg_col1[o2];
                        L1 v1_3; if constexpr(INDIRECT1) v1_3 = lookup1[agg_col1[o3]-1]; else v1_3 = agg_col1[o3];

                        L2 v2_0; if constexpr(INDIRECT2) v2_0 = lookup2[agg_col2[o0]-1]; else v2_0 = agg_col2[o0];
                        L2 v2_1; if constexpr(INDIRECT2) v2_1 = lookup2[agg_col2[o1]-1]; else v2_1 = agg_col2[o1];
                        L2 v2_2; if constexpr(INDIRECT2) v2_2 = lookup2[agg_col2[o2]-1]; else v2_2 = agg_col2[o2];
                        L2 v2_3; if constexpr(INDIRECT2) v2_3 = lookup2[agg_col2[o3]-1]; else v2_3 = agg_col2[o3];

                        local_res_ptr[g0] += ApplyAggregation<OP>(v1_0, v2_0);
                        local_res_ptr[g1] += ApplyAggregation<OP>(v1_1, v2_1);
                        local_res_ptr[g2] += ApplyAggregation<OP>(v1_2, v2_2);
                        local_res_ptr[g3] += ApplyAggregation<OP>(v1_3, v2_3);
                    }

                    // Phase 3C: Scalar Tail
                    for (; i < active_count; ++i) {
                        L1 val1; if constexpr(INDIRECT1) val1 = lookup1[agg_col1[local_OID[i]]-1]; else val1 = agg_col1[local_OID[i]];
                        L2 val2; if constexpr(INDIRECT2) val2 = lookup2[agg_col2[local_OID[i]]-1]; else val2 = agg_col2[local_OID[i]];
                        local_res_ptr[local_GroupID[i]] += ApplyAggregation<OP>(val1, val2);
                    }
                } 
                else 
                {
                    // Single-Column Fast Path (e.g., Q2/Q3 SUM)
                    size_t i = 0;
                    for (; i < prefetch_bound; i += UNROLL_FACTOR) 
                    {
                        __builtin_prefetch(&agg_col1[local_OID[i + PREFETCH_DIST]], 0, 1);
                        __builtin_prefetch(&agg_col1[local_OID[i + PREFETCH_DIST + 1]], 0, 1);

                        size_t o0 = local_OID[i],     o1 = local_OID[i + 1];
                        size_t o2 = local_OID[i + 2], o3 = local_OID[i + 3];

                        int g0 = local_GroupID[i],     g1 = local_GroupID[i + 1];
                        int g2 = local_GroupID[i + 2], g3 = local_GroupID[i + 3];

                        L1 v1_0; if constexpr(INDIRECT1) v1_0 = lookup1[agg_col1[o0]-1]; else v1_0 = agg_col1[o0];
                        L1 v1_1; if constexpr(INDIRECT1) v1_1 = lookup1[agg_col1[o1]-1]; else v1_1 = agg_col1[o1];
                        L1 v1_2; if constexpr(INDIRECT1) v1_2 = lookup1[agg_col1[o2]-1]; else v1_2 = agg_col1[o2];
                        L1 v1_3; if constexpr(INDIRECT1) v1_3 = lookup1[agg_col1[o3]-1]; else v1_3 = agg_col1[o3];

                        local_res_ptr[g0] += ApplyAggregation<OP>(v1_0, (T2)0);
                        local_res_ptr[g1] += ApplyAggregation<OP>(v1_1, (T2)0);
                        local_res_ptr[g2] += ApplyAggregation<OP>(v1_2, (T2)0);
                        local_res_ptr[g3] += ApplyAggregation<OP>(v1_3, (T2)0);
                    }
                    
                    for (; i < unroll_bound; i += UNROLL_FACTOR) 
                    {
                        size_t o0 = local_OID[i],     o1 = local_OID[i + 1];
                        size_t o2 = local_OID[i + 2], o3 = local_OID[i + 3];

                        int g0 = local_GroupID[i],     g1 = local_GroupID[i + 1];
                        int g2 = local_GroupID[i + 2], g3 = local_GroupID[i + 3];

                        L1 v1_0; if constexpr(INDIRECT1) v1_0 = lookup1[agg_col1[o0]-1]; else v1_0 = agg_col1[o0];
                        L1 v1_1; if constexpr(INDIRECT1) v1_1 = lookup1[agg_col1[o1]-1]; else v1_1 = agg_col1[o1];
                        L1 v1_2; if constexpr(INDIRECT1) v1_2 = lookup1[agg_col1[o2]-1]; else v1_2 = agg_col1[o2];
                        L1 v1_3; if constexpr(INDIRECT1) v1_3 = lookup1[agg_col1[o3]-1]; else v1_3 = agg_col1[o3];

                        local_res_ptr[g0] += ApplyAggregation<OP>(v1_0, (T2)0);
                        local_res_ptr[g1] += ApplyAggregation<OP>(v1_1, (T2)0);
                        local_res_ptr[g2] += ApplyAggregation<OP>(v1_2, (T2)0);
                        local_res_ptr[g3] += ApplyAggregation<OP>(v1_3, (T2)0);
                    }
                    
                    for (; i < active_count; ++i) {
                        L1 val1; if constexpr(INDIRECT1) val1 = lookup1[agg_col1[local_OID[i]]-1]; else val1 = agg_col1[local_OID[i]];
                        local_res_ptr[local_GroupID[i]] += ApplyAggregation<OP>(val1, (T2)0);
                    }
                }
            }
        } // Implicit OpenMP Barrier

        // =========================================================
        // Stage 4: High-Speed Sequential Matrix Reduction
        // =========================================================
           for (int t = 0; t < num_threads; ++t) {
            const unsigned long long* __restrict__ local_ptr = global_thread_results + t * PADDED_GROUPS;
            #pragma GCC ivdep
            for (int g = 0; g < total_groups; ++g) {
                res_vec[g] += local_ptr[g];
            }
        }

        free(global_thread_results);
    }
    // /**
    //  * @brief Vectorized Pipeline Join & Aggregate using Bitmaps.
    //  * Identical pipeline structure to Vector version, but uses Bitwise probes.
    //  */
    // inline void JoinAgg_Bitmap_Pipeline_CPU(
    //     size_t num_tuples, int join_col_num,
    //     const int** __restrict__ fks, 
    //     const uint8_t** __restrict__ bitmaps, 
    //     const int* __restrict__ agg_col1,
    //     const int* __restrict__ agg_col2,
    //     AggFunc agg_func,
    //     unsigned long long* __restrict__ res_vec,
    //     int num_threads)
    // {
    //     omp_set_num_threads(num_threads);
    //     const int VEC_LEN = 2048; 

    //     #pragma omp parallel
    //     {
    //         int64_t local_OID[VEC_LEN];

    //         #pragma omp for schedule(dynamic)
    //         for (size_t block_start = 0; block_start < num_tuples; block_start += VEC_LEN) 
    //         {
    //             size_t length = std::min((size_t)VEC_LEN, num_tuples - block_start);
    //             size_t active_count = 0;

    //             for (size_t i = 0; i < length; ++i) {
    //                 size_t global_idx = block_start + i;
    //                 if (ProbeBitmap(bitmaps[0], fks[0][global_idx]) != -1) {
    //                     local_OID[active_count++] = global_idx;
    //                 }
    //             }

    //             for (int j = 1; j < join_col_num; ++j) {
    //                 size_t new_active_count = 0;
    //                 for (size_t i = 0; i < active_count; ++i) {
    //                     size_t oid = local_OID[i];
    //                     if (ProbeBitmap(bitmaps[j], fks[j][oid]) != -1) {
    //                         local_OID[new_active_count++] = oid;
    //                     }
    //                 }
    //                 active_count = new_active_count;
    //             }

    //             // Since Bitmaps do not provide Group IDs, we accumulate to a Scalar result (Index 0)
    //             unsigned long long local_sum = 0;
    //             for (size_t i = 0; i < active_count; ++i) {
    //                 local_sum += agg_func(agg_col1, agg_col2, local_OID[i]);
    //             }
                
    //             #pragma omp atomic
    //             res_vec[0] += local_sum;
    //         }
    //     }
    // }

} // namespace CPU
} // namespace MOSS_DB