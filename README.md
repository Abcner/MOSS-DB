# MOSS-DB: GPU-Accelerated In-Memory Database Prototype
> **Targeting Star Schema Benchmark (SSB)**

* **Version**: 1.0.0 (Alpha - Research Prototype)
* **Author**: GPU Database Research Group -- Ruichen Han
* **License**: RUC License / Academic Use Only
* **Standard**: Compliant with TPDS (IEEE Trans. Parallel Distrib. Syst.) coding standards

---

## 1. OVERVIEW

MOSS-DB is a high-performance, GPU-accelerated in-memory database engine designed specifically for OLAP (Online Analytical Processing) workloads. It implements a complete execution pipeline for the Star Schema Benchmark (SSB), featuring a custom JSON-based Query Language (JQL).

**Key architectural features include:**
1. **Columnar Storage**: Utilizing dictionary encoding and bit-packing for efficient GPU memory utilization.
2. **JQL Parser**: A schema-agnostic parser supporting complex OLAP operations (Drill-Down, Rollup) and predicate pushdown.
3. **Late Materialization**: Processing strictly on ID-based compressed data until the final result generation phase.
4. **Optimized CUDA Kernels**: 
   - Template-based `ProbeDenseKernel` supporting dynamic aggregation modes (SUM, PRODUCT, SUBTRACT).
   - Fused Filter Operators (`FilterEqualWithMask`) to minimize memory bandwidth.
5. **Advanced Rollup Optimization**: A specialized Bottom-Up execution strategy for hierarchical dimensions (e.g., Date -> Month -> Year) to reduce join cardinality early.

---

## 2. SYSTEM REQUIREMENTS

### Hardware
* **CPU**: x86_64 Architecture (Intel/AMD)
* **GPU**: NVIDIA GPU with Compute Capability >= 6.0 (Pascal or newer)
* **RAM**: Sufficient to hold the SSB dataset (Scale Factor dependent)
* **VRAM**: At least 8GB recommended for SF=10

### Software
* **OS**: Linux (Ubuntu 20.04+ / CentOS 7+)
* **Compiler**: GCC 7.0+ / Clang 6.0+ (**Must support C++17**)
* **CUDA Toolkit**: 11.0 or higher
* **CMake**: 3.15 or higher
* **Third-party Libraries**:
  * `cJSON` (Included in `include/common/`) for JQL parsing.
  * `OpenMP` (For CPU-side parallelism in Phase 1 & 2).

---

## 3. PROJECT STRUCTURE

```text
MOSS-DB/
├── CMakeLists.txt              # [Build] CMake configuration defining C++17/CUDA flags, dependencies, and build rules.
├── README.md                   # [Doc] Project documentation, build instructions, and JQL specifications.
├── build/                      # [Artifacts] Directory for CMake-generated intermediate build files.
├── bin/                        # [Binaries] Directory for compiled executables (e.g., moss_db).
├── cmake/                      # [Build] Custom CMake modules (e.g., FindCUDA.cmake for environment detection).
├── data/                       # [Data] Raw SSB benchmark datasets (.tbl format) for memory loading and dictionary encoding.
├── jql/                        # [Query] JSON Query Language scripts covering SSB Q1-Q4 workloads.
│   ├── q11.json                # Q1.1: Tests scalar aggregation and high-selectivity filtering.
│   ├── q21.json                # Q2.1: Tests grouped aggregation and dimension Drill-down.
│   ├── q34.json                # Q3.4: Demonstrates complex execution plans with mixed Drill-down and Rollup operations.
│   └── q41.json                # Q4.1: Demonstrates templated subtraction aggregation (LO_REVENUE - LO_SUPPLYCOST).
├── src/                        # [Source] Core system source code.
│   └── main_engine.cu          # [Core] Main execution engine handling JQL parsing, plan generation, pipeline scheduling (Phase 1-5), and materialization.
├── lib/                        # [Lib] Compiled static/shared libraries (e.g., libcjson.a).
├── external/                   # [Deps] Third-party dependency source code.
│   └── cjson/                  # [Parser] cJSON library for parsing JQL structural components.
├── logs/                       # [Logs] Runtime logs recording query latency, memory consumption, and kernel profiling data.
└── include/                    # [Header] Header files defining data structures, macros, and operator interfaces.
    ├── common/                 # [Common] Utility functions and global configurations.
    │   ├── config.h            # Global parameters (e.g., CUDA Block Size, Items Per Thread, Max Joins).
    │   ├── gpu_db_utils.h      # CUDA error checking macros (CHECK_CUDA) and VRAM allocation/copy wrappers.
    │   └── timer.h             # High-precision timers for profiling execution phases (Phase 1-5).
    ├── ssb+/                   # [Schema] SSB business logic and schema definitions.
    │   └── ssb+_utils.h        # SSB table structures, column mappings, and data loading utilities.
    ├── operators/              # [Operators] Database operator implementations.
    │   ├── cpu/                # [CPU Ops] CPU-side operators for Phase 1 (Dimension Reduction) and Phase 2.
    │   │   ├── pred_cpu.h      # Predicate filters: FilterEqual, FilterRange, and fused FilterEqualWithMask.
    │   │   ├── compress_cpu.h  # Dictionary compression: Bitmap construction and Compressed Index generation.
    │   │   └── join_cpu.h      # CPU Join operators: JoinProbeAndMap for passing data between parent/child dimensions in Rollup.
    │   └── gpu/                # [GPU Ops] GPU-side operators for Phase 4 (Fact Table Processing).
    │       ├── build_kernels.cuh # Build phase kernels for Hash/Bitmap initialization (if GPU-built).
    │       ├── join_gpu.cuh      # GPU Join helpers for complex association logic and metadata preparation.
    │       ├── load_gpu.cuh      # Data loading kernels for efficient, aligned columnar memory access.
    │       ├── reduce_gpu.cuh    # Reduction kernels: BlockReduceSum for block-level aggregation (Scalar Aggregation).
    │       ├── term_gpu.cuh      # Termination/Cleanup kernels for resource management and early termination logic.
    │       └── probe_kernel.cuh  # [Kernel] Core ProbeDenseKernel fusing Hash Join probing, predicates, and multi-mode aggregation (SUM/PRODUCT/SUBTRACT).'''
```
## 4. BUILD INSTRUCTIONS

### Standard Build (Generic x86_64)
```bash
# 1. Create build directory
mkdir build && cd build

# 2. Configure with C++17 and default CUDA architecture
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 ..

# 3. Compile with all available cores
make -j $ (nproc)

# 4. Verify binary (Check ELF header and GPU architecture)
file bin/moss_db
# Expected output: ELF 64-bit LSB shared object, x86-64, version 1 (SYSV), ...
