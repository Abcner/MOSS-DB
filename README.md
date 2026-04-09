================================================================================
                MOSS-DB: GPU-Accelerated In-Memory Database Prototype
                     Targeting Star Schema Benchmark (SSB)
================================================================================

[Version]       1.0.0 (Alpha - Research Prototype)
[Author]        GPU Database Research Group--Ruichen Han
[License]       RUC License / Academic Use Only
[Standard]      Compliant with TPDS (IEEE Trans. Parallel Distrib. Syst.) coding standards

================================================================================
1. OVERVIEW
================================================================================

MOSS-DB is a high-performance, GPU-accelerated in-memory database engine designed 
specifically for OLAP (Online Analytical Processing) workloads. It implements a 
complete execution pipeline for the Star Schema Benchmark (SSB), featuring a custom 
JSON-based Query Language (JQL).

Key architectural features include:
1.  **Columnar Storage**: Utilizing dictionary encoding and bit-packing for efficient 
    GPU memory utilization.
2.  **JQL Parser**: A schema-agnostic parser supporting complex OLAP operations 
    (Drill-Down, Rollup) and predicate pushdown.
3.  **Late Materialization**: Processing strictly on ID-based compressed data until 
    the final result generation phase.
4.  **Optimized CUDA Kernels**: 
    - Template-based `ProbeDenseKernel` supporting dynamic aggregation modes 
      (SUM, PRODUCT, SUBTRACT).
    - Fused Filter Operators (`FilterEqualWithMask`) to minimize memory bandwidth.
5.  **Advanced Rollup Optimization**: A specialized Bottom-Up execution strategy 
    for hierarchical dimensions (e.g., Date -> Month -> Year) to reduce join 
    cardinality early.

================================================================================
2. SYSTEM REQUIREMENTS
================================================================================

[Hardware]
- CPU: x86_64 Architecture (Intel/AMD)
- GPU: NVIDIA GPU with Compute Capability >= 6.0 (Pascal or newer)
- RAM: Sufficient to hold the SSB dataset (Scale Factor dependent)
- VRAM: At least 8GB recommended for SF=10

[Software]
- OS: Linux (Ubuntu 20.04+ / CentOS 7+)
- Compiler: GCC 7.0+ / Clang 6.0+ (Must support C++17)
- CUDA Toolkit: 11.0 or higher
- CMake: 3.15 or higher
- Third-party Libraries:
  - cJSON (Included in `include/common/`) for JQL parsing.
  - OpenMP (For CPU-side parallelism in Phase 1 & 2).

================================================================================
3. PROJECT STRUCTURE
================================================================================

MOSS-DB/
├── CMakeLists.txt              # [Build] CMake 构建配置文件，定义编译选项(C++17/CUDA)、依赖库连接及可执行文件生成规则
├── README.md                   # [Doc] 项目说明文档，包含编译指南、JQL规范说明及系统架构概览
├── build                       # [Artifacts] 存放 CMake 生成的中间构建文件（Makefile, Cache 等）
├── bin                         # [Binaries] 存放编译生成的最终可执行文件（如 moss_db）
├── cmake                       # [Build] 自定义的 CMake 模块脚本（例如 FindCUDA.cmake 等环境检测脚本）
├── data/                       # [Data] 存放 SSB 基准测试的原始数据文件（.tbl 格式），用于加载到内存并构建字典编码
├── jql/                        # [Query] JQL (JSON Query Language) 查询脚本目录，涵盖 SSB Q1-Q4 测试集
│   ├── q11.json                # Q1.1 查询定义：测试标量聚合与高选择率过滤
│   ├── q21.json                # Q2.1 查询定义：测试分组聚合与维度 Drill-down
│   ├── q34.json                # Q3.4 查询定义：演示 Drill-down 与 Rollup 混合操作的复杂执行计划
│   └── q41.json                # Q4.1 查询定义：演示模板化减法聚合 (LO_REVENUE - LO_SUPPLYCOST)
├── src/                        # [Source] 系统核心源代码目录
│   └── main_engine.cu          # [Core] 核心执行引擎入口，负责 JQL 解析、查询计划生成、Pipeline 调度 (Phase 1-5) 及结果物化
├── lib/                        # [Lib] 编译好的静态库或共享库文件（如 libcjson.a）
├── external/                   # [Deps] 第三方依赖库源码
│   └── cjson                   # [Parser] cJSON 库源码，用于解析 JQL 的 JSON 结构
├── logs/                       # [Logs] 运行时生成的日志文件，记录查询延迟、内存使用及内核 Profiling 数据
└── include/                    # [Header] 头文件目录，定义数据结构、宏及算子接口
    ├── common/                 # [Common] 通用工具与配置
    │   ├── config.h            # 全局配置参数（如 CUDA Block Size, Items Per Thread, Max Joins）
    │   ├── gpu_db_utils.h      # CUDA 错误检查宏 (CHECK_CUDA)、显存分配/拷贝辅助函数
    │   └── timer.h             # 高精度计时器，用于统计各个执行阶段（Phase 1-5）的耗时
    ├── ssb+/                   # [Schema] SSB 业务相关定义
    │   └── ssb+_utils.h        # SSB 表结构定义、列映射关系及数据加载辅助工具
    ├── operators/              # [Operators] 数据库算子实现
    │   ├── cpu/                # [CPU Ops] CPU 端算子，主要用于 Phase 1 (Dimension Reduction) 和 Phase 2
    │   │   ├── pred_cpu.h      # 谓词过滤算子：包含 FilterEqual, FilterRange 及融合算子 FilterEqualWithMask
    │   │   ├── compress_cpu.h  # 字典压缩算子：用于构建 Bitmap 和生成 Compressed Index
    │   │   └── join_cpu.h      # CPU Join 算子：包含 JoinProbeAndMap，用于 Rollup 优化中的父子维度数据传递
    │   └── gpu/                # [GPU Ops] GPU 端算子，主要用于 Phase 4 (Fact Table Processing)
            ├── build_kernels.cuh # 构建阶段内核：用于 Hash 表或 Bitmap 的初始化（如需 GPU 构建）
            ├── join_gpu.cuh      # GPU Join 辅助内核：处理复杂的关联逻辑或 Metadata 准备
            ├── load_gpu.cuh      # 数据加载内核：负责列式数据的高效显存加载与对齐
            ├── reduce_gpu.cuh    # 归约内核：包含 BlockReduceSum，用于标量聚合（Scalar Aggregation）的 Block 级汇总
            ├── term_gpu.cuh      # 终止/清理内核：负责 GPU 资源的清理或特定条件的提前终止逻辑
            └── probe_kernel.cuh  # [Kernel] 核心探测内核 (ProbeDenseKernel)：融合了 Hash Join、谓词计算与多模式聚合 (SUM/PRODUCT/SUBTRACT)
    

================================================================================
4. BUILD INSTRUCTIONS
================================================================================

1.  Create a build directory:
    $ mkdir build && cd build

2.  Configure the project (Ensure C++17 is enabled):
    $ cmake -DCMAKE_BUILD_TYPE=Release ..     # Release
    $ cmake  ..                               # 80 for Ampere (A100/3090)

3.  Compile:
    $ make -j$(nproc)

    *Note: If you encounter "constexpr if" warnings, verify that -std=c++17 
     flag is correctly set in CMakeLists.txt.*

================================================================================
5. RUNNING THE SYSTEM
================================================================================

Usage:
    ./moss_db  <jql_query_file>

Example:
    # Run SSB Query 1.1
    $ ./moss_db  ../jql/q11.json

    # Run SSB Query 3.4 (With Rollup)
    $ ./moss_db  ../jql/q34.json

    # Run SSB Query 4.1 (With Profit Calculation)
    $ ./moss_db  ../jql/q41.json

[Expected Output]
The system prints the Logical Query Plan, Execution Phases (Time taken), 
Kernel Launch Configuration, and the Final Result Table.

================================================================================
6. JQL SPECIFICATION (JSON Query Language)
================================================================================

MOSS-DB uses a hierarchical JSON format to describe execution plans.

Structure:
{
  "SELECT": {
    "PROJECT": ["<Column1>", "<Column2>"],
    "ORDERBY": [ ... ]，
    "OLAP": [
      {
        "DRILLDOWN": [ ... ],   // Standard Top-Down Filtering
        "ROLLUP": [ ... ]       // Optimized Bottom-Up Filtering
      }
    ],
    "FILTING":[],
    "AGG": [
      {
        "FUNCTION": "SUM",
        "EXPRESSION": "LO_REVENUE-LO_SUPPLYCOST", // Supports *, -, +
        "ALIAS": "PROFIT",
        "TABLE": "LINEORDER"
      }
    ]
  }
}

[Filter Definition]
Filters can be single objects or arrays (for multi-predicate filtering on one dimension).
- "EXPRESSION": Supports "=", "<", ">", "BETWEEN", "BETWEENAND", "OR".
- "VALUE": Can be a single value, an array [v1, v2] for BETWEEN, or a list for IN.

================================================================================
7. ARCHITECTURAL HIGHLIGHTS & OPTIMIZATIONS
================================================================================

1.  **Schema-Agnostic Rollup**:
    The engine automatically resolves parent-child relationships in the Rollup 
    path by traversing the Join Graph, removing the need for explicit table 
    names in the query.

2.  **Kernel Fusion (CPU & GPU)**:
    - `FilterEqualWithMask`: Combines bitmap checking and value comparison into 
      a single pass to reduce memory traffic.
    - `ProbeDenseKernel`: Fuses Hash Join probing, Predicate evaluation, and 
      Aggregation into a single CUDA kernel.

3.  **Template Metaprogramming for Aggregation**:
    The CUDA Probe Kernel uses C++17 `if constexpr` and template parameters (`AggOp`) 
    to generate specialized kernels for SUM, PRODUCT, and SUBTRACT operations 
    at compile-time, avoiding runtime branching overhead on the GPU.

4.  **Benign Race Optimization**:
    In the Rollup phase (`FilterChildAndMarkParent`), the engine utilizes benign 
    data races to update parent validity bitmaps in parallel without expensive 
    atomic locks, significantly speeding up the dimension reduction phase.

================================================================================
8. TROUBLESHOOTING
================================================================================

Q: "Rollup nodes not found" error?
A: Ensure your `jql/join_path.json` correctly defines the hierarchy (e.g., 
   Date -> YearMonth -> Year) and that the columns in the JQL query match 
   the schema definition.

Q: Incorrect Result in Q4?
A: Check if the Aggregation Expression uses "-" and if the `ProbeDenseKernel` 
   was correctly instantiated with `AggOp::SUBTRACT` in `main_engine.cu`.

Q: Compilation fails with "constexpr if" warning?
A: Your compiler is not using C++17. Add `set(CMAKE_CXX_STANDARD 17)` to CMakeLists.txt.

================================================================================