# ==============================================================================
# MOSS-DB: GPU In-Memory Database Prototype Build Script
# Target: TPDS Journal Open Source Standards
# Version: 4.0 (Heterogeneous CUDA + OpenMP Support Edition)
# ==============================================================================

# 1. 初始化 CMake 版本
cmake_minimum_required(VERSION 3.17)

# 2. 定义项目
project(MOSS-DB 
    VERSION 1.0.0
    DESCRIPTION "High-Performance GPU In-Memory Database for SSB Benchmarks"
    LANGUAGES CXX C CUDA
)

# ==============================================================================
# 全局编译配置
# ==============================================================================

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

# CUDA 架构设置 (e.g., sm_80 for A100, sm_86 for RTX 3090)
if(NOT DEFINED GPU_ARCH)
    set(GPU_ARCH "sm_86") 
endif()

# --- CUDA 标志策略 ---
set(CUDA_COMMON_FLAGS "-arch=${GPU_ARCH} --std=c++14 -Xptxas -v")

set(CMAKE_CUDA_FLAGS_DEBUG "${CUDA_COMMON_FLAGS} -g -G -O0 -DDEBUG_MODE")
set(CMAKE_CUDA_FLAGS_RELEASE "${CUDA_COMMON_FLAGS} -O3 --use_fast_math -DNDEBUG")
set(CMAKE_CUDA_FLAGS "") # 依赖后缀自动附加

# Host 编译器优化 (GCC/Clang)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra)
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()

# ==============================================================================
# 依赖库管理 (OpenMP 核心修复区)
# ==============================================================================

# [FIX 1]: 查找 OpenMP CXX 和 C 支持
find_package(OpenMP REQUIRED)

# cJSON
add_library(cjson SHARED IMPORTED)
set_target_properties(cjson PROPERTIES
    IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/lib/libcjson.so"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/external/cjson"
)

# ==============================================================================
# 路径与源文件定义
# ==============================================================================

set(HEADER_DIRS
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/common
    ${CMAKE_SOURCE_DIR}/include/ssb+
    ${CMAKE_SOURCE_DIR}/include/operators/cpu
    ${CMAKE_SOURCE_DIR}/include/operators/gpu
)

file(GLOB_RECURSE SOURCES "src/*.cu" "src/*.cpp")

# ==============================================================================
# 构建目标与属性设置
# ==============================================================================

add_executable(moss_db ${SOURCES})

# Host 代码使用 C++17
set_property(TARGET moss_db PROPERTY CXX_STANDARD 17)
set_property(TARGET moss_db PROPERTY CXX_STANDARD_REQUIRED ON)
# Device 代码标准由 CUDA_COMMON_FLAGS 控制
set_property(TARGET moss_db PROPERTY CUDA_STANDARD_REQUIRED OFF)

target_include_directories(moss_db PRIVATE ${HEADER_DIRS})

# [FIX 2]: 为 NVCC 显式透传 OpenMP 编译选项
# 这一步极其关键。如果源文件是 .cu 且内部使用了 #pragma omp，
# NVCC 必须通过 -Xcompiler 将 OpenMP flag 传递给底层的主机编译器 (GCC)。
target_compile_options(moss_db PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>
    $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>
)

# [FIX 3]: 链接 OpenMP 库
# OpenMP::OpenMP_CXX 会自动处理链接器所需的 libgomp 等运行时库
target_link_libraries(moss_db 
    PRIVATE 
    cjson
    OpenMP::OpenMP_CXX 
)

# 开启 CUDA 分离编译 (支持 Relocatable Device Code)
set_target_properties(moss_db PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ==============================================================================
# RPATH 设置
# ==============================================================================
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_RPATH "$ORIGIN/../lib")
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")

# ==============================================================================
# 输出配置 & 状态打印
# ==============================================================================

file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/logs)
set_target_properties(moss_db PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

message(STATUS "=========================================================")
message(STATUS "  MOSS-DB Build Configuration (v4.0 Heterogeneous Ready)")
message(STATUS "=========================================================")
message(STATUS "  Build Type       : ${CMAKE_BUILD_TYPE}")
message(STATUS "  GPU Architecture : ${GPU_ARCH}")
message(STATUS "  CUDA Flags       : ${CMAKE_CUDA_FLAGS_${CMAKE_BUILD_TYPE}}") # 打印当前生效的 Flags
message(STATUS "  OpenMP CXX Flags : ${OpenMP_CXX_FLAGS}")                  # 确认 OpenMP Flags 是否正确提取
message(STATUS "  Output Path      : ${CMAKE_SOURCE_DIR}/bin/moss_db")
message(STATUS "=========================================================")