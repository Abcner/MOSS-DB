# MOSS-GDB: Seamless GPU Acceleration Integration for openGauss

This repository contains the system integration implementation of **MOSS-GDB**, demonstrating a seamless, zero-effect integration scheme of a GPU-accelerated execution engine into [openGauss](https://opengauss.org/), a widely adopted enterprise-grade relational database.

## 📖 About This Work

A major obstacle to embedding GPU-accelerated engines into legacy DBMSs lies in excessive engineering overhead and potential system instability, usually introduced by substantial modifications to the built-in SQL parser, query optimizer, and execution engine. 

To address this issue, we propose a loosely coupled integration paradigm featuring **zero impact on the host system**. MOSS-GDB functions as an independent plug-and-play GPU co-processor, requiring no changes to the core source code of openGauss.

### Key Architectural Features:
* **In-Process UDF Bridging Mechanism:** Encapsulated as a dynamically loadable shared library, MOSS-GDB operates safely within the openGauss extension sandbox. This eliminates context-switching/serialization overheads and establishes a zero-copy data interoperability channel.
* **Two-Phase Operational Lifecycle:** 
  1. *Persistent State Materialization:* Topologies and dictionaries are materialized in the CPU M-store, while heavily accessed columnar data is prefetched into the GPU V-store (HBM).
  2. *Accelerated Analytical Query Pipeline:* CPU parses JQL (JSON Query Language) semantics, generating compact vectors, and offloads V-computing to the GPU via super kernel fusion.

---

## 🛠️ Repository Structure

```text
.
├── src/
│   ├── main_engine.cu      # Native MOSS-DB backend (CUDA execution engine)
│   └── openjql_udf.cpp     # openGauss UDF bridge (C++ Wrapper)
├── jql/                    # Query plans (JQL JSON scripts)
└── sql/
    └── register_udf.sql    # SQL scripts to register the UDFs in openGauss
```

> **⚠️ Important Configuration Note before Building:**
> The source codes currently contain hardcoded absolute paths for demonstration purposes (e.g., `/home/xxx/workspace/...`). Before compiling, please globally search and replace these paths in `src/main_engine.cu` to match your local deployment environment.

---

## 🚀 Build & Deployment Guide

### Prerequisites
* OS: Linux (CentOS/Ubuntu)
* Database: openGauss Database Instance
* Compiler: GCC/G++ with C++11 or higher
* GPU: NVIDIA GPU with CUDA Toolkit installed
* Build Tool: CMake

### Step 1: Compile the Native GPU Engine
Compile the CUDA core logic into a dynamic linked library (`libmoss_db.so`):
```bash
cd MOSS-GDB/build
cmake ..
make -j
```

### Step 2: Compile the openGauss UDF Bridge
Link the C++ wrapper with openGauss headers and the MOSS-DB dynamic library to generate the sandbox plugin `moss_engine.so`:
```bash
g++ -O3 -fPIC -shared -o moss_engine.so src/openjql_udf.cpp \
    -I$(pg_config --includedir-server) \
    -L/path/to/your/MOSS-DB/build -lmoss_db \
    -Wl,-rpath=/path/to/your/MOSS-DB/build
```

### Step 3: Deploy to openGauss Security Sandbox
openGauss requires all C-extensions to be placed in the `proc_srclib` sandbox directory to prevent malicious code injection.
```bash
# Create directory if not exists
mkdir -p /path/to/openGauss/lib/postgresql/proc_srclib/

# Copy the generated plugin
cp moss_engine.so /path/to/openGauss/lib/postgresql/proc_srclib/
```

### Step 4: Register SQL Interfaces
Start your openGauss instance and register the underlying C functions as friendly SQL interfaces:
```bash
gs_ctl start -D /path/to/openGauss/data/single_node
gsql -d postgres -p 5432 -f sql/register_udf.sql
```

---

## 💡 Usage Manual

Once deployed, you can invoke hardware-accelerated OLAP queries directly via standard SQL in `gsql` or any openGauss client.

### 1. Engine Initialization (Execute once after DB restart)
Pre-warm data dictionaries, columnar data, and schema into GPU memory:
```sql
SELECT openjql_load();
```
*(Expected Output: Success message and initialization time)*

### 2. Standard Accelerated Query Execution
Executes the query and outputs the result with total elapsed time (Input files must exist in the `jql` directory):
```sql
\timing
SELECT openjql('Q11.json');
```

### 3. Debug / Performance Profiling Mode
Outputs detailed pipeline performance metrics, including parsing time, pure GPU computing time, and CPU post-processing overhead:
```sql
SELECT openjql('Q11.json', 'debug');
```
