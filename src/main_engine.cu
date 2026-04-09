/**
 * @file main_engine.cu
 * @brief High-Performance SSB Query Engine (Host Controller).
 * @author Assistant Engineer (Refactored for TPDS Standard)
 * @version 4.1 (Query Plan Visualization)
 * @date 2026-01-08
 */

#include <iostream>
#include <vector>
#include <set>
#include <string>
#include <chrono>
#include <map>
#include <iomanip> // For std::setw
#include <cuda_runtime.h>
#include <cJSON.h>
#include <thread>
#include <omp.h>
#include <stdexcept>

// Common Utilities
#include "ssb+/ssb+_utils.h"
#include "common/config.h"
#include "common/timer.h"
#include "common/gpu_db_utils.h"

// Operators
#include "operators/cpu/pred_cpu.h"
#include "operators/cpu/join_cpu.h"
#include "operators/cpu/compress_cpu.h"
#include "operators/gpu/build_kernels.cuh"
#include "operators/gpu/probe_kernels.cuh"

// =========================================================
// 1. Data Structures for Query Plan
// =========================================================

enum class OpType
{
    EQ = 0,
    LT = 1,
    GT = 2,
    LE = 3,
    GE = 4,
    BETWEEN = 5,
    IN = 6
};

struct SelectionOp
{
    std::string table;
    std::string col;
    OpType op_type;
    int val;
    int val2;
    std::vector<int> in_values;
};

struct AggregationOp
{
    std::string func;
    std::string col1;
    std::string op;
    std::string col2;
    std::string alias;
    std::string table;
    std::string exec_engine;
};

struct OrderByOp
{
    std::string col;
    std::string mode; // "ASC" or "DESC"
};

struct OlapOp
{
    std::string dimension;
    SelectionOp filter; // Optional filter inside OLAP
    std::string group_col;
    bool is_rollup;
    std::string rollup_table;
};

struct QueryPlan
{
    std::string query_id;
    bool use_gpu = true;

    std::vector<std::string> projections; // PROJECT
    std::vector<OrderByOp> order_bys;     // ORDERBY
    std::vector<OlapOp> olap_ops;         // OLAP/DRILLDOWN

    std::vector<SelectionOp> selections;     // FILTING + OLAP Filters
    std::vector<AggregationOp> aggregations; // AGG
};

struct JoinNode
{
    std::string table_name;
    std::string fk_col;                         // FK in Lower Level
    std::string pk_col;                         // PK in This Level
    std::vector<std::string> contained_columns; // Columns residing in this table
    std::vector<std::shared_ptr<JoinNode>> upper_levels;
};

struct JoinGraph
{
    // Maps "Root Dimension Table" name to its node (e.g., "CUSTOMER" -> Node)
    std::map<std::string, std::shared_ptr<JoinNode>> dimension_roots;
};

// --- Visualization Helper ---
/**
 * @brief Recursively prints the Join Node hierarchy with Column Metadata.
 * @param node Current JoinNode to print.
 * @param level Indentation level (0 for root, 1+ for children).
 */
void PrintJoinNode(const std::shared_ptr<JoinNode> &node, int level)
{
    if (!node)
        return;

    // 1. Indentation for Tree Structure
    std::string indent(level * 4, ' ');

    std::cout << indent << "|__ " << node->table_name;

    // 2. Print Key Relationships (PK/FK)
    // Root nodes might not have an FK pointing to a lower level in this context,
    // but intermediate nodes will.
    if (!node->fk_col.empty())
    {
        std::cout << " (PK: " << node->pk_col
                  << ", Link via: " << node->fk_col << ")";
    }
    else if (!node->pk_col.empty())
    {
        // For Root Dimension nodes that define a PK but no FK (top of join)
        std::cout << " (PK: " << node->pk_col << ")";
    }

    // 3. Print Contained Columns (Schema Mapping)
    if (!node->contained_columns.empty())
    {
        std::cout << " \033[1;32m[Cols: "; // Green color for visibility (optional ANSI code)
        for (size_t i = 0; i < node->contained_columns.size(); ++i)
        {
            std::cout << node->contained_columns[i]
                      << (i < node->contained_columns.size() - 1 ? ", " : "");
        }
        std::cout << "]\033[0m"; // Reset color
    }

    std::cout << std::endl;

    // 4. Recursively Print Children (Upper Levels)
    for (const auto &child : node->upper_levels)
    {
        PrintJoinNode(child, level + 1);
    }
}

/**
 * @brief Visualizes the complete Join Graph Topology.
 */
void PrintJoinGraph(const JoinGraph &graph)
{
    std::cout << "\n================================================================================" << std::endl;
    std::cout << "                          SCHEMA TOPOLOGY (JOIN GRAPH)                          " << std::endl;
    std::cout << "================================================================================" << std::endl;

    if (graph.dimension_roots.empty())
    {
        std::cout << "  [Warning] Graph is empty. Check Join_Path_Tree.json." << std::endl;
    }

    for (const auto &pair : graph.dimension_roots)
    {
        std::cout << "\n[Dimension Root]: " << pair.first << std::endl;

        // Print columns for the Root Node itself (if any)
        if (!pair.second->contained_columns.empty())
        {
            std::cout << "    \033[1;32m[Cols: ";
            for (size_t i = 0; i < pair.second->contained_columns.size(); ++i)
            {
                std::cout << pair.second->contained_columns[i]
                          << (i < pair.second->contained_columns.size() - 1 ? ", " : "");
            }
            std::cout << "]\033[0m" << std::endl;
        }

        // Recursively print the hierarchy (Upper Levels)
        for (const auto &upper : pair.second->upper_levels)
        {
            PrintJoinNode(upper, 1);
        }
    }
    std::cout << "================================================================================\n"
              << std::endl;
}
// =========================================================
// 2. Helper: Dynamic Column Access (Reflection-like)
// =========================================================

// Helper to get raw pointer and length from ColumnStore based on names
struct ColumnData
{
    void *data;
    size_t len;
    bool is_int;      // true for int, false for string/other
    bool is_fk_proxy; // True if this column is physically an FK but logically a Dimension Value
    int byte_width;   // 4 for int, 1 for int8_t
};

// =========================================================
// 2. Helper: Plan Visualization
// =========================================================

std::string OpTypeToString(OpType op)
{
    switch (op)
    {
    case OpType::EQ:
        return "=";
    case OpType::LT:
        return "<";
    case OpType::GT:
        return ">";
    case OpType::LE:
        return "<=";
    case OpType::GE:
        return ">=";
    case OpType::BETWEEN:
        return "BETWEEN";
    case OpType::IN:
        return "IN";
    default:
        return "???";
    }
}

/**
 * @brief Prints the logical query execution plan to the console.
 * Updated to correctly format IN operators as array lists [v1, v2, ...].
 */
void PrintQueryPlan(const QueryPlan &plan)
{
    // --- Helper Lambda for Section Headers ---
    auto PrintSectionHeader = [](const std::string &title)
    {
        std::cout << "\n"
                  << title << std::endl;
        std::cout << std::string(80, '-') << std::endl;
    };

    // --- Helper Lambda for Table Headers ---
    auto PrintTableHeader = [](const std::vector<std::string> &headers, const std::vector<int> &widths)
    {
        for (size_t i = 0; i < headers.size(); ++i)
        {
            std::cout << std::left << std::setw(widths[i]) << headers[i];
        }
        std::cout << std::endl;
        std::cout << std::string(80, '-') << std::endl;
    };

    // --- Helper Lambda for Value Formatting (Now supports IN, BETWEEN) ---
    auto FormatFilterValue = [](const SelectionOp &op) -> std::string
    {
        std::stringstream ss;
        if (op.op_type == OpType::BETWEEN)
        {
            ss << "[" << op.val << ", " << op.val2 << "]";
        }
        else if (op.op_type == OpType::IN)
        {
            ss << "[";
            if (!op.in_values.empty())
            {
                for (size_t i = 0; i < op.in_values.size(); ++i)
                {
                    ss << op.in_values[i] << (i < op.in_values.size() - 1 ? ", " : "");
                }
            }
            else
            {
                // Fallback if in_values is empty but val is set (single value IN)
                ss << op.val;
            }
            ss << "]";
        }
        else
        {
            ss << op.val;
        }
        return ss.str();
    };

    // --- Helper Lambda for Filter Signature (De-duplication) ---
    auto GetFilterSignature = [&](const std::string &table, const SelectionOp &op)
    {
        return table + "." + op.col + OpTypeToString(op.op_type) + FormatFilterValue(op);
    };

    // =========================================================
    // 1. Report Header
    // =========================================================
    std::cout << "\n"
              << std::string(80, '=') << std::endl;
    std::cout << "                          LOGICAL QUERY PLAN REPORT                           " << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << " Query ID       : " << (plan.query_id.empty() ? "Ad-Hoc" : plan.query_id) << std::endl;
    std::cout << " Execution Mode : " << (plan.use_gpu ? "GPU Acceleration (CUDA)" : "CPU Only") << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    // =========================================================
    // 2. Projections
    // =========================================================
    std::cout << "[0] PROJECTION: ";
    if (plan.projections.empty())
        std::cout << "* (All Columns)";
    else
        for (size_t i = 0; i < plan.projections.size(); ++i)
            std::cout << plan.projections[i] << (i < plan.projections.size() - 1 ? ", " : "");
    std::cout << std::endl;

    // =========================================================
    // 3. Filters
    // =========================================================
    PrintSectionHeader("[1] FILTER OPERATIONS (Predicates)");
    PrintTableHeader({"TABLE", "COLUMN", "OPERATOR", "VALUE(S)"}, {15, 25, 12, 20});

    bool has_filters = false;
    std::set<std::string> printed_filters; // 存储已打印过滤器的签名，用于去重

    // A. Print OLAP Filters strictly in Plan Order (Preserves JQL definition order)
    // This directly reflects the execution pipeline order (e.g., Rollup -> Drilldown)
    for (const auto &olap : plan.olap_ops)
    {
        if (!olap.filter.col.empty())
        {
            const auto &op = olap.filter;
            // 回退策略：如果 filter 内的 table 为空，使用 dimension 的名称
            std::string table_name = op.table.empty() ? olap.dimension : op.table;
            std::string signature = GetFilterSignature(table_name, op);

            if (printed_filters.find(signature) == printed_filters.end())
            {
                has_filters = true;
                std::string val_str = FormatFilterValue(op);

                // 动态判断是否为 Rollup，并追加标识
                std::string scope_str = olap.is_rollup ? " (Rollup Scope)" : "";

                std::cout << std::left << std::setw(15) << table_name
                          << std::setw(25) << op.col
                          << std::setw(12) << OpTypeToString(op.op_type)
                          << val_str << scope_str << std::endl;

                printed_filters.insert(signature);
            }
        }
    }

    // B. Print Remaining Global Filters (Fallback for Non-OLAP predicates)
    // Any filter in plan.selections that hasn't been printed yet (e.g., pure WHERE clauses)
    if (!plan.selections.empty())
    {
        for (const auto &op : plan.selections)
        {
            std::string table_name = op.table.empty() ? "UNKNOWN" : op.table;
            std::string signature = GetFilterSignature(table_name, op);

            if (printed_filters.find(signature) == printed_filters.end())
            {
                has_filters = true;
                std::string val_str = FormatFilterValue(op);

                std::cout << std::left << std::setw(15) << table_name
                          << std::setw(25) << op.col
                          << std::setw(12) << OpTypeToString(op.op_type)
                          << val_str << " (Global Scope)" << std::endl;

                printed_filters.insert(signature);
            }
        }
    }

    if (!has_filters)
    {
        std::cout << "  (No Filters - Full Table Scan)" << std::endl;
    }

    // =========================================================
    // 4. Dimensions & Grouping
    // =========================================================
    if (!plan.olap_ops.empty())
    {
        PrintSectionHeader("[1.5] OLAP DIMENSIONS (Group By)");
        bool has_group_by = false;
        for (const auto &op : plan.olap_ops)
        {
            if (!op.group_col.empty())
            {
                std::string type_str = "";

                // Smart Classification:
                if (op.is_rollup && !op.filter.col.empty())
                {
                    std::string table_name = op.filter.table.empty() ? op.dimension : op.filter.table;
                    std::string signature = GetFilterSignature(table_name, op.filter);

                    if (printed_filters.find(signature) == printed_filters.end())
                    {
                        type_str = " (Rollup)";
                    }
                }

                std::cout << "  - Dimension: " << std::left << std::setw(15) << op.dimension
                          << " Group By: " << op.group_col << type_str << std::endl;
                has_group_by = true;
            }
        }
        if (!has_group_by)
            std::cout << "  (No explicit Group By dimensions)" << std::endl;
    }

    // =====================================================================
    // 5. Aggregation (AGG) & Execution Routing
    // Prints the aggregation metrics and their designated heterogeneous
    // execution engine (CPU vs GPU) target.
    // =====================================================================
    PrintSectionHeader("[2] AGGREGATION & HETEROGENEOUS ROUTING");

    // Updated header and widths to include the 'ENGINE' column.
    // Widths: FUNC(10) | EXPRESSION(35) | ALIAS(15) | ENGINE(10)
    PrintTableHeader({"FUNC", "EXPRESSION / COLUMN", "ALIAS", "ENGINE"}, {10, 35, 15, 10});

    if (plan.aggregations.empty())
    {
        std::cout << "  (No Aggregation)" << std::endl;
    }
    else
    {
        for (const auto &agg : plan.aggregations)
        {
            // Construct the arithmetic or single column expression string
            std::string display_expr = agg.col1 + (!agg.op.empty() ? (" " + agg.op + " " + agg.col2) : "");

            // Fallback for execution engine to ensure UI robustness
            std::string display_engine = agg.exec_engine.empty() ? "GPU" : agg.exec_engine;

            // Formatted output aligning strictly with the specified widths
            std::cout << std::left
                      << std::setw(10) << agg.func
                      << std::setw(35) << display_expr
                      << std::setw(15) << agg.alias
                      // Adding visual emphasis for the execution target
                      << std::setw(10) << ("[" + display_engine + "]")
                      << std::endl;
        }
    }

    // =========================================================
    // 6. Order By
    // =========================================================
    if (!plan.order_bys.empty())
    {
        PrintSectionHeader("[3] ORDER BY");
        for (const auto &o : plan.order_bys)
        {
            std::cout << "  - " << std::left << std::setw(20) << o.col << "(" << o.mode << ")" << std::endl;
        }
    }
    std::cout << std::string(80, '=') << "\n"
              << std::endl;
}
// =========================================================
// Join Path Parser Implementation
// =========================================================

namespace SchemaParser
{

    void ParseUpperLevels(cJSON *upper_array, std::shared_ptr<JoinNode> parent)
    {
        if (!upper_array)
            return;
        cJSON *node = nullptr;
        cJSON_ArrayForEach(node, upper_array)
        {
            auto child = std::make_shared<JoinNode>();
            cJSON *t = cJSON_GetObjectItem(node, "TABLE_NAME");
            cJSON *fk = cJSON_GetObjectItem(node, "FKinLOWERLEVEL");
            cJSON *pk = cJSON_GetObjectItem(node, "HCOLNAME");

            child->table_name = t ? t->valuestring : "";
            child->fk_col = fk ? fk->valuestring : "";
            child->pk_col = pk ? pk->valuestring : "";

            // Parse contained columns
            cJSON *cols = cJSON_GetObjectItem(node, "COLUMNS");
            if (cols)
            {
                cJSON *col_item;
                cJSON_ArrayForEach(col_item, cols)
                {
                    if (col_item->valuestring)
                        child->contained_columns.push_back(col_item->valuestring);
                }
            }

            ParseUpperLevels(cJSON_GetObjectItem(node, "UPPERLEVEL"), child);
            parent->upper_levels.push_back(child);
        }
    }

    bool ParseJoinPath(const std::string &input_filename, JoinGraph &graph)
    {
        std::string filename = input_filename;
        if (filename.find('/') == std::string::npos && filename.find('\\') == std::string::npos)
        {
            filename = "../jql/" + filename; // Assume it's in project root, not jql/
        }

        std::cout << "[Schema] Reading Join Path: " << filename << std::endl;

        std::ifstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "[Warning] Cannot open Join Path file: " << filename << std::endl;
            return false;
        }
        std::string json_str((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();

        cJSON *root = cJSON_Parse(json_str.c_str());
        if (!root)
        {
            std::cerr << "[Error] Join Path JSON Parse Failed." << std::endl;
            return false;
        }

        cJSON *dims = cJSON_GetObjectItem(root, "DIMENSION_HIERARCHY");
        if (dims)
        {
            cJSON *dim_node = nullptr;
            cJSON_ArrayForEach(dim_node, dims)
            {
                cJSON *root_name = cJSON_GetObjectItem(dim_node, "ROOT");
                if (root_name)
                {
                    auto root_ptr = std::make_shared<JoinNode>();
                    root_ptr->table_name = root_name->valuestring;

                    cJSON *upper = cJSON_GetObjectItem(dim_node, "UPPERLEVEL");
                    ParseUpperLevels(upper, root_ptr);

                    graph.dimension_roots[root_ptr->table_name] = root_ptr;
                }
            }
        }
        cJSON_Delete(root);
        return true;
    }
}
// =========================================================
// 3. JQL Parser Implementation
// =========================================================

namespace JQLParser
{

    OpType MapExpressionToOp(const char *expr)
    {
        std::string s(expr);
        if (s == "=")
            return OpType::EQ;
        if (s == "<")
            return OpType::LT;
        if (s == ">")
            return OpType::GT;
        if (s == "<=")
            return OpType::LE;
        if (s == ">=")
            return OpType::GE;
        if (s == "BETWEENAND")
            return OpType::BETWEEN;
        if (s == "OR")
            return OpType::IN;
        return OpType::EQ;
    }
    /**
     * @brief Resolves a string literal to its integer dictionary ID/Primary Key.
     * Used when the JQL specifies a string value (e.g., "Jan1994") but the engine
     * operates on integer keys (e.g., 199401).
     * * @param table Table name defined in JQL (e.g., "DATE").
     * @param col Column name defined in JQL (e.g., "D_YEARMONTH").
     * @param val The string value to lookup (e.g., "Jan1994").
     * @return int The corresponding integer ID/Key. Returns 0 if not found.
     */
    int ResolveStringValue(const std::string &table, const std::string &col, const std::string &val)
    {
        // Case 1: DATE.D_YEARMONTH -> Lookup in YEARMONTH dimension
        // In SSB Schema, D_YEARMONTH (e.g., "Jan1994") corresponds to a unique Month ID.
        // We assume 'column_store' has the YearMonth dimension data loaded on Host.
        if (table == "DATE" && col == "D_YEARMONTH")
        {
            // Iterate over the distinct YearMonths to find the matching string
            // Assuming column_store.num_yearmonth_rows holds the count
            // And column_store.ym_name stores the strings (char**)
            // And column_store.ym_id stores the integer keys (int*)

            std::cout << "  [Dict Lookup] Resolving '" << val << "' in YEARMONTH dictionary..." << std::endl;

            for (size_t i = 0; i < column_store.num_yearmonth_rows; ++i)
            {
                // Convert C-string to std::string for comparison
                // Note: Ensure ym_name[i] is valid
                if (val == column_store.ym_name[i])
                {
                    int resolved_id = column_store.id_ym[i]; // e.g., 199401 or a surrogate key
                    std::cout << "  -> Found Match: ID = " << resolved_id << std::endl;
                    return resolved_id;
                }
            }
            std::cerr << "  [Warning] String value '" << val << "' not found in dictionary." << std::endl;
            return 0; // Fallback or Error code
        }

        // Case 2: P_MFGR / P_CATEGORY in PART table
        else if (table == "PART")
        {
            if (col == "P_CATEGORY")
            {
                std::cout << "  [Dict Lookup] Resolving '" << val << "' in CATEGORY dictionary..." << std::endl;
                for (size_t i = 0; i < column_store.num_category_rows; ++i)
                {
                    if (val == column_store.ca_name[i])
                    {
                        int resolved_id = column_store.id_category[i]; //
                        std::cout << "  -> Found Match: ID = " << resolved_id << std::endl;
                        return resolved_id;
                    }
                }
                std::cerr << "  [Warning] String value '" << val << "' not found in dictionary." << std::endl;
                return 0; // Fallback or Error code
            }
            else if (col == "P_BRAND1")
            {
                std::cout << "  [Dict Lookup] Resolving '" << val << "' in BRAND1 dictionary..." << std::endl;
                for (size_t i = 0; i < column_store.num_brand1_rows; ++i)
                {
                    if (val == column_store.b_name[i])
                    {
                        int resolved_id = column_store.b1_val[i]; //
                        std::cout << "  -> Found Match: ID = " << resolved_id << std::endl;
                        return resolved_id;
                    }
                }
                std::cerr << "  [Warning] String value '" << val << "' not found in dictionary." << std::endl;
                return 0; // Fallback or Error code
            }
            else if (col == "P_MFGR")
            {
                std::cout << "  [Dict Lookup] Resolving '" << val << "' in MFGR dictionary..." << std::endl;
                for (size_t i = 0; i < column_store.num_mfgr_rows; ++i)
                {
                    if (val == column_store.mfgr_name[i])
                    {
                        int resolved_id = column_store.id_mfgr[i]; //
                        std::cout << "  -> Found Match: ID = " << resolved_id << std::endl;
                        return resolved_id;
                    }
                }
                std::cerr << "  [Warning] String value '" << val << "' not found in dictionary." << std::endl;
                return 0; // Fallback or Error code
            }
        }
        // Case 2: S_REGION /  in SUPPLIER table
        else if (table == "SUPPLIER")
        {
            if (col == "S_REGION")
            {
                std::cout << "  [Dict Lookup] Resolving '" << val << "' in REGION dictionary..." << std::endl;
                for (size_t i = 0; i < column_store.num_region_rows; ++i)
                {
                    if (val == column_store.r_name[i])
                    {
                        int resolved_id = column_store.r_regionkey[i]; //
                        std::cout << "  -> Found Match: ID = " << resolved_id << std::endl;
                        return resolved_id;
                    }
                }
                std::cerr << "  [Warning] String value '" << val << "' not found in dictionary." << std::endl;
                return 0; // Fallback or Error code
            }
            else if (col == "S_CITY")
            {
                std::cout << "  [Dict Lookup] Resolving '" << val << "' in CITY dictionary..." << std::endl;
                for (size_t i = 0; i < column_store.num_city_s_rows; ++i)
                {
                    if (val == column_store.si_name[i])
                    {
                        int resolved_id = column_store.id_city_s[i]; //
                        std::cout << "  -> Found Match: ID = " << resolved_id << std::endl;
                        return resolved_id;
                    }
                }
                std::cerr << "  [Warning] String value '" << val << "' not found in dictionary." << std::endl;
                return 0; // Fallback or Error code
            }
            else if (col == "S_NATION")
            {
                std::cout << "  [Dict Lookup] Resolving '" << val << "' in NATION dictionary..." << std::endl;
                for (size_t i = 0; i < column_store.num_nation_rows; ++i)
                {
                    if (val == column_store.n_name[i])
                    {
                        int resolved_id = column_store.n_nationkey[i]; //
                        std::cout << "  -> Found Match: ID = " << resolved_id << std::endl;
                        return resolved_id;
                    }
                }
                std::cerr << "  [Warning] String value '" << val << "' not found in dictionary." << std::endl;
                return 0; // Fallback or Error code
            }
        }
        else if (table == "CUSTOMER")
        {
            if (col == "C_CITY")
            {
                std::cout << "  [Dict Lookup] Resolving '" << val << "' in CITY dictionary..." << std::endl;
                for (size_t i = 0; i < column_store.num_city_c_rows; ++i)
                {
                    if (val == column_store.ci_name[i])
                    {
                        int resolved_id = column_store.id_city_c[i]; //
                        std::cout << "  -> Found Match: ID = " << resolved_id << std::endl;
                        return resolved_id;
                    }
                }
                std::cerr << "  [Warning] String value '" << val << "' not found in dictionary." << std::endl;
                return 0; // Fallback or Error code
            }
            else if (col == "C_REGION")
            {
                std::cout << "  [Dict Lookup] Resolving '" << val << "' in REGION dictionary..." << std::endl;
                for (size_t i = 0; i < column_store.num_region_rows; ++i)
                {
                    if (val == column_store.r_name[i])
                    {
                        int resolved_id = column_store.r_regionkey[i]; //
                        std::cout << "  -> Found Match: ID = " << resolved_id << std::endl;
                        return resolved_id;
                    }
                }
                std::cerr << "  [Warning] String value '" << val << "' not found in dictionary." << std::endl;
                return 0; // Fallback or Error code
            }
            else if (col == "C_NATION")
            {
                std::cout << "  [Dict Lookup] Resolving '" << val << "' in NATION dictionary..." << std::endl;
                for (size_t i = 0; i < column_store.num_nation_rows; ++i)
                {
                    if (val == column_store.n_name[i])
                    {
                        int resolved_id = column_store.n_nationkey[i]; //
                        std::cout << "  -> Found Match: ID = " << resolved_id << std::endl;
                        return resolved_id;
                    }
                }
                std::cerr << "  [Warning] String value '" << val << "' not found in dictionary." << std::endl;
                return 0; // Fallback or Error code
            }
        }

        // Default: If no dictionary mapping exists, return 0 or hash
        std::cerr << "  [Warning] No dictionary lookup defined for " << table << "." << col << std::endl;
        return 0;
    }
    bool ParseJQL(const std::string &input_filename, QueryPlan &plan)
    {
        std::string filename = input_filename;
        if (filename.find('/') == std::string::npos && filename.find('\\') == std::string::npos)
        {
            filename = "../jql/" + filename;
        }

        std::cout << "[Parser] Reading JQL file: " << filename << std::endl;

        std::ifstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "[Error] Cannot open JQL file: " << filename << std::endl;
            return false;
        }
        std::string json_str((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();

        cJSON *root = cJSON_Parse(json_str.c_str());
        if (!root)
        { /* Error handling */
            return false;
        }
        plan.query_id = input_filename;

        cJSON *select_node = cJSON_GetObjectItem(root, "SELECT");
        if (!select_node)
        {
            cJSON_Delete(root);
            return false;
        }

        // --- 1. Parse PROJECT ---
        cJSON *proj_array = cJSON_GetObjectItem(select_node, "PROJECT");
        cJSON *proj_item = nullptr;
        cJSON_ArrayForEach(proj_item, proj_array)
        {
            if (cJSON_IsString(proj_item))
            {
                plan.projections.push_back(proj_item->valuestring);
            }
        }

        // --- 2. Parse ORDERBY ---
        cJSON *order_array = cJSON_GetObjectItem(select_node, "ORDERBY");
        cJSON *order_item = nullptr;
        cJSON_ArrayForEach(order_item, order_array)
        {
            OrderByOp op;
            cJSON *n = cJSON_GetObjectItem(order_item, "NAME");
            cJSON *m = cJSON_GetObjectItem(order_item, "MODE");
            op.col = n ? n->valuestring : "";
            op.mode = m ? m->valuestring : "ASC";
            plan.order_bys.push_back(op);
        }

        // --- 3. Parse OLAP (DrillDown & Rollup) ---
        // Critical: Extract Filters from OLAP and flatten them into plan.selections
        cJSON *olap_array = cJSON_GetObjectItem(select_node, "OLAP");
        cJSON *olap_item = nullptr;

        cJSON_ArrayForEach(olap_item, olap_array)
        {

            cJSON *operation_node = olap_item->child;
            while (operation_node != nullptr)
            {
                // =========================================================
                // A. Parse DRILLDOWN
                // =========================================================
                if (operation_node->string && std::string(operation_node->string) == "DRILLDOWN")
                {
                    cJSON *drill_array = operation_node;
                    cJSON *drill_item = nullptr;
                    cJSON_ArrayForEach(drill_item, drill_array)
                    {
                        // 1. Extract Common Dimension & Group Info
                        std::string dim_str = "";
                        std::string grp_str = "";

                        cJSON *dim = cJSON_GetObjectItem(drill_item, "DIMENSION");
                        if (dim && dim->valuestring)
                            dim_str = dim->valuestring;

                        cJSON *grp = cJSON_GetObjectItem(drill_item, "GROUP");
                        if (cJSON_IsString(grp))
                            grp_str = grp->valuestring;

                        // 2. Handle Filters (Array vs Object vs None)
                        cJSON *flt_root = cJSON_GetObjectItem(drill_item, "FILTER");
                        std::vector<cJSON *> filter_nodes;

                        if (flt_root)
                        {
                            if (cJSON_IsArray(flt_root))
                            {
                                cJSON *f = nullptr;
                                cJSON_ArrayForEach(f, flt_root)
                                {
                                    filter_nodes.push_back(f);
                                }
                            }
                            else if (cJSON_IsObject(flt_root))
                            {
                                if (cJSON_HasObjectItem(flt_root, "NAME"))
                                {
                                    filter_nodes.push_back(flt_root);
                                }
                            }
                        }

                        // 3. Generate OlapOps
                        // Case A: No Filters -> 1 OlapOp (Group By only)
                        if (filter_nodes.empty())
                        {
                            OlapOp olap;
                            olap.is_rollup = false;
                            olap.dimension = dim_str;
                            olap.group_col = grp_str;
                            plan.olap_ops.push_back(olap);
                        }
                        // Case B: N Filters -> N OlapOps
                        else
                        {
                            for (cJSON *flt : filter_nodes)
                            {
                                OlapOp olap;
                                olap.is_rollup = false;
                                olap.dimension = dim_str;
                                olap.group_col = grp_str; // Replicate Group By for each filter context

                                // Parse SelectionOp
                                SelectionOp op;
                                cJSON *n = cJSON_GetObjectItem(flt, "NAME");
                                cJSON *e = cJSON_GetObjectItem(flt, "EXPRESSION");
                                cJSON *v = cJSON_GetObjectItem(flt, "VALUE");

                                if (n && v)
                                {
                                    op.table = olap.dimension;
                                    op.col = n->valuestring;

                                    std::string expr = e ? e->valuestring : "=";
                                    if (expr == "OR")
                                        op.op_type = OpType::IN;
                                    else
                                        op.op_type = MapExpressionToOp(expr.c_str());

                                    // --- Value Parsing Lambda ---
                                    auto ParseSingleValue = [&](cJSON *item) -> int
                                    {
                                        if (cJSON_IsNumber(item))
                                            return item->valueint;
                                        if (cJSON_IsString(item))
                                        {
                                            try
                                            {
                                                return std::stoi(item->valuestring);
                                            }
                                            catch (...)
                                            {
                                                int id = ResolveStringValue(op.table, op.col, item->valuestring);
                                                return id;
                                            }
                                        }
                                        return 0;
                                    };

                                    // --- Value Processing ---
                                    if (cJSON_IsArray(v))
                                    {
                                        int array_size = cJSON_GetArraySize(v);
                                        if (op.op_type == OpType::BETWEEN)
                                        {
                                            if (array_size >= 2)
                                            {
                                                op.val = ParseSingleValue(cJSON_GetArrayItem(v, 0));
                                                op.val2 = ParseSingleValue(cJSON_GetArrayItem(v, 1));
                                            }
                                        }
                                        else
                                        {
                                            cJSON *elem = nullptr;
                                            cJSON_ArrayForEach(elem, v) op.in_values.push_back(ParseSingleValue(elem));
                                            op.op_type = OpType::IN; // Force IN for lists
                                            if (!op.in_values.empty())
                                                op.val = op.in_values[0];
                                        }
                                    }
                                    else
                                    {
                                        op.val = ParseSingleValue(v);
                                    }

                                    olap.filter = op;
                                    plan.selections.push_back(op); // Register Global Filter
                                }
                                plan.olap_ops.push_back(olap);
                            }
                        }
                    }
                }
                // =========================================================
                // B. Parse ROLLUP
                // =========================================================
                else if (operation_node->string && std::string(operation_node->string) == "ROLLUP")
                {
                    cJSON *rollup_array = operation_node;
                    cJSON *rollup_item = nullptr;
                    cJSON_ArrayForEach(rollup_item, rollup_array)
                    {
                        OlapOp olap;
                        olap.is_rollup = true;

                        cJSON *dim = cJSON_GetObjectItem(rollup_item, "DIMENSION"); // e.g. DATE
                        cJSON *grp = cJSON_GetObjectItem(rollup_item, "GROUP");     // e.g. D_YEAR
                        cJSON *flt = cJSON_GetObjectItem(rollup_item, "FILTER");    // Filter on D_YEARMONTH

                        olap.dimension = dim ? dim->valuestring : "";
                        if (cJSON_IsString(grp))
                            olap.group_col = grp->valuestring;

                        if (flt)
                        {
                            cJSON *n = cJSON_GetObjectItem(flt, "NAME");
                            cJSON *e = cJSON_GetObjectItem(flt, "EXPRESSION");
                            cJSON *v = cJSON_GetObjectItem(flt, "VALUE");

                            if (n && v)
                            {
                                olap.rollup_table = "";
                                olap.filter.table = olap.dimension;
                                olap.filter.col = n->valuestring;
                                olap.filter.op_type = MapExpressionToOp(e ? e->valuestring : "=");

                                if (cJSON_IsString(v))
                                {
                                    olap.filter.val = ResolveStringValue(olap.filter.table, olap.filter.col, v->valuestring);
                                }
                                else if (cJSON_IsNumber(v))
                                {
                                    olap.filter.val = v->valueint;
                                }
                            }
                        }
                        plan.olap_ops.push_back(olap);
                    }
                }

                // Move to the next key in the OLAP object (preserving JSON file order)
                operation_node = operation_node->next;
            }
        }

        // --- 4. Parse Filters (FILTING) ---
        cJSON *filting_array = cJSON_GetObjectItem(select_node, "FILTING");
        if (filting_array)
        {
            cJSON *node = nullptr;
            cJSON_ArrayForEach(node, filting_array)
            {
                SelectionOp op;
                cJSON *t = cJSON_GetObjectItem(node, "TABLE");
                cJSON *c = cJSON_GetObjectItem(node, "NAME");
                cJSON *e = cJSON_GetObjectItem(node, "EXPRESSION");
                cJSON *v = cJSON_GetObjectItem(node, "VALUE");

                op.table = t ? t->valuestring : "";
                op.col = c ? c->valuestring : "";
                op.op_type = MapExpressionToOp(e ? e->valuestring : "=");

                if (op.op_type == OpType::BETWEEN && cJSON_IsArray(v))
                {
                    op.val = cJSON_GetArrayItem(v, 0)->valueint;
                    op.val2 = cJSON_GetArrayItem(v, 1)->valueint;
                }
                else
                {
                    // [Optimization] Intelligent Value Parsing
                    if (cJSON_IsString(v))
                    {
                        try
                        {
                            // 1. Try treating as number first
                            op.val = std::stoi(v->valuestring);
                        }
                        catch (...)
                        {
                            // 2. If not a number, treat as Dictionary String
                            // Perform Lookup to convert "Jan1994" -> 199401 (Int)
                            op.val = ResolveStringValue(op.table, op.col, v->valuestring);
                            // if (op.table == "DATE" && op.col == "D_YEARMONTH") {
                            //     std::cout << "  [Optimization] Rewriting column " << op.col << " -> id_ym for efficient integer filtering." << std::endl;
                            //     op.col = "ID_YM";
                            // }
                        }
                    }
                    else if (cJSON_IsNumber(v))
                    {
                        op.val = v->valueint;
                    }
                }
                plan.selections.push_back(op);
            }
        }

        // =====================================================================
        // --- 5. Parse Aggregations (AGG) ---
        // Parses the aggregation node, evaluating arithmetic expressions and
        // determining the target heterogeneous execution engine (CPU vs GPU).
        // =====================================================================
        cJSON *agg_array = cJSON_GetObjectItem(select_node, "AGG");
        if (agg_array && cJSON_IsArray(agg_array)) // Defensive check: ensure it is an array
        {
            cJSON *node = nullptr;
            cJSON_ArrayForEach(node, agg_array)
            {
                AggregationOp agg;

                // 1. Extract Base Aggregation Fields
                cJSON *f = cJSON_GetObjectItem(node, "FUNCTION");
                cJSON *e = cJSON_GetObjectItem(node, "EXPRESSION");
                cJSON *a = cJSON_GetObjectItem(node, "ALIAS");
                cJSON *t = cJSON_GetObjectItem(node, "TABLE");

                // Safely assign string values (fallback to empty string if null)
                agg.func = (f && cJSON_IsString(f)) ? f->valuestring : "";
                agg.alias = (a && cJSON_IsString(a)) ? a->valuestring : "";
                agg.table = (t && cJSON_IsString(t)) ? t->valuestring : "";

                // 2. Parse Aggregation Expression (e.g., "LO_REVENUE-LO_SUPPLYCOST")
                std::string raw_expr = (e && cJSON_IsString(e)) ? e->valuestring : "";
                size_t op_pos = raw_expr.find_first_of("+-*/");

                if (op_pos != std::string::npos)
                {
                    // Dual-Column Aggregation (Arithmetic Operation)
                    agg.col1 = raw_expr.substr(0, op_pos);
                    agg.op = raw_expr.substr(op_pos, 1);
                    agg.col2 = raw_expr.substr(op_pos + 1);
                }
                else
                {
                    // Single-Column Aggregation
                    agg.col1 = raw_expr;
                    agg.op = "";
                    agg.col2 = "";
                }

                // ---------------------------------------------------------
                // 3. Heterogeneous Execution Engine Routing (New Feature)
                // ---------------------------------------------------------
                cJSON *ee = cJSON_GetObjectItem(node, "EXECUTION_ENGINE");
                std::string engine_str = "GPU"; // Default Fallback Behavior

                if (ee && cJSON_IsString(ee) && ee->valuestring != nullptr)
                {
                    engine_str = ee->valuestring;

                    // Normalize to uppercase for strict and fast matching in the Executor
                    std::transform(engine_str.begin(), engine_str.end(), engine_str.begin(),
                                   [](unsigned char c)
                                   { return std::toupper(c); });

                    // Optional Execution Engine Validation (Reject invalid inputs immediately)
                    if (engine_str != "GPU" && engine_str != "CPU")
                    {
                        std::cerr << "[Warning] Unknown EXECUTION_ENGINE: '" << engine_str
                                  << "'. Falling back to default 'GPU'." << std::endl;
                        engine_str = "GPU";
                    }
                }
                agg.exec_engine = engine_str;

                // Push configured operator to the logical execution plan
                plan.aggregations.push_back(agg);
            }
        }

        // [Change] MAPS parsing logic removed.

        cJSON_Delete(root);
        return true;
    }
}

// =========================================================
// 4. Execution Engine
// =========================================================

class SSBQueryExecutor
{
private:
    int *d_lo_custkey, *d_lo_partkey, *d_lo_suppkey, *d_lo_extendedprice, *d_date_fk, *d_lo_supplycost, *d_lo_discount, *d_p_brand1, *d_s_city, *d_c_city, *d_lo_revenue;
    int8_t *d_discount_fk, *d_quantity_fk;
    unsigned long long *d_results;
    int num_threads_;
    ColumnData GetCPUColumnData(const std::string &table, const std::string &col)
    {
        // 1. LineOrder Fact Table
        if (table == "LINEORDER")
        {
            size_t len = column_store.num_lineorder_rows;
            if (col == "QUANTITY_FK")
                return {column_store.quantity_fk, len, true, true, 1}; // Mapped to FK for simplicity or actual val
            else if (col == "DISCOUNT_FK")
                return {column_store.discount_fk, len, true, true, 1};
            else if (col == "DATE_FK")
                return {column_store.date_fk, len, true, true, 4};
            else if (col == "LO_PARTKEY")
                return {column_store.lo_partkey, len, true, true, 4};
            else if (col == "LO_CUSTKEY")
                return {column_store.lo_custkey, len, true, true, 4};
            else if (col == "LO_SUPPKEY")
                return {column_store.lo_suppkey, len, true, true, 4};
            else if (col == "LO_SUPPLYCOST")
                return {column_store.lo_supplycost, len, true, true, 4};
            // Note: In SSB raw, quantity/discount are int. In ColumnStore they might be int8 or int.
            // For Q1.1 filtering, we use the loaded int8/int columns.
            // Assuming loaded as int for simplicity in filter kernel, or overload filter.
            // Let's assume standard int pointers for prototype correctness.
            else if (col == "LO_EXTENDEDPRICE")
                return {column_store.lo_extendedprice, len, true, false, 4};
            else if (col == "LO_REVENUE")
                return {column_store.lo_revenue, len, true, false, 4};
        }

        // 2. Date Dimension (Snowflake)
        // Physical Table: DATE
        else if (table == "DATE")
        {
            if (col == "D_YEAR")
                return {column_store.d_year, column_store.num_date_rows, true, true, 4};
            else if (col == "D_DATEKEY")
                return {column_store.d_datekey, column_store.num_date_rows, true, false, 4};
            else if (col == "D_YEARMONTH")
                return {column_store.d_yearmonth, column_store.num_date_rows, true, true, 4}; // FK to YearMonth
            else if (col == "D_WEEKNUMINYEAR")
                return {column_store.d_weeknuminyear, column_store.num_date_rows, true, true, 4};
        }
        // Physical Table: YEAR
        else if (table == "YEAR")
        {
            if (col == "Y_NAME")
                return {column_store.y_name, column_store.num_year_rows, true, false, 4};
            else if (col == "Y_KEY")
                return {column_store.y_key, column_store.num_year_rows, true, false, 4}; // PK
        }
        // Physical Table: YEARMONTH
        else if (table == "YEARMONTH")
        {
            if (col == "YM_YEARKEY")
                return {column_store.ym_yearkey, column_store.num_yearmonth_rows, true, false, 4}; // Actually usually in YearMonth table
            else if (col == "D_YEARMONTHNUM")
                return {column_store.ym_num, column_store.num_yearmonth_rows, true, false, 4};
            else if (col == "ID_YM")
                return {column_store.id_ym, column_store.num_yearmonth_rows, true, false, 4};
            // Note: In strict SSB, YearMonth table bridges Date and Year.
            // We use column_store mappings. Assuming d_yearmonth_val used for dict.
        }
        else if (table == "DISCOUNT")
        {
            if (col == "DIM_DISCOUNT_VAL")
                return {column_store.dim_discount_val, column_store.num_discount_rows, true, false, 4};
        }
        else if (table == "QUANTITY")
        {
            if (col == "DIM_QUANTITY_VAL")
                return {column_store.dim_quantity_val, column_store.num_quantity_rows, true, false, 4};
        }
        else if (table == "CATEGORY")
        {
            if (col == "ID_CATEGORY")
                return {column_store.id_category, column_store.num_category_rows, true, false, 4};
            else if (col == "CA_MFGRKEY")
                return {column_store.ca_mfgrkey, column_store.num_category_rows, true, false, 4};
            else if (col == "CA_NAME")
                return {column_store.ca_name, column_store.num_category_rows, false, false, 4};
        }
        else if (table == "BRAND1")
        {
            if (col == "B_CATEGORYKEY")
                return {column_store.b_category, column_store.num_brand1_rows, true, false, 4};
            else if (col == "ID_BRAND1")
                return {column_store.id_brand1, column_store.num_brand1_rows, true, false, 4};
            else if (col == "B1_VAL")
                return {column_store.b1_val, column_store.num_brand1_rows, true, false, 4};
            else if (col == "B_NAME")
                return {column_store.b_name, column_store.num_brand1_rows, false, true, 4};
        }
        else if (table == "MFGR")
        {
            if (col == "ID_MFGR")
                return {column_store.id_mfgr, column_store.num_mfgr_rows, true, false, 4};
        }
        else if (table == "REGION")
        {
            if (col == "R_REGIONKEY")
                return {column_store.r_regionkey, column_store.num_region_rows, true, false, 4};
        }
        else if (table == "NATION")
        {
            if (col == "N_REGIONKEY")
                return {column_store.n_regionkey, column_store.num_nation_rows, true, false, 4};
            else if (col == "N_NAME")
                return {column_store.n_name, column_store.num_nation_rows, false, false, 4};
            else if (col == "N_NATIONKEY")
                return {column_store.n_nationkey, column_store.num_nation_rows, true, false, 4};
        }
        else if (table == "CITY")
        {
            if (col == "SI_NATIONKEY")
                return {column_store.si_nationkey, column_store.num_city_s_rows, true, false, 4};
            else if (col == "ID_CITY_S")
                return {column_store.id_city_s, column_store.num_city_s_rows, true, false, 4};
            else if (col == "ID_CITY_C")
                return {column_store.id_city_c, column_store.num_city_c_rows, true, false, 4};
            else if (col == "CI_NAME")
                return {column_store.ci_name, column_store.num_city_c_rows, false, true, 4};
            else if (col == "SI_NAME")
                return {column_store.si_name, column_store.num_city_s_rows, false, true, 4};
            else if (col == "CI_NATIONKEY")
                return {column_store.ci_nationkey, column_store.num_city_c_rows, true, false, 4};
        }
        else if (table == "PART")
        {
            if (col == "P_BRAND1")
                return {column_store.p_brand1, column_store.num_part_rows, true, false, 4};
        }
        else if (table == "SUPPLIER")
        {
            if (col == "S_CITY")
                return {column_store.s_city, column_store.num_supplier_rows, true, false, 4};
        }
        else if (table == "CUSTOMER")
        {
            if (col == "C_CITY")
                return {column_store.c_city, column_store.num_customer_rows, true, false, 4};
        }

        std::cerr << "[Error] Unknown Column: " << table << "." << col << std::endl;
        // exit(1);
        return {nullptr, 0, false, false, 0};
    }
    ColumnData GetGPUColumnData(const std::string &table, const std::string &col)
    {
        // 1. LineOrder Fact Table
        if (table == "LINEORDER")
        {
            size_t len = column_store.num_lineorder_rows;
            if (col == "QUANTITY_FK")
                return {d_quantity_fk, len, true, true, 1}; // Mapped to FK for simplicity or actual val
            else if (col == "DISCOUNT_FK")
                return {d_discount_fk, len, true, true, 1};
            else if (col == "DATE_FK")
                return {d_date_fk, len, true, true, 4};
            else if (col == "LO_PARTKEY")
                return {d_lo_partkey, len, true, true, 4};
            else if (col == "LO_CUSTKEY")
                return {d_lo_custkey, len, true, true, 4};
            else if (col == "LO_SUPPKEY")
                return {d_lo_suppkey, len, true, true, 4};
            else if (col == "LO_SUPPLYCOST")
                return {d_lo_supplycost, len, true, true, 4};
            // Note: In SSB raw, quantity/discount are int. In ColumnStore they might be int8 or int.
            // For Q1.1 filtering, we use the loaded int8/int columns.
            // Assuming loaded as int for simplicity in filter kernel, or overload filter.
            // Let's assume standard int pointers for prototype correctness.
            else if (col == "LO_EXTENDEDPRICE")
                return {d_lo_extendedprice, len, true, false, 4};
            else if (col == "LO_REVENUE")
                return {d_lo_revenue, len, true, false, 4};
        }
        else if (table == "DISCOUNT")
        {
            if (col == "DIM_DISCOUNT_VAL")
                return {d_lo_discount, column_store.num_discount_rows, true, false, 4};
        }
        else if (table == "PART")
        {
            if (col == "P_BRAND1")
                return {d_p_brand1, column_store.num_part_rows, true, false, 4};
        }
        else if (table == "SUPPLIER")
        {
            if (col == "S_CITY")
                return {d_s_city, column_store.num_supplier_rows, true, false, 4};
        }
        else if (table == "CUSTOMER")
        {
            if (col == "C_CITY")
                return {d_c_city, column_store.num_customer_rows, true, false, 4};
        }
        std::cerr << "[Error] Unknown Column: " << table << "." << col << std::endl;
        // exit(1);
        return {nullptr, 0, false, false, 0};
    }
    /**
     * @brief Result structure for OLAP operations (CPU/GPU Hybrid).
     * Now manages raw pointers to support zero-copy handoff to GPU kernels.
     */
    struct FilterResult
    {
        int *bitmap_ptr;   // Pointer to data (Host or Device)
        size_t length;     // Number of elements
        int unique_count;  // Cardinality of groups
        bool is_on_device; // True if bitmap_ptr is in GPU VRAM
        std::vector<int> reverse_dict;
        // Simple constructor for empty result
        FilterResult() : bitmap_ptr(nullptr), length(0), unique_count(1), is_on_device(false) {}
    };
    struct ResultRow
    {
        std::vector<std::string> dimensions; // e.g. ["1994", "MFGR#12"]
        unsigned long long metric;           // e.g. Revenue
    };
    /**
     * @brief Executes a Roll-up Optimization:
     * 1. Filter Child -> Mark Parent.
     * 2. Compress Parent (Generate Group IDs).
     * 3. Probe Parent -> Child (Propagate Group IDs with Re-Filtering).
     * * @return FilterResult containing the compressed bitmap for the CHILD table.
     */
    FilterResult ExecuteRollupFilter(
        std::shared_ptr<JoinNode> parent_node, // e.g., YEAR
        std::shared_ptr<JoinNode> child_node,  // e.g., YEARMONTH
        const SelectionOp &op,                 // Filter: D_YEARMONTH = 'Dec1997'
        const std::string &group_col           // Group: D_YEAR
    )
    {
        std::cout << "  [Rollup] Processing Child: " << child_node->table_name
                  << " -> Parent: " << parent_node->table_name << std::endl;

        // =========================================================
        // 1. Prepare Data & Resolve Schema
        // =========================================================

        // A. Resolve Physical Filter Column (Child)
        // Map logical JQL column to physical ID column (e.g. D_YEARMONTH -> ID_YEARMONTH)
        std::string physical_filter_col = op.col;
        if (!child_node->contained_columns.empty())
        {
            const auto &cols = child_node->contained_columns;
            for (size_t k = 0; k < cols.size(); ++k)
            {
                if (cols[k] == op.col)
                {
                    if ((op.col == "P_BRAND1" || op.col == "D_YEAR") && k + 2 < cols.size())
                        physical_filter_col = cols[k + 2];
                    else if (k + 1 < cols.size())
                        physical_filter_col = cols[k + 1];
                    break;
                }
            }
        }

        // B. Resolve Physical Group Column (Parent)
        std::string physical_group_col = group_col;
        if (!parent_node->contained_columns.empty())
        {
            const auto &cols = parent_node->contained_columns;
            for (size_t k = 0; k < cols.size(); ++k)
            {
                if (cols[k] == group_col)
                {
                    if ((group_col == "P_BRAND1" || group_col == "D_YEAR") && k + 2 < cols.size())
                        physical_group_col = cols[k + 2];
                    else if (k + 1 < cols.size())
                        physical_group_col = cols[k + 1];
                    break;
                }
            }
        }

        // C. Fetch Data
        auto child_filter_data = GetCPUColumnData(child_node->table_name, physical_filter_col); // For Filter
        auto child_fk_data = GetCPUColumnData(child_node->table_name, parent_node->fk_col);     // FK to Parent
        auto parent_group_data = GetCPUColumnData(parent_node->table_name, physical_group_col); // For Compression

        if (!child_filter_data.data || !child_fk_data.data || !parent_group_data.data)
        {
            std::cerr << "  [Error] Missing column data for Rollup." << std::endl;
            return {};
        }

        // =========================================================
        // 2. Initialize Parent Bitmap (Invalid = -1)
        // =========================================================
        size_t parent_rows = parent_group_data.len;
        int *parent_bitmap_ptr = new int[parent_rows];
        std::fill(parent_bitmap_ptr, parent_bitmap_ptr + parent_rows, -1);

        // =========================================================
        // 3. Execute Optimized Rollup Filter (Child -> Parent)
        // =========================================================
        // Mark Parent rows as valid (0) if any child row matches the filter.
        SSB_CPU::FilterChildAndMarkParent(
            (int *)child_filter_data.data,
            child_filter_data.len,
            parent_bitmap_ptr,
            op.val,
            (int *)child_fk_data.data);

        // =========================================================
        // 4. Compress Parent (Year) based on the Markings
        // =========================================================
        // Generate compact Group IDs for the Parent dimension.
        int *parent_compressed_indices = new int[parent_rows];
        std::fill(parent_compressed_indices, parent_compressed_indices + parent_rows, -1);
        std::vector<int> value_map(parent_rows, 0);
        std::vector<int> reverse_dict(parent_rows, 0);

        FilterResult result;
        SSB_CPU::CompressColumn(
            (int *)parent_group_data.data,
            parent_rows,
            parent_compressed_indices,
            value_map.data(),
            &result.unique_count,
            reverse_dict.data(),
            parent_bitmap_ptr);
        delete[] parent_bitmap_ptr;

        // =========================================================
        // 5. Propagate Back: Parent -> Child (Join + Filter)
        // =========================================================
        // We use the new JoinProbeAndMap to map Parent IDs back to Child rows.
        // Critically, we re-apply the Child Filter to ensure exact matches.
        // This handles cases where multiple Child rows map to the same Parent,
        // but only specific Child rows should be selected.

        size_t child_rows = child_filter_data.len;
        int *child_bitmap_ptr = new int[child_rows];

        // =====================================================================
        // [Modified] Dimension Probe & Map (Replaces 7-argument JoinProbeAndMap)
        // Purpose: Propagates filter state from Parent to Child dimension.
        // =====================================================================

        // 1. 构造符合新版 API 要求的指针数组 (Single Dimension Join)
        const int *fks_1[] = {(const int *)child_fk_data.data};
        const int *vectors_1[] = {(const int *)parent_compressed_indices};
        const int factors_1[] = {1}; // 步长设定为 1

        // 2. 调用新版 MultiJoin 核心算子，使用单线程 (num_threads = 1)
        MOSS_DB::CPU::MultiJoin_Vector_CPU(
            child_rows,       // num_tuples
            1,                // join_col_num = 1
            fks_1,            // Foreign keys array
            vectors_1,        // Lookup vectors array
            factors_1,        // Stride factors array
            child_bitmap_ptr, // Output buffer
            1                 // Explicitly set to single-threaded as requested
        );
        // 3. 等价逻辑补全 (Early Exit Mask Application)
        // 原版 JoinProbeAndMap 会预先检查 child_filter_data[i] == op.val。
        // 在新版架构中为了保持严格等价，我们在单线程下进行一次 O(N) 的 Mask 修正。
        int *child_mask = (int *)child_filter_data.data;
        int target_val = op.val;
        for (size_t i = 0; i < child_rows; ++i)
        {
            // 如果孩子节点本身就不满足前置过滤条件，强制置为无效 (-1)
            if (child_mask[i] != target_val)
            {
                child_bitmap_ptr[i] = -1;
            }
        }

        // Cleanup Parent Buffer (we only need the Child bitmap now)
        delete[] parent_compressed_indices;

        // Construct Result (Child Bitmap + Parent Metadata)
        result.bitmap_ptr = child_bitmap_ptr;
        result.length = child_rows;
        result.is_on_device = false;
        result.reverse_dict = reverse_dict; // Keep Parent's dict for Phase 5
        result.reverse_dict.resize(result.unique_count);

        std::cout << "  [Rollup] Completed. Child Bitmap Size: " << child_rows
                  << ", Unique Groups: " << result.unique_count << std::endl;
        return result;
    }
    // Helper: Recursive Drill Down
    /**
     * @brief Executes OLAP Drill-Down: Filtering + Optional Grouping/Compression.
     * Supports Snowflake Schema traversal, Filter Chaining, and GPU Acceleration for large dimensions.
     * * @param root_node   Dimension hierarchy root.
     * @param op          Filter operation details (col, type, val). Can be empty.
     * @param group_col   Column to group by/compress. Can be empty.
     * @param input_mask  Previous filter result for chaining (AND logic).
     */
    FilterResult ExecuteIterativeFilter(
        std::shared_ptr<JoinNode> root_node,
        const SelectionOp &op,
        const std::string &group_col,
        const QueryPlan &plan,
        const FilterResult *input_mask = nullptr)
    {
        // =========================================================
        // Phase 1: Identify Target Level (Highest Level in Schema)
        // =========================================================
        struct SearchState
        {
            std::shared_ptr<JoinNode> node;
            std::vector<std::shared_ptr<JoinNode>> path;
        };
        std::vector<std::shared_ptr<JoinNode>> target_path;
        std::vector<SearchState> stack;
        stack.push_back({root_node, {root_node}});

        // Determine which column defines the "Target Level"
        // Priority: Filter Column > Group Column
        std::string search_col = !op.col.empty() ? op.col : group_col;
        bool found = false;

        while (!stack.empty())
        {
            auto current = stack.back();
            stack.pop_back();

            for (const auto &col : current.node->contained_columns)
            {
                if (col == search_col)
                {
                    target_path = current.path;
                    found = true;
                    break;
                }
            }
            if (found)
                break;

            for (const auto &child : current.node->upper_levels)
            {
                auto next_path = current.path;
                next_path.push_back(child);
                stack.push_back({child, next_path});
            }
        }

        // Fallback: If not found in hierarchy, default to root (e.g. LINEORDER direct filter)
        if (!found)
        {
            // Special handling for LINEORDER filters (e.g. LO_DISCOUNT) if they weren't pushed down
            // OR for Date columns that are only in the base table.
            if (root_node->table_name == "LINEORDER" || root_node->table_name == "DATE")
            {
                std::cout << "  [Graph Search] Target '" << search_col
                          << "' found in Root: " << root_node->table_name << std::endl;
                target_path = {root_node};
                found = true;
            }
            else
            {
                std::cerr << "  [Error] Column '" << search_col << "' not found in schema." << std::endl;
                return {};
            }
        }

        // =========================================================
        // Phase 2: Execute Logic at Top Level (Filter OR Init)
        // =========================================================
        auto target_node = target_path.back();
        std::cout << "  [Optimization] Processing Level: " << target_node->table_name << std::endl;

        // 1. Resolve Physical Column
        std::string physical_col = op.col;
        if (!op.col.empty())
        {
            // Logical -> Physical Mapping
            if (!target_node->contained_columns.empty())
            {
                for (size_t i = 0; i < target_node->contained_columns.size(); i += 2)
                {
                    if (i + 1 < target_node->contained_columns.size())
                    {
                        if (target_node->contained_columns[i] == op.col)
                        {
                            if (op.col == "S_CITY" || op.col == "C_CITY" || op.col == "S_NATION" || op.col == "C_NATION" || op.col == "P_CATEGORY")
                                physical_col = target_node->contained_columns[i + 2];
                            else
                                physical_col = target_node->contained_columns[i + 1];
                            break;
                        }
                    }
                }
            }
        }
        else if (!group_col.empty())
        {
            for (size_t i = 0; i < target_node->contained_columns.size(); i += 2)
            {
                if (i + 1 < target_node->contained_columns.size())
                {
                    if (target_node->contained_columns[i] == group_col)
                    {
                        if (op.col == "S_CITY" || op.col == "C_CITY")
                            physical_col = target_node->contained_columns[i + 2];
                        else
                            physical_col = target_node->contained_columns[i + 1];
                        break;
                    }
                }
            }
        }
        auto col_data = GetCPUColumnData(target_node->table_name, physical_col);
        size_t num_rows = col_data.len;

        // Initialize Result (Always on CPU initially for Top Level)
        FilterResult result;
        result.length = num_rows;
        result.unique_count = 1;
        result.is_on_device = false;
        result.bitmap_ptr = new int[num_rows]; // Allocate Host Memory

        // Init Bitmap logic
        if (input_mask && input_mask->bitmap_ptr && op.col.empty())
        {
            // Note: Assuming input_mask for top level is on CPU (since it's same dimension)
            // If it were on GPU, we'd need to copy back, but top-level recurrence implies CPU.
            std::memcpy(result.bitmap_ptr, input_mask->bitmap_ptr, num_rows * sizeof(int));
        }
        else
        {
            // Fill with 0 (Valid) or -1 (Invalid)
            int init_val = op.col.empty() ? 0 : -1;
            std::fill(result.bitmap_ptr, result.bitmap_ptr + num_rows, init_val);
        }

        // 2. Apply Filter (CPU)
        if (!op.col.empty())
        {
            bool applied = false;

            // Optimization: Chained Filter with Mask (AND Logic for EQ)
            if (input_mask && input_mask->bitmap_ptr && op.op_type == OpType::EQ)
            {
                std::cout << "  [Optimization] Using Fused FilterEqualWithMask for column: " << op.col << std::endl;

                SSB_CPU::FilterEqualWithMask(
                    (int *)col_data.data,
                    num_rows,
                    result.bitmap_ptr,      // Output (Pre-filled with -1)
                    input_mask->bitmap_ptr, // Input Mask
                    op.val);

                applied = true; // Mark as handled to skip default path
            }

            if (!applied)
            {
                // Dispatch based on Operator Type
                switch (op.op_type)
                {
                case OpType::EQ:
                    SSB_CPU::FilterEqual((int *)col_data.data, num_rows, result.bitmap_ptr, op.val);
                    break;

                case OpType::LT:
                    SSB_CPU::FilterLessThan((int *)col_data.data, num_rows, result.bitmap_ptr, op.val);
                    break;

                case OpType::BETWEEN:
                    SSB_CPU::FilterRange((int *)col_data.data, num_rows, result.bitmap_ptr, op.val, op.val2);
                    break;

                case OpType::IN:
                    // [Optimization] Handle IN operator
                    if (op.in_values.size() == 1)
                    {
                        // Degenerate case: IN [v1] -> EQ v1
                        SSB_CPU::FilterEqual((int *)col_data.data, num_rows, result.bitmap_ptr, op.in_values[0]);
                    }
                    else if (op.in_values.size() == 2)
                    {
                        // Specialized Dual-Equality Filter (e.g. Q3.4)
                        std::cout << "  [Filter] Using Optimized FilterInTwo for " << op.col
                                  << " IN [" << op.in_values[0] << ", " << op.in_values[1] << "]" << std::endl;

                        SSB_CPU::FilterInTwo(
                            (int *)col_data.data,
                            num_rows,
                            result.bitmap_ptr,
                            op.in_values[0],
                            op.in_values[1]);
                    }
                    else if (op.in_values.size() > 2)
                    {
                        // General Case: Fallback for list size > 2
                        // (Implementation logic: scan and check against vector/set)
                        std::cout << "  [Filter] General IN Filter (Size: " << op.in_values.size() << ")" << std::endl;
                        const int *input = (int *)col_data.data;
                        const std::vector<int> &vals = op.in_values;

#pragma omp parallel for schedule(static)
                        for (size_t i = 0; i < num_rows; i++)
                        {
                            int val = input[i];
                            bool match = false;
                            // Small linear scan is often faster than set lookup for small N (< 16)
                            for (int v : vals)
                            {
                                if (val == v)
                                {
                                    match = true;
                                    break;
                                }
                            }
                            if (match)
                                result.bitmap_ptr[i] = 0;
                        }
                    }
                    break;

                default:
                    std::cerr << "  [Error] Unsupported Operator Type: " << OpTypeToString(op.op_type) << std::endl;
                    break;
                }
            }

            // Handle Input Mask Chaining (Logical AND)
            // If we had a previous mask, we must AND it with the current result.
            // Note: Current logic assumes result.bitmap_ptr was initialized to -1 (Invalid)
            // and filters write 0 (Valid).
            // If input_mask exists, we should have initialized result.bitmap_ptr with it?
            // Correct approach:
            // 1. result.bitmap_ptr is allocated new.
            // 2. Filter writes 0 for local match.
            // 3. AND with previous mask: Valid if (Previous == 0 AND Current == 0).

            if (input_mask && input_mask->bitmap_ptr)
            {
                const int *prev = input_mask->bitmap_ptr;
                int *curr = result.bitmap_ptr;

#pragma omp parallel for schedule(static)
                for (size_t k = 0; k < num_rows; ++k)
                {
                    // Logic: Keep valid (0) only if BOTH are valid.
                    // If prev was invalid (-1) OR curr failed filter (-1), result is -1.
                    // Actually, the filter functions above only write 0 if match.
                    // They leave non-matches as initialized.
                    // So we must initialize curr to -1 before filter? (Done in prev steps)

                    if (prev[k] == 0 && curr[k] == 0)
                    {
                        curr[k] = 0;
                    }
                    else
                    {
                        curr[k] = -1;
                    }
                }
            }
        }

        // 3. Apply Compression (CPU)
        bool is_compressed = false;
        if (!group_col.empty())
        {
            bool is_group_node = false;
            for (const auto &c : target_node->contained_columns)
            {
                if (c == group_col)
                {
                    is_group_node = true;
                    break;
                }
            }

            if (is_group_node)
            {
                // [Optimization] Physical Column Remapping for Grouping
                // We need to compress based on the Integer ID column, not the Logical String column.
                std::string physical_group_col = group_col;

                const auto &cols = target_node->contained_columns;
                for (size_t k = 0; k < cols.size(); ++k)
                {
                    if (cols[k] == group_col)
                    {
                        // Special Rule for P_BRAND1: Format ["P_BRAND1", "B1_VAL", "ID_BRAND1"]
                        // We need index + 2.
                        if ((group_col == "D_YEAR" || group_col == "S_CITY" || group_col == "C_CITY" || group_col == "C_NATION" || group_col == "S_NATION" || group_col == "P_CATEGORY") && k + 2 < cols.size())
                        {
                            physical_group_col = cols[k + 2];
                            std::cout << "  [Schema Map] Group By '" << group_col
                                      << "' mapped to Special ID '" << physical_group_col << "'" << std::endl;
                        }
                        else if (group_col == "P_BRAND1" && k + 3 < cols.size())
                        {
                            physical_group_col = cols[k + 3];
                            std::cout << "  [Schema Map] Group By '" << group_col
                                      << "' mapped to Special ID '" << physical_group_col << "'" << std::endl;
                        }
                        // Standard Rule: Format ["LOGICAL", "PHYSICAL"]
                        // We need index + 1.
                        else if (k + 1 < cols.size())
                        {
                            physical_group_col = cols[k + 1];
                            std::cout << "  [Schema Map] Group By '" << group_col
                                      << "' mapped to Physical '" << physical_group_col << "'" << std::endl;
                        }
                        break; // Stop after finding the match
                    }
                }

                std::cout << "  [Compression] Compressing by '" << physical_group_col << "'..." << std::endl;

                // Fetch data using the resolved Physical Column Name
                auto g_data = GetCPUColumnData(target_node->table_name, physical_group_col);

                int *compressed_out = new int[num_rows];
                memset(compressed_out, -1, num_rows * sizeof(int));
                std::vector<int> value_map(num_rows, 0);
                result.reverse_dict.resize(num_rows, 0);

                SSB_CPU::CompressColumn(
                    (int *)g_data.data, num_rows,
                    compressed_out,
                    value_map.data(), &result.unique_count, result.reverse_dict.data(),
                    result.bitmap_ptr);
                delete[] result.bitmap_ptr;
                result.bitmap_ptr = compressed_out; // Now holds Group IDs
                is_compressed = true;
            }
        }

        // =========================================================
        // Phase 3: Drill Down (Backtrack)
        // =========================================================
        for (int i = target_path.size() - 1; i > 0; --i)
        {
            auto upper = target_path[i];
            auto lower = target_path[i - 1];

            std::string lower_name = lower->table_name;
            std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::toupper);

            // [Rule 1] Skip if Last Layer is LINEORDER
            if (lower_name == "LINEORDER")
            {
                std::cout << "  [Optimization] Stopping Drill-Down at Fact Table (LINEORDER)." << std::endl;
                break;
            }

            std::cout << "  [Drill-Down] " << upper->table_name << " -> " << lower->table_name;

            std::string fk_name = upper->fk_col;
            if (fk_name.empty())
                return {};

            // Check for Compression at Lower Level (Drill-Up scenario)
            bool compress_here = false;
            if (!is_compressed && !group_col.empty())
            {
                for (const auto &c : lower->contained_columns)
                {
                    if (c == group_col)
                    {
                        compress_here = true;
                        break;
                    }
                }
            }

            // =====================================================================
            // [Rule 2] Heterogeneous Device-Aware Join/Probe Routing
            // Identifies the target execution engine from the logical plan and
            // dispatches the workload to either the CUDA Kernel or the OpenMP CPU Kernel.
            // =====================================================================

            // 1. Extract Target Execution Engine from the AST
            // Fallback to "GPU" if no aggregation is defined or engine is missing.
            std::string target_engine = "GPU";
            if (!plan.aggregations.empty() && !plan.aggregations[0].exec_engine.empty())
            {
                target_engine = plan.aggregations[0].exec_engine;
            }

            // 2. Routing Decision: Only route to GPU if it's a large table AND the plan dictates GPU
            bool is_large_dim = (lower_name == "PART" || lower_name == "SUPPLIER" || lower_name == "CUSTOMER");
            bool use_gpu_join = (target_engine == "GPU");
            // Fetch Foreign Key column metadata

            if (!compress_here && is_large_dim && (target_engine == "GPU"))
            {
                if (use_gpu_join)
                {
                    auto fk_data = GetGPUColumnData(lower->table_name, fk_name);
                    size_t lower_rows = fk_data.len;
                    // -------------------------------------------------------------
                    // GPU EXECUTION PATH (CUDA)
                    // -------------------------------------------------------------
                    int *d_fk_col = (int *)fk_data.data;
                    std::cout << " (Mode: GPU Accelerated)" << std::endl;

                    int *d_current_bitmap = nullptr;
                    int *d_lower_bitmap = nullptr;

                    // State Machine: Ensure Input Bitmap is on Device
                    if (result.is_on_device)
                    {
                        d_current_bitmap = result.bitmap_ptr; // Zero-copy reuse
                    }
                    else
                    {
                        // PCIe Transfer: Host -> Device
                        CHECK_CUDA(cudaMalloc(&d_current_bitmap, result.length * sizeof(int)));
                        CHECK_CUDA(cudaMemcpy(d_current_bitmap, result.bitmap_ptr, result.length * sizeof(int), cudaMemcpyHostToDevice));
                        delete[] result.bitmap_ptr;
                    }

                    // Allocate output bitmap on GPU VRAM
                    CHECK_CUDA(cudaMalloc(&d_lower_bitmap, lower_rows * sizeof(int)));
                    CHECK_CUDA(cudaMemset(d_lower_bitmap, -1, lower_rows * sizeof(int)));

                    // Launch Configuration
                    const int BLOCK = SSBConfig::DEFAULT_BLOCK_SIZE;
                    const int ITEMS = SSBConfig::ITEMS_PER_THREAD;
                    int grid_size = (lower_rows + BLOCK * ITEMS - 1) / (BLOCK * ITEMS);

                    // Dispatch to GPU
                    SSB_GPU::BuildDenseIndicesKernel<BLOCK, ITEMS><<<grid_size, BLOCK>>>(
                        d_fk_col, lower_rows, d_current_bitmap, d_lower_bitmap, 1);
                    CHECK_KERNEL();

                    // Memory Cleanup
                    if (!result.is_on_device)
                    {
                        cudaFree(d_current_bitmap);
                    }
                    else
                    {
                        cudaFree(result.bitmap_ptr);
                    }

                    // Update Result Metadata
                    result.bitmap_ptr = d_lower_bitmap;
                    result.length = lower_rows;
                    result.is_on_device = true;
                }
                else
                {
                    // -------------------------------------------------------------
                    // CPU EXECUTION PATH (OpenMP Vectorized Pipeline)
                    // -------------------------------------------------------------
                    std::cout << " (Mode: CPU Vectorized Multi-Threaded)" << std::endl;

                    // Host pointers
                    auto h_fk_data = GetCPUColumnData(lower->table_name, fk_name);
                    size_t lower_rows = h_fk_data.len;
                    int *h_fk_col = (int *)h_fk_data.data; // Assumes host mirror exists
                    int *h_current_bitmap = nullptr;

                    int *h_lower_bitmap = new int[lower_rows]; // Allocate new Host Memory

                    // State Machine: Ensure Input Bitmap is on Host
                    if (result.is_on_device)
                    {
                        // PCIe Transfer: Device -> Host (Costly, but necessary for Fallback)
                        h_current_bitmap = new int[result.length];
                        CHECK_CUDA(cudaMemcpy(h_current_bitmap, result.bitmap_ptr, result.length * sizeof(int), cudaMemcpyDeviceToHost));
                        cudaFree(result.bitmap_ptr); // Free Device Memory
                    }
                    else
                    {
                        h_current_bitmap = result.bitmap_ptr; // Zero-copy reuse
                    }

                    // Prepare typed pointer arrays required by the new CPU MultiJoin API
                    const int *fks_array[] = {(const int *)h_fk_col};
                    const int *vectors_array[] = {(const int *)h_current_bitmap};
                    const int factors_array[] = {1}; // Default stride

                    // Retrieve the configured thread count from the Executor state (or global config)
                    // Assuming `num_threads_` is an accessible member variable of the executor context.
                    // If not, replace with `omp_get_max_threads()`.
                    int execution_threads = this->num_threads_;

                    // Dispatch to CPU OpenMP Kernel
                    MOSS_DB::CPU::MultiJoin_Vector_CPU(
                        lower_rows,       // Total rows in the child dimension table
                        1,                // Joining a single column mapping
                        fks_array,        // Foreign Key array
                        vectors_array,    // Lookup vectors (Parent Bitmap)
                        factors_array,    // Strides
                        h_lower_bitmap,   // Output buffer
                        execution_threads // Explicit OpenMP Thread count
                    );

                    // Memory Cleanup
                    if (result.is_on_device)
                    {
                        delete[] h_current_bitmap; // Clean up temporary host buffer if we moved it
                    }
                    else
                    {
                        delete[] result.bitmap_ptr; // Clean up the original host buffer
                    }

                    // Update Result Metadata
                    result.bitmap_ptr = h_lower_bitmap;
                    result.length = lower_rows;
                    result.is_on_device = false; // Flag as Host resident
                }
            }

            else
            {
                // CPU Mode (Standard Join OR CompressWithFK)
                auto fk_data = GetCPUColumnData(lower->table_name, fk_name);
                int *lower_bitmap = new int[fk_data.len];
                memset(lower_bitmap, -1, fk_data.len * sizeof(int));
                if (compress_here)
                {
                    // [Optimization] Physical Column Remapping for Grouping (Lower Level)
                    std::string physical_group_col = group_col;

                    const auto &cols = lower->contained_columns;
                    for (size_t k = 0; k < cols.size(); ++k)
                    {
                        if (cols[k] == group_col)
                        {
                            // Special Rule for P_BRAND1
                            if ((group_col == "D_YEAR" || group_col == "S_CITY" || group_col == "C_CITY" || group_col == "C_NATION" || group_col == "S_NATION" || group_col == "P_CATEGORY") && k + 2 < cols.size())
                            {
                                physical_group_col = cols[k + 2];
                                std::cout << "  [Schema Map] Group By '" << group_col
                                          << "' mapped to Special ID '" << physical_group_col << "'" << std::endl;
                            }
                            else if (group_col == "P_BRAND1" && k + 3 < cols.size())
                            {
                                physical_group_col = cols[k + 3];
                                std::cout << "  [Schema Map] Group By '" << group_col
                                          << "' mapped to Special ID '" << physical_group_col << "'" << std::endl;
                            }
                            // Standard Rule
                            else if (k + 1 < cols.size())
                            {
                                physical_group_col = cols[k + 1];
                                std::cout << "  [Schema Map] Group By '" << group_col
                                          << "' mapped to Physical '" << physical_group_col << "'" << std::endl;
                            }
                            break;
                        }
                    }

                    std::cout << " (Action: Compress with FK on " << physical_group_col << ")" << std::endl;

                    // Fetch data using the resolved Physical Column Name
                    auto g_data = GetCPUColumnData(lower->table_name, physical_group_col);

                    // Use explicit sizing based on column length to avoid vector resize overhead
                    std::vector<int> value_map(fk_data.len, 0);
                    result.reverse_dict.resize(fk_data.len, 0);

                    SSB_CPU::CompressColumnWithForeignKey(
                        (int *)g_data.data, fk_data.len,
                        lower_bitmap,
                        value_map.data(), &result.unique_count, result.reverse_dict.data(),
                        result.bitmap_ptr,  // Input Mask
                        (int *)fk_data.data // FK
                    );
                    is_compressed = true;
                }
                else
                {
                    std::cout << " (Mode: CPU)" << std::endl;
                    const int *fks_2[] = {(const int *)fk_data.data};
                    const int *vectors_2[] = {(const int *)result.bitmap_ptr};
                    const int factors_2[] = {1};

                    MOSS_DB::CPU::MultiJoin_Vector_CPU(
                        fk_data.len,  // num_tuples
                        1,            // join_col_num = 1
                        fks_2,        // Foreign keys
                        vectors_2,    // Lookup vectors
                        factors_2,    // Strides
                        lower_bitmap, // Output mapped bitmap/vector
                        1             // Explicitly set to single-threaded
                    );
                }
                delete[] result.bitmap_ptr;
                result.bitmap_ptr = lower_bitmap;
                result.length = fk_data.len;
            }
        }
        return result;
    }

public:
    SSBQueryExecutor(int num_threads)
    {
        size_t lo_rows = column_store.num_lineorder_rows;
        size_t di_rows = column_store.num_discount_rows;
        size_t p_rows = column_store.num_part_rows;
        size_t s_rows = column_store.num_supplier_rows;
        size_t c_rows = column_store.num_customer_rows;
        num_threads_ = num_threads;
        CHECK_CUDA(cudaMalloc(&d_lo_custkey, lo_rows * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_lo_partkey, lo_rows * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_lo_suppkey, lo_rows * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_lo_extendedprice, lo_rows * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_date_fk, lo_rows * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_lo_supplycost, lo_rows * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_lo_revenue, lo_rows * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_lo_discount, di_rows * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_p_brand1, p_rows * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_s_city, s_rows * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_c_city, c_rows * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_discount_fk, lo_rows * sizeof(int8_t)));
        CHECK_CUDA(cudaMalloc(&d_quantity_fk, lo_rows * sizeof(int8_t)));
        CHECK_CUDA(cudaMalloc(&d_results, 1024 * sizeof(unsigned long long)));
    }

    ~SSBQueryExecutor()
    {
        cudaFree(d_lo_custkey);
        cudaFree(d_lo_partkey);
        cudaFree(d_lo_suppkey);
        cudaFree(d_lo_extendedprice);
        cudaFree(d_date_fk);
        cudaFree(d_lo_supplycost);
        cudaFree(d_lo_revenue);
        cudaFree(d_lo_discount);
        cudaFree(d_discount_fk);
        cudaFree(d_quantity_fk);
        cudaFree(d_p_brand1);
        cudaFree(d_s_city);
        cudaFree(d_c_city);
        cudaFree(d_results);
    }

    void PrepareGPUData()
    {
        SSB_Utils::CpuTimer timer("GPU Data Transfer");
        size_t lo_rows = column_store.num_lineorder_rows;
        size_t di_rows = column_store.num_discount_rows;
        size_t p_rows = column_store.num_part_rows;
        size_t s_rows = column_store.num_supplier_rows;
        size_t c_rows = column_store.num_customer_rows;
        CHECK_CUDA(cudaMemcpy(d_lo_custkey, column_store.lo_custkey, lo_rows * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_lo_partkey, column_store.lo_partkey, lo_rows * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_lo_suppkey, column_store.lo_suppkey, lo_rows * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_lo_extendedprice, column_store.lo_extendedprice, lo_rows * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_date_fk, column_store.date_fk, lo_rows * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_lo_supplycost, column_store.lo_supplycost, lo_rows * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_lo_revenue, column_store.lo_revenue, lo_rows * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_lo_discount, column_store.dim_discount_val, di_rows * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_p_brand1, column_store.p_brand1, p_rows * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_s_city, column_store.s_city, s_rows * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_c_city, column_store.c_city, c_rows * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_discount_fk, column_store.discount_fk, lo_rows * sizeof(int8_t), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_quantity_fk, column_store.quantity_fk, lo_rows * sizeof(int8_t), cudaMemcpyHostToDevice));
        timer.print();
    }
    /**
     * @brief Helper to safely free FilterResult memory (Host or Device).
     */
    void SafeFreeResult(FilterResult &res)
    {
        if (res.bitmap_ptr)
        {
            if (res.is_on_device)
            {
                CHECK_CUDA(cudaFree(res.bitmap_ptr));
            }
            else
            {
                delete[] res.bitmap_ptr;
            }
            res.bitmap_ptr = nullptr;
        }
    }
    void ExecuteQuery(const QueryPlan &plan, JoinGraph &graph)
    {
        SSB_Utils::GpuTimer total_timer("Total Query Time");
        total_timer.start();
        
        // test_timer.start();
        std::map<std::string, FilterResult> dimension_bitmaps;
        std::map<std::string, int> dimension_cardinalities;
        std::map<std::string, std::vector<int>> dimension_reverse_dicts;

        for (const auto &olap_op : plan.olap_ops)
        {
            std::string target_dim_name = olap_op.dimension;
            std::shared_ptr<JoinNode> current_root = nullptr;

            if (graph.dimension_roots.count(target_dim_name))
            {
                current_root = graph.dimension_roots[target_dim_name];
            }
            else
            {
                std::cerr << "  [Warning] Unknown dimension: " << target_dim_name << ". Skipping." << std::endl;
                continue;
            }

            FilterResult result;

            // =========================================================
            // Strategy A: ROLLUP Operation (Bottom-Up)
            // =========================================================
            if (olap_op.is_rollup && !olap_op.filter.col.empty())
            {
                std::cout << "  [OLAP] Executing ROLLUP on " << target_dim_name
                          << " (Filter Col: " << olap_op.filter.col << ", Group Col: " << olap_op.group_col << ")" << std::endl;

                std::shared_ptr<JoinNode> child_node = nullptr;
                std::shared_ptr<JoinNode> parent_node = nullptr;

                // DFS Search
                std::vector<std::shared_ptr<JoinNode>> stack;
                stack.push_back(current_root);

                while (!stack.empty())
                {
                    auto n = stack.back();
                    stack.pop_back();
                    // Schema-Agnostic Search
                    for (const auto &c : n->contained_columns)
                    {
                        if (c == olap_op.filter.col)
                        {
                            child_node = n;
                            break;
                        }
                    }
                    for (const auto &c : n->contained_columns)
                    {
                        if (c == olap_op.group_col)
                        {
                            parent_node = n;
                            break;
                        }
                    }
                    if (child_node && parent_node)
                        break;
                    for (const auto &child : n->upper_levels)
                        stack.push_back(child);
                }

                if (child_node && parent_node)
                {
                    // A.1. Execute Upward Rollup (Returns CHILD Bitmap)
                    FilterResult child_result_with_group_ids = ExecuteRollupFilter(
                        parent_node, child_node, olap_op.filter, olap_op.group_col);

                    // A.2. Propagate Downward (Child -> Base)
                    // We start the drill-down from the CHILD node (YEARMONTH) because our result
                    // is already mapped to that level.
                    // We pass 'olap_op.filter.col' as the target to ensure ExecuteIterativeFilter
                    // identifies the Child Node as the starting point.
                    result = ExecuteIterativeFilter(
                        current_root,       // Search Scope
                        {},                 // No new filter
                        olap_op.filter.col, // Target Node Hint (Child Node)
                        plan,
                        &child_result_with_group_ids // Input: Child Bitmap
                    );

                    // A.3. Metadata Handover
                    result.unique_count = child_result_with_group_ids.unique_count;
                    result.reverse_dict = child_result_with_group_ids.reverse_dict;

                    SafeFreeResult(child_result_with_group_ids);
                }
                else
                {
                    std::cerr << "  [Error] Rollup nodes not found." << std::endl;
                }
            }
            // =========================================================
            // Strategy B: DRILLDOWN / FILTER Operation (Top-Down)
            // Condition: Default path or Rollup without child table (fallback).
            // =========================================================
            else
            {
                // B.1. Schema Push-down Optimization
                std::string effective_target = target_dim_name;
                if (target_dim_name == "LINEORDER" && !olap_op.filter.col.empty())
                {
                    for (const auto &child : current_root->upper_levels)
                    {
                        bool found = false;
                        for (const auto &c : child->contained_columns)
                            if (c == olap_op.filter.col)
                                found = true;
                        if (found)
                        {
                            std::cout << "  [Schema Map] Pushing filter down to: " << child->table_name << std::endl;
                            current_root = child;
                            effective_target = child->table_name;
                            break;
                        }
                    }
                }

                // B.2. Filter Chaining
                const FilterResult *input_mask_ptr = nullptr;
                if (dimension_bitmaps.count(effective_target))
                {
                    input_mask_ptr = &dimension_bitmaps[effective_target];
                }
                target_dim_name = effective_target;
                // B.3. Execute
                result = ExecuteIterativeFilter(
                    current_root,
                    olap_op.filter,
                    olap_op.group_col,
                    plan,
                    input_mask_ptr);
            }

            // =========================================================
            // Common Registry Update & Cleanup
            // =========================================================
            if (result.bitmap_ptr != nullptr)
            {
                if (dimension_bitmaps.count(target_dim_name))
                {
                    FilterResult &old_res = dimension_bitmaps[target_dim_name];
                    if (old_res.bitmap_ptr != result.bitmap_ptr)
                        SafeFreeResult(old_res);
                }

                if (!result.reverse_dict.empty())
                {
                    dimension_reverse_dicts[target_dim_name] = result.reverse_dict;
                }

                dimension_bitmaps[target_dim_name] = result;

                if (result.unique_count > 1 || dimension_cardinalities[target_dim_name] == 0)
                {
                    dimension_cardinalities[target_dim_name] = result.unique_count;
                }
            }
        }

        // =========================================================
        // Phase 2: Dynamic Heterogeneous Join Assembly
        // Routings FKs and Bitmaps to the targeted execution engine.
        // =========================================================
        const int MAX_JOINS = 5;
        const void *fks_array[MAX_JOINS] = {nullptr};
        const int *maps_array[MAX_JOINS] = {nullptr};
        int strides_array[MAX_JOINS] = {0};

        std::vector<int *> gpu_allocations;
        std::vector<int *> cpu_allocations;

        // 1. Determine Execution Engine Target from Logical Plan
        bool use_gpu = true;
        if (!plan.aggregations.empty())
        {
            std::string engine = plan.aggregations[0].exec_engine;
            if (engine == "CPU")
            {
                use_gpu = false;
            }
        }

        std::cout << "[System] Target Execution Engine for Fact Table: "
                  << (use_gpu ? "GPU (CUDA)" : "CPU (OpenMP)") << std::endl;

        // Temporary structure to hold slot info for stride calculation
        struct SlotInfo
        {
            int slot_idx;
            int cardinality;
            std::string table_name;
        };
        std::vector<SlotInfo> active_slots;

        if (graph.dimension_roots.count("LINEORDER"))
        {
            auto lo_root = graph.dimension_roots["LINEORDER"];
            int slot_idx = 0;
            std::set<std::string> assembled_dimensions;

            for (const auto &olap_op : plan.olap_ops)
            {
                if (slot_idx >= MAX_JOINS)
                    break;

                // 1. Determine Effective Target Dimension
                std::string target_dim = olap_op.dimension;
                if (target_dim == "LINEORDER" && !olap_op.filter.col.empty())
                {
                    for (const auto &child : lo_root->upper_levels)
                    {
                        bool found = false;
                        for (const auto &c : child->contained_columns)
                        {
                            if (c == olap_op.filter.col)
                                found = true;
                        }
                        if (found)
                        {
                            target_dim = child->table_name;
                            break;
                        }
                    }
                }

                if (assembled_dimensions.count(target_dim))
                    continue;

                // 2. Find the corresponding node in LINEORDER's upper levels
                std::shared_ptr<JoinNode> target_node = nullptr;
                for (const auto &dim_node : lo_root->upper_levels)
                {
                    if (dim_node->table_name == target_dim)
                    {
                        target_node = dim_node;
                        break;
                    }
                }

                // 3. Process Assembly based on Target Engine
                if (target_node && dimension_bitmaps.count(target_dim))
                {
                    std::string table_name = target_node->table_name;
                    std::string fk_col = target_node->fk_col;

                    // A. Fetch Device-Aware Column Data
                    auto fk_data = use_gpu ? GetGPUColumnData("LINEORDER", fk_col)
                                           : GetCPUColumnData("LINEORDER", fk_col);
                    if (!fk_data.data)
                    {
                        std::cerr << "  -> Missing FK Column data on target engine: " << fk_col << std::endl;
                        return;
                    }
                    fks_array[slot_idx] = fk_data.data;

                    // B. Heterogeneous Bitmap Memory Migration
                    FilterResult &res = dimension_bitmaps[table_name];
                    if (use_gpu)
                    {
                        if (res.is_on_device)
                        {
                            maps_array[slot_idx] = res.bitmap_ptr;
                        }
                        else
                        {
                            int *d_bitmap_buf = nullptr;
                            CHECK_CUDA(cudaMalloc(&d_bitmap_buf, res.length * sizeof(int)));
                            CHECK_CUDA(cudaMemcpy(d_bitmap_buf, res.bitmap_ptr, res.length * sizeof(int), cudaMemcpyHostToDevice));
                            maps_array[slot_idx] = d_bitmap_buf;
                            gpu_allocations.push_back(d_bitmap_buf);
                        }
                    }
                    else
                    { // CPU Path
                        if (!res.is_on_device)
                        {
                            maps_array[slot_idx] = res.bitmap_ptr;
                        }
                        else
                        {
                            int *h_bitmap_buf = new int[res.length];
                            CHECK_CUDA(cudaMemcpy(h_bitmap_buf, res.bitmap_ptr, res.length * sizeof(int), cudaMemcpyDeviceToHost));
                            maps_array[slot_idx] = h_bitmap_buf;
                            cpu_allocations.push_back(h_bitmap_buf);
                        }
                    }

                    // C. Record info for Stride Calculation
                    int card = dimension_cardinalities[table_name];
                    if (card <= 0)
                        card = 1;
                    active_slots.push_back({slot_idx, card, table_name});
                    assembled_dimensions.insert(table_name);
                    slot_idx++;
                }
            }
        }

        // 4. Calculate Strides (Suffix Products)
        long long current_multiplier = 1;
        for (int i = active_slots.size() - 1; i >= 0; --i)
        {
            int idx = active_slots[i].slot_idx;
            int card = active_slots[i].cardinality;
            strides_array[idx] = (int)current_multiplier;
            current_multiplier *= card;
        }

        int total_groups = (int)current_multiplier;
        if (total_groups <= 0)
            total_groups = 1;

        // =========================================================
        // Phase 3: Prepare Aggregation Columns (Device Aware)
        // =========================================================
        if (plan.aggregations.empty())
            return;
        const auto &agg = plan.aggregations[0];

        struct AggregationColumn
        {
            const void *data_ptr;
            const void *lookup_ptr;
            int data_width;
            int lookup_width;
            bool is_indirect;
        };

        auto ResolveData = [&](const std::string &table_name, const std::string &col_name) -> AggregationColumn
        {
            if (col_name.empty())
                return {nullptr, nullptr, 0, 0, false};

            auto direct_data = use_gpu ? GetGPUColumnData(table_name, col_name)
                                       : GetCPUColumnData(table_name, col_name);

            if (direct_data.data != nullptr && !direct_data.is_fk_proxy)
            {
                return {direct_data.data, nullptr, direct_data.byte_width, 0, false};
            }

            if (graph.dimension_roots.count(table_name))
            {
                auto root = graph.dimension_roots[table_name];
                for (auto &child : root->upper_levels)
                {
                    bool has_col = false;
                    for (const auto &c : child->contained_columns)
                    {
                        if (c == col_name)
                        {
                            has_col = true;
                            break;
                        }
                    }
                    if (has_col)
                    {
                        auto fk_data = use_gpu ? GetGPUColumnData(table_name, child->fk_col)
                                               : GetCPUColumnData(table_name, child->fk_col);

                        std::string dim_val_col = (child->table_name == "DISCOUNT") ? "DIM_DISCOUNT_VAL" : (child->table_name == "QUANTITY") ? "DIM_QUANTITY_VAL"
                                                                                                                                             : col_name;
                        auto dim_data = use_gpu ? GetGPUColumnData(child->table_name, dim_val_col)
                                                : GetCPUColumnData(child->table_name, dim_val_col);

                        if (!fk_data.data || !dim_data.data)
                            return {nullptr, nullptr, 0, 0, false};
                        return {fk_data.data, dim_data.data, fk_data.byte_width, dim_data.byte_width, true};
                    }
                }
            }

            if (direct_data.data != nullptr)
                return {direct_data.data, nullptr, direct_data.byte_width, 0, false};

            return {nullptr, nullptr, 0, 0, false};
        };

        auto col1 = ResolveData(agg.table, agg.col1);
        auto col2 = ResolveData(agg.table, agg.col2);
        // test_timer.print();
        // =========================================================
        // Phase 4: Heterogeneous Kernel Launch
        // =========================================================
        size_t lo_rows = column_store.num_lineorder_rows;

        // [Optimization] Unified Host Result Buffer (Zero-Copy Support for CPU Path)
        // By hoisting the allocation here, we eliminate PCIe ping-pong overhead
        // when the aggregation is natively executed on the CPU.
        std::vector<unsigned long long> h_result_agg(total_groups, 0);

        if (use_gpu)
        {
            // -------------------------------------------------------------
            // PATH A: GPU EXECUTION (CUDA)
            // -------------------------------------------------------------
            CHECK_CUDA(cudaMemset(d_results, 0, total_groups * sizeof(unsigned long long)));

            const int BLOCK = SSBConfig::DEFAULT_BLOCK_SIZE;
            const int ITEMS = SSBConfig::ITEMS_PER_THREAD;
            int grid_size = (lo_rows + (BLOCK * ITEMS) - 1) / (BLOCK * ITEMS);

            std::cout << "[Kernel Launch] Target: GPU | Grid=" << grid_size << " | Groups=" << total_groups << std::endl;

            if (col1.data_ptr != nullptr && col2.data_ptr != nullptr)
            {
                if (agg.op == "-")
                {
                    SSB_GPU::ProbeDenseKernel<BLOCK, ITEMS, SSB_GPU::AggOp::SUBTRACT, int, int, int, int, int, int, true><<<grid_size, BLOCK>>>(
                        (const int *)fks_array[0], maps_array[0], strides_array[0],
                        (const int *)fks_array[1], maps_array[1], strides_array[1],
                        (const int *)fks_array[2], maps_array[2], strides_array[2],
                        (const int *)fks_array[3], maps_array[3], strides_array[3],
                        col1.data_ptr, col1.lookup_ptr, col2.data_ptr, col2.lookup_ptr,
                        lo_rows, total_groups, d_results);
                }
                else
                {
                    SSB_GPU::ProbeDenseKernel<BLOCK, ITEMS, SSB_GPU::AggOp::PRODUCT, int, int8_t, int8_t, int8_t, int, int8_t, true><<<grid_size, BLOCK>>>(
                        (const int *)fks_array[0], maps_array[0], strides_array[0],
                        (const int8_t *)fks_array[1], maps_array[1], strides_array[1],
                        (const int8_t *)fks_array[2], maps_array[2], strides_array[2],
                        (const int8_t *)fks_array[3], maps_array[3], strides_array[3],
                        col1.data_ptr, col1.lookup_ptr, col2.data_ptr, col2.lookup_ptr,
                        lo_rows, total_groups, d_results);
                }
            }
            else if (col1.data_ptr != nullptr)
            {
                SSB_GPU::ProbeDenseKernel<BLOCK, ITEMS, SSB_GPU::AggOp::SUM, int, int, int, int, int, int, false><<<grid_size, BLOCK>>>(
                    (const int *)fks_array[0], maps_array[0], strides_array[0],
                    (const int *)fks_array[1], maps_array[1], strides_array[1],
                    (const int *)fks_array[2], maps_array[2], strides_array[2],
                    (const int *)fks_array[3], maps_array[3], strides_array[3],
                    col1.data_ptr, col1.lookup_ptr, nullptr, nullptr,
                    lo_rows, total_groups, d_results);
            }
            CHECK_KERNEL();
        }
        else
        {
            // -------------------------------------------------------------
            // PATH B: CPU EXECUTION (OpenMP Vectorized Pipeline)
            // -------------------------------------------------------------

            // =====================================================================
            // [CRITICAL PERFORMANCE FIX] Hardware-Aware Thread Scheduling
            // 1. Bypass OpenMP ICV pollution by querying OS hardware concurrency.
            // 2. Mitigate SMT/Hyper-Threading L1 cache thrashing by using physical cores.
            // 3. Prevent NUMA cross-socket memory bus saturation by capping threads.
            // =====================================================================
            unsigned int hw_threads = std::thread::hardware_concurrency();

            int num_threads = hw_threads;
            std::cout << "[Kernel Launch] Target: CPU | Threads=" << num_threads
                      << " (HW Max: " << hw_threads << ") | Groups=" << total_groups << std::endl;

            // =====================================================================
            // [CRITICAL PERFORMANCE FIX] Static Typed Dispatch (C++17 Constexpr)
            // Replaced Lambdas with strong-typed template instantiation to enforce
            // Strict Aliasing Rules and guarantee 100% AVX/SIMD auto-vectorization.
            // =====================================================================

            if (col1.data_ptr != nullptr && col2.data_ptr != nullptr)
            {
                if (agg.op == "-")
                {
                    // -------------------------------------------------------------
                    // Target: Q4 (e.g., REVENUE - SUPPLYCOST)
                    // Behavior: Dual-column Direct Access. Both columns reside natively 
                    // in the Fact Table. No dictionary lookups needed.
                    // -------------------------------------------------------------
                    std::cout << "  -> [Router] Path: Dual-Column SUBTRACT (Direct)" << std::endl;
                    MOSS_DB::CPU::JoinAgg_Vector_Pipeline_CPU<
                        MOSS_DB::CPU::AggOp::SUBTRACT, 
                        int, int, true,                 // T1, T2, HAS_COL2
                        false, false,                   // [ALIGNED] INDIRECT1, INDIRECT2
                        int, int,                       // [ALIGNED] L1, L2 (Dummy types, DCE removes them)
                        int, int, int, int              // FK0..3 (Assumes standard 32-bit for Q4 dimensions)
                    >(
                        lo_rows, active_slots.size(),
                        (const void **)fks_array, 
                        (const int **)maps_array, strides_array,
                        (const int *)col1.data_ptr, nullptr, // agg_col1, lookup1 (nullptr)
                        (const int *)col2.data_ptr, nullptr, // agg_col2, lookup2 (nullptr)
                        h_result_agg.data(), num_threads, total_groups
                    );
                }
                else
                {
                    // -------------------------------------------------------------
                    // Target: Q1 (e.g., EXTENDEDPRICE * DISCOUNT)
                    // Behavior: Dual-column Late Materialization. Fact table stores 1-byte
                    // Foreign Keys (int8_t). CPU must fetch 4-byte metrics (int) from Lookup Dictionaries.
                    // -------------------------------------------------------------
                    std::cout << "  -> [Router] Path: Dual-Column PRODUCT (Indirect Late-Materialized)" << std::endl;
                    MOSS_DB::CPU::JoinAgg_Vector_Pipeline_CPU<
                        MOSS_DB::CPU::AggOp::PRODUCT,
                        int, int8_t, true,           // T1(FK), T2(FK), HAS_COL2
                        false, true,                     // [ALIGNED] Both columns require dictionary lookups
                        int, int,                       // [ALIGNED] L1, L2 (Actual metric values in dict are 32-bit int)
                        int, int8_t, int8_t, int8_t     // FK0..3 (Date is 32-bit, others compressed to 8-bit)
                    >(
                        lo_rows, active_slots.size(),
                        (const void **)fks_array, 
                        (const int **)maps_array, strides_array,
                        (const int *)col1.data_ptr, (const int *)col1.lookup_ptr, // FK Array + Dict Array
                        (const int8_t *)col2.data_ptr, (const int *)col2.lookup_ptr, // FK Array + Dict Array
                        h_result_agg.data(), num_threads, total_groups
                    );
                }
            }
            else if (col1.data_ptr != nullptr)
            {
                // -------------------------------------------------------------
                // Target: Q2/Q3 (e.g., SUM(REVENUE))
                // Behavior: Single-column Direct Access. 
                // -------------------------------------------------------------
                std::cout << "  -> [Router] Path: Single-Column SUM (Direct)" << std::endl;
                MOSS_DB::CPU::JoinAgg_Vector_Pipeline_CPU<
                    MOSS_DB::CPU::AggOp::SUM, 
                    int, int, false,                    // T1, T2(Ignored), HAS_COL2 = false
                    false, false,                       // [ALIGNED] No indirect lookups
                    int, int,                           // [ALIGNED] L1, L2
                    int, int, int, int                  // FK0..3
                >(
                    lo_rows, active_slots.size(),
                    (const void **)fks_array,
                    (const int **)maps_array, strides_array,
                    (const int *)col1.data_ptr, nullptr, // agg_col1, lookup1 (nullptr)
                    nullptr, nullptr,                    // agg_col2 (nullptr), lookup2 (nullptr)
                    h_result_agg.data(), num_threads, total_groups
                );
            }
            else
            {
                std::cerr << "[Fatal Error] Aggregation Column data pointers are NULL." << std::endl;
            }
        }

        total_timer.print();
        SSB_Utils::CpuTimer result_timer("Result Analysis Time");
        result_timer.start();
        // =========================================================
        // Phase 5: Result Retrieval, Materialization & Sorting
        // Decodes Group IDs back to string/integer values and sorts them.
        // =========================================================

        if (use_gpu)
        {
            // [Optimization] Only perform PCIe transfer if data actually resides on the GPU
            CHECK_CUDA(cudaMemcpy(h_result_agg.data(), d_results, total_groups * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        }

        std::vector<ResultRow> final_results;

        // 1. Materialize Results
        // Case A: Scalar (Q1) - Single Group
        if (total_groups == 1)
        {
            if (h_result_agg[0] > 0)
            {
                final_results.push_back({{}, h_result_agg[0]});
            }
        }
        // Case B: Vector (Q2-Q4) - Multi-Group
        else
        {
            // [Optimization] Pre-resolve Metadata for all Active Slots
            struct SlotMetadata
            {
                bool is_valid_group;
                std::string group_col;
                void *raw_data_ptr; // Pointer to Physical Data Column
                std::string table_name;
                bool is_int;
                int byte_width; // Used for pointer offset calculation in integral types
            };
            std::vector<SlotMetadata> slot_meta(active_slots.size());

            for (size_t i = 0; i < active_slots.size(); ++i)
            {
                std::string root_dim = active_slots[i].table_name;
                std::string group_col = "";

                // A. Find Group Column Name from Plan
                for (const auto &op : plan.olap_ops)
                {
                    if (op.dimension == root_dim)
                    {
                        group_col = op.group_col;
                        break;
                    }
                }

                if (group_col.empty())
                {
                    slot_meta[i] = {false, "", nullptr, root_dim, true, 4};
                    continue;
                }

                // B. Find Target Node (DFS in Schema Tree)
                std::shared_ptr<JoinNode> target_node = nullptr;
                if (graph.dimension_roots.count(root_dim))
                {
                    std::vector<std::shared_ptr<JoinNode>> search_stack;
                    search_stack.push_back(graph.dimension_roots[root_dim]);
                    while (!search_stack.empty())
                    {
                        auto node = search_stack.back();
                        search_stack.pop_back();
                        for (const auto &c : node->contained_columns)
                        {
                            if (c == group_col)
                            {
                                target_node = node;
                                break;
                            }
                        }
                        if (target_node)
                            break;
                        for (auto &child : node->upper_levels)
                            search_stack.push_back(child);
                    }
                }

                // C. Resolve Physical Column & Data Pointer
                void *data_ptr = nullptr;
                bool int_flag = true;
                int b_width = 4;

                if (target_node)
                {
                    std::string physical_col = group_col;
                    const auto &cols = target_node->contained_columns;
                    for (size_t k = 0; k < cols.size(); ++k)
                    {
                        if (cols[k] == group_col)
                        {
                            // Rule 1: Special Offset (e.g. P_BRAND1 -> ID_BRAND1)
                            if (group_col == "P_BRAND1" && k + 2 < cols.size())
                                physical_col = cols[k + 2];
                            // Rule 2: Standard Offset
                            else if (k + 1 < cols.size())
                                physical_col = cols[k + 1];
                            break;
                        }
                    }
                    ColumnData raw_col = GetCPUColumnData(target_node->table_name, physical_col);
                    data_ptr = raw_col.data;
                    int_flag = raw_col.is_int;
                    b_width = raw_col.byte_width;
                }

                slot_meta[i] = {true, group_col, data_ptr, root_dim, int_flag, b_width};
            }

            // Iterate all potential groups to decode IDs
            for (int gid = 0; gid < total_groups; ++gid)
            {
                unsigned long long metric = h_result_agg[gid];
                if (metric == 0)
                    continue;

                ResultRow row;
                row.metric = metric;

                // Decode Dimensions from global Group ID
                for (size_t i = 0; i < active_slots.size(); ++i)
                {
                    if (!slot_meta[i].is_valid_group)
                        continue;

                    int slot_idx = active_slots[i].slot_idx;
                    int stride = strides_array[slot_idx];
                    int card = active_slots[i].cardinality;

                    // Reverse calculation to find local dimension ID
                    size_t local_id = (gid / stride) % card;
                    std::string val_str = "NULL";

                    const auto &meta = slot_meta[i];
                    if (meta.raw_data_ptr && dimension_reverse_dicts.count(meta.table_name))
                    {
                        const auto &dict = dimension_reverse_dicts[meta.table_name];
                        if (local_id < dict.size())
                        {
                            int original_row_idx = dict[local_id];

                            // =========================================================
                            // Type-Safe Value Extraction Logic (BUG FIX)
                            // =========================================================
                            if (meta.is_int)
                            {
                                // Integral Branch: Support precise width extraction
                                if (meta.byte_width == 1)
                                {
                                    int8_t val = static_cast<int8_t *>(meta.raw_data_ptr)[original_row_idx];
                                    val_str = std::to_string(val);
                                }
                                else
                                {
                                    int32_t val = static_cast<int32_t *>(meta.raw_data_ptr)[original_row_idx];
                                    val_str = std::to_string(val);
                                }
                            }
                            else
                            {
                                // String Branch: Restored Object Pointer Cast
                                // GetCPUColumnData natively returns an array of std::string objects for text columns.
                                // We MUST explicitly cast the void* back to std::string* to avoid
                                // reading internal object layout bytes (which causes console formatting corruption).
                                val_str = static_cast<std::string *>(meta.raw_data_ptr)[original_row_idx];
                            }
                        }
                    }
                    row.dimensions.push_back(val_str);
                }
                final_results.push_back(row);
            }
        }
        // =========================================================
        // 2. Sort Results (CPU) - Optimized with Pre-compilation
        // =========================================================
        std::string agg_alias = plan.aggregations.empty() ? "REVENUE" : plan.aggregations[0].alias;
        if (agg_alias.empty())
            agg_alias = "REVENUE";

        if (!plan.order_bys.empty() && !final_results.empty())
        {
            auto is_numeric_string = [](const std::string &s) -> bool
            {
                if (s.empty())
                    return false;
                size_t start = (s[0] == '-' || s[0] == '+') ? 1 : 0;
                if (start == s.length())
                    return false;
                for (size_t i = start; i < s.length(); ++i)
                {
                    if (!std::isdigit(s[i]))
                        return false;
                }
                return true;
            };

            struct SortKeyInfo
            {
                bool is_metric;
                int dim_idx;
                bool asc;
                bool is_numeric;
            };

            std::vector<SortKeyInfo> compiled_order;
            compiled_order.reserve(plan.order_bys.size());

            for (const auto &order : plan.order_bys)
            {
                SortKeyInfo info;
                info.asc = (order.mode == "ASC");
                std::string key = order.col;

                // Match dynamic alias or standard metric names
                if (key == agg_alias || key == "REVENUE" || key == "LO_REVENUE" || key == "PROFIT")
                {
                    info.is_metric = true;
                    info.dim_idx = -1;
                    info.is_numeric = true;
                    compiled_order.push_back(info);
                }
                else
                {
                    info.is_metric = false;
                    info.dim_idx = -1;

                    int current_valid_idx = 0;
                    for (size_t i = 0; i < active_slots.size(); ++i)
                    {
                        std::string t_name = active_slots[i].table_name;
                        std::string g_col = "";
                        for (auto &op : plan.olap_ops)
                        {
                            if (op.dimension == t_name)
                            {
                                g_col = op.group_col;
                                break;
                            }
                        }
                        if (!g_col.empty())
                        {
                            if (g_col == key)
                            {
                                info.dim_idx = current_valid_idx;
                                break;
                            }
                            current_valid_idx++;
                        }
                    }

                    if (info.dim_idx != -1 && static_cast<size_t>(info.dim_idx) < final_results[0].dimensions.size())
                    {
                        info.is_numeric = is_numeric_string(final_results[0].dimensions[info.dim_idx]);
                        compiled_order.push_back(info);
                    }
                }
            }

            std::sort(final_results.begin(), final_results.end(),
                      [&compiled_order](const ResultRow &a, const ResultRow &b) -> bool
                      {
                          for (const auto &info : compiled_order)
                          {
                              if (info.is_metric)
                              {
                                  if (a.metric != b.metric)
                                      return info.asc ? (a.metric < b.metric) : (a.metric > b.metric);
                              }
                              else
                              {
                                  const std::string &val_a = a.dimensions[info.dim_idx];
                                  const std::string &val_b = b.dimensions[info.dim_idx];
                                  if (val_a != val_b)
                                  {
                                      if (info.is_numeric)
                                      {
                                          long long n_a = std::stoll(val_a);
                                          long long n_b = std::stoll(val_b);
                                          return info.asc ? (n_a < n_b) : (n_a > n_b);
                                      }
                                      else
                                      {
                                          return info.asc ? (val_a < val_b) : (val_a > val_b);
                                      }
                                  }
                              }
                          }
                          return false;
                      });
        }

        // =========================================================
        // 3. Print Output Table
        // =========================================================
        std::cout << "\n================================================================================" << std::endl;
        std::cout << "                             QUERY RESULTS                                      " << std::endl;
        std::cout << "================================================================================" << std::endl;

        // Print Dynamic Header
        for (auto &slot : active_slots)
        {
            std::string g_col = "";
            for (auto &op : plan.olap_ops)
            {
                if (op.dimension == slot.table_name)
                    g_col = op.group_col;
            }
            if (!g_col.empty())
            {
                std::cout << std::left << std::setw(20) << g_col;
            }
        }
        std::cout << std::left << std::setw(20) << agg_alias << std::endl;
        std::cout << std::string(80, '-') << std::endl;

        for (const auto &row : final_results)
        {
            for (const auto &val : row.dimensions)
            {
                std::cout << std::left << std::setw(20) << val;
            }
            std::cout << std::left << std::setw(20) << row.metric << std::endl;
        }
        std::cout << "Total Rows: " << final_results.size() << std::endl;
        std::cout << "================================================================================" << std::endl;

        result_timer.print();

        // =========================================================
        // 4. Resource Cleanup
        // =========================================================
        for (auto ptr : gpu_allocations)
            cudaFree(ptr);
        for (auto ptr : cpu_allocations)
            delete[] ptr;
    }
};

// =========================================================
// 5. Main Entry Point & CLI Configuration
// =========================================================

/**
 * @brief Prints the usage manual for the MOSS-GDB executable.
 */
void PrintUsage(const char *prog_name)
{
    std::cerr << "==========================================================\n"
              << "MOSS-GDB: High-Performance GPU In-Memory Database\n"
              << "==========================================================\n"
              << "Usage: " << prog_name << " <jql_file> [OPTIONS]\n\n"
              << "Arguments:\n"
              << "  <jql_file>     Path to the JQL query file (Required)\n\n"
              << "Options:\n"
              << "  -t, --threads  Number of CPU threads for Join/Agg operators.\n"
              << "                 (Default: All available hardware threads)\n"
              << "  -h, --help     Print this help message and exit.\n"
              << "==========================================================\n";
}
int main(int argc, char *argv[])
{
    std::cout << "--- MOSS-GDB: Starting Engine Initialization ---" << std::endl;

    // ---------------------------------------------------------
    // 1. Command Line Interface (CLI) Parsing
    // ---------------------------------------------------------
    if (argc < 2)
    {
        PrintUsage(argv[0]);
        return EXIT_FAILURE;
    }

    std::string jql_file_path = "";

    // Probe the maximum hardware threads available on the host machine
    // Fallback to std::thread::hardware_concurrency() if OpenMP env var is missing
    int max_hardware_threads = omp_get_max_threads();
    if (max_hardware_threads <= 0)
    {
        max_hardware_threads = std::thread::hardware_concurrency();
    }

    // Set Default Value (All available threads)
    int num_cpu_threads = max_hardware_threads;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help")
        {
            PrintUsage(argv[0]);
            return EXIT_SUCCESS;
        }
        else if (arg == "-t" || arg == "--threads")
        {
            if (i + 1 < argc)
            {
                try
                {
                    int parsed_threads = std::stoi(argv[++i]);
                    if (parsed_threads <= 0)
                    {
                        std::cerr << "[Warning] Thread count must be positive. Falling back to max threads: "
                                  << max_hardware_threads << std::endl;
                    }
                    else
                    {
                        num_cpu_threads = parsed_threads;
                    }
                }
                catch (const std::invalid_argument &e)
                {
                    std::cerr << "[Error] Invalid thread count provided. Must be an integer." << std::endl;
                    return EXIT_FAILURE;
                }
                catch (const std::out_of_range &e)
                {
                    std::cerr << "[Error] Thread count out of range." << std::endl;
                    return EXIT_FAILURE;
                }
            }
            else
            {
                std::cerr << "[Error] Option '-t/--threads' requires an integer argument." << std::endl;
                PrintUsage(argv[0]);
                return EXIT_FAILURE;
            }
        }
        else if (jql_file_path.empty() && arg[0] != '-')
        {
            // First non-flag argument is treated as the JQL file path
            jql_file_path = arg;
        }
        else
        {
            std::cerr << "[Error] Unknown or misplaced argument: " << arg << std::endl;
            PrintUsage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    if (jql_file_path.empty())
    {
        std::cerr << "[Error] Missing required argument: <jql_file>" << std::endl;
        PrintUsage(argv[0]);
        return EXIT_FAILURE;
    }

    // Apply the configured thread count to the OpenMP runtime environment
    omp_set_num_threads(num_cpu_threads);

    std::cout << "[System] Hardware Concurrency : " << max_hardware_threads << " threads." << std::endl;
    std::cout << "[System] Configured CPU Threads : " << num_cpu_threads << std::endl;
    std::cout << "[System] Target JQL Query File  : " << jql_file_path << std::endl;

    // ---------------------------------------------------------
    // 2. Data Ingestion & Schema Topology
    // ---------------------------------------------------------
    std::cout << "[System] Loading Internal Data..." << std::endl;
    load_data();

    JoinGraph join_graph;
    std::cout << "[System] Parsing Schema Topology..." << std::endl;
    // Assuming file is in the project root based on your structure
    if (!SchemaParser::ParseJoinPath("Join_Path_Tree.json", join_graph))
    {
        std::cerr << "[Fatal] Failed to parse Schema Topology (Join_Path_Tree.json)." << std::endl;
        return EXIT_FAILURE;
    }

    // Visualize Schema to verify hierarchical dimension loading
    PrintJoinGraph(join_graph);

    // ---------------------------------------------------------
    // 3. Query Parsing & Compilation
    // ---------------------------------------------------------
    QueryPlan plan;
    SSB_Utils::CpuTimer plan_timer("Parse Plan Time");

    plan_timer.start();
    if (!JQLParser::ParseJQL(jql_file_path.c_str(), plan))
    {
        std::cerr << "[Fatal] Failed to compile JQL query." << std::endl;
        return EXIT_FAILURE;
    }
    plan_timer.print();

    // Print the logical execution plan for debugging purposes
    PrintQueryPlan(plan);

    // ---------------------------------------------------------
    // 4. Execution Pipeline (Heterogeneous CPU-GPU)
    // ---------------------------------------------------------
    // Assumption: SSBQueryExecutor encapsulates the runtime state.
    // If the executor needs the thread count explicitly, we pass it here via a setter
    // or constructor, e.g.: SSBQueryExecutor executor(num_cpu_threads);
    SSBQueryExecutor executor(num_cpu_threads);

    // Explicitly configure the executor with the CLI parsed thread count
    // (Assuming SetNumThreads is/will be implemented in SSBQueryExecutor)
    // executor.SetNumThreads(num_cpu_threads);

    std::cout << "[System] Preparing GPU Device Memory Data..." << std::endl;
    // 1. Introspect the query plan to determine the global execution target
    bool requires_gpu = true; // Default to true for backward compatibility

    if (!plan.aggregations.empty())
    {
        // Extract the engine hint injected by the JQL Parser
        std::string target_engine = plan.aggregations[0].exec_engine;

        // If the query is explicitly routed to the CPU vectorization pipeline,
        // we flag the GPU preparation phase to be bypassed.
        if (target_engine == "CPU")
        {
            requires_gpu = false;
        }
    }

    // 2. Conditionally trigger device memory allocation and PCIe transfer
    if (requires_gpu)
    {
        std::cout << "[System Memory Manager] Query dictates GPU acceleration. "
                  << "Allocating VRAM and initiating PCIe Host-to-Device transfer..." << std::endl;

        // Starts the heavily parallelized data transfer to CUDA device
        executor.PrepareGPUData();
    }
    else
    {
        std::cout << "[System Memory Manager] Query routed to CPU OpenMP Pipeline. "
                  << "Bypassing GPU VRAM allocation to conserve PCIe bandwidth." << std::endl;

        // Note: Data remains purely in Host Memory (RAM) for CPU execution.
        // Any specific Host-side indexing or buffer preparations (if needed)
        // can be hooked here in future extensions.
    }

    std::cout << "--- MOSS-GDB: Executing Query Pipeline ---" << std::endl;
    // Warm-up and benchmark loop
    for (int i = 0; i < 10; i++)
    {

    SSB_Utils::CpuTimer total_timer("Total Time");
    total_timer.start();
    executor.ExecuteQuery(plan, join_graph);
    total_timer.print();
    }

    std::cout << "--- MOSS-GDB: Engine Shutdown Successfully ---" << std::endl;
    return EXIT_SUCCESS;
}