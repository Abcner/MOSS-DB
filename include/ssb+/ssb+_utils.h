/**
 * @file ssb_utils.h
 * @brief Utilities for Star Schema Benchmark (SSB) Data Loading.
 * @author Assistant Engineer (Refactored for TPDS Standard)
 * @version 3.4 (Fixed Unreachable Allocation Bug)
 * @date 2026-01-07
 */

#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <memory>
#include <type_traits>
#include <sys/stat.h>
#include <cstdint> 

// 确保路径正确，如果还没移动文件，请尽快调整目录结构
#include "common/config.h"
#include "common/timer.h"

// =========================================================
// 1. System Configuration
// =========================================================

namespace SSBConfig {
    constexpr size_t NATION_COUNT = 25;
    constexpr size_t REGION_COUNT = 5;
    constexpr size_t BRAND1_COUNT = 1000;
    constexpr size_t CATEGORY_COUNT = 25;
    constexpr size_t CITY_C_COUNT = 250;
    constexpr size_t CITY_S_COUNT = 250;
    constexpr size_t DISCOUNT_COUNT = 11;
    constexpr size_t MFGR_COUNT  = 5;
    constexpr size_t QUANTITY_COUNT = 50;
    constexpr size_t YEAR_COUNT = 7;
    constexpr size_t YEARMONTH_COUNT = 7;
}

// =========================================================
// 2. Data Structures
// =========================================================

struct Column_store
{
    // Runtime Metadata
    size_t num_lineorder_rows = 0;
    size_t num_part_rows      = 0;
    size_t num_customer_rows  = 0;
    size_t num_supplier_rows  = 0;
    size_t num_date_rows      = 0;
    
    size_t num_nation_rows    = 0;
    size_t num_region_rows    = 0;
    size_t num_city_s_rows    = 0;
    size_t num_city_c_rows    = 0;
    size_t num_brand1_rows    = 0;
    size_t num_category_rows  = 0;
    size_t num_mfgr_rows      = 0;
    size_t num_year_rows      = 0;
    size_t num_yearmonth_rows      = 0;
    size_t num_discount_rows  = 0;
    size_t num_quantity_rows  = 0;

    // Fact Table
    int *lo_orderkey;
    int *lo_linenumber;
    int *lo_custkey;
    int *lo_partkey;
    int *lo_suppkey;
    int *lo_orderdate;
    std::string *lo_orderpriority;
    std::string *lo_shippriority;
    int *lo_extendedprice;
    int *lo_ordertotalprice;
    int8_t *lo_discount;   
    int *lo_revenue;
    int *lo_supplycost;
    int *lo_tax;
    int *lo_commitdate;
    std::string *lo_shipmode;
    
    int *date_fk;
    int8_t *discount_fk; 
    int8_t *quantity_fk; 

    // Dimensions
    int *p_partkey;
    int *p_brand1;

    int *s_suppkey;
    int *s_city;

    int *c_custkey;
    int *c_city;

    // Date
    int *d_datekey;
    int *d_yearmonth;
    int *d_year;
    int *d_weeknuminyear;

    // Hierarchy
    int *n_nationkey; std::string *n_name; int *n_regionkey;
    int *r_regionkey; std::string *r_name;
    std::string *si_name; int *id_city_s; int *si_nationkey;
    std::string *ci_name; int *id_city_c; int *ci_nationkey;
    std::string *b_name; int *id_brand1; int *b_category; int *b1_val;
    std::string *ca_name; int *id_category; int *ca_mfgrkey;
    std::string *mfgr_name; int *id_mfgr;
    std::string *ym_name; int *ym_num; int *id_ym; int *ym_yearkey;
    int *y_name; int *y_key;
    int *dim_discount_val; int *id_discount;
    int *dim_quantity_val; int *id_quantity;

    Column_store() { std::memset(this, 0, sizeof(Column_store)); }
};

extern Column_store column_store;
Column_store column_store;

// =========================================================
// 3. Loading Utilities
// =========================================================

namespace LoaderUtils {

    inline int fast_atoi(const char* str) {
        int val = 0; bool neg = false;
        if (!str) return 0;
        if (*str == '-') { neg = true; ++str; }
        while (*str >= '0' && *str <= '9') val = val * 10 + (*str++ - '0');
        return neg ? -val : val;
    }

    /**
     * @brief Smart Binary Loader.
     */
    template<typename T>
    T* load_binary_smart(const std::string& filename, size_t& out_count) {
        std::string full_path = std::string(BASE_DATA_PATH) + filename;
        std::ifstream file(full_path, std::ios::binary | std::ios::ate);
        
        if (!file.is_open()) {
            std::cerr << "[Error] Cannot open binary: " << full_path << std::endl;
            exit(1);
        }
        
        std::streamsize file_size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        size_t element_count = file_size / sizeof(T);
        T* data = new T[element_count];
        
        if (!file.read(reinterpret_cast<char*>(data), file_size)) {
            delete[] data;
            throw std::runtime_error("Failed to read: " + filename);
        }
        
        file.close();
        out_count = element_count;
        std::cout << "[Info] Loaded binary " << filename << ": " << element_count << " rows." << std::endl;
        return data;
    }

    void verify_length(size_t actual, size_t expected, const std::string& name) {
        if (actual != expected) {
            std::cerr << "[Fatal] Length mismatch " << name << ": expected " << expected << ", got " << actual << std::endl;
            exit(1);
        }
    }
}

// =========================================================
// 4. Main Loading Logic
// =========================================================

void load_data() {
    using namespace LoaderUtils;
    std::cout << "--- Starting Data Load ---" << std::endl;
    
    // [Fix] Redesigned Macros to separate Allocation from Parsing
    // -----------------------------------------------------------
    
    // Macro 1: Prepare (Count rows -> Provide 'count' variable -> Scope for allocation)
    #define CSV_PREPARE_LOAD(FILENAME, ROW_COUNTER_REF) \
        { \
            std::string _path = std::string(BASE_DATA_PATH) + FILENAME; \
            FILE* _file = fopen(_path.c_str(), "r"); \
            if(!_file) { std::cerr << "[Error] Failed to open " << FILENAME << std::endl; exit(1); } \
            size_t _line_count = 0; \
            char _buf[1024]; \
            while(fgets(_buf, sizeof(_buf), _file)) _line_count++; \
            rewind(_file); \
            ROW_COUNTER_REF = _line_count; \
            std::cout << "[Info] Detected " << FILENAME << ": " << _line_count << " rows." << std::endl; \
            size_t count = _line_count; /* User code runs after this */

    // Macro 2: Start Parse (Ends allocation scope -> Starts loop -> Starts switch)
    #define CSV_PARSE_START \
            size_t row = 0; \
            while(fgets(_buf, sizeof(_buf), _file) && row < count) { \
                _buf[strcspn(_buf, "\n")] = 0; \
                char* token = std::strtok(_buf, "|"); \
                int col = 0; \
                while(token) { \
                    switch(col) {

    // Macro 3: End Parse (Ends switch -> Ends loop -> Closes file)
    #define CSV_PARSE_END \
                    } \
                    token = std::strtok(NULL, "|"); \
                    col++; \
                } \
                row++; \
            } \
            fclose(_file); \
        }

    // --- Nation ---
    CSV_PREPARE_LOAD("nation_olap.csv", column_store.num_nation_rows)
        // [Safe] Allocation happens strictly before parsing loop
        column_store.n_nationkey = new int[count];
        column_store.n_name = new std::string[count];
        column_store.n_regionkey = new int[count];
    CSV_PARSE_START
        case 0: column_store.n_nationkey[row] = fast_atoi(token) + 1; break;
        case 1: column_store.n_name[row] = token; break;
        case 2: column_store.n_regionkey[row] = fast_atoi(token) + 1; break;
    CSV_PARSE_END

    // --- Region ---
    CSV_PREPARE_LOAD("region_olap.csv", column_store.num_region_rows)
        column_store.r_regionkey = new int[count];
        column_store.r_name = new std::string[count];
    CSV_PARSE_START
        case 0: column_store.r_regionkey[row] = fast_atoi(token) + 1; break;
        case 1: column_store.r_name[row] = token; break;
    CSV_PARSE_END

    // --- City_S ---
    CSV_PREPARE_LOAD("city_S.csv", column_store.num_city_s_rows)
        column_store.si_name = new std::string[count];
        column_store.id_city_s = new int[count];
        column_store.si_nationkey = new int[count];
    CSV_PARSE_START
        case 0: column_store.si_name[row] = token; break;
        case 1: column_store.id_city_s[row] = fast_atoi(token); break;
        case 2: column_store.si_nationkey[row] = fast_atoi(token) + 1; break;
    CSV_PARSE_END

    // --- City_C ---
    CSV_PREPARE_LOAD("city_C.csv", column_store.num_city_c_rows)
        column_store.ci_name = new std::string[count];
        column_store.id_city_c = new int[count];
        column_store.ci_nationkey = new int[count];
    CSV_PARSE_START
        case 0: column_store.ci_name[row] = token; break;
        case 1: column_store.id_city_c[row] = fast_atoi(token); break;
        case 2: column_store.ci_nationkey[row] = fast_atoi(token) + 1; break;
    CSV_PARSE_END

    // --- Brand1 ---
    CSV_PREPARE_LOAD("brand1.csv", column_store.num_brand1_rows)
        column_store.b_name = new std::string[count];
        column_store.id_brand1 = new int[count];
        column_store.b_category = new int[count];
        column_store.b1_val = new int[count];
    CSV_PARSE_START
        case 0: 
            column_store.b_name[row] = token; 
            column_store.b1_val[row] = fast_atoi(token + 5); 
            break;
        case 1: column_store.id_brand1[row] = fast_atoi(token); break;
        case 2: column_store.b_category[row] = fast_atoi(token); break;
    CSV_PARSE_END

    // --- Category ---
    CSV_PREPARE_LOAD("category.csv", column_store.num_category_rows)
        column_store.ca_name = new std::string[count];
        column_store.id_category = new int[count];
        column_store.ca_mfgrkey = new int[count];
    CSV_PARSE_START
        case 0: column_store.ca_name[row] = token; break;
        case 1: column_store.id_category[row] = fast_atoi(token); break;
        case 2: column_store.ca_mfgrkey[row] = fast_atoi(token); break;
    CSV_PARSE_END

    // --- MFGR ---
    CSV_PREPARE_LOAD("mfgr.csv", column_store.num_mfgr_rows)
        column_store.mfgr_name = new std::string[count];
        column_store.id_mfgr = new int[count];
    CSV_PARSE_START
        case 0: column_store.mfgr_name[row] = token; break;
        case 1: column_store.id_mfgr[row] = fast_atoi(token); break;
    CSV_PARSE_END
    // --- Yearmonth ---
    CSV_PREPARE_LOAD("yearmonth.csv", column_store.num_yearmonth_rows)
        column_store.ym_name = new std::string[count];
        column_store.ym_num = new int[count];
        column_store.id_ym = new int[count];
        column_store.ym_yearkey = new int[count];
    CSV_PARSE_START
        case 0: column_store.ym_name[row] = token; break;
        case 1: column_store.ym_num[row] = fast_atoi(token); break;
        case 2: column_store.id_ym[row] = fast_atoi(token); break;
        case 3: column_store.ym_yearkey[row] = fast_atoi(token); break;
    CSV_PARSE_END
    // --- Year ---
    CSV_PREPARE_LOAD("year.csv", column_store.num_year_rows)
        column_store.y_name = new int[count];
        column_store.y_key = new int[count];
    CSV_PARSE_START
        case 0: column_store.y_name[row] = fast_atoi(token); break;
        case 1: column_store.y_key[row] = fast_atoi(token); break;
    CSV_PARSE_END
    
    // --- Discount ---
    CSV_PREPARE_LOAD("discount.csv", column_store.num_discount_rows)
        column_store.dim_discount_val = new int[count];
        column_store.id_discount = new int[count];
    CSV_PARSE_START
        case 0: column_store.dim_discount_val[row] = fast_atoi(token); break;
        case 1: column_store.id_discount[row] = fast_atoi(token); break;
    CSV_PARSE_END

    // --- Quantity ---
    CSV_PREPARE_LOAD("quantity.csv", column_store.num_quantity_rows)
        column_store.dim_quantity_val = new int[count];
        column_store.id_quantity = new int[count];
    CSV_PARSE_START
        case 0: column_store.dim_quantity_val[row] = fast_atoi(token); break;
        case 1: column_store.id_quantity[row] = fast_atoi(token); break;
    CSV_PARSE_END

    // --- Date ---
    CSV_PREPARE_LOAD("date_olap.csv", column_store.num_date_rows)
        column_store.d_datekey = new int[count];
        // column_store.d_date = new std::string[count];
        // column_store.d_dayofweek = new std::string[count];
        // column_store.d_month = new std::string[count];
        // column_store.d_yearmonthnum = new int[count];
        column_store.d_yearmonth = new int[count];
        column_store.d_year = new int[count];
        column_store.d_weeknuminyear = new int[count];
        // column_store.dic_d_yearmonth_len = 0;
        // column_store.dic_d_yearmonth = new std::string[count];
        // column_store.d_yearmonth_val = new int[count];
        // column_store.d_daynuminweek = new int[count];
        // column_store.d_daynuminmonth = new int[count];
        // column_store.d_daynuminyear = new int[count];
        // column_store.d_monthnuminyear = new int[count];
        // column_store.d_weeknuminyear = new int[count];
        // column_store.d_sellingseason = new std::string[count];
        // column_store.d_lastdayinweekfl = new std::string[count];
        // column_store.d_lastdayinmonthfl = new std::string[count];
        // column_store.d_holidayfl = new std::string[count];
        // column_store.d_weekdayfl = new std::string[count];
        // column_store.d_PK = new int[count];
        // column_store.d_y_fk = new int[count];
    CSV_PARSE_START
        case 0: column_store.d_datekey[row] = fast_atoi(token); break;
        case 1: column_store.d_yearmonth[row] = fast_atoi(token); break;
        case 2: column_store.d_year[row] = fast_atoi(token); break;
        case 3: column_store.d_weeknuminyear[row] = fast_atoi(token); break;
        // case 1: column_store.d_date[row] = token; break;
        // case 2: column_store.d_dayofweek[row] = token; break;
        // case 3: column_store.d_month[row] = token; break;
        // case 4: column_store.d_yearmonthnum[row] = fast_atoi(token); break;
        // case 5: column_store.d_yearmonth[row] = token; 
        //         for (i = 0; i < column_store.dic_d_yearmonth_len; i++)
        //         {
        //             if (column_store.dic_d_yearmonth[i].compare(token) == 0)
        //             {
        //                 column_store.d_yearmonth_val[row] = i;
        //                 break;
        //             }
        //         }
        //         if (i == column_store.dic_d_yearmonth_len)
        //         {
        //             column_store.dic_d_yearmonth[i] = token;
        //             column_store.d_yearmonth_val[row] = i;
        //             column_store.dic_d_yearmonth_len++;
        //         }
        //         break;
        
        // case 6: column_store.d_daynuminweek[row] = fast_atoi(token); break;
        // case 7: column_store.d_daynuminmonth[row] = fast_atoi(token); break;
        // case 8: column_store.d_daynuminyear[row] = fast_atoi(token); break;
        // case 9: column_store.d_monthnuminyear[row] = fast_atoi(token); break;
        // case 10: column_store.d_weeknuminyear[row] = fast_atoi(token); break;
        // case 11: column_store.d_sellingseason[row] = token; break;
        // case 12: column_store.d_lastdayinweekfl[row] = token; break;
        // case 13: column_store.d_lastdayinmonthfl[row] = token; break;
        // case 14: column_store.d_holidayfl[row] = token; break;
        // case 15: column_store.d_weekdayfl[row] = token; break;
        // case 16: column_store.d_PK[row] = fast_atoi(token); break;
        // case 17: column_store.d_y_fk[row] = fast_atoi(token); break;
    CSV_PARSE_END

    // --- Customer ---
    CSV_PREPARE_LOAD("customer_olap.csv", column_store.num_customer_rows)
        column_store.c_custkey = new int[count];
        column_store.c_city = new int[count];
    CSV_PARSE_START
        case 0: column_store.c_custkey[row] = fast_atoi(token); break;
        case 1: column_store.c_city[row] = fast_atoi(token); break;
    CSV_PARSE_END

    // --- Supplier ---
    CSV_PREPARE_LOAD("supplier_olap.csv", column_store.num_supplier_rows)
        column_store.s_suppkey = new int[count];
        column_store.s_city = new int[count];
    CSV_PARSE_START
        case 0: column_store.s_suppkey[row] = fast_atoi(token); break;
        case 1: column_store.s_city[row] = fast_atoi(token); break;
    CSV_PARSE_END

    // --- Part ---
    CSV_PREPARE_LOAD("part_olap.csv", column_store.num_part_rows)
        column_store.p_partkey = new int[count];
        column_store.p_brand1 = new int[count];
    CSV_PARSE_START
        case 0: column_store.p_partkey[row] = fast_atoi(token); break;
        case 1: column_store.p_brand1[row] = fast_atoi(token); break;
    CSV_PARSE_END

    // --- Fact Table (Binary) ---
    {
        size_t len = 0;
        column_store.lo_custkey = load_binary_smart<int>("lineorder.lo_custkey", len);
        column_store.num_lineorder_rows = len;

        // Example loading other columns
        size_t tmp_len;
        column_store.lo_discount = load_binary_smart<int8_t>("lineorder.lo_discount", tmp_len);
        verify_length(tmp_len, len, "lineorder_lo_discount");
        
        column_store.date_fk = load_binary_smart<int>("lineorder.date_FK", tmp_len);
        verify_length(tmp_len, len, "lineorder_date_FK");
        
        column_store.discount_fk = load_binary_smart<int8_t>("lineorder.discount_FK", tmp_len);
        verify_length(tmp_len, len, "lineorder_discount_FK");

        column_store.lo_extendedprice = load_binary_smart<int>("lineorder.lo_extendedprice", tmp_len);
        verify_length(tmp_len, len, "lineorder_lo_extendedprice");

        column_store.lo_partkey = load_binary_smart<int>("lineorder.lo_partkey", tmp_len);
        verify_length(tmp_len, len, "lineorder_lo_partkey");

        column_store.lo_revenue = load_binary_smart<int>("lineorder.lo_revenue", tmp_len);
        verify_length(tmp_len, len, "lineorder_lo_revenue");

        column_store.lo_suppkey = load_binary_smart<int>("lineorder.lo_suppkey", tmp_len);
        verify_length(tmp_len, len, "lineorder_lo_suppkey");

        column_store.lo_supplycost = load_binary_smart<int>("lineorder.lo_supplycost", tmp_len);
        verify_length(tmp_len, len, "lineorder_lo_supplycost");

        column_store.quantity_fk = load_binary_smart<int8_t>("lineorder.quantity_FK", tmp_len);
        verify_length(tmp_len, len, "lineorder_quantity_FK");
    }
    
    std::cout << "--- Load Complete ---" << std::endl;
    std::cout << "Fact Rows: " << column_store.num_lineorder_rows << std::endl;
}