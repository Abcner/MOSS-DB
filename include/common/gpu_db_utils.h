/**
 * @file gpu_db_utils.h
 * @brief General Purpose String and System Utilities for GPU DB.
 * @author Assistant Engineer (Refactored for TPDS Standard)
 * @version 1.0
 * @date 2026-01-14
 * * @note Contains lightweight, header-only utility functions used across
 * the Host-side engine code (e.g., JQL parsing, Metadata handling).
 */

#pragma once

#include <string>
#include <algorithm>
#include <cctype>
#include <vector>

namespace MOSS_DB {
namespace Utils {

    /**
     * @brief Converts a standard string to lowercase.
     * * @note Marked 'inline' to permit definition in header file without 
     * causing ODR (One Definition Rule) violations during linking.
     * Uses RVO (Return Value Optimization) to minimize copying.
     * * @param s Input string (passed by const reference to avoid unnecessary copy).
     * @return std::string New string with all characters converted to lowercase.
     */
    inline std::string StrToLower(const std::string& s) {
        std::string result = s;
        // std::tolower requires unsigned char cast to avoid Undefined Behavior 
        // with negative char values (e.g., extended ASCII).
        std::transform(result.begin(), result.end(), result.begin(),
            [](unsigned char c) { return std::tolower(c); });
        return result;
    }

    /**
     * @brief Case-insensitive string comparison helper.
     * Useful for checking JQL keywords like "SELECT", "select", "Select".
     */
    inline bool StrEqualsIgnoreCase(const std::string& a, const std::string& b) {
        return StrToLower(a) == StrToLower(b);
    }

} // namespace Utils
} // namespace MOSS_DB