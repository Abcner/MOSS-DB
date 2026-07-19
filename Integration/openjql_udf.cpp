#include "postgres.h"
#include "fmgr.h"
#include <sys/time.h>
#include <string>
#include <cstring>

extern "C" {
    void moss_backend_load();
    
    const char* moss_backend_execute(const char* query_json, double* parse_ms, double* gpu_ms, double* analysis_ms);

    PG_MODULE_MAGIC;

    PG_FUNCTION_INFO_V1(moss_load_func);
    Datum moss_load_func(PG_FUNCTION_ARGS) {
        struct timeval t0, t1;
        gettimeofday(&t0, NULL);

        moss_backend_load();

        gettimeofday(&t1, NULL);
        double load_time = ((t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_usec - t0.tv_usec) / 1000.0);

        std::string res = "MOSS-DB Engine successfully loaded into Memory & GPU!\nInitialization Time: " + std::to_string(load_time) + " ms.";
        
        text *result_text = (text *)palloc(res.length() + VARHDRSZ);
        SET_VARSIZE(result_text, res.length() + VARHDRSZ);
        memcpy(VARDATA(result_text), res.c_str(), res.length());
        
        PG_RETURN_TEXT_P(result_text);
    }
    
    PG_FUNCTION_INFO_V1(moss_query_func);
    Datum moss_query_func(PG_FUNCTION_ARGS) {
        struct timeval t0, t1;
        gettimeofday(&t0, NULL);

        text *jql_text = PG_GETARG_TEXT_PP(0);
        size_t jql_len = VARSIZE_ANY_EXHDR(jql_text);
        std::string jql_string(VARDATA_ANY(jql_text), jql_len);

        bool debug_mode = false; 
        if (PG_NARGS() == 2) {
            text *mode_text = PG_GETARG_TEXT_PP(1);
            size_t mode_len = VARSIZE_ANY_EXHDR(mode_text);
            std::string mode_string(VARDATA_ANY(mode_text), mode_len);
            if (mode_string == "debug") {
                debug_mode = true;
            }
        }

        double parse_ms = 0.0, gpu_ms = 0.0, analysis_ms = 0.0;
        const char* engine_output = moss_backend_execute(jql_string.c_str(), &parse_ms, &gpu_ms, &analysis_ms);

        gettimeofday(&t1, NULL);
        double total_udf_ms = ((t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_usec - t0.tv_usec) / 1000.0);
        
        double pure_time_ms = parse_ms + gpu_ms + analysis_ms;

        std::string final_display = "\n"; 
        final_display += engine_output;

        if (debug_mode) {
            final_display += "================================================================================\n";
            final_display += "        ⚙️  Query Performance Summary        \n";
            final_display += "================================================================================\n";
            final_display += " [Time] 1. Plan Parsing Time (CPU side): " + std::to_string(parse_ms) + " ms\n";
            final_display += " [Time] 2. GPU Exec Time (GPU side): " + std::to_string(gpu_ms) + " ms\n";
            final_display += " [Time] 3. Result Analy Time (CPU side): " + std::to_string(analysis_ms) + " ms\n";
            final_display += " -------------------------------------------------------------------------------\n";
            final_display += " [Time] Total Pure Exec Time: " + std::to_string(pure_time_ms) + " ms\n";
            final_display += "================================================================================\n";
        } else {
            final_display += "================================================================================\n";
            final_display += " [Result Analyze Time (CPU side)]: " + std::to_string(analysis_ms) + " ms\n";
            final_display += "================================================================================\n";
        }

        text *result_text = (text *)palloc(final_display.length() + VARHDRSZ);
        SET_VARSIZE(result_text, final_display.length() + VARHDRSZ);
        memcpy(VARDATA(result_text), final_display.c_str(), final_display.length());

        PG_RETURN_TEXT_P(result_text);
    }
}
