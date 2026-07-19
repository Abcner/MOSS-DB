#include <iostream>
#include <fstream>
#include <chrono>
#include <sstream>
#include <string>


static bool g_is_loaded = false;
static JoinGraph* g_join_graph = nullptr;
static SSBQueryExecutor* g_executor = nullptr;
static std::string g_last_result = "";

extern "C" {

void moss_backend_load() {
    if (g_is_loaded) return; 
    
    std::cout << "[MOSS-DB] Native Engine Initialization..." << std::endl;
    
    g_join_graph = new JoinGraph();
    
    std::string join_path = "/home/xxx/workspace/moss-db_test/MOSS-DB/jql/Join_Path_Tree.json";
    SchemaParser::ParseJoinPath(join_path, *g_join_graph);
    
    g_executor = new SSBQueryExecutor();
    g_executor->PrepareGPUData();
    
    g_is_loaded = true;
    std::cout << "[MOSS-DB] Native Engine Loaded to GPU Memory." << std::endl;
}

const char* moss_backend_execute(const char* filename_c_str, double* parse_ms, double* gpu_ms, double* analysis_ms) {
    if (!g_is_loaded) {
        g_last_result = "Error: Engine not loaded. Run SELECT openjql_load() first.";
        return g_last_result.c_str();
    }
    
    std::string filename(filename_c_str); 
    
    
    std::string absolute_path = "/home/xxx/workspace/jql/" + filename;
    std::ifstream check_file(absolute_path);
    if (!check_file.good()) {
        g_last_result = "[Error] File not found or permission denied: " + absolute_path;
        return g_last_result.c_str();
    }
    check_file.close();

    std::string hack_filename = "../../../../../../../../../../../../../../../../home/xxx/workspace/jql/" + filename;

    auto start_parse = std::chrono::high_resolution_clock::now();
    QueryPlan plan;
    
    JQLParser::ParseJQL(hack_filename, plan);
    
    auto end_parse = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> parse_time = end_parse - start_parse;
    
    std::stringstream buffer;
    std::streambuf* old_cout = std::cout.rdbuf(buffer.rdbuf());
    
    try {
        g_executor->ExecuteQuery(plan, *g_join_graph);
    } catch (const std::exception& e) {
        std::cout.rdbuf(old_cout);
        g_last_result = std::string("\n[MOSS-DB Execution Error]: ") + e.what() + "\n";
        return g_last_result.c_str();
    } catch (...) {
        std::cout.rdbuf(old_cout);
        g_last_result = "\n[MOSS-DB Execution Error]: Unknown C++ Exception!\n";
        return g_last_result.c_str();
    }
    
    std::cout.rdbuf(old_cout);
    
    *parse_ms = parse_time.count();
    *gpu_ms = g_executor->last_gpu_time_ms;
    *analysis_ms = g_executor->last_analysis_time_ms;
    
    g_last_result = buffer.str();
    return g_last_result.c_str();
}

} // end extern "C"
