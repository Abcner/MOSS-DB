-- 1. Register Engine Initialization Function
CREATE OR REPLACE FUNCTION openjql_load() 
RETURNS TEXT 
AS 'moss_engine.so', 'moss_load_func' 
LANGUAGE C STRICT VOLATILE;

-- 2. Register Query Execution Function (Standard Mode)
CREATE OR REPLACE FUNCTION openjql(query_file TEXT) 
RETURNS TEXT 
AS 'moss_engine.so', 'moss_query_func' 
LANGUAGE C STRICT VOLATILE;

-- 3. Register Query Execution Function (Debug/Profiling Mode)
CREATE OR REPLACE FUNCTION openjql(query_file TEXT, mode TEXT) 
RETURNS TEXT 
AS 'moss_engine.so', 'moss_query_func' 
LANGUAGE C STRICT VOLATILE;