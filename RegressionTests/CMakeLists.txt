cmake_minimum_required(VERSION 3.23)

project(MLPCPP)

add_executable(run_unit_tests)
target_sources(run_unit_tests
PRIVATE 
unit_tests.cpp 
network_output_tests.cpp
input_output_mapping.cpp
)
