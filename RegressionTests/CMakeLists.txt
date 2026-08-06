# cmake_minimum_required(VERSION 3.23)

# project(MLPCPP)

# add_executable(run_unit_tests)
# target_sources(run_unit_tests
# PRIVATE
# unit_tests.cpp
# network_output_tests.cpp
# input_output_mapping.cpp
# gradient_correctness.cpp
# )


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Locate roots relative to this directory (UnitTests)
set(CCATCH_PATH "/home/evert/SU2_Develop/externals/catch2")
set(MLPCPP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")


add_executable(test_gradients gradient_correctness.cpp)

set(TEST_INCLUDE_DIRS
    ${MLPCPP_ROOT}/include
    ${CCATCH_PATH}
)

target_include_directories(test_gradients PRIVATE ${TEST_INCLUDE_DIRS})

# Register with CTest
include(CTest)
add_test(NAME test_gradients COMMAND test_gradients)
