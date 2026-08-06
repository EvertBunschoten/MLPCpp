cmake_minimum_required(VERSION 3.23)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Locate roots relative to this directory (UnitTests)
set(MLPCPP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")

Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.8.1 # or a later release
)

FetchContent_MakeAvailable(Catch2)

add_executable(test_gradients gradient_correctness.cpp)
add_executable(test_input_output_mapping input_output_mapping.cpp)
add_executable(test_activation_functions activation_function_tests.cpp)

set(TEST_INCLUDE_DIRS
    ${MLPCPP_ROOT}/include
)

target_include_directories(test_gradients PRIVATE ${TEST_INCLUDE_DIRS})
target_include_directories(test_input_output_mapping PRIVATE ${TEST_INCLUDE_DIRS})
target_include_directories(test_activation_functions PRIVATE ${TEST_INCLUDE_DIRS})
target_link_libraries(test_gradients PRIVATE Catch2::Catch2WithMain)
target_link_libraries(test_input_output_mapping PRIVATE Catch2::Catch2WithMain)
target_link_libraries(test_activation_functions PRIVATE Catch2::Catch2WithMain)
# Register with CTest
include(CTest)
add_test(NAME test_gradients COMMAND test_gradients)
add_test(NAME test_input_output_mapping COMMAND test_input_output_mapping)
add_test(NAME test_activation_functions COMMAND test_activation_functions)
