cmake_minimum_required(VERSION 3.23)

project(MLPCppTestCase LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MLPCPP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")

include(FetchContent)

FetchContent_Declare(
    CoDiPack
    GIT_REPOSITORY https://github.com/SciCompKL/CoDiPack
)

FetchContent_MakeAvailable(CoDiPack)


add_executable(mlp_test_case mlp_test_case.cpp)
add_executable(lid_cavity_trainer LidCavity/LidCavityTrainer.cpp)


set(TEST_CASE_INCLUDE_DIRS
    ${MLPCPP_ROOT}/include
)

target_include_directories(mlp_test_case PRIVATE ${TEST_CASE_INCLUDE_DIRS})
target_include_directories(lid_cavity_trainer PRIVATE ${TEST_CASE_INCLUDE_DIRS})


target_link_libraries(mlp_test_case PRIVATE CoDiPack)
target_link_libraries(lid_cavity_trainer PRIVATE CoDiPack)
