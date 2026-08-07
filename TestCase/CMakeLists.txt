cmake_minimum_required(VERSION 3.23)

project(MLPCppTestCase LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
    CoDiPack
    GIT_REPOSITORY https://github.com/SciCompKL/CoDiPack
)

FetchContent_MakeAvailable(CoDiPack)

add_executable(mlp_test_case
    mlp_test_case.cpp
)

target_include_directories(mlp_test_case PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(mlp_test_case PRIVATE
    CoDiPack
)