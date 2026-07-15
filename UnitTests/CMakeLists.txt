set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Locate roots relative to this directory (UnitTests)
set(CODI_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../externals/codi" CACHE PATH "Path to the CoDi library")
set(CCATCH_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../externals/catch2" CACHE PATH "Path to the Catch2 library")
set(MLPCPP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." CACHE PATH "Path to the MLP-C++ root directory")

# Define the test executables
add_executable(test_adam CAdam_tests.cpp)
add_executable(test_annealer CGradientAnnealer_tests.cpp)
add_executable(test_codi_jacobian test_codi_neuralnetwork.cpp)



# Common include directories for all tests in this subproject
set(TEST_INCLUDE_DIRS
    ${MLPCPP_ROOT}/include
    ${CODI_PATH}/include
    ${CCATCH_PATH}
)

target_include_directories(test_adam PRIVATE ${TEST_INCLUDE_DIRS})
target_include_directories(test_annealer PRIVATE ${TEST_INCLUDE_DIRS})
target_include_directories(test_codi_jacobian PRIVATE ${TEST_INCLUDE_DIRS})



# Register with CTest
include(CTest)
add_test(NAME test_adam COMMAND test_adam)
add_test(NAME test_annealer COMMAND test_annealer)
add_test(NAME test_codi_jacobian COMMAND test_codi_jacobian)

