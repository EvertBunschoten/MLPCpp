set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Locate roots relative to this directory (UnitTests)
set(SU2_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
set(MLPCPP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")

# Define the test executables
add_executable(test_adam CAdam_tests.cpp)
add_executable(test_annealer CGradientAnnealer_tests.cpp)

# Common include directories for all tests in this subproject
set(TEST_INCLUDE_DIRS
    ${MLPCPP_ROOT}/include
    ${SU2_ROOT}/Common/include
    ${SU2_ROOT}/externals/codi/include
    ${SU2_ROOT}/externals/catch2 
)

target_include_directories(test_adam PRIVATE ${TEST_INCLUDE_DIRS})
target_include_directories(test_annealer PRIVATE ${TEST_INCLUDE_DIRS})

# Register with CTest
include(CTest)
add_test(NAME test_adam COMMAND test_adam)
add_test(NAME test_annealer COMMAND test_annealer)