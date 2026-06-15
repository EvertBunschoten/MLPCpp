# Locate roots relative to this directory
set(SU2_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(MLPCPP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")

# Define the test executable
add_executable(test_adam CAdam_tests.cpp)

# Add required include directories
target_include_directories(test_adam PRIVATE
    ${MLPCPP_ROOT}/include
    ${SU2_ROOT}/Common/include
    ${SU2_ROOT}/externals/codi/include
    ${SU2_ROOT}/externals/catch2
)


# Register with CTest
add_test(NAME test_adam COMMAND test_adam)