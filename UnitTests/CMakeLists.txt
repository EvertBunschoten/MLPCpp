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

FetchContent_Declare(
  CoDiPack
  GIT_REPOSITORY https://github.com/SciCompKL/CoDiPack
)

FetchContent_MakeAvailable(Catch2)
FetchContent_MakeAvailable(CoDiPack)

add_executable(test_gradients gradient_correctness.cpp)
add_executable(test_input_output_mapping input_output_mapping.cpp)
add_executable(test_activation_functions activation_function_tests.cpp)
add_executable(neural_network_tests CNeuralNetwork_unittests.cpp)
add_executable(test_scaler_functions scaler_function_tests.cpp)
add_executable(test_automatic_differentiation ad_tests.cpp)
add_executable(test_adam adam_tests.cpp)
add_executable(test_mean_squared_error_loss mean_squared_error_tests.cpp)
add_executable(test_gradient_annealer gradient_annealer_tests.cpp)
add_executable(test_physics_loss physics_loss_tests.cpp)
add_executable(test_mlp_trainer mlp_trainer_tests.cpp)


set(TEST_INCLUDE_DIRS
    ${MLPCPP_ROOT}/include
)

target_include_directories(test_gradients PRIVATE ${TEST_INCLUDE_DIRS})
target_include_directories(test_input_output_mapping PRIVATE ${TEST_INCLUDE_DIRS})
target_include_directories(test_activation_functions PRIVATE ${TEST_INCLUDE_DIRS})
target_include_directories(neural_network_tests PRIVATE ${TEST_INCLUDE_DIRS})
target_include_directories(test_scaler_functions PRIVATE ${TEST_INCLUDE_DIRS})
target_include_directories(test_automatic_differentiation PRIVATE ${TEST_INCLUDE_DIRS})
target_include_directories(test_adam PRIVATE ${TEST_INCLUDE_DIRS})
target_include_directories(test_mean_squared_error_loss PRIVATE ${TEST_INCLUDE_DIRS})
target_include_directories(test_gradient_annealer PRIVATE ${TEST_INCLUDE_DIRS})
target_include_directories(test_physics_loss PRIVATE ${TEST_INCLUDE_DIRS})
target_include_directories(test_mlp_trainer PRIVATE ${TEST_INCLUDE_DIRS})


target_link_libraries(test_gradients PRIVATE Catch2::Catch2WithMain)
target_link_libraries(test_input_output_mapping PRIVATE Catch2::Catch2WithMain)
target_link_libraries(test_activation_functions PRIVATE Catch2::Catch2WithMain)
target_link_libraries(neural_network_tests PRIVATE Catch2::Catch2WithMain)
target_link_libraries(test_scaler_functions PRIVATE Catch2::Catch2WithMain)
target_link_libraries(test_automatic_differentiation PRIVATE Catch2::Catch2WithMain PRIVATE CoDiPack)
target_link_libraries(test_mean_squared_error_loss PRIVATE Catch2::Catch2WithMain)
target_link_libraries(test_adam PRIVATE Catch2::Catch2WithMain)
target_link_libraries(test_gradient_annealer PRIVATE Catch2::Catch2WithMain)
target_link_libraries(test_physics_loss PRIVATE Catch2::Catch2WithMain PRIVATE CoDiPack)
target_link_libraries(test_mlp_trainer PRIVATE Catch2::Catch2WithMain PRIVATE CoDiPack)


# Register with CTest
include(CTest)
add_test(NAME test_gradients COMMAND test_gradients)
add_test(NAME test_input_output_mapping COMMAND test_input_output_mapping)
add_test(NAME test_activation_functions COMMAND test_activation_functions)
add_test(NAME neural_network_tests COMMAND neural_network_tests)
add_test(NAME test_scaler_functions COMMAND test_scaler_functions)
add_test(NAME test_automatic_differentiation COMMAND test_automatic_differentiation)
add_test(NAME test_mean_squared_error_loss COMMAND test_mean_squared_error_loss)
add_test(NAME test_adam COMMAND test_adam)
add_test(NAME test_gradient_annealer COMMAND test_gradient_annealer)
add_test(NAME test_physics_loss COMMAND test_physics_loss)
add_test(NAME test_mlp_trainer COMMAND test_mlp_trainer)
