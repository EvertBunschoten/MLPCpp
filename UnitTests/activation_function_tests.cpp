#include "../include/ActivationFunctions.hpp"
#include "test_subjects.hpp"
#include <cmath>
#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#define REQUIRE_EQUAL_TOL(a, b, tol)                                           \
  REQUIRE_THAT(static_cast<double>(a),                                         \
               Catch::Matchers::WithinAbs(static_cast<double>(b), tol))

TEST_CASE("ReLu", "[CActivationFunction]") {
  MLPToolbox::Relu relufunction = MLPToolbox::Relu();

  REQUIRE(relufunction(-100) == 0.0);
  REQUIRE(relufunction(1.0) == 1.0);
  REQUIRE(relufunction(0.0) == 0.0);
}

TEST_CASE("ELu", "[CActivationFunction]") {
  MLPToolbox::Elu elufunction = MLPToolbox::Elu();

  REQUIRE(elufunction(-100) == -1.0);
  REQUIRE(elufunction(-0.5) == (exp(-0.5) - 1));
  REQUIRE(elufunction(0.0) == 0.0);
  REQUIRE(elufunction(1.0) == 1.0);
}

TEST_CASE("Sigmoid", "[CActivationFunction]") {
  MLPToolbox::Sigmoid sigmoidfunction = MLPToolbox::Sigmoid();

  REQUIRE_EQUAL_TOL(sigmoidfunction(-100), 0.0, 1e-8);
  REQUIRE_EQUAL_TOL(sigmoidfunction(100), 1.0, 1e-8);
  REQUIRE(sigmoidfunction(0.5) == (exp(0.5) / (1 + exp(0.5))));
}

TEST_CASE("Exponential", "[CActivationFunction]") {
  MLPToolbox::Exponential expfunction = MLPToolbox::Exponential();

  REQUIRE_EQUAL_TOL(expfunction(-100), 0.0, 1e-8);
  auto val_y = expfunction(0.8, true, true);
  auto jac = expfunction.GetJacobian();
  auto hess = expfunction.GetHessian();
  REQUIRE(val_y == jac);
  REQUIRE(val_y == hess);
}

TEST_CASE("Jacobians", "[CActivationFunction]") {

  std::random_device
      rd; // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dis(-1.0, 1.0);

  for (auto a = MLPToolbox::activation_function_map.begin();
       a != MLPToolbox::activation_function_map.end(); a++) {
    std::string function_name = a->first;
    auto f = MLPToolbox::RetrieveActivationFunction(function_name);

    double x_base = dis(gen);
    double delta_x_fd = 1e-6;
    auto y_base = f->operator()(x_base, true, false);
    double dydx_a = f->GetJacobian();

    auto y_plus = f->operator()(x_base + delta_x_fd);
    auto y_minus = f->operator()(x_base - delta_x_fd);
    double dydx_fd = (y_plus - y_minus) / (2 * delta_x_fd);
    REQUIRE_EQUAL_TOL(dydx_a, dydx_fd, 1e-5);
  }
}

TEST_CASE("Hessian", "[CActivationFunction]") {

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0, 1.0);

  for (auto a = MLPToolbox::activation_function_map.begin();
       a != MLPToolbox::activation_function_map.end(); a++) {
    std::string function_name = a->first;
    auto f = MLPToolbox::RetrieveActivationFunction(function_name);

    double x_base = dis(gen);
    double delta_x_fd = 1e-5;
    auto y_base = f->operator()(x_base, true, true);
    double d2ydx2_a = f->GetHessian();

    auto y_plus = f->operator()(x_base + delta_x_fd);
    auto y_minus = f->operator()(x_base - delta_x_fd);
    double d2ydx2_fd =
        (y_plus - 2 * y_base + y_minus) / (delta_x_fd * delta_x_fd);
    REQUIRE_EQUAL_TOL(d2ydx2_a, d2ydx2_fd, 1e-5);
  }
}
