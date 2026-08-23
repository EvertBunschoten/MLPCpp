#define CATCH_CONFIG_MAIN
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "CBaseLoss.hpp"
#include "CMeanSquaredErrorLoss.hpp"
#include "variable_def.hpp"

#define REQUIRE_EQUAL_TOL(a, b, tol)                                           \
  REQUIRE(static_cast<double>(a) ==                                            \
          Catch::Approx(static_cast<double>(b)).margin(tol))

// Helper to construct PredictionResult with inputs and outputs for testing
// Note: Derivatives (jacobian/hessian) are omitted as they are not used by MSE
// loss
static MLPToolbox::PredictionResult make_pred(std::vector<mlpdouble> in,
                                              std::vector<mlpdouble> out) {
  MLPToolbox::PredictionResult p;
  p.inputs = std::move(in);
  p.outputs = std::move(out);
  p.jacobian = nullptr;
  p.hessian = nullptr;
  return p;
}

// ============================================================
//  CMeanSquaredErrorLoss Unit Tests
// ============================================================

TEST_CASE("CMeanSquaredErrorLoss empty predictions",
          "[CMeanSquaredErrorLoss]") {
  MLPToolbox::CMeanSquaredErrorLoss loss;
  std::vector<MLPToolbox::PredictionResult> preds;
  std::vector<std::vector<mlpdouble>> ref;

  mlpdouble val = loss.Evaluate(preds, ref);
  REQUIRE_EQUAL_TOL(val, 0.0, 1e-12);
}

TEST_CASE("CMeanSquaredErrorLoss size mismatch throws",
          "[CMeanSquaredErrorLoss]") {
  MLPToolbox::CMeanSquaredErrorLoss loss;
  std::vector<MLPToolbox::PredictionResult> preds(1);
  preds[0].outputs = {1.0};
  std::vector<std::vector<mlpdouble>> ref(2, {1.0});

  REQUIRE_THROWS_AS(loss.Evaluate(preds, ref), std::runtime_error);
}

TEST_CASE("CMeanSquaredErrorLoss inconsistent outputs throws",
          "[CMeanSquaredErrorLoss]") {
  MLPToolbox::CMeanSquaredErrorLoss loss;
  std::vector<MLPToolbox::PredictionResult> preds(2);
  preds[0].outputs = {1.0, 2.0};
  preds[1].outputs = {1.0};
  std::vector<std::vector<mlpdouble>> ref = {{1.0, 2.0}, {1.0, 2.0}};

  REQUIRE_THROWS_AS(loss.Evaluate(preds, ref), std::runtime_error);
}

TEST_CASE("CMeanSquaredErrorLoss ref_data size mismatch throws",
          "[CMeanSquaredErrorLoss]") {
  MLPToolbox::CMeanSquaredErrorLoss loss;
  std::vector<MLPToolbox::PredictionResult> preds(1);
  preds[0].outputs = {1.0, 2.0};
  std::vector<std::vector<mlpdouble>> ref = {{1.0}};

  REQUIRE_THROWS_AS(loss.Evaluate(preds, ref), std::runtime_error);
}

TEST_CASE("CMeanSquaredErrorLoss simple MSE calculation",
          "[CMeanSquaredErrorLoss]") {
  MLPToolbox::CMeanSquaredErrorLoss loss;
  std::vector<MLPToolbox::PredictionResult> preds = {
      make_pred({0.0}, {3.0, 4.0})};
  std::vector<std::vector<mlpdouble>> ref = {{1.0, 2.0}};

  // diff = {2, 2}, sq = {4, 4}, sum = 8, MSE = 8 / (1*2) = 4
  mlpdouble val = loss.Evaluate(preds, ref);
  REQUIRE_EQUAL_TOL(val, 4.0, 1e-12);
}

TEST_CASE("CMeanSquaredErrorLoss multiple points MSE",
          "[CMeanSquaredErrorLoss]") {
  MLPToolbox::CMeanSquaredErrorLoss loss;
  std::vector<MLPToolbox::PredictionResult> preds = {make_pred({0.0}, {2.0}),
                                                     make_pred({1.0}, {4.0})};
  std::vector<std::vector<mlpdouble>> ref = {{1.0}, {2.0}};

  // diffs: 1, 2 -> sq: 1, 4 -> sum: 5, MSE = 5 / (2*1) = 2.5
  mlpdouble val = loss.Evaluate(preds, ref);
  REQUIRE_EQUAL_TOL(val, 2.5, 1e-12);
}
