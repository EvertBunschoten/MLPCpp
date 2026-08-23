/*! \brief Unit tests for the correct functioning of the supported scaling
 * functions. */
#include "../include/ScalarFunctions.hpp"
#include "test_subjects.hpp"
#include <cmath>
#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

TEST_CASE("Min-max scaler test", "[ScalerFunction]") {
  MLPToolbox::MinMaxScaler scaler = MLPToolbox::MinMaxScaler(2);
  scaler.SetScaling(0, 0, 1);
  scaler.SetScaling(1, -1, 0);

  double val_dim = 0.3;
  double val_norm = scaler.Normalize(val_dim, 0);
  REQUIRE(val_dim == val_norm);

  val_norm = scaler.Normalize(val_dim, 1);
  double val_norm_ref = (val_dim + 1.0);
  REQUIRE(val_norm == val_norm_ref);

  double vals_input[] = {0.5, -0.3};
  auto dist = scaler.Distance(vals_input);
  REQUIRE(dist == 0);

  vals_input[0] = 10;
  auto dist_outside = scaler.Distance(vals_input);
  REQUIRE(dist_outside > 0);
}

TEST_CASE("Standard scaler test", "[ScalerFunction]") {
  MLPToolbox::StandardScaler scaler = MLPToolbox::StandardScaler(2);
  scaler.SetScaling(0, 0.0, 1.0);
  scaler.SetScaling(1, 0.0, 2.0);

  double val_dim = 0.6;
  double val_norm = scaler.Normalize(val_dim, 0);
  REQUIRE(val_dim == val_norm);

  val_norm = scaler.Normalize(val_dim, 1);
  REQUIRE(val_dim == 2 * val_norm);
}

TEST_CASE("Normalization and de-normalization", "[ScalerFunction]") {
  /*! \brief For all scaler functions, check if de-normalized output equals the
   * original input. */

  size_t n_inputs{3};

  for (auto scalerfunction : MLPToolbox::scaling_map) {
    auto f = MLPToolbox::RetrieveScalerFunction(scalerfunction.first, n_inputs);
    auto r = RandomInputs(n_inputs);
    auto val_dim = r[0];
    auto norm = f->Normalize(val_dim, 0);
    auto denorm = f->Dimensionalize(norm, 0);
    REQUIRE(val_dim == denorm);
  }
}
