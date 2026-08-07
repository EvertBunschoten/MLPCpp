/*! \brief Unit tests that evaluate whether Jacobians and Hessians are correctly
 * calculated. */
#include "../include/CNeuralNetwork.hpp"
#include "test_subjects.hpp"
#include <iostream>
#include <string>
#include <vector>
#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

TEST_CASE("Jacobian correctness", "[CNeuralNetwork]") {
  /*! \brief Compare analytically evaluated Jacobian with Jacobian calculated
   * through finite-difference approximation. */

  std::vector<std::string> input_names = {"a", "b", "c"};
  std::vector<std::string> output_names = {"x", "y", "z"};
  MLPToolbox::CNeuralNetwork *mlp =
      CreateRandomNetwork(input_names, output_names);
  mlp->CalcJacobian(true);

  /* Select random input and output components for which to evaluate the
   * Jacobian */
  const size_t iIn = (rand() % mlp->GetnInputs());
  const size_t iOut = (rand() % mlp->GetnOutputs());

  /* Analytically evaluate the Jacobian. */
  const auto inputs_base = RandomInputs(mlp->GetnInputs());
  mlp->SetInput(inputs_base);
  mlp->Predict();
  const double Jac_analytical = mlp->GetJacobian(iOut, iIn);

  /* Approximate the Jacobian with central finite-differneces. */
  std::vector<double> inputs_plus = inputs_base, inputs_minus = inputs_base;
  const double delta_inp{1e-6};
  inputs_plus[iIn] += delta_inp;
  inputs_minus[iIn] -= delta_inp;
  mlp->SetInput(inputs_plus);
  mlp->Predict();
  const double outp_plus = mlp->GetOutput(iOut);
  mlp->SetInput(inputs_minus);
  mlp->Predict();
  const double outp_minus = mlp->GetOutput(iOut);
  const double Jac_FD = (outp_plus - outp_minus) / (2 * delta_inp);

  REQUIRE_EQUAL_TOL(Jac_analytical, Jac_FD, 1e-6);
  delete mlp;
}

TEST_CASE("Hessian correctness", "[CNeuralNetwork]") {
  /*! \brief Compare analytically evaluated Hessian with Hessian calculated
   * through finite-difference approximation. */

  std::vector<std::string> input_names = {"a", "b", "c"};
  std::vector<std::string> output_names = {"x", "y", "z"};
  MLPToolbox::CNeuralNetwork *mlp =
      CreateRandomNetwork(input_names, output_names);
  mlp->CalcJacobian(true);
  mlp->CalcHessian(true);

  /* Evaluate the Hessian of iOut w.r.t iIn and jIn. */
  const size_t iIn = (rand() % mlp->GetnInputs());
  const size_t jIn = (rand() % mlp->GetnInputs());
  const size_t iOut = (rand() % mlp->GetnOutputs());

  /* Analytically evaluate the Hessian. */
  const auto inputs_base = RandomInputs(mlp->GetnInputs());
  mlp->SetInput(inputs_base);
  mlp->Predict();
  const double Hes_analytical = mlp->GetHessian(iOut, iIn, jIn);

  /* Approximate the Hessian with central finite-differences. */
  std::vector<double> inputs_plus = inputs_base, inputs_minus = inputs_base;
  const double delta_inp{1e-6};
  inputs_plus[iIn] += delta_inp;
  inputs_minus[iIn] -= delta_inp;
  mlp->SetInput(inputs_plus);
  mlp->Predict();
  const double jac_outp_plus = mlp->GetJacobian(iOut, jIn);
  mlp->SetInput(inputs_minus);
  mlp->Predict();
  const double jac_outp_minus = mlp->GetJacobian(iOut, jIn);
  const double Hes_FD = (jac_outp_plus - jac_outp_minus) / (2 * delta_inp);

  REQUIRE_EQUAL_TOL(Hes_analytical, Hes_FD, 1e-6);

  /* Check if cross terms of the Hessian are the same */
  for (auto iInput = 0; iInput < mlp->GetnInputs(); iInput++) {
    REQUIRE_EQUAL_TOL(mlp->GetHessian(iOut, iInput, jIn),
                      mlp->GetHessian(iOut, jIn, iInput), 1e-6);
  }
  delete mlp;
}
