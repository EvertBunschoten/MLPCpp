#include <vector>
#include <string>
#include <iostream>
#include "test_subjects.hpp"
#include "../include/CNeuralNetwork.hpp"
#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#define REQUIRE_EQUAL_TOL(a, b, tol) \
    REQUIRE_THAT(static_cast<double>(a), Catch::Matchers::WithinAbs(static_cast<double>(b), tol))


TEST_CASE("Jacobian correctness", "[CNeuralNetwork]") {
    /* Create randomized network and enable Jacobian calculation */
    std::vector<std::string> input_names = {"a","b","c"};
    std::vector<std::string> output_names = {"x","y","z"};
    const double delta_inp{1e-6};
    MLPToolbox::CNeuralNetwork * mlp = CreateRandomNetwork(input_names, output_names);
    mlp->CalcJacobian(true);

    /* Calculate the Jacobian of iOut w.r.t. iIn */
    const size_t iIn = (rand() % mlp->GetnInputs());
    const size_t iOut = (rand() % mlp->GetnOutputs());
    
    /* Evaluate the Jacobian analytically. */
    const auto inputs_base = RandomInputs(mlp->GetnInputs());
    mlp->SetInput(inputs_base);
    mlp->Predict();
    const double Jac_analytical = mlp->GetJacobian(iOut, iIn);

    /* Approximate the Jacobian with central finite-differneces. */
    std::vector<double> inputs_plus = inputs_base,
                        inputs_minus = inputs_base;
    inputs_plus[iIn] += delta_inp;
    inputs_minus[iIn] -= delta_inp;
    mlp->SetInput(inputs_plus);
    mlp->Predict();
    const double outp_plus = mlp->GetOutput(iOut);
    mlp->SetInput(inputs_minus);
    mlp->Predict();
    const double outp_minus = mlp->GetOutput(iOut);
    const double Jac_FD = (outp_plus - outp_minus) / (2*delta_inp);

    REQUIRE_EQUAL_TOL(Jac_analytical, Jac_FD, 1e-6);
    //REQUIRE_THAT(Jac_analytical, WithinAbs(Jac_FD, 1e-6));
}

TEST_CASE("Hessian correctness", "[CNeuralNetwork]") {
    std::vector<std::string> input_names = {"a","b","c"};
    std::vector<std::string> output_names = {"x","y","z"};
    const double delta_inp{1e-6};
    MLPToolbox::CNeuralNetwork * mlp = CreateRandomNetwork(input_names, output_names);
    mlp->CalcJacobian(true);
    mlp->CalcHessian(true);

    /* Evaluate the Hessian of iOut w.r.t iIn and jIn. */
    const size_t iIn = (rand() % mlp->GetnInputs());
    const size_t jIn = (rand() % mlp->GetnInputs());
    const size_t iOut = (rand() % mlp->GetnOutputs());
    
    /* Evaluate the Hessian analytically. */
    const auto inputs_base = RandomInputs(mlp->GetnInputs());
    mlp->SetInput(inputs_base);
    mlp->Predict();
    const double Hes_analytical = mlp->GetHessian(iOut, iIn, jIn);

    /* Approximate the Hessian with central finite-differences. */
    std::vector<double> inputs_plus = inputs_base,
                        inputs_minus = inputs_base;
    inputs_plus[iIn] += delta_inp;
    inputs_minus[iIn] -= delta_inp;
    mlp->SetInput(inputs_plus);
    mlp->Predict();
    const double jac_outp_plus = mlp->GetJacobian(iOut, jIn);
    mlp->SetInput(inputs_minus);
    mlp->Predict();
    const double jac_outp_minus = mlp->GetJacobian(iOut, jIn);

    const double Hes_FD = (jac_outp_plus - jac_outp_minus) / (2*delta_inp);

    REQUIRE_EQUAL_TOL(Hes_analytical, Hes_FD, 1e-6);
}
