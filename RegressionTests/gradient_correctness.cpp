#include <vector>
#include <string>
#include <iostream>
#include "unit_test.hpp"
#include "../include/CLookUp_ANN.hpp"

bool GradientCorrectness::JacobianCorrectness() {

    /* Create randomized network and enable Jacobian calculation */
    std::vector<std::string> input_names = {"a","b","c"};
    std::vector<std::string> output_names = {"x","y","z"};
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

    /* Compare approximate and analytical Jacobian values. */
    const double rel_diff = std::abs((Jac_analytical - Jac_FD)/Jac_FD);
    bool passed_test = rel_diff < 1e-6;
    summary << "Analytical Jacobian: " << Jac_analytical << std::endl;
    summary << "Approximate Jacobian (central differences): " << Jac_FD << std::endl;
    summary << "Relative difference: " << rel_diff*100 << "%" << std::endl;
    delete mlp;
    return passed_test;
}

bool GradientCorrectness::HessianCorrectness() {

    /* Create randomized network and enable Hessian calculation. */
    std::vector<std::string> input_names = {"a","b","c"};
    std::vector<std::string> output_names = {"x","y","z"};
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
    const double rel_diff = std::abs((Hes_analytical - Hes_FD)/Hes_FD);
    bool passed_test = rel_diff < 1e-6;
    summary << "Analytical Hessian: " << Hes_analytical << std::endl;
    summary << "Approximate Hessian (central differences): " << Hes_FD << std::endl;
    summary << "Relative difference: " << rel_diff*100 << "%" << std::endl;
    delete mlp;
    return passed_test;
}


bool GradientCorrectness::RunTest() {
    bool passed_jac = JacobianCorrectness();
    bool passed_hes = HessianCorrectness();
    passed = (passed_jac && passed_hes);
    return passed;
}