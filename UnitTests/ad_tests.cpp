#include "codi.hpp"
#define MLP_CUSTOM_TYPE codi::RealReverse;

#include "../include/CNeuralNetwork.hpp"
#include "test_subjects.hpp"
#include <iostream>
#include <string>
#include <vector>
#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

TEST_CASE("Network Jacobian differentiation test", "[CNeuralNetwork]") {
  /*! \brief Check whether the Jacobian of the network output equals the
   * differentiated network output. */
  std::vector<std::string> input_names = {"a", "b"}, output_names = {"x", "y"};
  MLPToolbox::CNeuralNetwork *mlp =
      CreateRandomNetwork(input_names, output_names);
  mlpdouble val_a = 0.5, val_b = 2.0;
  mlpdouble::getTape().reset();
  mlpdouble::getTape().setActive();
  mlpdouble::getTape().registerInput(val_a);
  mlpdouble::getTape().registerInput(val_b);

  mlp->SetInput(0, val_a);
  mlp->SetInput(1, val_b);
  mlp->CalcJacobian(true);
  mlp->Predict();
  mlpdouble val_x = mlp->GetOutput(0);
  mlpdouble val_y = mlp->GetOutput(1);

  mlpdouble::getTape().registerOutput(val_x);
  mlpdouble::getTape().setPassive();
  val_x.setGradient(1.0);

  mlpdouble::getTape().evaluate();
  mlpdouble jac_ad = val_a.getGradient();
  mlpdouble jac_analytical = mlp->GetJacobian(0, 0);

  double a = jac_ad.getValue();
  double b = jac_analytical.getValue();
  REQUIRE_EQUAL_TOL(a, b, 1e-8);
}

TEST_CASE("Network Hessian differentiation test", "[CNeuralNetwork]") {
  /*! \brief Check whether the Hessian of the network output equals the
   * differentiated network output. */
  std::vector<std::string> input_names = {"a", "b"}, output_names = {"x", "y"};
  MLPToolbox::CNeuralNetwork *mlp =
      CreateRandomNetwork(input_names, output_names);
  mlpdouble val_a = 0.5, val_b = 2.0;

  mlpdouble::getTape().reset();
  mlpdouble::getTape().setActive();
  mlpdouble::getTape().registerInput(val_a);
  mlpdouble::getTape().registerInput(val_b);

  mlp->SetInput(0, val_a);
  mlp->SetInput(1, val_b);
  mlp->CalcJacobian(true);
  mlp->CalcHessian(true);
  mlp->Predict();
  mlpdouble val_dxda = mlp->GetJacobian(0, 0);

  mlpdouble::getTape().registerOutput(val_dxda);
  mlpdouble::getTape().setPassive();
  val_dxda.setGradient(1.0);

  mlpdouble::getTape().evaluate();
  mlpdouble hess_ad = val_b.getGradient();
  mlpdouble hess_analytical = mlp->GetHessian(0, 0, 1);

  double a = hess_ad.getValue();
  double b = hess_analytical.getValue();
  REQUIRE_EQUAL_TOL(a, b, 1e-8);
}
