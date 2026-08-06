#include "../include/CNeuralNetwork.hpp"
#include "test_subjects.hpp"
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#define REQUIRE_EQUAL_TOL(a, b, tol)                                           \
  REQUIRE_THAT(static_cast<double>(a),                                         \
               Catch::Matchers::WithinAbs(static_cast<double>(b), tol))

TEST_CASE("Copy constructor", "[CNeuralNetwork]") {
  MLPToolbox::CNeuralNetwork *mlp = CreateRandomNetwork();

  std::vector<double> network_inputs = RandomInputs(mlp->GetnInputs());
  mlp->Predict(network_inputs);

  const auto network_output_ref = mlp->GetOutput(0);

  MLPToolbox::CNeuralNetwork mlp_copy = MLPToolbox::CNeuralNetwork(*mlp);
  mlp_copy.Predict(network_inputs);

  const auto network_output_copy = mlp_copy.GetOutput(0);

  REQUIRE(network_output_ref == network_output_copy);
  delete mlp;
};

TEST_CASE("File writer test", "[CNeuralNetwork]") {
  MLPToolbox::CNeuralNetwork *mlp = CreateRandomNetwork();
  std::string file_out_name = "mlp_test.mlp";
  mlp->WriteNeuralNetwork(file_out_name);

  MLPToolbox::CNeuralNetwork mlp_from_file =
      MLPToolbox::CNeuralNetwork(file_out_name);

  std::vector<double> network_inputs = RandomInputs(mlp->GetnInputs());
  mlp->Predict(network_inputs);
  const auto outp_ref = mlp->GetOutput(0);
  mlp_from_file.Predict(network_inputs);
  const auto outp_read = mlp_from_file.GetOutput(0);

  REQUIRE(outp_ref == outp_read);
  delete mlp;
};

TEST_CASE("Flattened weights and biases", "[CNeuralNetwork]") {
  MLPToolbox::CNeuralNetwork *mlp = CreateRandomNetwork();
  auto weightsbiases = mlp->GetWeightsBiases();
  MLPToolbox::CNeuralNetwork mlp_copy = MLPToolbox::CNeuralNetwork(*mlp);
  mlp_copy.RandomWeights();

  mlp_copy.SetWeightsBiases(weightsbiases);
  std::vector<double> network_inputs = RandomInputs(mlp->GetnInputs());
  mlp->Predict(network_inputs);
  const auto outp_ref = mlp->GetOutput(0);
  mlp_copy.Predict(network_inputs);
  const auto outp_copy = mlp_copy.GetOutput(0);

  REQUIRE(outp_ref == outp_copy);

  delete mlp;
};

TEST_CASE("Vector-wise and member-wise input", "[CNeuralNetwork]") {
  MLPToolbox::CNeuralNetwork *mlp = CreateRandomNetwork();
  auto network_inputs_vec = RandomInputs(mlp->GetnInputs());

  mlp->Predict(network_inputs_vec);

  const auto output_ref = mlp->GetOutput(0);

  /* Reset network output */
  mlp->Predict(RandomInputs(mlp->GetnInputs()));

  /* Set network input member-wise. */
  for (auto iInput = 0u; iInput < network_inputs_vec.size(); iInput++)
    mlp->SetInput(iInput, network_inputs_vec[iInput]);

  mlp->Predict();

  const auto output_p = mlp->GetOutput(0);

  REQUIRE(output_ref == output_p);
};

TEST_CASE("Ill-defined networks", "[CNeuralNetworks]"){
    {std::vector<size_t> bad_architecture = {3, 4, 0, 10};
REQUIRE_THROWS_AS(MLPToolbox::CNeuralNetwork(bad_architecture),
                  MLPToolbox::ShouldBePositiveException);
}

{
  MLPToolbox::CNeuralNetwork *mlp = CreateRandomNetwork();
  auto weights_biases = mlp->GetWeightsBiases();
  weights_biases.push_back(1.0);
  REQUIRE_THROWS_AS(mlp->SetWeightsBiases(weights_biases),
                    MLPToolbox::WeightsMisMatchException);
}
}
;
