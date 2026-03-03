#include <string>
#include <vector>
#include <random>
#include <iostream>
#include "../include/CLookUp_ANN.hpp" 
#include "unit_test.hpp"

bool OutputCorrectness::CopyConstructorTest() {
    MLPToolbox::CNeuralNetwork * mlp = CreateRandomNetwork();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    std::vector<double> network_inputs = RandomInputs(mlp->GetnInputs());
    mlp->Predict(network_inputs);

    double network_output_ref = mlp->GetOutput(0);

    MLPToolbox::CNeuralNetwork mlp_copy = MLPToolbox::CNeuralNetwork(*mlp);
    mlp_copy.Predict(network_inputs);

    
    double network_output_copy = mlp_copy.GetOutput(0);
    delete mlp;
    bool passed = (network_output_ref == network_output_copy);

    summary << "From copy constructor: " << (passed ? "Passed" : "Failed") << std::endl;
    return passed;
}

bool OutputCorrectness::FileWriterReaderTest() {
    MLPToolbox::CNeuralNetwork * mlp = CreateRandomNetwork();
    std::string file_out_name = "mlp_test.mlp";
    mlp->WriteNeuralNetwork(file_out_name);

    MLPToolbox::CNeuralNetwork mlp_from_file = MLPToolbox::CNeuralNetwork(file_out_name);

    std::vector<double> network_inputs = RandomInputs(mlp->GetnInputs());
    mlp->Predict(network_inputs);
    double outp_ref = mlp->GetOutput(0);
    mlp_from_file.Predict(network_inputs);
    double outp_read = mlp_from_file.GetOutput(0);
    delete mlp;
    bool passed = (outp_ref == outp_read);
    summary << "From writing-reading input file: " << (passed ? "Passed" : "Failed") << std::endl;
    return passed;

}

bool OutputCorrectness::WeightsBiasesTest() {
    MLPToolbox::CNeuralNetwork * mlp = CreateRandomNetwork();
    auto weightsbiases = mlp->GetWeightsBiases();
    MLPToolbox::CNeuralNetwork mlp_copy = MLPToolbox::CNeuralNetwork(*mlp);
    mlp_copy.RandomWeights();

    mlp_copy.SetWeightsBiases(weightsbiases);
    std::vector<double> network_inputs = RandomInputs(mlp->GetnInputs());
    mlp->Predict(network_inputs);
    double outp_ref = mlp->GetOutput(0);
    mlp_copy.Predict(network_inputs);
    double outp_copy = mlp_copy.GetOutput(0);

    delete mlp;
    bool passed = (outp_ref == outp_copy);
    summary << "From weights and biases test: " << (passed ? "Passed" : "Failed") << std::endl;
    return passed;
}

bool OutputCorrectness::RunTest() {
    
    bool passed_copy = CopyConstructorTest();
    bool passed_file = FileWriterReaderTest();
    bool passed_weights = WeightsBiasesTest();

    passed = (passed_copy && passed_file && passed_weights);
    return passed;
}