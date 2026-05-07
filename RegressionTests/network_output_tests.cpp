#include <string>
#include <vector>
#include <random>
#include <iostream>
#include "../include/CLookUp_ANN.hpp" 
#include "unit_test.hpp"

bool OutputCorrectness::CopyConstructorTest() {
    auto mlp = CreateRandomNetwork();
    
    auto network_inputs = RandomInputs(mlp->GetnInputs());
    mlp->Predict(network_inputs);

    const auto network_output_ref = mlp->GetOutput(0);

    auto mlp_copy = MLPToolbox::CNeuralNetwork(*mlp);
    mlp_copy.Predict(network_inputs);
    
    const double network_output_copy = mlp_copy.GetOutput(0);

    delete mlp;
    bool passed = (network_output_ref == network_output_copy);
    if (!passed)
        summary << "From copy constructor: " << (passed ? "Passed" : "Failed") << std::endl;
    return passed;
}

bool OutputCorrectness::FileWriterReaderTest() {
    auto mlp = CreateRandomNetwork();
    std::string file_out_name = "mlp_test.mlp";
    mlp->WriteNeuralNetwork(file_out_name);

    auto mlp_from_file = MLPToolbox::CNeuralNetwork(file_out_name);

    auto network_inputs = RandomInputs(mlp->GetnInputs());
    mlp->Predict(network_inputs);
    const auto outp_ref = mlp->GetOutput(0);
    mlp_from_file.Predict(network_inputs);
    const auto outp_read = mlp_from_file.GetOutput(0);
    delete mlp;
    bool passed = (outp_ref == outp_read);
    if (!passed)
        summary << "From writing-reading input file: " << (passed ? "Passed" : "Failed") << std::endl;
    return passed;

}

bool OutputCorrectness::WeightsBiasesTest() {
    auto mlp = CreateRandomNetwork();
    auto weightsbiases = mlp->GetWeightsBiases();
    auto mlp_copy = MLPToolbox::CNeuralNetwork(*mlp);
    mlp_copy.RandomWeights();

    mlp_copy.SetWeightsBiases(weightsbiases);
    auto network_inputs = RandomInputs(mlp->GetnInputs());
    mlp->Predict(network_inputs);
    const auto outp_ref = mlp->GetOutput(0);
    mlp_copy.Predict(network_inputs);
    const auto outp_copy = mlp_copy.GetOutput(0);

    delete mlp;
    bool passed = (outp_ref == outp_copy);
    if (!passed)
        summary << "From weights and biases test: " << (passed ? "Passed" : "Failed") << std::endl;
    return passed;
}

bool OutputCorrectness::VectorInputOutputs() {

    /* Create randomized network and a vector with random inputs. */
    auto mlp = CreateRandomNetwork();
    auto network_inputs_vec = RandomInputs(mlp->GetnInputs());
    
    mlp->Predict(network_inputs_vec);

    const auto output_ref = mlp->GetOutput(0);

    /* Reset network output */
    mlp->Predict(RandomInputs(mlp->GetnInputs()));

    /* Set network input member-wise. */
    for (auto iInput=0u; iInput < network_inputs_vec.size(); iInput++)
        mlp->SetInput(iInput, network_inputs_vec[iInput]);

    mlp->Predict();

    const auto output_p = mlp->GetOutput(0);

    /* Outputs from vector and piece-wise input should be the same. */
    bool passed_test = (output_ref == output_p);

    if (!passed_test) {
        mlp->DisplayNetwork(summary);
        summary << "Network input values: \n";
        for (auto iInput=0u; iInput<mlp->GetnInputs(); iInput++)
            summary << mlp->GetInputName(iInput) << " : " << network_inputs_vec[iInput] << std::endl;
        summary << "Network output from vector input: " << output_ref << std::endl;
        summary << "Network output from piece-wise input: " << output_p << std::endl;
    }
    delete mlp;
    return passed_test;
    
}

bool OutputCorrectness::RunTest() {
    
    bool passed_copy = CopyConstructorTest();
    bool passed_file = FileWriterReaderTest();
    bool passed_weights = WeightsBiasesTest();
    bool passed_vector = VectorInputOutputs();

    passed = (passed_copy && passed_file && passed_weights && passed_vector);
    return passed;
}