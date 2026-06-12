/*!
 * \file test_codi_neuralnetwork.cpp
 *
 * Validate CoDi reverse-mode derivatives (w.r.t. inputs) against
 * CNeuralNetwork::GetJacobian().
 */

#include <iostream>
#include <vector>
#include <iomanip>
#include <cassert>

#include "codi.hpp"
#include "variable_def.hpp"

#include "CLookUp_ANN.hpp"
#include "CNeuralNetwork.hpp"
#include "../RegressionTests/unit_test.hpp"

using Scalar = codi::RealReverse;
using Tape   = Scalar::Tape;


//  Forward pass implemented with CoDi scalars                   

static std::vector<Scalar> codi_forward_inputs_active(
    const MLPToolbox::CNeuralNetwork* net,
    const std::vector<Scalar>& inputs)
{
    const size_t n_inputs = net->GetnInputs();
    const size_t n_outputs = net->GetnOutputs();
    const size_t n_layers = net->GetnLayers();
    const size_t n_hidden = n_layers - 1;

    assert(inputs.size() == n_inputs);

    std::vector<std::vector<Scalar>> layer_outputs(n_layers);
    for (size_t l = 0; l < n_layers; ++l) {
        layer_outputs[l].resize(net->GetnNodes(l));
    }

    // Input Layer (scaling)
    for (size_t i = 0; i < n_inputs; ++i) {
        auto norm = net->GetInputNorm(i);
        mlpdouble offset = norm.first;
        mlpdouble scale  = norm.second;
        Scalar scaled = (inputs[i] - Scalar(offset)) / Scalar(scale);
        layer_outputs[0][i] = scaled;
    }

    // Hidden + Output Layers
    for (size_t iLayer = 1; iLayer < n_layers; ++iLayer) {
        const size_t prev = iLayer - 1;
        const size_t n_prev = net->GetnNodes(prev);
        const size_t n_cur  = net->GetnNodes(iLayer);

        std::string act_name = net->GetActivationFunction(iLayer);
        auto activation = [&](Scalar x) -> Scalar {
            if (act_name == "linear") return x;
            if (act_name == "tanh")   return codi::tanh(x);
            if (act_name == "relu")   return codi::max(Scalar(0.0), x);
            if (act_name == "sigmoid") {
                return Scalar(1.0) / (Scalar(1.0) + codi::exp(-x));
            }
            std::cerr << "Warning: activation " << act_name << " not fully implemented in CoDi test. Using linear.\n";
            return x;
        };

        for (size_t iNeuron = 0; iNeuron < n_cur; ++iNeuron) {
            Scalar node_input = net->GetBias(iLayer, iNeuron);  

            // Weighted sum from previous layer
            for (size_t j = 0; j < n_prev; ++j) {
                mlpdouble w = net->GetWeight(prev, j, iNeuron);  
                node_input += Scalar(w) * layer_outputs[prev][j];
            }

            layer_outputs[iLayer][iNeuron] = activation(node_input);
        }
    }

    std::vector<Scalar> final_outputs(n_outputs);
    for (size_t i = 0; i < n_outputs; ++i) {
        auto norm = net->GetOutputNorm(i);
        mlpdouble offset = norm.first;
        mlpdouble scale  = norm.second;
        // Reverse of normalization: dimensionalize = scaled * scale + offset
        final_outputs[i] = layer_outputs.back()[i] * Scalar(scale) + Scalar(offset);
    }

    return final_outputs;
}

int main()
{
    // Create a random network for testing
    auto* net = CreateRandomNetwork();
    net->CalcJacobian(true);   // Enable internal Jacobian

    const size_t n_inputs  = net->GetnInputs();
    const size_t n_outputs = net->GetnOutputs();

    std::vector<mlpdouble> x(n_inputs);
    for (size_t i = 0; i < n_inputs; ++i) {
        auto norm = net->GetInputNorm(i);
        x[i] = 0.5 * (norm.first + norm.second);  // midpoint
    }

    net->SetInput(x);
    net->Predict();

    std::cout << "\n====================================\n";
    std::cout << "CNeuralNetwork Jacobian vs CoDi Reverse AD\n";
    std::cout << "====================================\n\n";

    bool all_passed = true;
    const double tol = 1e-8;

    for (size_t iOut = 0; iOut < n_outputs; ++iOut) {
        for (size_t iIn = 0; iIn < n_inputs; ++iIn) {
            Tape& tape = Scalar::getTape();
            tape.setActive();

            std::vector<Scalar> x_active(n_inputs);
            for (size_t j = 0; j < n_inputs; ++j) {
                x_active[j] = x[j];
                tape.registerInput(x_active[j]);
            }

            auto y = codi_forward_inputs_active(net, x_active);

            tape.registerOutput(y[iOut]);
            tape.setPassive();

            y[iOut].setGradient(1.0);
            tape.evaluate();

            double jac_codi = x_active[iIn].getGradient();
            double jac_internal = net->GetJacobian(iOut, iIn);

            double rel_err = std::abs(jac_codi - jac_internal) /
                            (std::abs(jac_internal) + 1e-14);

            std::cout << "Out " << iOut << " In " << iIn
                      << "   Internal Jacobian = " << std::setw(12) << jac_internal
                      << "   CoDi = " << std::setw(12) << jac_codi
                      << "   RelErr = " << rel_err << "\n";

            if (rel_err > tol) {
                all_passed = false;
            }

            tape.reset();
        }
    }

    std::cout << "\nTest " << (all_passed ? "PASSED" : "FAILED") << "\n";


    delete net;
    return all_passed ? 0 : 1;
}