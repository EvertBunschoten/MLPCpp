/*!
 * \file CNeuralNetwork.hpp
 * \brief Declaration of the neural network class
 * \author E.C.Bunschoten
 * \version 1.0.0
 */

#pragma once

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>

#include "CLayer.hpp"

namespace MLPToolbox {
class CNeuralNetwork {
  /*!
   *\class CNeuralNetwork
   *\brief The CNeuralNetwork class allows for the evaluation of a loaded MLP
   *architecture for a given set of inputs. The class also contains a list of
   *the various supported activation function types (linear, relu, elu, gelu,
   *selu, sigmoid, swish, tanh, exp)which can be applied to the layers in the
   *network. Currently, only dense, feed-forward type neural nets are supported
   *in this implementation.
   */
private:
  std::vector<std::string> input_names, /*!< MLP input variable names. */
      output_names;                     /*!< MLP output variable names. */

  unsigned long n_hidden_layers = 0; /*!< Number of hidden layers (layers
                                        between input and output layer). */

  CLayer *inputLayer = nullptr, /*!< Pointer to network input layer. */
      *outputLayer = nullptr;   /*!< Pointer to network output layer. */

  std::vector<CLayer *> hiddenLayers; /*!< Hidden layer collection. */
  std::vector<CLayer *>
      total_layers; /*!< Hidden layers plus in/output layers */

  // std::vector<su2activematrix> weights_mat; /*!< Weights of synapses
  // connecting layers */
  std::vector<std::vector<std::vector<double>>> weights_mat;

  std::vector<std::pair<double, double>>
      input_norm,  /*!< Normalization factors for network inputs */
      output_norm; /*!< Normalization factors for network outputs */

  std::vector<double> last_inputs; /*!< Inputs from previous lookup operation.
                                      Evaluation of the network */
  /*!< is skipped if current inputs are the same as the last inputs. */

  double *ANN_outputs; /*!< Pointer to network outputs */
  std::vector<std::vector<double>>
      dOutputs_dInputs; /*!< Network output derivatives w.r.t inputs */

  /*!
   * \brief Available activation function enumeration.
   */
  enum class ENUM_ACTIVATION_FUNCTION {
    NONE = 0,
    LINEAR = 1,
    RELU = 2,
    ELU = 3,
    GELU = 4,
    SELU = 5,
    SIGMOID = 6,
    SWISH = 7,
    TANH = 8,
    EXPONENTIAL = 9
  };

  /*!
   * \brief Available activation function map.
   */
  std::map<std::string, ENUM_ACTIVATION_FUNCTION> activation_function_map{
      {"none", ENUM_ACTIVATION_FUNCTION::NONE},
      {"linear", ENUM_ACTIVATION_FUNCTION::LINEAR},
      {"elu", ENUM_ACTIVATION_FUNCTION::ELU},
      {"relu", ENUM_ACTIVATION_FUNCTION::RELU},
      {"gelu", ENUM_ACTIVATION_FUNCTION::GELU},
      {"selu", ENUM_ACTIVATION_FUNCTION::SELU},
      {"sigmoid", ENUM_ACTIVATION_FUNCTION::SIGMOID},
      {"swish", ENUM_ACTIVATION_FUNCTION::SWISH},
      {"tanh", ENUM_ACTIVATION_FUNCTION::TANH},
      {"exponential", ENUM_ACTIVATION_FUNCTION::EXPONENTIAL}};

  std::vector<ENUM_ACTIVATION_FUNCTION>
      activation_function_types; /*!< Activation function type for each layer in
                                    the network. */
  std::vector<std::string>
      activation_function_names; /*!< Activation function name for each layer in
                                    the network. */

public:
  ~CNeuralNetwork() {
    delete inputLayer;
    delete outputLayer;
    for (std::size_t i = 1; i < total_layers.size() - 1; i++) {
      delete total_layers[i];
    }
    delete[] ANN_outputs;
  };
  /*!
   * \brief Set the input layer of the network.
   * \param[in] n_neurons - Number of inputs
   */
  void DefineInputLayer(unsigned long n_neurons);

  /*!
   * \brief Set the output layer of the network.
   * \param[in] n_neurons - Number of outputs
   */
  void DefineOutputLayer(unsigned long n_neurons);

  /*!
   * \brief Add a hidden layer to the network
   * \param[in] n_neurons - Hidden layer size.
   */
  void PushHiddenLayer(unsigned long n_neurons);

  /*!
   * \brief Set the weight value of a specific synapse.
   * \param[in] i_layer - current layer.
   * \param[in] i_neuron - neuron index in current layer.
   * \param[in] j_neuron - neuron index of connecting neuron.
   * \param[in] value - weight value.
   */
  void SetWeight(unsigned long i_layer, unsigned long i_neuron,
                 unsigned long j_neuron, double value) {
    weights_mat[i_layer][j_neuron][i_neuron] = value;
  };

  /*!
   * \brief Set bias value at a specific neuron.
   * \param[in] i_layer - Layer index.
   * \param[in] i_neuron - Neuron index of current layer.
   * \param[in] value - Bias value.
   */
  void SetBias(unsigned long i_layer, unsigned long i_neuron, double value) {
    total_layers[i_layer]->SetBias(i_neuron, value);
  }

  /*!
   * \brief Set layer activation function.
   * \param[in] i_layer - Layer index.
   * \param[in] input - Activation function name.
   */
  void SetActivationFunction(unsigned long i_layer, std::string input);

  /*!
   * \brief Display the network architecture in the terminal.
   */
  void DisplayNetwork() const;

  /*!
   * \brief Size the weight layers in the network according to its architecture.
   */
  void SizeWeights();

  /*!
   * \brief Size the std::vector of previous inputs.
   * \param[in] n_inputs - Number of inputs.
   */
  void SizeInputs(unsigned long n_inputs) {
    last_inputs.resize(n_inputs);
    for (unsigned long iInput = 0; iInput < n_inputs; iInput++)
      last_inputs[iInput] = 0.0;
  }

  /*!
   * \brief Get the number of connecting regions in the network.
   * \returns number of spaces in between layers.
   */
  unsigned long GetNWeightLayers() const { return total_layers.size() - 1; }

  /*!
   * \brief Get the total number of layers in the network
   * \returns number of netowork layers.
   */
  unsigned long GetNLayers() const { return total_layers.size(); }

  /*!
   * \brief Get neuron count in a layer.
   * \param[in] iLayer - Layer index.
   * \returns number of neurons in the layer.
   */
  unsigned long GetNNeurons(unsigned long iLayer) const {
    return total_layers[iLayer]->GetNNeurons();
  }

  /*!
   * \brief Evaluate the network.
   * \param[in] inputs - Network input variable values.
   * \param[in] compute_gradient - Compute the derivatives of the outputs wrt
   * inputs.
   */
  void Predict(std::vector<double> &inputs, bool compute_gradient = false);

  /*!
   * \brief Set the normalization factors for the input layer
   * \param[in] iInput - Input index.
   * \param[in] input_min - Minimum input value.
   * \param[in] input_max - Maximum input value.
   */
  void SetInputNorm(unsigned long iInput, double input_min, double input_max) {
    input_norm[iInput] = std::make_pair(input_min, input_max);
  }

  /*!
   * \brief Set the normalization factors for the output layer
   * \param[in] iOutput - Input index.
   * \param[in] input_min - Minimum output value.
   * \param[in] input_max - Maximum output value.
   */
  void SetOutputNorm(unsigned long iOutput, double output_min,
                     double output_max) {
    output_norm[iOutput] = std::make_pair(output_min, output_max);
  }

  std::pair<double, double> GetInputNorm(unsigned long iInput) const {
    return input_norm[iInput];
  }

  std::pair<double, double> GetOutputNorm(unsigned long iOutput) const {
    return output_norm[iOutput];
  }
  /*!
   * \brief Add an output variable name to the network.
   * \param[in] input - Input variable name.
   */
  void SetOutputName(size_t iOutput, std::string input) {
    output_names[iOutput] = input;
  }

  /*!
   * \brief Add an input variable name to the network.
   * \param[in] output - Output variable name.
   */
  void SetInputName(size_t iInput, std::string input) {
    input_names[iInput] = input;
  }

  /*!
   * \brief Get network input variable name.
   * \param[in] iInput - Input variable index.
   * \returns input variable name.
   */
  std::string GetInputName(std::size_t iInput) const {
    return input_names[iInput];
  }

  /*!
   * \brief Get network output variable name.
   * \param[in] iOutput - Output variable index.
   * \returns output variable name.
   */
  std::string GetOutputName(std::size_t iOutput) const {
    return output_names[iOutput];
  }

  /*!
   * \brief Get network number of inputs.
   * \returns Number of network inputs
   */
  std::size_t GetnInputs() const { return input_names.size(); }

  /*!
   * \brief Get network number of outputs.
   * \returns Number of network outputs
   */
  std::size_t GetnOutputs() const { return output_names.size(); }

  /*!
   * \brief Get network evaluation output.
   * \param[in] iOutput - output index.
   * \returns Prediction value.
   */
  double GetANNOutput(std::size_t iOutput) const {
    return ANN_outputs[iOutput];
  }

  /*!
   * \brief Get network output derivative w.r.t specific input.
   * \param[in] iOutput - output variable index.
   * \param[in] iInput - input variable index.
   * \returns Output derivative w.r.t input.
   */
  double GetdOutputdInput(std::size_t iOutput, std::size_t iInput) const {
    return dOutputs_dInputs[iOutput][iInput];
  }

  /*!
   * \brief Set the activation function array size.
   * \param[in] n_layers - network layer count.
   */
  void SizeActivationFunctions(unsigned long n_layers) {
    activation_function_types.resize(n_layers);
    activation_function_names.resize(n_layers);
  }

  /*!
   * \brief Compute neuron activation function input.
   * \param[in] iLayer - Network layer index.
   * \param[in] iNeuron - Layer neuron index.
   * \returns Neuron activation function input.
   */
  double ComputeX(std::size_t iLayer, std::size_t iNeuron) const {
    double x;
    x = total_layers[iLayer]->GetBias(iNeuron);
    std::size_t nNeurons_previous = total_layers[iLayer - 1]->GetNNeurons();
    for (std::size_t jNeuron = 0; jNeuron < nNeurons_previous; jNeuron++) {
      x += weights_mat[iLayer - 1][iNeuron][jNeuron] *
           total_layers[iLayer - 1]->GetOutput(jNeuron);
    }
    return x;
  }
  double ComputedOutputdInput(std::size_t iLayer, std::size_t iNeuron,
                              std::size_t iInput) const {
    double doutput_dinput = 0;
    for (auto jNeuron = 0u; jNeuron < total_layers[iLayer - 1]->GetNNeurons();
         jNeuron++) {
      doutput_dinput += weights_mat[iLayer - 1][iNeuron][jNeuron] *
                        total_layers[iLayer - 1]->GetdYdX(jNeuron, iInput);
    }
    return doutput_dinput;
  }
};

} // namespace MLPToolbox