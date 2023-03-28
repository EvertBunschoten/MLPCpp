/*!
 * \file CReadNeuralNetwork.hpp
 * \brief Declaration of MLP input file reader class
 * \author E.C.Bunschoten
 * \version 1.0.0

 */
#pragma once

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#if defined(HAVE_OMP)
using su2double = codi::RealReverseIndexOpenMP;
#else
#if defined(CODI_INDEX_TAPE)
using su2double = codi::RealReverseIndex;
#else
using su2double = codi::RealReverse;
#endif
#endif
#elif defined(CODI_FORWARD_TYPE)  // forward mode AD
#include "codi.hpp"
using su2double = codi::RealForward;

#else  // primal / direct / no AD
using su2double = double;
#endif


namespace MLPToolbox {

class CReadNeuralNetwork {
private:
  std::vector<std::string> input_names, /*!< Input variable names. */
      output_names;                     /*!< Output variable names. */

  std::string filename; /*!< MLP input filename. */

  unsigned long n_layers; /*!< Network total layer count. */

  std::vector<unsigned long> n_neurons; /*!<  Neuron count per layer. */

  std::vector<std::vector<std::vector<su2double>>>
      weights_mat; /*!< Network synapse weights. */

  std::vector<std::vector<su2double>> biases_mat; /*!< Bias values per neuron. */

  std::vector<std::string>
      activation_functions; /*!< Activation function per layer. */

  std::vector<std::pair<su2double, su2double>>
      input_norm,  /*!< Input variable normalization values (min, max). */
      output_norm; /*!< Output variable normalization values (min, max). */
public:
  /*!
   * \brief CReadNeuralNetwork class constructor
   * \param[in] filename_in - .mlp input file name containing network
   * information.
   */
  CReadNeuralNetwork(std::string filename_in);

  /*!
   * \brief Read input file and store necessary information
   */
  void ReadMLPFile();

  /*!
   * \brief Go to a specific line in file.
   * \param[in] file_stream - input file stream.
   * \param[in] flag - line to be skipped to.
   * \returns flag line
   */
  std::string SkipToFlag(std::ifstream *file_stream, std::string flag);

  /*!
   * \brief Get number of read input variables.
   * \returns Number of neurons in the input layer.
   */
  unsigned long GetNInputs() const { return n_neurons[0]; }

  /*!
   * \brief Get number of read output variables.
   * \returns Number of neurons in the output layer.
   */
  unsigned long GetNOutputs() const { return n_neurons[n_layers - 1]; }

  /*!
   * \brief Get total number of layers in the network.
   * \returns Total layer count.
   */
  unsigned long GetNlayers() const { return n_layers; }

  /*!
   * \brief Get neuron count of a specific layer.
   * \param[in] iLayer - Total layer index.
   * \returns Number of neurons in the layer.
   */
  unsigned long GetNneurons(std::size_t iLayer) const {
    return n_neurons[iLayer];
  }

  /*!
   * \brief Get synapse weight between two neurons in subsequent layers.
   * \param[in] iLayer - Total layer index.
   * \param[in] iNeuron - Neuron index in layer with index iLayer.
   * \param[in] jNeuron - Neuron index in subsequent layer.
   * \returns Weight value
   */
  su2double GetWeight(std::size_t iLayer, std::size_t iNeuron,
                   std::size_t jNeuron) const {
    return weights_mat[iLayer][iNeuron][jNeuron];
  }

  /*!
   * \brief Get bias value of specific neuron.
   * \param[in] iLayer - Total layer index.
   * \param[in] iNeuron - Neuron index.
   * \returns Bias value
   */
  su2double GetBias(std::size_t iLayer, std::size_t iNeuron) const {
    return biases_mat[iLayer][iNeuron];
  }

  /*!
   * \brief Get input variable normalization values.
   * \param[in] iInput - Input variable index.
   * \returns Input normalization values (min first, max second)
   */
  std::pair<su2double, su2double> GetInputNorm(std::size_t iInput) const {
    return input_norm[iInput];
  }

  /*!
   * \brief Get output variable normalization values.
   * \param[in] iOutput - Input variable index.
   * \returns Output normalization values (min first, max second)
   */
  std::pair<su2double, su2double> GetOutputNorm(std::size_t iOutput) const {
    return output_norm[iOutput];
  }

  /*!
   * \brief Get layer activation function type.
   * \param[in] iLayer - Total layer index.
   * \returns Layer activation function type.
   */
  std::string GetActivationFunction(std::size_t iLayer) const {
    return activation_functions[iLayer];
  }

  /*!
   * \brief Get input variable name.
   * \param[in] iInput - Input variable index.
   * \returns Input variable name.
   */
  std::string GetInputName(std::size_t iInput) const {
    return input_names[iInput];
  }

  /*!
   * \brief Get output variable name.
   * \param[in] iOutput - Output variable index.
   * \returns Output variable name.
   */
  std::string GetOutputName(std::size_t iOutput) const {
    return output_names[iOutput];
  }
};
} // namespace MLPToolbox
