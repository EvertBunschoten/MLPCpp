/*!
* \file CLookUp_ANN.hpp
* \brief Declaration of the main MLP evaluation class.
* \author E.C.Bunschoten
* \version 1.2.0
*
* MLPCpp Project Website: https://github.com/EvertBunschoten/MLPCpp
*
* Copyright (c) 2023 Evert Bunschoten

* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:

* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.

* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/
#pragma once

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "CIOMap.hpp"
#include "CNeuralNetwork.hpp"
#include "CReadNeuralNetwork.hpp"
namespace MLPToolbox {

class CLookUp_ANN {
  /*!
   *\class CLookUp_ANN
   *\brief This class allows for the evaluation of one or more multi-layer
   *perceptrons in for example thermodynamic state look-up operations. The
   *multi-layer perceptrons are loaded in the order listed in the MLP collection
   *file. Each multi-layer perceptron is generated based on the architecture
   *described in its respective input file. When evaluating the MLP collection,
   *an input-output map is used to find the correct MLP corresponding to the
   *call function inputs and outputs.
   */

private:
  std::vector<CNeuralNetwork*> NeuralNetworks; /*!< std::std::vector containing
  //                                                all loaded neural networks. */

  unsigned short number_of_variables; /*!< Number of loaded ANNs. */

public:

  CLookUp_ANN() = default;

  /*!
   * \brief ANN collection class constructor
   * \param[in] n_inputs - Number of MLP files to be loaded.
   * \param[in] input_filenames - String array containing MLP input file names.
   */
  CLookUp_ANN(const unsigned short n_inputs,
              const std::string *input_filenames) {
    /*--- Define collection of MLPs for regression purposes ---*/
    number_of_variables = n_inputs;

    /*--- Generate an MLP for every filename provided ---*/
    for (auto i_MLP = 0u; i_MLP < n_inputs; i_MLP++) {
      MLPToolbox::CNeuralNetwork * mlp = new MLPToolbox::CNeuralNetwork(input_filenames[i_MLP]);
      NeuralNetworks.push_back(mlp);
    }
  }

  CLookUp_ANN(const std::vector<MLPToolbox::CNeuralNetwork*> &mlps) {
    for (auto mlp : mlps) AddNetwork(mlp);
  }

  void AddNetwork(MLPToolbox::CNeuralNetwork * mlp) {
    NeuralNetworks.push_back(mlp);
  }

  CLookUp_ANN(const CLookUp_ANN &copy_class) {
    NeuralNetworks.reserve(copy_class.GetNANNs());
    for (auto i_MLP=0u; i_MLP<NeuralNetworks.size(); i_MLP++)
      NeuralNetworks[i_MLP] = new MLPToolbox::CNeuralNetwork(*copy_class.NeuralNetworks[i_MLP]);
  }

  ~CLookUp_ANN() {
    for (auto MLP : NeuralNetworks) delete MLP;
  }

  void PairVariableswithMLPs(MLPToolbox::CIOMap &query) {
    query.FindNetworksForQuery(NeuralNetworks);
  }

  void Predict(const MLPToolbox::CIOMap &query) const {
    query();
  }

  void Predict(const MLPToolbox::CIOMap &query, const std::vector<mlpdouble> &vals_input, const std::vector<mlpdouble*> &refs_output) const {
    query(vals_input, refs_output);
  }

  /*!
   * \brief Get number of loaded ANNs
   * \return number of loaded ANNs
   */
  std::size_t GetNANNs() const { return NeuralNetworks.size(); }

  /*!
   * \brief Display architectural information on the loaded MLPs
   */
  void DisplayNetworkInfo() const {
    /*--- Display network information on the loaded MLPs ---*/

    std::cout << std::setfill(' ');
    std::cout << std::endl;
    std::cout << "+------------------------------------------------------------"
                 "------+"
                 "\n";
    std::cout
        << "|                 Multi-Layer Perceptron (MLP) info                "
           "|\n";
    std::cout << "+------------------------------------------------------------"
                 "------+"
              << std::endl;

    /* For every loaded MLP, display the inputs, outputs, activation functions,
     * and architecture. */
    for (auto i_MLP = 0u; i_MLP < NeuralNetworks.size(); i_MLP++) {
      NeuralNetworks[i_MLP]->DisplayNetwork();
    }
  }
};

} // namespace MLPToolbox
