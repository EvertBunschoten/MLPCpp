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
   *\brief This class allows for the inference of multiple multi-layer perceptrons for queries.
   */

private:
  std::vector<CNeuralNetwork*> NeuralNetworks; /*!< std::std::vector containing all loaded neural networks. */

public:

  CLookUp_ANN() = default;

  /*!
   * \brief ANN collection class constructor
   * \param[in] n_inputs - Number of MLP files to be loaded.
   * \param[in] input_filenames - String array containing MLP input file names.
   */
  CLookUp_ANN(const unsigned short n_inputs,
              const std::string *input_filenames) {
    /*--- Generate an MLP for every filename provided ---*/
    for (auto i_MLP = 0u; i_MLP < n_inputs; i_MLP++) {
      MLPToolbox::CNeuralNetwork * mlp = new MLPToolbox::CNeuralNetwork(input_filenames[i_MLP]);
      NeuralNetworks.push_back(mlp);
    }
  }

  /*!
   * \brief ANN collection class constructor
   * \param[in] input_filenames - String array containing MLP input file names.
   */
  CLookUp_ANN(const std::vector<std::string> &input_filenames) {
    NeuralNetworks.resize(input_filenames.size());
    for (auto i_MLP=0u; i_MLP<input_filenames.size(); i_MLP++)
      NeuralNetworks[i_MLP] = new MLPToolbox::CNeuralNetwork(input_filenames[i_MLP]);
  }

  /*!
   * \brief ANN collection class constructor
   * \param[in] mlps - vector with pointers to network objects.
   */
  CLookUp_ANN(const std::vector<MLPToolbox::CNeuralNetwork*> &mlps) {
    NeuralNetworks.clear();
    for (auto mlp : mlps) AddNetwork(mlp);
  }

  /*!
  * \brief Add a network to the collection.
  * \param[in] mlp - pointer to network object.
  */
  void AddNetwork(MLPToolbox::CNeuralNetwork * mlp) {
    NeuralNetworks.push_back(mlp);
  }

  /*!
  * \brief Copy constructor
  */
  CLookUp_ANN(const CLookUp_ANN &copy_class) {
    NeuralNetworks.resize(copy_class.GetNANNs());
    for (auto i_MLP=0u; i_MLP<NeuralNetworks.size(); i_MLP++)
      NeuralNetworks[i_MLP] = new MLPToolbox::CNeuralNetwork(*copy_class.NeuralNetworks[i_MLP]);
  }

  ~CLookUp_ANN() {
    for (auto MLP : NeuralNetworks) delete MLP;
  }

  /*!
  * \brief Find the networks in the collection with the inputs and outputs needed for a query.
  * \param[in] query - query class
  */
  void PairVariableswithMLPs(MLPToolbox::CIOMap &query) {
    query.FindNetworksForQuery(NeuralNetworks);
  }

  /*!
  * \brief Evaluate the output of the networks selected for a query.
  * \param[in] query - query class with input-output information
  */
  bool Predict(const MLPToolbox::CIOMap &query) const {
    return query();
  }

  /*!
  * \brief Evaluate the output of the networks selected for a query with inputs and outputs.
  * \param[in] query - query class with input-output information.
  * \param[in] vals_input - network inputs.
  * \param[in] refs_output - pointers to output variables.
  */
  bool Predict(const MLPToolbox::CIOMap &query, const std::vector<mlpdouble> &vals_input, const std::vector<mlpdouble*> &refs_output) const {
    return query(vals_input, refs_output);
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
    for (auto MLP : NeuralNetworks) MLP->DisplayNetwork();
  }
};

} // namespace MLPToolbox
