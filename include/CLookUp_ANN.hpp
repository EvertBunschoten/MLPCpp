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
  std::vector<IteratorNetwork*> NeuralNetworks; /*!< std::std::vector containing
  //                                                all loaded neural networks. */

  unsigned short number_of_variables; /*!< Number of loaded ANNs. */

  /*!
   * \brief Load ANN architecture
   * \param[in] ANN - pointer to target NeuralNetwork class
   * \param[in] filename - filename containing ANN architecture information
   */
  void GenerateANN(std::string filename) {
    /*--- Generate MLP architecture based on information in MLP input file ---*/

    /* Read MLP input file */
    CReadNeuralNetwork Reader = CReadNeuralNetwork(filename);

    /* Read MLP input file */
    Reader.ReadMLPFile();

    IteratorNetwork * MLP = new IteratorNetwork(Reader.GetNneurons());

    for (auto iInput=0u; iInput<Reader.GetNInputs(); iInput++)
      MLP->SetInputName(iInput, Reader.GetInputName(iInput));
    for (auto iOutput=0u; iOutput<Reader.GetNOutputs(); iOutput++)
      MLP->SetOutputName(iOutput, Reader.GetOutputName(iOutput));
    
    
    for (auto iLayer=0u; iLayer<Reader.GetNlayers(); iLayer++) {
      for (auto iNode = 0u; iNode < Reader.GetNneurons()[iLayer];
           iNode++) {
        if (iLayer < (Reader.GetNlayers()-1)) {
          for (auto jNode = 0u; jNode < Reader.GetNneurons()[iLayer+1];
             jNode++) {
          MLP
          ->SetWeight(iLayer, iNode, jNode,Reader.GetWeight(iLayer, iNode, jNode));
        }
        }
        
        MLP->SetBias(iLayer, iNode, Reader.GetBias(iLayer, iNode));
      }
    }
    /* Define input and output layer normalization values */
    MLP->SetInputRegularization(Reader.GetInputRegularization());
    MLP->SetOutputRegularization(Reader.GetOutputRegularization());
    
    for (auto iInput = 0u; iInput < Reader.GetNInputs(); iInput++) {
      MLP->SetInputNorm(iInput, Reader.GetInputNorm(iInput).first,
                       Reader.GetInputNorm(iInput).second);
    }
    for (auto iOutput = 0u; iOutput < Reader.GetNOutputs(); iOutput++) {
      MLP->SetOutputNorm(iOutput, Reader.GetOutputNorm(iOutput).first,
                        Reader.GetOutputNorm(iOutput).second);
    }
    for (size_t iLayer=0; iLayer < Reader.GetNlayers(); iLayer++)
      MLP->SetActivationFunction(iLayer, Reader.GetActivationFunction(iLayer));
    
    NeuralNetworks.push_back(MLP);
  }

public:
  /*!
   * \brief ANN collection class constructor
   * \param[in] n_inputs - Number of MLP files to be loaded.
   * \param[in] input_filenames - String array containing MLP input file names.
   */
  CLookUp_ANN(const unsigned short n_inputs,
              const std::string *input_filenames) {
    /*--- Define collection of MLPs for regression purposes ---*/
    number_of_variables = n_inputs;

    //NeuralNetworks.resize(n_inputs);

    /*--- Generate an MLP for every filename provided ---*/
    for (auto i_MLP = 0u; i_MLP < n_inputs; i_MLP++) {
      GenerateANN(input_filenames[i_MLP]);
      //GenerateANN(NeuralNetworks[i_MLP], input_filenames[i_MLP]);
    }
  }


  std::pair<mlpdouble, mlpdouble>
  GetInputNorm(MLPToolbox::CQuery *input_output_map,
               std::string varname) const {
    mlpdouble CV_min{0.0}, CV_max{0.0};
    for (const auto &q : input_output_map->GetNetworksInQuery()) {
      auto loc = std::find(q.MLP->GetInputVars().begin(), q.MLP->GetInputVars().end(), varname);
      size_t iInput = std::distance(q.MLP->GetInputVars().begin(), loc);

      CV_min += q.MLP->GetInputNorm(iInput).first;
      CV_max += q.MLP->GetInputNorm(iInput).second;
    }
    int n_networks = input_output_map->GetNetworksInQuery().size();
    CV_min /= n_networks;
    CV_max /= n_networks;
    return std::make_pair(CV_min, CV_max);
  } ///TODO: make compatible with CQuery

  
  
  /*!
   * \brief Get the median input regularization value for a specific look-up operation.
   * \param[in] input_output_map - Pointer to input-output
   * map for look-up operation. 
   * \param[in] input_index - Input variable index
   * for which to get the input median.
   * \returns Input median value.
   */
  mlpdouble GetInputOffset(MLPToolbox::CQuery *input_output_map,
               std::string varname) const {
    mlpdouble CV_offset{0};
    for (const auto &q : input_output_map->GetNetworksInQuery()) {
      auto loc = std::find(q.MLP->GetInputVars().begin(), q.MLP->GetInputVars().end(), varname);
      size_t iInput = std::distance(q.MLP->GetInputVars().begin(), loc);
      CV_offset += q.MLP->GetRegularizationOffset(iInput);
    }
    int n_networks = input_output_map->GetNetworksInQuery().size();
    return CV_offset / n_networks;
  }

  void PairVariableswithMLPs(MLPToolbox::CQuery &query) {
    query.FindNetworksForQuery(NeuralNetworks);
  }

  void Predict(MLPToolbox::CQuery &query) {
    query();
  }

  /*!
   * \brief Get number of loaded ANNs
   * \return number of loaded ANNs
   */
  std::size_t GetNANNs() const { return NeuralNetworks.size(); }

  // /*!
  //  * \brief Map variable names to ANN inputs or outputs
  //  * \param[in] i_ANN - loaded ANN index
  //  * \param[in] variable_names - variable names to map to ANN inputs or outputs
  //  * \param[in] input - map to inputs (true) or outputs (false)
  //  */
  // std::vector<std::pair<std::size_t, std::size_t>>
  // FindVariableIndices(std::size_t i_ANN,
  //                     std::vector<std::string> variable_names,
  //                     bool input) const {
  //   /*--- Find loaded MLPs that have the same input variable names as the
  //    * variables listed in variable_names ---*/

  //   std::vector<std::pair<size_t, size_t>> variable_indices;
  //   auto nVar = input ? NeuralNetworks[i_ANN]->GetnInputs()
  //                     : NeuralNetworks[i_ANN]->GetnOutputs();

  //   // auto nVar = input ? NeuralNetworks[i_ANN].GetnInputs()
  //   //                   : NeuralNetworks[i_ANN].GetnOutputs();

  //   for (auto iVar = 0u; iVar < nVar; iVar++) {
  //     for (auto jVar = 0u; jVar < variable_names.size(); jVar++) {
  //       std::string ANN_varname =
  //           input ? NeuralNetworks[i_ANN]->GetInputName(iVar)
  //                 : NeuralNetworks[i_ANN]->GetOutputName(iVar);
  //       // std::string ANN_varname =
  //       //     input ? NeuralNetworks[i_ANN].GetInputName(iVar)
  //       //           : NeuralNetworks[i_ANN].GetOutputName(iVar);
  //       if (variable_names[jVar].compare(ANN_varname) == 0) {
  //         variable_indices.push_back(std::make_pair(jVar, iVar));
  //       }
  //     }
  //   }
  //   return variable_indices;
  // }

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
      // NeuralNetworks[i_MLP].DisplayNetwork();
    }
  }
};

} // namespace MLPToolbox
