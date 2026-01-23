/*!
* \file CNeuralNetwork.hpp
* \brief Declaration of the CNeuralNetwork class.
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
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include "variable_def.hpp"

#include "ActivationFunctions.hpp"
#include "CReadNeuralNetwork.hpp"
#include "ScalarFunctions.hpp"
#include "option_maps.hpp"

namespace MLPToolbox {
  class IteratorNetwork {
    private:
    size_t n_layers{0},         /*!< \brief Total number of layers in the network. */
           n_hidden_layers{0};  /*!< \brief Number of hidden layers. */

    mlpdouble *** weights_mat {nullptr};    /*!<\brief Weights values. */
    mlpdouble ** biases_mat {nullptr},      /*!<\brief Biases values. */
              ** layer_outputs {nullptr};   /*!<\brief Output values of the nodes in the network. */
    mlpdouble *** layer_Jacobian {nullptr}; /*!<\brief Jacobian values of the node output in the hidden layers. */
    mlpdouble **** layer_Hessian {nullptr}; /*!<\brief Hessian values of the node output in the hidden layers. */
    mlpdouble * input_layer {nullptr},      /*!<\brief Scaled values stored at the input layer. */
              * network_inputs{nullptr};    /*!<\brief Unscaled values of the network input. */

    mlpdouble * output_layer {nullptr};     /*!<\brief Scaled values stored at the output layer. */
    mlpdouble ** output_Jacobian {nullptr}; /*!<\brief Jacobian of the network output w.r.t. the network input. */
    mlpdouble *** output_Hessian {nullptr}; /*!<\brief Hessian of the network output w.r.t. the network input. */

    std::vector<std::pair<mlpdouble, mlpdouble>> input_norm,    /*!<\brief Scaling values used to normalize the network input. */
                                                 output_norm;   /*!<\brief Scaling values used to dimensionalize the network output. */

    ENUM_SCALING_FUNCTIONS input_reg_method {ENUM_SCALING_FUNCTIONS::MINMAX},   /*!<\brief Scaling method tag used for the network input. */
                           output_reg_method {ENUM_SCALING_FUNCTIONS::MINMAX};  /*!<\brief Scaling method tag used for the network output. */
    
    ScalerFunction * input_scaler{nullptr},     /*!<\brief Function used to scale the network input. */
                   * output_scaler{nullptr};    /*!<\brief Function used to scale the network output. */

    ActivationFunctionBase** activation_functions {nullptr};    /*!<\brief Hidden layer activation functions. */
    
    size_t * NN {nullptr},  /*!<\brief Hidden layer architecture. */
             n_inputs{0},   /*!<\brief Number of input nodes. */
             n_outputs{0};  /*!<\brief Number of output nodes. */

    std::vector<std::string> input_names,   /*!<\brief Names of the network input variables. */
                             output_names;  /*!<\brief Names of the network output variables. */

    bool calc_Jacobian {false}, /*!<\brief Evaluate the network Jacobian. */
         calc_Hessian {false};  /*!<\brief Evaluate the network Hessian. */
    
    /*!<\brief Default values for network hyperparameters. */
    const mlpdouble default_offset{0.0},
                    default_scale{1.0},
                    default_weight{0.0},
                    default_bias{0.0};

    /*!
    * \brief Set the values of the network input nodes.
    */
    void InputLayer() const {
        for (auto iInput=0u; iInput < n_inputs; iInput++) {
            // Scale the network input values to normalized inputs.
            input_layer[iInput] = input_scaler->Normalize(network_inputs[iInput], iInput);
            if (calc_Jacobian) {
                // Jacobian of the input layer is a diagonal matrix.
                std::fill(layer_Jacobian[0][iInput], layer_Jacobian[0][iInput]+n_inputs, 0.0);
                layer_Jacobian[0][iInput][iInput] = 1.0 / input_scaler->GetScale(iInput);
                if (calc_Hessian) {
                // Hessian of the input layer is always zero.
                for (auto jInput=0u; jInput < n_inputs; jInput++) {
                        std::fill(layer_Hessian[0][iInput][jInput], layer_Hessian[0][iInput][jInput]+n_inputs, 0.0);
                    }
                }
            }
        }
    }

    /*!
    * \brief Dimensionalize the network output, Jacobian, and Hessian
    */
    void OutputLayer() const {
        for (auto iOutput=0u; iOutput < n_outputs; iOutput++) {
            output_layer[iOutput] = output_scaler->Dimensionalize(output_layer[iOutput], iOutput);
            if (calc_Jacobian) {
                for (auto iInput=0u; iInput < n_inputs; iInput++) {
                    output_Jacobian[iInput][iOutput] *= output_scaler->GetScale(iOutput);
                    if (calc_Hessian) {
                        for (auto jInput=0u; jInput < n_inputs; jInput++) 
                            output_Hessian[iInput][jInput][iOutput] *= output_scaler->GetScale(iOutput);
                    }
                }  
            }
        }
    }

    /*!
   * \brief Calculate the values of the output, Jacobian, and Hessian of the nodes in the hidden layers.
   * \param[in] iLayer - layer index
   */
    void CalcLayerOutputs(const size_t iLayer) const {
        if (iLayer==0){
            /* Nothing done at the input layer. */
            return;
        }else {
            const size_t prev_layer = iLayer-1;
            /* Recursive function call to the previous layer. */
            CalcLayerOutputs(prev_layer);
            
            for (auto iNeuron=0u; iNeuron < NN[iLayer]; ++iNeuron){
                /* Calculate the node input through matrix-vector multiplcation of the weights with the output of the previous layer. */
                const mlpdouble node_input = WeightsMultiplication(prev_layer, iNeuron, layer_outputs[prev_layer], biases_mat[iLayer][iNeuron]);

                /* Evaluate the activation function output and store in the hidden layer node output. */
                layer_outputs[iLayer][iNeuron] = activation_functions[iLayer]->operator()(node_input, calc_Jacobian, calc_Hessian);

                if (calc_Jacobian) {
                    for (auto iInput=0u; iInput < n_inputs; iInput++){
                        /* Calculate the Jacobian of the node output w.r.t. the network input. */
                        const mlpdouble psi = WeightsMultiplication(prev_layer, iNeuron, layer_Jacobian[prev_layer][iInput]);

                        /* Calculate the derivative of the activation function. */
                        const mlpdouble phi_prime = activation_functions[iLayer]->GetJacobian();

                        layer_Jacobian[iLayer][iInput][iNeuron] = psi * phi_prime;
                        if (calc_Hessian) {
                            for (auto jInput=0u; jInput < n_inputs; jInput++){
                                /* Calculate the Hessian of the node output w.r.t. the network input. */
                                const mlpdouble psi_j = (jInput==iInput) ? psi : WeightsMultiplication(prev_layer, iNeuron, layer_Jacobian[prev_layer][jInput]);
                                
                                /* Calculate the second-order derivative of the activation function. */
                                const mlpdouble phi_dprime = activation_functions[iLayer]->GetHessian();
                                const mlpdouble chi = WeightsMultiplication(prev_layer, iNeuron, layer_Hessian[prev_layer][iInput][jInput]);
                                layer_Hessian[iLayer][iInput][jInput][iNeuron] = phi_dprime * psi_j * psi + phi_prime * chi;
                            }
                        }
                    }
                }
            }
            return;
        }
    }

    
    /*!
    * \brief Size the weights arrays and set the network hyperparameters.
    * \param[in] copy_network - Pointer to reference network used in copy constructor.
    * \param[in] reader - Pointer to reader class.
    */
    void SizeWeights(const IteratorNetwork * copy_network=nullptr, const CReadNeuralNetwork * reader=nullptr) {
        const bool from_reader = (reader != nullptr);       /* Retrieve weight information from reader. */
        const bool from_copy = (copy_network != nullptr);   /* Copy weight information from reference MLP. */

        n_hidden_layers = n_layers - 1;
        n_inputs = NN[0];
        n_outputs = NN[n_hidden_layers];

        /* Size weights, biases, Jacobian, and Hessians. */
        weights_mat = new mlpdouble**[n_hidden_layers];
        biases_mat = new mlpdouble * [n_layers];
        layer_outputs = new mlpdouble*[n_layers];
        layer_Jacobian = new mlpdouble**[n_layers];
        layer_Hessian = new mlpdouble***[n_layers];
        for (auto iLayer=0u; iLayer<n_layers; iLayer++) {
            if (iLayer < n_hidden_layers){
                weights_mat[iLayer] = new mlpdouble*[NN[iLayer+1]];
                for (auto iNeuron=0u; iNeuron<NN[iLayer+1]; iNeuron++){
                    weights_mat[iLayer][iNeuron] = new mlpdouble[NN[iLayer]];

                    /* Set weights values */ 
                    std::fill(weights_mat[iLayer][iNeuron], weights_mat[iLayer][iNeuron]+NN[iLayer], default_weight);
                    if (from_reader){
                        for (auto jNeuron=0u; jNeuron<NN[iLayer]; jNeuron++)
                            SetWeight(iLayer, jNeuron, iNeuron, reader->GetWeight(iLayer, jNeuron, iNeuron));
                    }else if (from_copy){
                        for (auto jNeuron=0u; jNeuron<NN[iLayer]; jNeuron++)
                            SetWeight(iLayer, jNeuron, iNeuron, copy_network->weights_mat[iLayer][iNeuron][jNeuron]);
                    }
                }
            }
            biases_mat[iLayer] = new mlpdouble[NN[iLayer]];

            /* Set bias values. */
            std::fill(biases_mat[iLayer], biases_mat[iLayer]+NN[iLayer], default_bias);
            if (from_reader){
                for (auto iNeuron=0u; iNeuron<NN[iLayer]; iNeuron++) 
                    SetBias(iLayer, iNeuron, reader->GetBias(iLayer, iNeuron));
            } else if (from_copy) {
                for (auto iNeuron=0u; iNeuron<NN[iLayer]; iNeuron++) 
                    SetBias(iLayer, iNeuron, copy_network->biases_mat[iLayer][iNeuron]);
            }

            /* Size hidden layer Jacobian and Hessian. */
            layer_outputs[iLayer] = new mlpdouble[NN[iLayer]];
            layer_Jacobian[iLayer] = new mlpdouble*[n_inputs];
            layer_Hessian[iLayer] = new mlpdouble**[n_inputs];
            for (auto iInput=0u; iInput<n_inputs; iInput++){
                layer_Jacobian[iLayer][iInput] = new mlpdouble[NN[iLayer]];
                layer_Hessian[iLayer][iInput] = new mlpdouble*[n_inputs];
                for (auto jInput=0u; jInput < n_inputs; jInput++)
                    layer_Hessian[iLayer][iInput][jInput] = new mlpdouble[NN[iLayer]];
            }
        }

        /* Set default scaling values for network input and output. */
        input_norm.resize(n_inputs);
        std::fill(input_norm.begin(), input_norm.end(), std::make_pair(default_offset, default_scale));
        output_norm.resize(n_outputs);
        std::fill(output_norm.begin(), output_norm.end(), std::make_pair(default_offset, default_scale));
        network_inputs = new mlpdouble[n_inputs];
        output_Jacobian = layer_Jacobian[n_hidden_layers];
        output_Hessian = layer_Hessian[n_hidden_layers];
        input_names.resize(n_inputs);
        output_names.resize(n_outputs);
        
        input_layer = layer_outputs[0];
        output_layer = layer_outputs[n_hidden_layers];

        /* Set hidden layer activation functions. */
        activation_functions = new ActivationFunctionBase*[n_layers];
        std::fill(activation_functions, activation_functions+n_layers, nullptr);
        SetActivationFunction();
        if (from_reader){
            for (auto iLayer=0u; iLayer<n_layers; iLayer++)
                SetActivationFunction(iLayer, reader->GetActivationFunction(iLayer));
        } else if (from_copy){
            for (auto iLayer=0u; iLayer<n_layers; iLayer++){
                SetActivationFunction(iLayer, copy_network->activation_functions[iLayer]->GetTag());
            }
        }
    }
    
    /*!
    * \brief Matrix-vector multiplication between the weights and node output of the previous layer.
    * \param[in] iLayer - layer index
    * \param[in] iNeuron - node index
    * \param[in] array_in - vector for multiplication
    * \param[in] bias - node bias value, defaults to 0.0
    * \returns - innder product between array and weights.
    */
    const mlpdouble WeightsMultiplication(const size_t iLayer, const size_t iNeuron, const mlpdouble*array_in, const mlpdouble bias=0.0) const {
        return std::inner_product(weights_mat[iLayer][iNeuron], weights_mat[iLayer][iNeuron] + NN[iLayer], array_in, bias);
    }
    
    public:
        IteratorNetwork() = default;

        /*!
        * \brief Constructor from ASCII file name
        * \param[in] MLP_filename - MLPCpp ASCII file name from which to read network information.
        */
        IteratorNetwork(const std::string MLP_filename) {

            /* Read content of MLP file. */
            CReadNeuralNetwork reader = CReadNeuralNetwork(MLP_filename);
            reader.ReadMLPFile();

            /* Retrieve hidden layer information. */
            const auto N_H = reader.GetNneurons();
            n_layers = N_H.size();
            NN = new size_t[n_layers];
            for (auto iLayer=0u; iLayer<(n_layers); iLayer++)
                NN[iLayer] = N_H[iLayer];
            
            /* Retrieve other network hyperparameters. */
            SizeWeights(nullptr, &reader);

            /* Retrieve input and output normalization functions and scaling values. */
            SetInputRegularization(reader.GetInputRegularization());
            for (auto iInput=0u; iInput<n_inputs; iInput++){
                SetInputName(iInput, reader.GetInputName(iInput));
                SetInputNorm(iInput, reader.GetInputNorm(iInput).first, reader.GetInputNorm(iInput).second);
            }
            SetOutputRegularization(reader.GetOutputRegularization());
            for (auto iOutput=0u; iOutput<n_outputs; iOutput++){
                SetOutputName(iOutput, reader.GetOutputName(iOutput));
                SetOutputNorm(iOutput, reader.GetOutputNorm(iOutput).first, reader.GetOutputNorm(iOutput).second);
            }

        }

        /*!
        * \brief Copy constructor
        * \param[in] copy_network - network from which to copy information.
        */
        IteratorNetwork(const IteratorNetwork & copy_network) {
            n_layers = copy_network.n_layers;
            if (n_layers > 1){
                NN = new size_t[n_layers];
                std::copy(copy_network.NN, copy_network.NN+n_layers, NN);
                SizeWeights(&copy_network);
                input_layer = layer_outputs[0];
                output_layer = layer_outputs[n_hidden_layers];
                SetInputRegularization(copy_network.input_reg_method);
                std::copy(copy_network.input_names.begin(), copy_network.input_names.end(), input_names.begin());
                std::copy(copy_network.input_norm.begin(), copy_network.input_norm.end(), input_norm.begin());
                
                SetOutputRegularization(copy_network.output_reg_method);
                std::copy(copy_network.output_names.begin(), copy_network.output_names.end(), output_names.begin());
                std::copy(copy_network.output_norm.begin(), copy_network.output_norm.end(), output_norm.begin());
            }
        }
        
        /*!
        * \brief Constructor from network architecture.
        * \param[in] NN_input - vector describing number of nodes per layer in the network.
        */
        IteratorNetwork(const std::vector<size_t> &NN_input) {
            if (std::find(NN_input.begin(), NN_input.end(), 0) != NN_input.end()){
                throw std::exception();
                return;
            }
            n_layers = NN_input.size();
            NN = new size_t[n_layers];
            std::copy(NN_input.begin(), NN_input.end(), NN);
            
            /* Set default values for network hyperparameters. */
            SizeWeights();
            SetInputRegularization();
            SetOutputRegularization();
        }

        /*!
        * \brief Specify method used to scale the network input.
        * \param[in] reg_method_tag - scaling method tag, defaults to "minmax"
        */
        void SetInputRegularization(const std::string reg_method_tag="minmax") {
            const auto it = scaling_map.find(reg_method_tag);
            if (it == scaling_map.end())
                throw std::exception();
            else
                input_reg_method = it->second;
            SetInputRegularization(input_reg_method);
        }

        std::string GetInputRegularization() const {return input_scaler->GetTag();}
        std::string GetOutputRegularization() const {return output_scaler->GetTag();}
        /*!
        * \brief Specify method used to scale the network output.
        * \param[in] reg_method_tag - scaling method tag, defaults to "minmax"
        */
        void SetOutputRegularization(const std::string reg_method_tag="minmax") {
            const auto it = scaling_map.find(reg_method_tag);
            if (it == scaling_map.end())
                throw std::exception();
            else
                output_reg_method = it->second;
            SetOutputRegularization(output_reg_method);
        }
        
        /*!
        * \brief Define the regularization method used to normalize the inputs before feeding them to the network.
        * \param[in] reg_method_input - regularization method (minmax, standard, or robust).
        */
        void SetInputRegularization(const ENUM_SCALING_FUNCTIONS reg_method_input) {
            if (input_scaler != nullptr) delete input_scaler;
          switch (reg_method_input)
          {
          
          case ENUM_SCALING_FUNCTIONS::STANDARD:
            input_scaler = new StandardScaler(n_inputs);
            break;
          case ENUM_SCALING_FUNCTIONS::ROBUST:
            input_scaler = new RobustScaler(n_inputs);
            break;
          case ENUM_SCALING_FUNCTIONS::MINMAX:
          default:
            input_scaler = new MinMaxScaler(n_inputs);
            break;
          };
          input_reg_method = reg_method_input;
          return;
        }

        /*!
        * \brief Define the regularization method used to normalize the training data before training. 
        * \param[in] reg_method_input - regularization method (minmax, standard, or robust).
        */
        void SetOutputRegularization(const ENUM_SCALING_FUNCTIONS reg_method_input) {
          if (output_scaler != nullptr) delete output_scaler;
          switch (reg_method_input)
          {
          
          case ENUM_SCALING_FUNCTIONS::STANDARD:
            output_scaler = new StandardScaler(n_outputs);
            break;
          case ENUM_SCALING_FUNCTIONS::ROBUST:
            output_scaler = new RobustScaler(n_outputs);
            break;
          case ENUM_SCALING_FUNCTIONS::MINMAX:
          default:
            output_scaler = new MinMaxScaler(n_outputs);
            break;
          };
          output_reg_method = reg_method_input;
          return;
        }

        /*!
        * \brief Set the normalization factors for the input layer
        * \param[in] iInput - Input index.
        * \param[in] scale_val_1 - First scaling value (mean for "robust", "standard", minimum for "minmax")
        * \param[in] scale_val_2 - Second scaling value (standard deviation for "standard", inter-quantile range for "robust", maximum for "minmax")
        */
        void SetInputNorm(const size_t iInput, const mlpdouble scale_val_1=0.0,
                            const mlpdouble scale_val_2=1.0) {
            input_norm[iInput] = std::make_pair(scale_val_1, scale_val_2);
            input_scaler->SetScaling(iInput, scale_val_1, scale_val_2);
        }

        std::pair<mlpdouble, mlpdouble> GetInputNorm(const size_t iInput) const {return input_norm[iInput];}
        std::pair<mlpdouble, mlpdouble> GetOutputNorm(const size_t iOutput) const {return output_norm[iOutput];}
        /*!
        * \brief Set the normalization factors for the output layer
        * \param[in] iOutput - Input index.
        * \param[in] scale_val_1 - First scaling value (mean for "robust", "standard", minimum for "minmax")
        * \param[in] scale_val_2 - Second scaling value (standard deviation for "standard", inter-quantile range for "robust", maximum for "minmax")
        */
        void SetOutputNorm(const size_t iOutput, const mlpdouble scale_val_1=0.0,
                            const mlpdouble scale_val_2=1.0) {
            output_norm[iOutput] = std::make_pair(scale_val_1, scale_val_2);
            output_scaler->SetScaling(iOutput, scale_val_1, scale_val_2);
        }
        
        /*!
        * \brief Retrieve relative distance between the network input range and query input.
        * \param[in] query - network input values.
        * \returns - relative distance between query and network feature range.
        */
        mlpdouble QueryDistance(const std::vector<mlpdouble> &query) const {
            return input_scaler->Distance(query);
        }

        /*!
        * \brief Retrieve relative distance between the network input range and network input values.
        * \returns - relative distance between query and network feature range.
        */
        mlpdouble QueryDistance() const {
            return input_scaler->Distance(network_inputs);
        }

        /*!
        * \brief Set random values for the network weights and biases.
        */
        void RandomWeights(){
            std::random_device rd;  
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(-1.0, 1.0);
            for (auto iLayer=0u; iLayer<n_layers; iLayer++) {
                if (iLayer < (n_hidden_layers)){
                    for (auto iNeuron=0u; iNeuron<NN[iLayer]; iNeuron++){
                        for (auto jNeuron=0u; jNeuron<NN[iLayer+1]; jNeuron++){
                            SetWeight(iLayer, iNeuron, jNeuron, dis(gen));
                        }
                    }
                }
                for (auto iNeuron=0u; iNeuron<NN[iLayer]; iNeuron++) {
                    SetBias(iLayer, iNeuron, dis(gen));
                }
            }
        }
        

        ~IteratorNetwork() {
            for (auto iLayer=0u; iLayer<n_layers; iLayer++) {
                delete [] biases_mat[iLayer];
                delete [] layer_outputs[iLayer];
                for (auto iInput=0u; iInput<n_inputs; iInput++){
                    delete [] layer_Jacobian[iLayer][iInput];
                    for (auto jInput=0u; jInput < n_inputs; jInput++) {
                        delete [] layer_Hessian[iLayer][iInput][jInput];
                    }
                    delete [] layer_Hessian[iLayer][iInput];
                }
                delete [] layer_Jacobian[iLayer];
                delete [] layer_Hessian[iLayer];

                if (iLayer < (n_hidden_layers)) {
                    for (auto iNeuron=0u; iNeuron < NN[iLayer+1]; iNeuron++) 
                        delete [] weights_mat[iLayer][iNeuron];
                    delete [] weights_mat[iLayer]; 
                }
                delete activation_functions[iLayer];
            }
            delete [] layer_Jacobian;
            delete [] layer_Hessian;
            delete [] activation_functions;
            delete [] layer_outputs;
            delete [] biases_mat;
            delete [] weights_mat;
            delete [] NN;
            delete [] network_inputs;
            if (input_scaler != nullptr) delete input_scaler;
            if (output_scaler != nullptr) delete output_scaler;
            
            return;
        }

        
        /*!
        * \brief Set the weight value connecting two nodes in the network.
        * \param[in] iLayer - layer index
        * \param[in] iNode - node index
        * \param[in] jNode - connecting node index
        * \param[in] val_weight - weight value
        */
        void SetWeight(const size_t iLayer, const size_t iNode, const size_t jNode, const mlpdouble val_weight) const {
            weights_mat[iLayer][jNode][iNode] = val_weight;
        }

        mlpdouble GetWeight(const size_t iLayer, const size_t iNode, const size_t jNode) const {
            return weights_mat[iLayer][jNode][iNode];
        }
        /*!
        * \brief Set the bias value for a node in the network.
        * \param[in] iLayer - layer index
        * \param[in] iNode - node index
        * \param[in] val_bias - bias value
        */
        void SetBias(const size_t iLayer, const size_t iNode, const mlpdouble val_bias) const {
            biases_mat[iLayer][iNode] = val_bias;
        }
        
        mlpdouble GetBias(const size_t iLayer, const size_t iNode) const {
            return biases_mat[iLayer][iNode];
        }
        /*!
        * \brief Enable Jacobian calculation while evaluating the network output.
        * \param[in] enable_Jacobian - calculate the Jacobian, defaults to false.
        */
        void CalcJacobian(const bool enable_Jacobian=false) {
            calc_Jacobian = enable_Jacobian;
        }

        /*!
        * \brief Enable Hessian calculation while evaluating the network output.
        * \param[in] enable_Hessian - calculate the Hessian, defaults to false.
        */
        void CalcHessian(const bool enable_Hessian=false) {
            calc_Hessian = enable_Hessian;
        }

        /*!
        * \brief Set the value for one of the network inputs.
        * \param[in] iInput - input node index
        * \param[in] val_input - input value
        */
        void SetInput(const size_t iInput, const mlpdouble val_input) const {
            network_inputs[iInput] = val_input;
            return;
        }

        /*!
        * \brief Set the network inputs
        * \param[in] input_vals - pointer to array of input values
        */
        void SetInput(const mlpdouble* const input_vals) const {
            std::copy(input_vals, input_vals + n_inputs, input_layer);
            return;
        }

        /*!
        * \brief Set the network inputs
        * \param[in] input_vals - vector of input values
        */
        void SetInput(const std::vector<mlpdouble> &input_vals) const {
            std::copy(input_vals.begin(), input_vals.end(), input_layer);
            return;
        }

        /*!
        * \brief Retrieve the pointer to one of the network inputs.
        * \param[in] iInput - input node index
        * \returns - pointer to network input node
        */
        mlpdouble * InputLayer(const size_t iInput) const { return &network_inputs[iInput];}

        /*!
        * \brief Retrieve the pointer to one of the network outputs.
        * \param[in] iOutput - output node index
        * \returns - pointer to network output node
        */
        mlpdouble * OutputLayer(const size_t iOutput) const {return &output_layer[iOutput];}

        /*!
        * \brief Retrieve the pointer to one of the network Jacobians.
        * \param[in] iOutput - output node index
        * \param[in] iInput - input node index
        * \returns - pointer to network Jacobian
        */
        mlpdouble * Jacobian(const size_t iOutput, const size_t iInput) const 
        {return &output_Jacobian[iInput][iOutput];}

        /*!
        * \brief Retrieve the pointer to one of the network Hessians
        * \param[in] iOutput - output node index
        * \param[in] iInput - input node index
        * \param[in] jInput - second input node index
        * \returns - pointer to network Hessian
        */
        mlpdouble * Hessian(const size_t iOutput, const size_t iInput, const size_t jInput) const
        {return &output_Hessian[iInput][jInput][iOutput];}


        /*!
        * \brief Retrieve the value for one of the network outputs.
        * \param[in] iOutput - output node index
        * \returns - network output value
        */
        mlpdouble GetOutput(const size_t iOutput) const {
            return output_layer[iOutput];
        }

        /*!
        * \brief Retrieve the value for one of the network Jacobians.
        * \param[in] iOutput - output node index
        * \param[in] iInput - input node index
        * \returns - network Jacobian value
        */
        mlpdouble GetJacobian(const size_t iOutput, const size_t iInput) const {
            return output_Jacobian[iInput][iOutput];
        }

        /*!
        * \brief Retrieve the value for one of the network Hessian
        * \param[in] iOutput - output node index
        * \param[in] iInput - input node index
        * \param[in] jInput - second input node index
        * \returns - network Hessian value
        */
        mlpdouble GetHessian(const size_t iOutput, const size_t iInput, const size_t jInput) const {
            return output_Hessian[iInput][jInput][iOutput];
        }

        /*!
        * \brief Calculate the network output, Jacobian, and Hessian from a provided set of inputs.
        * \param[in] X_in - vector of network inputs
        * \param[in] evaluate_Jacobian - calculate the network Jacobian, defaults to false
        * \param[in] evaluate_Hessian - calculate the network Hessian, defaults to false
        */
        void Predict(const std::vector<mlpdouble> &X_in, const bool evaluate_Jacobian=false, const bool evaluate_Hessian=false) {
            SetInput(X_in);
            calc_Jacobian=evaluate_Jacobian;
            calc_Hessian=evaluate_Hessian;
            Predict();
        }
        
        /*!
        * \brief Calculate the network output, Jacobian, and Hessian from the stored network inputs.
        */
        void Predict() const {
            InputLayer();
            CalcLayerOutputs(n_hidden_layers);
            OutputLayer();
        }
        /*!
        * \brief Set the activation function for the nodes in one of the layers in the network.
        * \param[in] iLayer - layer index
        * \param[in] name_activation_function - activation function tag
        */
        void SetActivationFunction(const size_t iLayer, const std::string name_activation_function="linear") {
            /* Check if activation function tag is valid */
            const auto it = activation_function_map.find(name_activation_function);
            if (it == activation_function_map.end())
                throw std::exception();
            
            const auto i_phi = it->second;
            ActivationFunctionBase * function_out;
            switch (i_phi)
            {
            case ENUM_ACTIVATION_FUNCTION::LINEAR:
                function_out = new Lin();
                break;
            case ENUM_ACTIVATION_FUNCTION::ELU:
                function_out = new Elu();
                break;
            case ENUM_ACTIVATION_FUNCTION::EXPONENTIAL:
                function_out = new Exponential();
                break;
            case ENUM_ACTIVATION_FUNCTION::RELU:
                function_out = new Relu();
                break;
            case ENUM_ACTIVATION_FUNCTION::SWISH:
                function_out = new Swish();
                break;
            case ENUM_ACTIVATION_FUNCTION::TANH:
                function_out = new Tanh();
                break;
            case ENUM_ACTIVATION_FUNCTION::SIGMOID:
                function_out = new Sigmoid();
                break;
            case ENUM_ACTIVATION_FUNCTION::SELU:
                function_out = new SeLu();
                break;
            case ENUM_ACTIVATION_FUNCTION::GELU:
                function_out = new GeLu();
                break;
            default:
                function_out = new Lin();
                break;
            }
            if (activation_functions[iLayer] != nullptr){
                delete activation_functions[iLayer];
            }
            activation_functions[iLayer] = function_out;
        }

        /*!
        * \brief Set the activation function for all hidden layers
        * \param[in] name_activation_function - activation function tag
        */
        void SetActivationFunction(const std::string name_activation_function="linear") {
            for (auto iLayer=0u; iLayer < n_layers; iLayer++) 
                SetActivationFunction(iLayer, name_activation_function);
        }

        /*!
        * \brief Add an output variable name to the network.
        * \param[in] input - Output variable name.
        */
        void SetOutputName(const size_t iOutput, const std::string input) {
            output_names[iOutput] = input;
        }

        /*!
        * \brief Add an input variable name to the network.
        * \param[in] output - Input variable name.
        */
        void SetInputName(const size_t iInput, const std::string input) {
            input_names[iInput] = input;
        }
    /*!
    * \brief Display the network architecture in the terminal.
    */
    void DisplayNetwork() const {
        /*--- Display information on the MLP architecture ---*/
        const int display_width{54};
        const int column_width = int(display_width / 3.0) - 1;

        /*--- Input layer information ---*/
        std::cout << "+" << std::setfill('-') << std::setw(display_width)
                << std::right << "+" << std::endl;
        std::cout << std::setfill(' ');
        std::cout << "|" << std::left << std::setw(display_width - 1)
                << "Input Layer Information:"
                << "|" << std::endl;
        std::cout << "+" << std::setfill('-') << std::setw(display_width)
                << std::right << "+" << std::endl;
        input_scaler->PrintInfo(display_width, input_names);
        std::cout << "+" << std::setfill('-') << std::setw(display_width)
                << std::right << "+" << std::endl;

        std::cout << std::setfill(' ');
        std::cout << "|" << std::left << std::setw(display_width - 1)
                << "Hidden Layers Information:"
                << "|" << std::endl;
        std::cout << "+" << std::setfill('-') << std::setw(display_width)
                << std::right << "+" << std::endl;
        std::cout << std::setfill(' ');
        std::cout << "|" << std::setw(column_width) << std::left << "Layer index"
                << "|" << std::setw(column_width) << std::left << "Neuron count"
                << "|" << std::setw(column_width) << std::left << "Function"
                << "|" << std::endl;
        std::cout << "+" << std::setfill('-') << std::setw(display_width)
                << std::right << "+" << std::endl;
        std::cout << std::setfill(' ');
        for (size_t iLayer = 1; iLayer < n_hidden_layers; iLayer++)
        std::cout << "|" << std::setw(column_width) << std::right << iLayer + 1
                    << "|" << std::setw(column_width) << std::right
                    << NN[iLayer] << "|"
                    << std::setw(column_width) << std::right
                    << activation_functions[iLayer]->GetName() << "|" << std::endl;
        std::cout << "+" << std::setfill('-') << std::setw(display_width)
                << std::right << "+" << std::endl;
        std::cout << std::setfill(' ');

        std::cout << "|" << std::left << std::setw(display_width - 1)
                << "Output Layer Information:"
                << "|" << std::endl;
        std::cout << "+" << std::setfill('-') << std::setw(display_width)
                << std::right << "+" << std::endl;
        output_scaler->PrintInfo(display_width, output_names);
        std::cout << "+" << std::setfill('-') << std::setw(display_width)
                << std::right << "+" << std::endl;
        std::cout << std::setfill(' ');
        std::cout << std::endl;
    }
  /*!
   * \brief Get network number of inputs.
   * \returns Number of network inputs
   */
  std::size_t GetnInputs() const { return n_inputs; }

  /*!
   * \brief Get network number of outputs.
   * \returns Number of network outputs
   */
  std::size_t GetnOutputs() const { return n_outputs; }

  std::size_t GetnLayers() const { return n_layers; }
  std::size_t GetnNodes(const size_t iLayer) const { return NN[iLayer]; }
  std::string GetActivationFunction(const size_t iLayer) const {return activation_functions[iLayer]->GetTag();}
  /*!
   * \brief Get network input variable name.
   * \param[in] iInput - Input variable index.
   * \returns input variable name.
   */
  std::string GetInputName(const std::size_t iInput) const {
    return input_names[iInput];
  }

  /*!
  * \brief Retrieve the network input variable names
  * \returns vector of input variables
  */
  std::vector<std::string> GetInputVars() const { return input_names; }

  /*!
  * \brief Retrieve the network output variable names
  * \returns vector of output variables
  */
  std::vector<std::string> GetOutputVars() const { return output_names; }
  

  /*!
   * \brief Get network output variable name.
   * \param[in] iOutput - Output variable index.
   * \returns output variable name.
   */
  std::string GetOutputName(const size_t iOutput=0) const {
    return output_names[iOutput];
  }

  /*!
  * \brief Check whether specified input is within the feature range of the network
  * \param[in] X_in - network input
  * \returns whether input is within network feature range
  */
  bool CheckInputInclusion(const std::vector<mlpdouble> &X_in) const {
    SetInput(X_in);
    return CheckInputInclusion();
  }

  /*!
  * \brief Check whether input is within the feature range of the network
  * \returns whether input is within network feature range
  */
  bool CheckInputInclusion() const {
    const auto dist = QueryDistance();
    if ((dist > 0) && (input_reg_method==ENUM_SCALING_FUNCTIONS::MINMAX))
        return false;
    else return true;
  }
};

} // namespace MLPToolbox
