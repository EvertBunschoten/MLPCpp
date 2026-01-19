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
     /*!
    * \brief Available activation function map.
    */
    // std::map<std::string, ENUM_ACTIVATION_FUNCTION> activation_function_map{
    //     {"none", ENUM_ACTIVATION_FUNCTION::NONE},
    //     {"linear", ENUM_ACTIVATION_FUNCTION::LINEAR},
    //     {"elu", ENUM_ACTIVATION_FUNCTION::ELU},
    //     {"relu", ENUM_ACTIVATION_FUNCTION::RELU},
    //     {"gelu", ENUM_ACTIVATION_FUNCTION::GELU},
    //     {"selu", ENUM_ACTIVATION_FUNCTION::SELU},
    //     {"sigmoid", ENUM_ACTIVATION_FUNCTION::SIGMOID},
    //     {"swish", ENUM_ACTIVATION_FUNCTION::SWISH},
    //     {"tanh", ENUM_ACTIVATION_FUNCTION::TANH},
    //     {"exponential", ENUM_ACTIVATION_FUNCTION::EXPONENTIAL}};

    size_t n_layers{0};
    size_t last_layer{0};

    mlpdouble *** weights_mat {nullptr};
    mlpdouble ** biases_mat {nullptr};
    mlpdouble ** layer_outputs {nullptr};
    mlpdouble *** layer_Jacobian {nullptr};
    mlpdouble **** layer_Hessian {nullptr};
    mlpdouble * input_layer {nullptr};
    mlpdouble * network_inputs{nullptr};

    mlpdouble * output_layer {nullptr};
    mlpdouble ** output_Jacobian {nullptr};
    mlpdouble *** Jacobian_refs {nullptr};
    mlpdouble **** Hessian_refs {nullptr};

    mlpdouble *** output_Hessian {nullptr};
    mlpdouble * input_norm_offset {nullptr};
    mlpdouble * input_norm_scale {nullptr};
    mlpdouble * output_norm_offset {nullptr};
    mlpdouble * output_norm_scale {nullptr};
    std::vector<std::pair<mlpdouble, mlpdouble>> input_norm ;
    std::vector<std::pair<mlpdouble, mlpdouble>> output_norm ;

    ENUM_SCALING_FUNCTIONS input_reg_method {ENUM_SCALING_FUNCTIONS::MINMAX},
                         output_reg_method {ENUM_SCALING_FUNCTIONS::MINMAX};
    
    ScalerFunction * input_scaler{nullptr}, *output_scaler{nullptr};
    //ActivationFunctionBase * activation_functions {nullptr};
    ActivationFunctionBase** activation_functions {nullptr};
    std::vector<std::string> activation_function_tags;
    
    size_t * NN {nullptr};
    size_t n_inputs{0};
    size_t n_outputs{0};

    std::vector<std::string> input_names;
    std::vector<std::string> output_names;

    bool calc_Jacobian {false};
    bool calc_Hessian {false};
    
    const mlpdouble default_offset{0.0},
                    default_scale{1.0},
                    default_weight{0.0},
                    default_bias{0.0};

    /*!
    * \brief Define network nodes and synapses.
    * \param[in] copy_network - Pointer to reference network used in copy constructor.
    * \param[in] reader - Pointer to reader class.
    */
    void SizeWeights(const IteratorNetwork * copy_network=nullptr, const CReadNeuralNetwork * reader=nullptr) {
        const bool from_reader = (reader != nullptr);       // Retrieve weight information from reader
        const bool from_copy = (copy_network != nullptr);   // Copy weight information from reference MLP.

        last_layer = n_layers - 1;
        n_inputs = NN[0];
        n_outputs = NN[last_layer];
        weights_mat = new mlpdouble**[last_layer];
        biases_mat = new mlpdouble * [n_layers];
        layer_outputs = new mlpdouble*[n_layers];
        layer_Jacobian = new mlpdouble**[n_layers];
        layer_Hessian = new mlpdouble***[n_layers];
        for (size_t iLayer=0; iLayer<n_layers; iLayer++) {
            if (iLayer < last_layer){
                weights_mat[iLayer] = new mlpdouble*[NN[iLayer+1]];
                for (size_t iNeuron=0; iNeuron<NN[iLayer+1]; iNeuron++){
                    weights_mat[iLayer][iNeuron] = new mlpdouble[NN[iLayer]];
                    std::fill(weights_mat[iLayer][iNeuron], weights_mat[iLayer][iNeuron]+NN[iLayer], default_weight);
                    if (from_reader){
                        for (size_t jNeuron=0; jNeuron<NN[iLayer]; jNeuron++)
                            SetWeight(iLayer, jNeuron, iNeuron, reader->GetWeight(iLayer, jNeuron, iNeuron));
                    }else if (from_copy){
                        for (size_t jNeuron=0; jNeuron<NN[iLayer]; jNeuron++)
                            SetWeight(iLayer, jNeuron, iNeuron, copy_network->weights_mat[iLayer][iNeuron][jNeuron]);
                    }
                }
            }
            biases_mat[iLayer] = new mlpdouble[NN[iLayer]];
            std::fill(biases_mat[iLayer], biases_mat[iLayer]+NN[iLayer], default_bias);
            if (from_reader){
                for (size_t iNeuron=0; iNeuron<NN[iLayer]; iNeuron++) 
                    SetBias(iLayer, iNeuron, reader->GetBias(iLayer, iNeuron));
            } else if (from_copy) {
                for (size_t iNeuron=0; iNeuron<NN[iLayer]; iNeuron++) 
                    SetBias(iLayer, iNeuron, copy_network->biases_mat[iLayer][iNeuron]);
            }
            layer_outputs[iLayer] = new mlpdouble[NN[iLayer]];
            layer_Jacobian[iLayer] = new mlpdouble*[n_inputs];
            layer_Hessian[iLayer] = new mlpdouble**[n_inputs];
            for (size_t iInput=0; iInput<n_inputs; iInput++){
                layer_Jacobian[iLayer][iInput] = new mlpdouble[NN[iLayer]];
                layer_Hessian[iLayer][iInput] = new mlpdouble*[n_inputs];
                for (size_t jInput=0; jInput < n_inputs; jInput++)
                    layer_Hessian[iLayer][iInput][jInput] = new mlpdouble[NN[iLayer]];
            }
        }
        
        input_norm_offset = new mlpdouble [n_inputs];
        std::fill(input_norm_offset, input_norm_offset+n_inputs, default_offset);
        input_norm_scale = new mlpdouble [n_inputs];
        std::fill(input_norm_scale, input_norm_scale+n_inputs, default_scale);
        output_norm_offset = new mlpdouble [n_outputs];
        std::fill(output_norm_offset, output_norm_offset+n_outputs, default_offset);
        output_norm_scale = new mlpdouble [n_outputs];
        std::fill(output_norm_scale, output_norm_scale+n_outputs, default_scale);


        input_norm.resize(n_inputs);
        std::fill(input_norm.begin(), input_norm.end(), std::make_pair(default_offset, default_scale));
        output_norm.resize(n_outputs);
        std::fill(output_norm.begin(), output_norm.end(), std::make_pair(default_offset, default_scale));
        network_inputs = new mlpdouble[n_inputs];
        output_Jacobian = layer_Jacobian[last_layer];
        output_Hessian = layer_Hessian[last_layer];
        input_names.resize(n_inputs);
        output_names.resize(n_outputs);
        
        input_layer = layer_outputs[0];
        output_layer = layer_outputs[last_layer];
        activation_functions = new ActivationFunctionBase*[n_layers];
        activation_function_tags.resize(n_layers);
        std::fill(activation_functions, activation_functions+n_layers, nullptr);
        SetActivationFunction();
        if (from_reader){
            for (size_t iLayer=0; iLayer<n_layers; iLayer++)
                SetActivationFunction(iLayer, reader->GetActivationFunction(iLayer));
        } else if (from_copy){
            for (size_t iLayer=0; iLayer<n_layers; iLayer++){
                SetActivationFunction(iLayer, copy_network->activation_function_tags[iLayer]);
            }
        }
    }

    void InputLayer(){
        for (auto iInput=0u; iInput < n_inputs; iInput++) {
            input_layer[iInput] = input_scaler->Normalize(network_inputs[iInput], iInput);// NormalizeInput(network_inputs[iInput], iInput);
            if (calc_Jacobian) {
                std::fill(layer_Jacobian[0][iInput], layer_Jacobian[0][iInput]+n_inputs, 0.0);
                layer_Jacobian[0][iInput][iInput] = 1.0 / input_scaler->GetScale(iInput);
                for (auto jInput=0u; jInput < n_inputs; jInput++) {
                    if (calc_Hessian) 
                        std::fill(layer_Hessian[0][iInput][jInput], layer_Hessian[0][iInput][jInput]+n_inputs, 0.0);
                }   
            }
        }
    }
    void OutputLayer() {
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

    // /*!
    // * \brief Normalize the network input.
    // * \param[in] val_input_dim - Dimensional input value.
    // * \param[in] iInput - Input index.
    // * \returns Normalized network input value.
    // */
    // mlpdouble NormalizeInput(const mlpdouble val_input_dim, const size_t iInput) const {
    //     mlpdouble val_norm_input{0};
    //     switch(input_reg_method)
    //     {
    //     case ENUM_SCALING_FUNCTIONS::MINMAX:
    //     val_norm_input = (val_input_dim - input_norm[iInput].first) / (input_norm[iInput].second - input_norm[iInput].first);
    //     break;
    //     case ENUM_SCALING_FUNCTIONS::STANDARD:
    //     case ENUM_SCALING_FUNCTIONS::ROBUST:
    //     default:
    //     val_norm_input= (val_input_dim - input_norm[iInput].first) / input_norm[iInput].second;
    //     break;
    //     }
    //     return val_norm_input;
    // }
    
    // mlpdouble DimensionalizeOutput(const mlpdouble val_output_norm, const size_t iOutput) const {
    //     mlpdouble val_dim_output{0};
    //     switch(input_reg_method)
    //     {
    //     case ENUM_SCALING_FUNCTIONS::MINMAX:
    //     val_dim_output = (output_norm[iOutput].second - output_norm[iOutput].first) * val_output_norm + output_norm[iOutput].first;
    //     break;
    //     case ENUM_SCALING_FUNCTIONS::STANDARD:
    //     case ENUM_SCALING_FUNCTIONS::ROBUST:
    //     default:
    //     val_dim_output = output_norm[iOutput].second * val_output_norm + output_norm[iOutput].first;
    //     break;
    //     }
    //     return val_dim_output;
    // }

    void CalcLayerOutputs(const size_t iLayer) const {
        if (iLayer==0){
            return;
        }else {
            const size_t prev_layer = iLayer-1;
            CalcLayerOutputs(prev_layer);
            
            for (size_t iNeuron=0; iNeuron < NN[iLayer]; ++iNeuron){
                const mlpdouble node_input = WeightsMultiplication(prev_layer, iNeuron, layer_outputs[prev_layer], biases_mat[iLayer][iNeuron]);
                layer_outputs[iLayer][iNeuron] = activation_functions[iLayer]->call(node_input, calc_Jacobian, calc_Hessian);
                if (calc_Jacobian) {
                    for (size_t iInput=0; iInput < n_inputs; iInput++){
                        const mlpdouble psi = WeightsMultiplication(prev_layer, iNeuron, layer_Jacobian[prev_layer][iInput]);
                        const mlpdouble phi_prime = activation_functions[iLayer]->GetJacobian();
                        layer_Jacobian[iLayer][iInput][iNeuron] = psi * phi_prime;
                        if (calc_Hessian) {
                            for (size_t jInput=0; jInput < n_inputs; jInput++){
                                const mlpdouble psi_j = (jInput==iInput) ? psi : WeightsMultiplication(prev_layer, iNeuron, layer_Jacobian[prev_layer][jInput]);
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
    void CalcLayerJacobian(const size_t iLayer, const size_t iNeuron) const {
        const mlpdouble phi_prime = activation_functions[iLayer]->GetJacobian();
        for (size_t iInput=0; iInput < n_inputs; iInput++){
            const mlpdouble psi = WeightsMultiplication(iLayer-1, iNeuron, layer_Jacobian[iLayer-1][iInput]);
            layer_Jacobian[iLayer][iInput][iNeuron] = psi * phi_prime;
        }
    }
    void CalcLayerHessian(const size_t iLayer, const size_t iNeuron) const {
        const mlpdouble phi_dprime = activation_functions[iLayer]->GetHessian();
        const mlpdouble phi_prime = activation_functions[iLayer]->GetJacobian();
        for (size_t iInput=0; iInput<n_inputs; iInput++){
            for (size_t jInput=0; jInput<n_inputs; jInput++){
                const mlpdouble psi_i = layer_Jacobian[iLayer][iInput][iNeuron];
                const mlpdouble psi_j = layer_Jacobian[iLayer][jInput][iNeuron];
                const mlpdouble chi = WeightsMultiplication(iLayer-1, iNeuron, layer_Hessian[iLayer-1][iInput][jInput]);
                layer_Hessian[iLayer][iInput][jInput][iNeuron] = phi_dprime * psi_j * psi_i + phi_prime * chi;
            }
        }
    }
    const mlpdouble WeightsMultiplication(size_t iLayer, size_t iNeuron, mlpdouble*array_in, const mlpdouble bias=0.0) const {
        return std::inner_product(weights_mat[iLayer][iNeuron], weights_mat[iLayer][iNeuron] + NN[iLayer], array_in, bias);
    }
    
    public:
        IteratorNetwork() = default;

        /*!
        * \brief Constructor from ASCII file name
        * \param[in] MLP_filename - MLPCpp ASCII file name from which to read network information.
        */
        IteratorNetwork(const std::string MLP_filename) {

            CReadNeuralNetwork reader = CReadNeuralNetwork(MLP_filename);
            reader.ReadMLPFile();
            const auto N_H = reader.GetNneurons();
            n_layers = N_H.size();
            NN = new size_t[n_layers];
            for (auto iLayer=0u; iLayer<(n_layers); iLayer++)
                NN[iLayer] = N_H[iLayer];

            SizeWeights(nullptr, &reader);

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
                output_layer = layer_outputs[last_layer];
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
                std::cout << "Error! Layers must contain at least one neuron" << std::endl;
                return;
            }
            n_layers = NN_input.size();
            NN = new size_t[n_layers];
            std::copy(NN_input.begin(), NN_input.end(), NN);
            
            SizeWeights();
            SetInputRegularization();
            SetOutputRegularization();
        }

        void SetInputRegularization(const std::string reg_method_tag="minmax") {
            input_reg_method = scaling_map[reg_method_tag];
            SetInputRegularization(input_reg_method);
        }
        void SetOutputRegularization(const std::string reg_method_tag="minmax") {
            output_reg_method = scaling_map[reg_method_tag];
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
          return;
        }

        /*!
        * \brief Set the normalization factors for the input layer
        * \param[in] iInput - Input index.
        * \param[in] input_min - Minimum input value.
        * \param[in] input_max - Maximum input value.
        */
        void SetInputNorm(const size_t iInput, const mlpdouble input_min=0.0,
                            const mlpdouble input_max=1.0) {
            input_scaler->SetScaling(iInput, input_min, input_max);
        }

        /*!
        * \brief Set the normalization factors for the output layer
        * \param[in] iOutput - Input index.
        * \param[in] input_min - Minimum output value.
        * \param[in] input_max - Maximum output value.
        */
        void SetOutputNorm(const size_t iOutput, const mlpdouble output_min=0.0,
                            const mlpdouble output_max=1.0) {
            output_scaler->SetScaling(iOutput, output_min, output_max);
        }
        

        mlpdouble GetRegularizationScale(const std::size_t iInput, const bool is_input=true) const {
            return is_input ? input_scaler->GetScale(iInput) : output_scaler->GetScale(iInput);
        }

        mlpdouble GetRegularizationOffset(const std::size_t iInput, const bool is_input=true) const {
            return is_input ? input_scaler->GetOffset(iInput) : output_scaler->GetOffset(iInput);
        }

        
        
        mlpdouble QueryDistance(const std::vector<mlpdouble> &query) const {
            return input_scaler->Distance(query);
        }
        mlpdouble QueryDistance() const {
            return input_scaler->Distance(network_inputs);
        }
        void RandomWeights(){
            std::random_device rd;  
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(-1.0, 1.0);
            for (size_t iLayer=0; iLayer<n_layers; iLayer++) {
                if (iLayer < (last_layer)){
                    for (size_t iNeuron=0; iNeuron<NN[iLayer]; iNeuron++){
                        for (size_t jNeuron=0; jNeuron<NN[iLayer+1]; jNeuron++){
                            SetWeight(iLayer, iNeuron, jNeuron, dis(gen));
                        }
                    }
                }
                for (size_t iNeuron=0; iNeuron<NN[iLayer]; iNeuron++) {
                    SetBias(iLayer, iNeuron, dis(gen));
                }
            }
        }
        

        ~IteratorNetwork() {
            for (size_t iLayer=0; iLayer<n_layers; iLayer++) {
                delete [] biases_mat[iLayer];
                delete [] layer_outputs[iLayer];
                for (size_t iInput=0; iInput<n_inputs; iInput++){
                    delete [] layer_Jacobian[iLayer][iInput];
                    for (size_t jInput=0; jInput < n_inputs; jInput++) {
                        delete [] layer_Hessian[iLayer][iInput][jInput];
                    }
                    delete [] layer_Hessian[iLayer][iInput];
                }
                delete [] layer_Jacobian[iLayer];
                delete [] layer_Hessian[iLayer];

                if (iLayer < (n_layers - 1)) {
                    for (size_t iNeuron=0; iNeuron < NN[iLayer+1]; iNeuron++) 
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
            delete [] input_norm_offset;
            delete [] input_norm_scale;
            delete [] output_norm_offset;
            delete [] output_norm_scale;
            delete [] network_inputs;
            if (input_scaler != nullptr) delete input_scaler;
            if (output_scaler != nullptr) delete output_scaler;
            
            return;
        }

        

        void SetWeight(const size_t iLayer, const size_t iNode, const size_t jNode, const mlpdouble val_weight) {
            weights_mat[iLayer][jNode][iNode] = val_weight;
        }

        void SetBias(const size_t iLayer, const size_t iNode, const mlpdouble val_bias) {
            biases_mat[iLayer][iNode] = val_bias;
        }

        void CalcJacobian(const bool enable_Jacobian=false) {
            calc_Jacobian = enable_Jacobian;
        }
        
        void CalcHessian(const bool enable_Hessian=false) {
            calc_Hessian = enable_Hessian;
        }

        void SetInput(const size_t iInput, const mlpdouble val_input) {
            network_inputs[iInput] = val_input;
            return;
        }
        void SetInput(const mlpdouble* const input_vals) {
            std::copy(input_vals, input_vals + n_inputs, input_layer);
            return;
        }
        void SetInput(const std::vector<mlpdouble> &input_vals) {
            std::copy(input_vals.begin(), input_vals.end(), input_layer);
            return;
        }
        mlpdouble * InputLayer(const size_t iInput) const { return &network_inputs[iInput];}

        mlpdouble * OutputLayer(const size_t iOutput) const {return &output_layer[iOutput];}

        mlpdouble * Jacobian(const size_t iOutput, const size_t iInput) const {return &output_Jacobian[iInput][iOutput];}
        mlpdouble * Hessian(const size_t iOutput, const size_t iInput, const size_t jInput) const {return &output_Hessian[iInput][jInput][iOutput];}

        mlpdouble GetOutput(const size_t iOutput) const {
            return output_layer[iOutput];
        }

        mlpdouble GetJacobian(const size_t iOutput, const size_t iInput) const {
            return output_Jacobian[iInput][iOutput];
        }
        mlpdouble GetHessian(const size_t iOutput, const size_t iInput, const size_t jInput) const {
            return output_Hessian[iInput][jInput][iOutput];
        }

        void Predict(const std::vector<mlpdouble> &X_in, const bool evaluate_Jacobian=false, const bool evaluate_Hessian=false) {
            std::copy(X_in.begin(), X_in.end(), network_inputs);
            calc_Jacobian=evaluate_Jacobian;
            calc_Hessian=evaluate_Hessian;
            Predict();
        }
        
        void Predict() {
            InputLayer();
            CalcLayerOutputs(last_layer);
            OutputLayer();
        }

        void SetActivationFunction(const size_t iLayer, const std::string name_activation_function="") {
            ENUM_ACTIVATION_FUNCTION i_phi = activation_function_map[name_activation_function];
            activation_function_tags[iLayer] = name_activation_function;
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

        void SetActivationFunction(const std::string name_activation_function="") {
            for (size_t iLayer=0; iLayer < n_layers; iLayer++) {
                SetActivationFunction(iLayer, name_activation_function);
            }
        }
        /*!
        * \brief Add an output variable name to the network.
        * \param[in] input - Input variable name.
        */
        void SetOutputName(const size_t iOutput, const std::string input) {
            output_names[iOutput] = input;
        }

        /*!
        * \brief Add an input variable name to the network.
        * \param[in] output - Output variable name.
        */
        void SetInputName(const size_t iInput, const std::string input) {
            input_names[iInput] = input;
        }
    /*!
    * \brief Display the network architecture in the terminal.
    */
    void DisplayNetwork() const {
        /*--- Display information on the MLP architecture ---*/
        int display_width = 54;
        int column_width = int(display_width / 3.0) - 1;

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
        for (size_t iLayer = 1; iLayer < last_layer; iLayer++)
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

  /*!
   * \brief Get network input variable name.
   * \param[in] iInput - Input variable index.
   * \returns input variable name.
   */
  std::string GetInputName(const std::size_t iInput) const {
    return input_names[iInput];
  }

  std::vector<std::string> GetInputVars() const { return input_names; }
  std::vector<std::string> GetOutputVars() const { return output_names; }
  

  /*!
   * \brief Get network output variable name.
   * \param[in] iOutput - Output variable index.
   * \returns output variable name.
   */
  std::string GetOutputName(const size_t iOutput=0) const {
    return output_names[iOutput];
  }

  std::pair<mlpdouble, mlpdouble> GetInputNorm(const size_t iInput) const {
    return input_norm[iInput];
  }

  std::pair<mlpdouble, mlpdouble> GetOutputNorm(const size_t iOutput=0) const {
    return output_norm[iOutput];
  }

  bool CheckInputInclusion() const {
    auto dist = QueryDistance();
    if ((dist > 0) && (input_reg_method==ENUM_SCALING_FUNCTIONS::MINMAX))
        return false;
    else return true;
  }

  bool CheckInputInclusion(const mlpdouble val_input, const size_t iInput) const {
    bool inside {true};
    mlpdouble val_input_norm;
    switch(input_reg_method)
    {
    case ENUM_SCALING_FUNCTIONS::MINMAX:
      if ((val_input < input_norm[iInput].first) || 
          (val_input > input_norm[iInput].second))
          inside = false;
      break;
    case ENUM_SCALING_FUNCTIONS::STANDARD:
      val_input_norm = (val_input - input_norm[iInput].first) / input_norm[iInput].second;
      if ((val_input_norm < -2.0) || (val_input_norm > 2.0))
        inside = false;
      break;
    case ENUM_SCALING_FUNCTIONS::ROBUST:
      val_input_norm = (val_input - input_norm[iInput].first) / input_norm[iInput].second;
      if ((val_input_norm < -10.0) || (val_input_norm > 10.0))
        inside = false;
      break;
    default:
      break;
    }
    
    return inside;
  }
};

} // namespace MLPToolbox
