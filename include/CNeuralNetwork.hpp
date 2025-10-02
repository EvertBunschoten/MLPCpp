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
#include "variable_def.hpp"

#include "ActivationFunctions.hpp"
#include "option_maps.hpp"

namespace MLPToolbox {
  class IteratorNetwork {
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

    size_t n_layers =0;
    size_t last_layer;
    mlpdouble *** weights_mat {nullptr};
    mlpdouble ** biases_mat {nullptr};
    mlpdouble ** layer_outputs {nullptr};
    mlpdouble *** layer_Jacobian {nullptr};
    mlpdouble **** layer_Hessian {nullptr};
    mlpdouble * input_layer {nullptr};
    mlpdouble * output_layer {nullptr};
    mlpdouble ** output_Jacobian {nullptr};
    mlpdouble *** output_Hessian {nullptr};
    mlpdouble * input_norm_offset {nullptr};
    mlpdouble * input_norm_scale {nullptr};
    mlpdouble * output_norm_offset {nullptr};
    mlpdouble * output_norm_scale {nullptr};
    std::vector<std::pair<mlpdouble, mlpdouble>> input_norm ;
    std::vector<std::pair<mlpdouble, mlpdouble>> output_norm ;

    ENUM_SCALING_FUNCTIONS input_reg_method {ENUM_SCALING_FUNCTIONS::MINMAX},
                         output_reg_method {ENUM_SCALING_FUNCTIONS::MINMAX};
    //ActivationFunctionBase * activation_functions {nullptr};
    ActivationFunctionBase** activation_functions {nullptr};
    
    size_t * NN {nullptr};
    size_t n_inputs{0};
    size_t n_outputs{0};

    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    
    public:
        IteratorNetwork(std::vector<size_t> NN_input) {
            n_layers = NN_input.size();
            last_layer = n_layers - 1;
            NN = new size_t[n_layers];
            activation_functions = new ActivationFunctionBase*[n_layers]; 
            for (auto iLayer=0u; iLayer<(n_layers); iLayer++)
                NN[iLayer] = NN_input[iLayer];
            
            n_inputs = NN[0];
            n_outputs = NN[last_layer];
            SizeWeights();
            input_layer = layer_outputs[0];
            output_layer = layer_outputs[last_layer];
        }
        /*!
        * \brief Define the regularization method used to normalize the inputs before feeding them to the network.
        * \param[in] reg_method_input - regularization method (minmax, standard, or robust).
        */
        void SetInputRegularization(ENUM_SCALING_FUNCTIONS reg_method_input) {
          input_reg_method = reg_method_input;
          return;
        }

        /*!
        * \brief Define the regularization method used to normalize the training data before training. 
        * \param[in] reg_method_input - regularization method (minmax, standard, or robust).
        */
        void SetOutputRegularization(ENUM_SCALING_FUNCTIONS reg_method_input) {
          output_reg_method = reg_method_input;
          return;
        }

        /*!
        * \brief Set the normalization factors for the input layer
        * \param[in] iInput - Input index.
        * \param[in] input_min - Minimum input value.
        * \param[in] input_max - Maximum input value.
        */
        void SetInputNorm(size_t iInput, mlpdouble input_min,
                            mlpdouble input_max) {
            input_norm[iInput] = std::make_pair(input_min, input_max);
        }

        /*!
        * \brief Set the normalization factors for the output layer
        * \param[in] iOutput - Input index.
        * \param[in] input_min - Minimum output value.
        * \param[in] input_max - Maximum output value.
        */
        void SetOutputNorm(size_t iOutput, mlpdouble output_min,
                            mlpdouble output_max) {
            output_norm[iOutput] = std::make_pair(output_min, output_max);
        }
        /*!
        * \brief Normalize the network input.
        * \param[in] val_input_dim - Dimensional input value.
        * \param[in] iInput - Input index.
        * \returns Normalized network input value.
        */
        mlpdouble NormalizeInput(mlpdouble val_input_dim, std::size_t iInput) const {
            mlpdouble val_norm_input{0};
            switch(input_reg_method)
            {
            case ENUM_SCALING_FUNCTIONS::MINMAX:
            val_norm_input = (val_input_dim - input_norm[iInput].first) / (input_norm[iInput].second - input_norm[iInput].first);
            break;
            case ENUM_SCALING_FUNCTIONS::STANDARD:
            case ENUM_SCALING_FUNCTIONS::ROBUST:
            default:
            val_norm_input= (val_input_dim - input_norm[iInput].first) / input_norm[iInput].second;
            break;
            }
            return val_norm_input;
        }

        mlpdouble GetRegularizationScale(std::size_t iInput, bool is_input=true) const {
            switch(input_reg_method)
            {
            case ENUM_SCALING_FUNCTIONS::MINMAX:
            if (is_input) {
                return input_norm[iInput].second - input_norm[iInput].first;
            } else {
                return output_norm[iInput].second - output_norm[iInput].first;
            }
            break;
            case ENUM_SCALING_FUNCTIONS::STANDARD:
            case ENUM_SCALING_FUNCTIONS::ROBUST:
            if (is_input) {
                return input_norm[iInput].second;
            } else {
                return output_norm[iInput].second;
            }
            break;
            default:
            return 0;
            break;
            }
        }

        mlpdouble GetRegularizationOffset(std::size_t iInput, bool is_input=true) const {
            switch(input_reg_method)
            {
            case ENUM_SCALING_FUNCTIONS::MINMAX:
            if (is_input) {
                return 0.5*(input_norm[iInput].second + input_norm[iInput].first);
            } else {
                return 0.5*(output_norm[iInput].second + output_norm[iInput].first);
            }
            break;
            case ENUM_SCALING_FUNCTIONS::STANDARD:
            case ENUM_SCALING_FUNCTIONS::ROBUST:
            if (is_input) {
                return input_norm[iInput].first;
            } else {
                return output_norm[iInput].first;
            }
            break;
            default:
            return 0;
            break;
            }
        }

        mlpdouble DimensionalizeOutput(mlpdouble val_output_norm, std::size_t iOutput) const {
            mlpdouble val_dim_output{0};
            switch(input_reg_method)
            {
            case ENUM_SCALING_FUNCTIONS::MINMAX:
            val_dim_output = (output_norm[iOutput].second - output_norm[iOutput].first) * val_output_norm + output_norm[iOutput].first;
            break;
            case ENUM_SCALING_FUNCTIONS::STANDARD:
            case ENUM_SCALING_FUNCTIONS::ROBUST:
            default:
            val_dim_output = output_norm[iOutput].second * val_output_norm + output_norm[iOutput].first;
            break;
            }
            return val_dim_output;
        }


        void SizeWeights() {
            weights_mat = new mlpdouble**[n_layers-1];
           
            biases_mat = new mlpdouble * [n_layers];
            layer_outputs = new mlpdouble*[n_layers];
            layer_Jacobian = new mlpdouble**[n_layers];
            layer_Hessian = new mlpdouble***[n_layers];
            for (size_t iLayer=0; iLayer<n_layers; iLayer++) {
                if (iLayer < (n_layers - 1)){
                    weights_mat[iLayer] = new mlpdouble*[NN[iLayer+1]];
                    for (size_t iNeuron=0; iNeuron<NN[iLayer+1]; iNeuron++)
                        weights_mat[iLayer][iNeuron] = new mlpdouble[NN[iLayer]];
                }
                biases_mat[iLayer] = new mlpdouble[NN[iLayer]];
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
            input_norm_scale = new mlpdouble [n_inputs];
            output_norm_offset = new mlpdouble [NN[last_layer]];
            output_norm_scale = new mlpdouble [NN[last_layer]];

            input_norm.resize(n_inputs);
            output_norm.resize(n_outputs);

            output_Jacobian = layer_Jacobian[last_layer];
            output_Hessian = layer_Hessian[last_layer];
            input_names.resize(n_inputs);
            output_names.resize(NN[last_layer]);
            return;
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
            return;
        }

        void CalcLayerOutputs(int iLayer, bool calc_Jacobian=false, bool calc_Hessian=false) {
            
            if (iLayer==0){
                return;
            }else {
                size_t prev_layer = iLayer-1;
                CalcLayerOutputs(prev_layer, calc_Jacobian, calc_Hessian);
                
                for (size_t iNeuron=0; iNeuron < NN[iLayer]; ++iNeuron){
                    mlpdouble node_input = WeightsMultiplication(prev_layer, iNeuron, layer_outputs[prev_layer], biases_mat[iLayer][iNeuron]);
                    layer_outputs[iLayer][iNeuron] = activation_functions[iLayer]->call(node_input, calc_Jacobian, calc_Hessian);
                    
                    if (calc_Jacobian) {
                        for (size_t iInput=0; iInput < n_inputs; iInput++){
                            mlpdouble psi = WeightsMultiplication(prev_layer, iNeuron, layer_Jacobian[prev_layer][iInput]);
                            mlpdouble phi_prime = activation_functions[iLayer]->GetJacobian();
                            layer_Jacobian[iLayer][iInput][iNeuron] = psi * phi_prime;
                            if (calc_Hessian) {
                                for (size_t jInput=0; jInput < n_inputs; jInput++){
                                    mlpdouble psi_j = (jInput==iInput) ? psi : WeightsMultiplication(prev_layer, iNeuron, layer_Jacobian[prev_layer][jInput]);
                                    mlpdouble phi_dprime = activation_functions[iLayer]->GetHessian();
                                    mlpdouble chi = WeightsMultiplication(prev_layer, iNeuron, layer_Hessian[prev_layer][iInput][jInput]);
                                    layer_Hessian[iLayer][iInput][jInput][iNeuron] = phi_dprime * psi_j * psi + phi_prime * chi;
                                }
                            }
                        }
                    }
                }       
                return;
            }
        }
        mlpdouble WeightsMultiplication(size_t iLayer, size_t iNeuron, mlpdouble*array_in, const mlpdouble bias=0.0) const {
            mlpdouble y = std::inner_product(weights_mat[iLayer][iNeuron], weights_mat[iLayer][iNeuron] + NN[iLayer], array_in, bias);
            return y;
        }
        

        void SetWeight(size_t iLayer, size_t iNode, size_t jNode, const mlpdouble val_weight) {
            weights_mat[iLayer][jNode][iNode] = val_weight;
        }

        void SetBias(size_t iLayer, size_t iNode, const mlpdouble val_bias) {
            biases_mat[iLayer][iNode] = val_bias;
        }

        void SetInput(size_t iInput, const mlpdouble val_input) {
            input_layer[iInput] = val_input;
            return;
        }

        void SetInput(const mlpdouble* const input_vals) {
            std::copy(input_vals, input_vals + n_inputs, input_layer);
            return;
        }

        mlpdouble GetOutput(size_t iOutput) {
            return output_layer[iOutput];
        }

        mlpdouble GetJacobian(size_t iOutput, size_t iInput) {
            return output_Jacobian[iInput][iOutput];
        }
        mlpdouble GetHessian(size_t iOutput, size_t iInput, size_t jInput) {
            return output_Hessian[iInput][jInput][iOutput];
        }
        // /*!
        // * \brief Set the normalization factors for the input layer
        // * \param[in] iInput - Input index.
        // * \param[in] input_min - Minimum input value.
        // * \param[in] input_max - Maximum input value.
        // */
        // void SetInputNorm(size_t iInput, mlpdouble input_offset,
        //                   mlpdouble input_scale) {
        //   input_norm_offset[iInput] = input_offset;
        //   input_norm_scale[iInput] = input_scale;
        // }

        // /*!
        // * \brief Set the normalization factors for the output layer
        // * \param[in] iOutput - Input index.
        // * \param[in] input_min - Minimum output value.
        // * \param[in] input_max - Maximum output value.
        // */
        // void SetOutputNorm(size_t iOutput, mlpdouble output_offset,
        //                   mlpdouble output_scale) {
        //   output_norm_offset[iOutput] = output_offset;
        //   output_norm_scale[iOutput] = output_scale;
        // }

        
        void Predict(std::vector<mlpdouble> &X_in, bool calc_Jacobian=false, bool calc_Hessian=false) {
            for (auto iInput=0u; iInput < n_inputs; iInput++) {
                input_layer[iInput] = NormalizeInput(X_in[iInput], iInput);//(X_in[iInput] - input_norm_offset[iInput])/input_norm_scale[iInput];
                if (calc_Jacobian) {
                    for (auto jInput=0u; jInput < n_inputs; jInput++) {
                        layer_Jacobian[0][iInput][jInput] = 0.0;
                        if (calc_Hessian) {
                            for (auto kInput=0u; kInput < n_inputs; kInput++)
                                layer_Hessian[0][iInput][jInput][kInput] = 0.0;
                        }
                    }   
                    layer_Jacobian[0][iInput][iInput] = 1.0 / GetRegularizationScale(iInput, true);
                }
            }
            CalcLayerOutputs(int(last_layer), calc_Jacobian, calc_Hessian);
            

            for (auto iOutput=0u; iOutput < n_outputs; iOutput++) {
                output_layer[iOutput] = DimensionalizeOutput(output_layer[iOutput], iOutput);// [iOutput] + output_norm_scale[iOutput]*output_layer[iOutput];
                if (calc_Jacobian) {
                    for (auto iInput=0u; iInput < n_inputs; iInput++) {
                        output_Jacobian[iInput][iOutput] *= GetRegularizationScale(iOutput, false);
                        for (auto jInput=0u; jInput < n_inputs; jInput++) {
                            output_Hessian[iInput][jInput][iOutput] *= GetRegularizationScale(iOutput, false);
                        }
                    }
                        
                }
            }
        }

        void SetActivationFunction(size_t iLayer, std::string name_activation_function) {
            ENUM_ACTIVATION_FUNCTION i_phi = activation_function_map[name_activation_function];
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
            activation_functions[iLayer] = function_out;
        }

        void SetActivationFunction(std::string name_activation_function) {
            for (size_t iLayer=0; iLayer < n_layers; iLayer++) {
                SetActivationFunction(iLayer, name_activation_function);
            }
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
        
        
        std::string val_disp_1, val_disp_2, label_disp_1, label_disp_2, norm_method;
        switch (input_reg_method)
        {
        case ENUM_SCALING_FUNCTIONS::MINMAX:
        label_disp_1 = "Lower limit";
        label_disp_2 = "Upper limit";
        norm_method = "minimum-maximum";
        break;
        case ENUM_SCALING_FUNCTIONS::STANDARD:
        label_disp_1 = "Mean";
        label_disp_2 = "std";
        norm_method = "mean-standard deviation";
        break;
        case ENUM_SCALING_FUNCTIONS::ROBUST:
        label_disp_1 = "Mean";
        label_disp_2 = "IQ range";
        norm_method = "Quantile range";
        break;
        default:
        break;
        }
        std::cout << "|" << std::left << std::setw(column_width)
                << "Input Normalization:";
        std::cout << std::setfill(' ') << std::right << std::setw(column_width) << norm_method
                << std::setfill(' ') << std::setw(column_width) << std::right << "|" << std::endl;
        std::cout << "+" << std::setfill('-') << std::setw(display_width)
                << std::right << "+" << std::endl;
        std::cout << std::setfill(' ');
        std::cout << "|" << std::left << std::setw(column_width)
                << "Input Variable:";
        std::cout << "|" << std::left << std::setw(column_width) << label_disp_1
                << "|" << std::left << std::setw(column_width) << label_disp_2
                << "|" << std::endl;      
        std::cout << "+" << std::setfill('-') << std::setw(display_width)
                << std::right << "+" << std::endl;
        std::cout << std::setfill(' ');

        /*--- Hidden layer information ---*/
        for (auto iInput = 0u; iInput < n_inputs; iInput++)
        std::cout << "|" << std::left << std::setw(column_width)
                    << std::to_string(iInput + 1) + ": " + input_names[iInput]
                    << "|" << std::right << std::setw(column_width)
                    << input_norm[iInput].first << "|" << std::right
                    << std::setw(column_width) << input_norm[iInput].second << "|"
                    << std::endl;
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

        /*--- Output layer information ---*/
        switch (output_reg_method)
        {
        case ENUM_SCALING_FUNCTIONS::MINMAX:
        label_disp_1 = "Lower limit";
        label_disp_2 = "Upper limit";
        norm_method = "minimum-maximum";
        break;
        case ENUM_SCALING_FUNCTIONS::STANDARD:
        label_disp_1 = "Mean";
        label_disp_2 = "std";
        norm_method = "mean-standard deviation";
        break;
        case ENUM_SCALING_FUNCTIONS::ROBUST:
        label_disp_1 = "Mean";
        label_disp_2 = "IQ range";
        norm_method = "Quantile range";
        break;
        default:
        break;
        }
        std::cout << "|" << std::left << std::setw(display_width - 1)
                << "Output Layer Information:"
                << "|" << std::endl;
        std::cout << "+" << std::setfill('-') << std::setw(display_width)
                << std::right << "+" << std::endl;
        std::cout << std::setfill(' ');
        std::cout << "|" << std::left << std::setw(column_width)
                << "Output Variable:";
        std::cout << "|" << std::left << std::setw(column_width) << label_disp_1
                << "|" << std::left << std::setw(column_width) << label_disp_2
                << "|" << std::endl;
        std::cout << "+" << std::setfill('-') << std::setw(display_width)
                << std::right << "+" << std::endl;
        std::cout << std::setfill(' ');
        for (auto iOutput = 0u; iOutput < n_outputs; iOutput++)
        std::cout << "|" << std::left << std::setw(column_width)
                    << std::to_string(iOutput + 1) + ": " + output_names[iOutput]
                    << "|" << std::right << std::setw(column_width)
                    << output_norm[iOutput].first << "|" << std::right
                    << std::setw(column_width) << output_norm[iOutput].second << "|"
                    << std::endl;
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

  std::pair<mlpdouble, mlpdouble> GetInputNorm(unsigned long iInput) const {
    return input_norm[iInput];
  }

  std::pair<mlpdouble, mlpdouble> GetOutputNorm(unsigned long iOutput) const {
    return output_norm[iOutput];
  }

  bool CheckInputInclusion(mlpdouble val_input, size_t iInput) const {
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
