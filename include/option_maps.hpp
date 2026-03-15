/*!
* \file option_maps.hpp
* \brief General enumerations, options, and exceptions.
* \author E.C.Bunschoten
* \version 2.1.0
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
#include <string>
#include <map>

/*!
* \brief Available activation function enumeration.
*/
enum class ENUM_SCALING_FUNCTIONS {
MINMAX = 0,
STANDARD = 1,
ROBUST = 2,
};

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
  
static const std::map<std::string, ENUM_ACTIVATION_FUNCTION> activation_function_map{
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

static const std::map<std::string, ENUM_SCALING_FUNCTIONS> scaling_map{
      {"minmax", ENUM_SCALING_FUNCTIONS::MINMAX},
      {"standard", ENUM_SCALING_FUNCTIONS::STANDARD},
      {"robust", ENUM_SCALING_FUNCTIONS::ROBUST},
  };



static void ErrorMessage(const std::string ErrorMsg, const std::string FunctionName) {
  std::cerr << std::endl << std::endl;
  std::cerr << "Error in \"" << FunctionName << "\": " << std::endl;
  std::cerr << "+" << std::setfill('-') << std::setw(54) << std::right << "+" << std::endl;
  std::cerr << ErrorMsg << std::endl;
  std::cerr << "+" << std::setfill('-') << std::setw(54) << std::right << "+" << std::endl;
  std::cerr << std::endl << std::endl;
  exit(EXIT_FAILURE);
} 
