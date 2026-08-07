/*!
* \file ActivationFunctions.hpp
* \brief Activation functions supported by MLPCpp
* \author E.C.Bunschoten
* \version 2.1.1
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
#include "variable_def.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace MLPToolbox {
class ActivationFunctionBase {
  /*! \brief Base class for the hidden layer activation function.
  The activation function calculates the node output, Jacobian, and Hessian
   based on the weighted output of the nodes in the previous layer of the
  network. */
protected:
  std::string name{"None"},  /* Activation function display name. */
      tag{"none"};           /* Tag used to identify the activation function. */
  mlpdouble output{0};       /* Activation function output. */
  mlpdouble Jacobian{0};     /* Derivative of output w.r.t input. */
  mlpdouble Hessian{0};      /* Second derivative of output w.r.t input. */
  bool calc_gradient{false}; /* Enable derivative calculation. */
  bool calc_gradient_2{false}; /* Enable second derivative calculation. */
public:
  ActivationFunctionBase() = delete;
  ActivationFunctionBase(const std::string &name_in) : name(name_in) {
    tag = name_in;
    std::transform(tag.begin(), tag.end(), tag.begin(),
                   [](unsigned char c) { return std::tolower(c); });
  }

  virtual ~ActivationFunctionBase() = default;

  /*!
   * \brief Retrieve activation function output.
   * \returns - activation function output.
   */
  mlpdouble GetOutput() const { return output; }

  /*!
   * \brief Retrieve activation function derivative.
   * \returns - activation function derivative.
   */
  mlpdouble GetJacobian() const { return Jacobian; }

  /*!
   * \brief Retrieve activation function second derivative.
   * \returns - activation function second derivative.
   */
  mlpdouble GetHessian() const { return Hessian; }

  /*!
   * \brief Retrieve the activation function display name
   * \returns - activation function display name.
   */
  std::string GetName() const { return name; }

  /*!
   * \brief Retrieve the activation function ID tag.
   * \returns - activation function ID tag.
   */
  std::string GetTag() const { return tag; }

  /*!
   * \brief Call operator to evaluate function output and derivatives.
   * \param[in] x - activation function input.
   * \param[in] calc_Jacobian - calculate derivative value.
   * \param[in] calc_Hessian - calculate second-order derivative.
   * \returns - activation function output.
   */
  virtual mlpdouble operator()(const mlpdouble x,
                               const bool calc_Jacobian = false,
                               const bool calc_Hessian = false) = 0;
};

class Lin final : public ActivationFunctionBase {
  /*! \brief Linear activation function (output = input), used for input and
   * output layer.*/
public:
  Lin() : ActivationFunctionBase("Linear") {}
  mlpdouble operator()(const mlpdouble x, const bool calc_Jacobian = false,
                       const bool calc_Hessian = false) override {
    output = x;
    if (calc_Jacobian)
      Jacobian = 1.0;
    if (calc_Hessian)
      Hessian = 0.0;
    return output;
  }
};

class Elu final : public ActivationFunctionBase {
  /*! \brief Exponential linear unit function. */
public:
  Elu() : ActivationFunctionBase("Elu") {}
  mlpdouble operator()(const mlpdouble x, const bool calc_Jacobian = false,
                       const bool calc_Hessian = false) override {
    if (x > 0) {
      if (calc_Jacobian)
        Jacobian = 1.0;
      if (calc_Hessian)
        Hessian = 0.0;
      output = x;
    } else {
      mlpdouble exp_x = exp(x);
      if (calc_Jacobian)
        Jacobian = exp_x;
      if (calc_Hessian)
        Hessian = exp_x;
      output = exp_x - 1;
    }
    return output;
  }
};

class Sigmoid final : public ActivationFunctionBase {
  /*! \brief Sigmoid activation function. */
public:
  Sigmoid() : ActivationFunctionBase("Sigmoid") {}
  virtual mlpdouble operator()(const mlpdouble x,
                               const bool calc_Jacobian = false,
                               const bool calc_Hessian = false) {
    const mlpdouble exp_x = exp(x);
    output = exp_x / (1 + exp_x);
    if (calc_Jacobian) {
      Jacobian = exp_x / pow(1 + exp_x, 2);
      if (calc_Hessian) {
        Hessian = -(exp_x * (exp_x - 1)) / pow(exp_x + 1, 3);
      }
    }
    return output;
  }
};

class Exponential final : public ActivationFunctionBase {
  /*! \brief Exponential function. */
public:
  Exponential() : ActivationFunctionBase("Exponential") {}
  virtual mlpdouble operator()(const mlpdouble x,
                               const bool calc_Jacobian = false,
                               const bool calc_Hessian = false) {
    output = exp(x);
    if (calc_Jacobian)
      Jacobian = output;
    if (calc_Hessian)
      Hessian = output;
    return output;
  }
};

class Relu final : public ActivationFunctionBase {
  /*! \brief Rectified linear unit activation function. */
public:
  Relu() : ActivationFunctionBase("ReLu") {}
  virtual mlpdouble operator()(const mlpdouble x,
                               const bool calc_Jacobian = false,
                               const bool calc_Hessian = false) {
    if (x > 0) {
      output = x;
      if (calc_Jacobian)
        Jacobian = 1.0;
    } else {
      output = 0.0;
      if (calc_Jacobian)
        Jacobian = 0.0;
    }
    if (calc_Hessian)
      Hessian = 0.0;
    return output;
  }
};

class Swish final : public ActivationFunctionBase {
  /*! \brief Swish or sigmoid linear unit activation function. */
public:
  Swish() : ActivationFunctionBase("Swish") {}
  virtual mlpdouble operator()(const mlpdouble x,
                               const bool calc_Jacobian = false,
                               const bool calc_Hessian = false) {
    const mlpdouble exp_x = exp(x);
    output = x * exp_x / (1 + exp_x);
    if (calc_Jacobian) {
      Jacobian = exp_x * (x + exp_x + 1) / pow(exp_x + 1, 2);
      if (calc_Hessian)
        Hessian = exp_x * (-exp_x * (x - 2) + x + 2) / pow(exp_x + 1, 3);
    }
    return output;
  }
};

class Tanh final : public ActivationFunctionBase {
public:
  Tanh() : ActivationFunctionBase("Tanh") {}
  virtual mlpdouble operator()(const mlpdouble x,
                               const bool calc_Jacobian = false,
                               const bool calc_Hessian = false) {
    const mlpdouble tnh = tanh(x);
    output = tnh;
    if (calc_Jacobian) {
      Jacobian = pow(cosh(x), -2);
      if (calc_Hessian) {
        Hessian = -2 * tnh * Jacobian;
      }
    }
    return output;
  }
};

class SeLu final : public ActivationFunctionBase {
  /*! \brief Scaled exponential linear unit activation function. */
private:
  const mlpdouble lambda{1.05070098};
  const mlpdouble alpha{1.67326324};

public:
  SeLu() : ActivationFunctionBase("SeLu") {}
  virtual mlpdouble operator()(const mlpdouble x,
                               const bool calc_Jacobian = false,
                               const bool calc_Hessian = false) {
    if (x > 0) {
      output = lambda * x;
      if (calc_Jacobian) {
        Jacobian = lambda;
        if (calc_Hessian) {
          Hessian = 0.0;
        }
      }
    } else {
      const mlpdouble exp_x = exp(x);
      output = lambda * alpha * (exp_x - 1);
      if (calc_Jacobian) {
        Jacobian = output + lambda * alpha;
        if (calc_Hessian) {
          Hessian = Jacobian;
        }
      }
    }
    return output;
  }
};

class GeLu final : public ActivationFunctionBase {
  /*! \brief Gaussian error linear unit activation function. */
private:
  const mlpdouble gelu_c{0.5 * sqrt(2)}, pi_sqrt{sqrt(2 / M_PI)};

public:
  GeLu() : ActivationFunctionBase("GeLu") {}
  virtual mlpdouble operator()(const mlpdouble x,
                               const bool calc_Jacobian = false,
                               const bool calc_Hessian = false) {
    output = 0.5 * x * (1 + erf(x / sqrt(2)));
    if (calc_Jacobian) {
      Jacobian = 0.5 + 0.5 * pi_sqrt * exp(-0.5 * pow(x, 2)) * x +
                 0.5 * erf(x / sqrt(2));
      if (calc_Hessian)
        Hessian = pi_sqrt * exp(-0.5 * pow(x, 2)) * (1 - 0.5 * pow(x, 2));
    }
    return output;
  }
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

static const std::map<std::string, ENUM_ACTIVATION_FUNCTION>
    activation_function_map{
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

class UnknownActivationFunctionException : public std::exception {
private:
  std::string function_name;

public:
  UnknownActivationFunctionException(const std::string &f) noexcept
      : function_name{f} {};
  ~UnknownActivationFunctionException() noexcept = default;
  virtual const char *what() const noexcept {
    std::string msg =
        "Activation function not recognized (" + function_name + ")";
    return msg.c_str();
  }
};

static ENUM_ACTIVATION_FUNCTION
RetrieveActivationEnum(const std::string &tag_function) {
  const auto it = activation_function_map.find(tag_function);
  if (it == activation_function_map.end())
    throw UnknownActivationFunctionException(tag_function);
  return it->second;
};

static ActivationFunctionBase *
RetrieveActivationFunction(const std::string &function_name) {
  const auto i_phi = RetrieveActivationEnum(function_name);
  ActivationFunctionBase *function_out;
  switch (i_phi) {
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
  return function_out;
}
} // namespace MLPToolbox
