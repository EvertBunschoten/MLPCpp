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
