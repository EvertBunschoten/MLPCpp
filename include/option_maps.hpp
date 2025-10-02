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

// std::map<std::string, ENUM_SCALING_FUNCTIONS> scaling_map{
//     {"minmax", ENUM_SCALING_FUNCTIONS::MINMAX},
//     {"standard", ENUM_SCALING_FUNCTIONS::STANDARD},
//     {"robust", ENUM_SCALING_FUNCTIONS::ROBUST},
// };
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

 

