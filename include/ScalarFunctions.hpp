#pragma once
#include <vector>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <random>
#include <limits>
#include <cmath>
#include <cstdlib>
#include <string>
#include <map>
#include "variable_def.hpp"
#include "option_maps.hpp"

namespace MLPToolbox {
class ScalerFunction {
    size_t n_scalars{0};
    std::vector<mlpdouble> norm_scalars;
    public:
    ScalerFunction(const size_t n_in) : n_scalars {n_in} {norm_scalars.resize(n_scalars);};
    virtual mlpdouble Normalize(const mlpdouble scalar_dim, const size_t i_scalar) const =0;
    virtual mlpdouble Dimensionalize(const mlpdouble scalar_dim, const size_t i_scalar) const =0;
    virtual mlpdouble Distance(const std::vector<mlpdouble>) const = 0;
    virtual mlpdouble Distance(const mlpdouble*) const = 0;
    virtual void SetScaling(const size_t, const mlpdouble, const mlpdouble) =0;
    virtual mlpdouble GetScale(const size_t) const =0;
    virtual mlpdouble GetOffset(const size_t) const = 0;
    virtual void PrintInfo(const int display_width, const std::vector<std::string>&) const =0;
};


class StandardScaler : public ScalerFunction {
    public: 

    std::vector<mlpdouble> vals_mu;
    std::vector<mlpdouble> vals_std;
    StandardScaler(const size_t n_in) : ScalerFunction(n_in) {
        vals_mu.resize(n_in); 
        vals_std.resize(n_in);
        std::fill(vals_mu.begin(), vals_mu.end(), 0.0);
        std::fill(vals_std.begin(), vals_std.end(), 1.0);
    };
    virtual mlpdouble GetScale(const size_t i_scalar) const {return vals_std[i_scalar]; }
    virtual mlpdouble GetOffset(const size_t i_scalar) const {return vals_mu[i_scalar]; }
    
    virtual void SetScaling(const size_t i_in, const mlpdouble val_mu=0.0, const mlpdouble val_std=1.0) 
    {
        if (val_std < 0) 
            std::cout << "Error: standard deviation should be positive" << std::endl;
        vals_mu[i_in] = val_mu;
        vals_std[i_in] = val_std;
    }

    virtual mlpdouble Normalize(const mlpdouble scalar_dim, const size_t i_scalar) const
    {
        mlpdouble val_norm = (scalar_dim - vals_mu[i_scalar])/(vals_std[i_scalar]);
        return val_norm;
    };

    virtual mlpdouble Dimensionalize(const mlpdouble scalar_norm, const size_t i_scalar) const {
        mlpdouble val_dim = vals_mu[i_scalar] + scalar_norm * vals_std[i_scalar];
        return val_dim;
    };

    virtual mlpdouble Distance(const std::vector<mlpdouble> scalar_dim) const 
    {
        mlpdouble val_dist{0};
        for (auto iDim=0u; iDim<vals_mu.size(); iDim++) {
            val_dist += pow(Normalize(scalar_dim[iDim], iDim), 2);
        }
        return val_dist;
    };

    virtual mlpdouble Distance(const mlpdouble* scalar_dim) const 
    {
        mlpdouble val_dist{0};
        for (auto iDim=0u; iDim<vals_mu.size(); iDim++) {
            val_dist += pow(Normalize(scalar_dim[iDim], iDim), 2);
        }
        return val_dist;
    };
    virtual void PrintInfo(const int display_width, const std::vector<std::string> &input_names) const {
        const int column_width = int(display_width / 3.0) - 1;
        std::cout << "|" << std::setfill(' ') << std::left << std::setw(display_width - 1)
                << "Standard scaling" 
                << "|" << std::endl;std::cout << "+" << std::setfill('-') << std::setw(display_width)
                << std::right << "+" << std::endl;
        std::cout << std::setfill(' ');
        std::cout << "|" << std::left << std::setw(column_width)
                << "Variable:";
        std::cout << "|" << std::left << std::setw(column_width) << "Mean"
                << "|" << std::left << std::setw(column_width) << "Std"
                << "|" << std::endl;      
        std::cout << "+" << std::setfill('-') << std::setw(display_width)
                << std::right << "+" << std::endl;
        std::cout << std::setfill(' ');

        /*--- Hidden layer information ---*/
        for (auto iInput = 0u; iInput < vals_mu.size(); iInput++)
        std::cout << "|" << std::left << std::setw(column_width)
                    << std::to_string(iInput + 1) + ": " + input_names[iInput]
                    << "|" << std::right << std::setw(column_width)
                    << vals_mu[iInput] << "|" << std::right
                    << std::setw(column_width) << vals_std[iInput] << "|"
                    << std::endl;
    };
};

class RobustScaler : public StandardScaler {
    public:
    RobustScaler(const size_t n_in) : StandardScaler(n_in) {};
    virtual void PrintInfo(const int display_width, const std::vector<std::string> &input_names) const {
        const int column_width = int(display_width / 3.0) - 1;
        std::cout << "|" << std::setfill(' ') << std::left << std::setw(display_width - 1)
                << "Inter-quantile range scaling" 
                << "|" << std::endl;std::cout << "+" << std::setfill('-') << std::setw(display_width)
                << std::right << "+" << std::endl;
        std::cout << std::setfill(' ');
        std::cout << "|" << std::left << std::setw(column_width)
                << "Variable:";
        std::cout << "|" << std::left << std::setw(column_width) << "Mean"
                << "|" << std::left << std::setw(column_width) << "IQ range"
                << "|" << std::endl;      
        std::cout << "+" << std::setfill('-') << std::setw(display_width)
                << std::right << "+" << std::endl;
        std::cout << std::setfill(' ');

        /*--- Hidden layer information ---*/
        for (auto iInput = 0u; iInput < vals_mu.size(); iInput++)
        std::cout << "|" << std::left << std::setw(column_width)
                    << std::to_string(iInput + 1) + ": " + input_names[iInput]
                    << "|" << std::right << std::setw(column_width)
                    << vals_mu[iInput] << "|" << std::right
                    << std::setw(column_width) << vals_std[iInput] << "|"
                    << std::endl;
    };
};

class MinMaxScaler :  public ScalerFunction {
    std::vector<mlpdouble> vals_min;
    std::vector<mlpdouble> vals_max;
    
    public:
    MinMaxScaler(const size_t n_in) : ScalerFunction(n_in) {
        vals_min.resize(n_in); 
        vals_max.resize(n_in);
        std::fill(vals_min.begin(), vals_min.end(), 0.0);
        std::fill(vals_max.begin(), vals_max.end(), 1.0);
    };
    virtual mlpdouble GetScale(const size_t i_scalar) const {return (vals_max[i_scalar] - vals_min[i_scalar]); }
    virtual mlpdouble GetOffset(const size_t i_scalar) const {return vals_min[i_scalar]; }
    
    virtual void SetScaling(const size_t i_in, const mlpdouble min=0, const mlpdouble max=1)
    {
        if (min >= max) 
            std::cout << "Error: maximum scaling value should be higher than minimum scaling value" << std::endl;
        vals_min[i_in] = min;
        vals_max[i_in] = max;
    };
    virtual mlpdouble Normalize(const mlpdouble scalar_dim, const size_t i_scalar) const
    {
        mlpdouble val_norm = (scalar_dim - vals_min[i_scalar])/(vals_max[i_scalar] - vals_min[i_scalar]);
        return val_norm;
    };

    virtual mlpdouble Dimensionalize(const mlpdouble scalar_norm, const size_t i_scalar) const {
        mlpdouble val_dim = scalar_norm * (vals_max[i_scalar] - vals_min[i_scalar]) + vals_min[i_scalar];
        return val_dim;
    }

    virtual mlpdouble Distance(const std::vector<mlpdouble> scalar_dim) const 
    {
        mlpdouble val_dist{0};
        for (auto iDim=0u; iDim<vals_min.size(); iDim++) {
            if ((scalar_dim[iDim] < vals_min[iDim]) || (scalar_dim[iDim] > vals_max[iDim])){
                mlpdouble norm = Normalize(scalar_dim[iDim], iDim);
                val_dist += pow(norm - 0.5, 2);
            }
        }
        return val_dist;
    };

    virtual mlpdouble Distance(const mlpdouble* scalar_dim) const 
    {
        mlpdouble val_dist{0};
        for (auto iDim=0u; iDim<vals_min.size(); iDim++) {
            if ((scalar_dim[iDim] < vals_min[iDim]) || (scalar_dim[iDim] > vals_max[iDim])){
                mlpdouble norm = Normalize(scalar_dim[iDim], iDim);
                val_dist += pow(norm - 0.5, 2);
            }
        }
        return val_dist;
    };
    virtual void PrintInfo(const int display_width, const std::vector<std::string> &input_names) const {
        const int column_width = int(display_width / 3.0) - 1;
        std::cout << "|" << std::setfill(' ') << std::left << std::setw(display_width - 1)
                << "Min-max scaling" 
                << "|" << std::endl;
        std::cout << "+" << std::setfill('-') << std::setw(display_width)
                << std::right << "+" << std::endl;
        std::cout << std::setfill(' ');
        std::cout << "|" << std::left << std::setw(column_width)
                << "Variable:";
        std::cout << "|" << std::left << std::setw(column_width) << "min"
                << "|" << std::left << std::setw(column_width) << "max"
                << "|" << std::endl;      
        std::cout << "+" << std::setfill('-') << std::setw(display_width)
                << std::right << "+" << std::endl;
        std::cout << std::setfill(' ');

        /*--- Hidden layer information ---*/
        for (auto iInput = 0u; iInput < vals_min.size(); iInput++)
        std::cout << "|" << std::left << std::setw(column_width)
                    << std::to_string(iInput + 1) + ": " + input_names[iInput]
                    << "|" << std::right << std::setw(column_width)
                    << vals_min[iInput] << "|" << std::right
                    << std::setw(column_width) << vals_max[iInput] << "|"
                    << std::endl;
    };
};
}