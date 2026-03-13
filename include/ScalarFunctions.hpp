/*!
* \file ScalerFunctions.hpp
* \brief Functions used to scale network input and output.
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
    /*! \brief Base class for scaler functions. */
    protected:
    size_t n_scalars{0}; /* Number of variables to scale. */
    std::string tag;     /* Scaling function ID tag. */
    public:
    ScalerFunction(const size_t n_in) : n_scalars {n_in} {};
    virtual ~ScalerFunction() = default;

    /*!
    * \brief Normalize network input.
    * \param[in] scalar_dim - Unscaled network input.
    * \param[in] i_scalar - Network input index.
    * \returns - Normalized network input.
    */
    virtual mlpdouble Normalize(const mlpdouble scalar_dim, const size_t i_scalar) const =0;

    /*!
    * \brief Unscale network output
    * \param[in] scalar_norm - Scaled network output.
    * \param[in] i_scalar - Network output index.
    * \returns - Unscaled network output.
    */
    virtual mlpdouble Dimensionalize(const mlpdouble scalar_norm, const size_t i_scalar) const =0;

    /*!
    * \brief Calculate normalized distance from feature range
    * \param[in] inputs - Unscaled network inputs
    * \returns - Normalized distance.
    */
    virtual mlpdouble Distance(const std::vector<mlpdouble> inputs) const = 0;
    virtual mlpdouble Distance(const mlpdouble* inputs) const = 0;

    /*!
    * \brief Set linear scaling values.
    * \param[in] i_scalar - node index.
    * \param[in] val_1 - first linear scaling value.
    * \param[in] val_2 - second linear scaling value.
    */
    virtual void SetScaling(const size_t i_scalar, const mlpdouble val_1, const mlpdouble val_2) =0;

    /*!
    * \brief Return data multiplication factor.
    * \param[in] i_scalar - node index.
    */
    virtual mlpdouble GetScale(const size_t i_scalar) const =0;

    /*!
    * \brief Return data offset.
    * \param[in] i_scalar - node index.
    */
    virtual mlpdouble GetOffset(const size_t i_scaler) const = 0;

    /*!
    * \brief Display scaler function information in the terminal.
    * \param[in] display_width - maximum column width.
    * \param[in] input_names - names of input/output variables.
    * \param[in] outp - output stream.
    */
    virtual void PrintInfo(const int display_width, const std::vector<std::string>& input_names, std::ostream &outp=std::cout) const =0;

    /*!
    * \brief Get scaler function ID tag.
    * \returns - tag name.
    */
    std::string GetTag() const {return tag;}
};


class StandardScaler : public ScalerFunction {
    /*! \brief Scaler function using standard deviation. n = (d - mu)/std */

    public: 
    std::vector<mlpdouble> vals_mu;  /*! Mean values */
    std::vector<mlpdouble> vals_std; /*! Standard deviation values. */

    StandardScaler(const size_t n_in) : ScalerFunction(n_in) {
        tag = "standard";
        vals_mu.resize(n_in); 
        vals_std.resize(n_in);
        std::fill(vals_mu.begin(), vals_mu.end(), 0.0);
        std::fill(vals_std.begin(), vals_std.end(), 1.0);
    };
    virtual mlpdouble GetScale(const size_t i_scalar) const {return vals_std[i_scalar]; }
    virtual mlpdouble GetOffset(const size_t i_scalar) const {return vals_mu[i_scalar]; }
    
    virtual void SetScaling(const size_t i_in, const mlpdouble val_mu=0.0, const mlpdouble val_std=1.0) 
    {
        if (val_std < 0) {
            ErrorMessage("Standard deviation value should be positive.", "StandardScaler:SetScaling");
        }
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
    virtual void PrintInfo(const int display_width, const std::vector<std::string> &input_names, std::ostream &outp=std::cout) const {
        const int column_width = int(display_width / 3.0) - 1;
        outp << "|" << std::setfill(' ') << std::left << std::setw(display_width - 1)
                << "Standard scaling" 
                << "|" << std::endl;outp << "+" << std::setfill('-') << std::setw(display_width)
                << std::right << "+" << std::endl;
        outp << std::setfill(' ');
        outp << "|" << std::left << std::setw(column_width)
                << "Variable:";
        outp << "|" << std::left << std::setw(column_width) << "Mean"
                << "|" << std::left << std::setw(column_width) << "Std"
                << "|" << std::endl;      
        outp << "+" << std::setfill('-') << std::setw(display_width)
                << std::right << "+" << std::endl;
        outp << std::setfill(' ');

        /*--- Hidden layer information ---*/
        for (auto iInput = 0u; iInput < vals_mu.size(); iInput++)
        outp << "|" << std::left << std::setw(column_width)
                    << std::to_string(iInput + 1) + ": " + input_names[iInput]
                    << "|" << std::right << std::setw(column_width)
                    << vals_mu[iInput] << "|" << std::right
                    << std::setw(column_width) << vals_std[iInput] << "|"
                    << std::endl;
    };
};

class RobustScaler : public StandardScaler {
    /*! \brief Inter-quantile range scaling. Similar to standard scaling, but using inter-quantile range rather than standard deviation. */
    public:
    RobustScaler(const size_t n_in) : StandardScaler(n_in) {tag = "robust";};
    virtual void PrintInfo(const int display_width, const std::vector<std::string> &input_names, std::ostream &outp=std::cout) const {
        const int column_width = int(display_width / 3.0) - 1;
        outp << "|" << std::setfill(' ') << std::left << std::setw(display_width - 1)
                << "Inter-quantile range scaling" 
                << "|" << std::endl;outp << "+" << std::setfill('-') << std::setw(display_width)
                << std::right << "+" << std::endl;
        outp << std::setfill(' ');
        outp << "|" << std::left << std::setw(column_width)
                << "Variable:";
        outp << "|" << std::left << std::setw(column_width) << "Mean"
                << "|" << std::left << std::setw(column_width) << "IQ range"
                << "|" << std::endl;      
        outp << "+" << std::setfill('-') << std::setw(display_width)
                << std::right << "+" << std::endl;
        outp << std::setfill(' ');

        /*--- Hidden layer information ---*/
        for (auto iInput = 0u; iInput < vals_mu.size(); iInput++)
        outp << "|" << std::left << std::setw(column_width)
                    << std::to_string(iInput + 1) + ": " + input_names[iInput]
                    << "|" << std::right << std::setw(column_width)
                    << vals_mu[iInput] << "|" << std::right
                    << std::setw(column_width) << vals_std[iInput] << "|"
                    << std::endl;
    };
};

class MinMaxScaler :  public ScalerFunction {
    /*! \brief min-max scaler function: n = (d - min)/(max - min). */

    std::vector<mlpdouble> vals_min;
    std::vector<mlpdouble> vals_max;
    
    public:
    MinMaxScaler(const size_t n_in) : ScalerFunction(n_in) {
        tag="minmax";
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
            ErrorMessage("Maximum scaling value should be higher than minimum scaling value", "MinMaxScaler:SetScaling");
            
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
        /* Returns positive value if value lies outside range. */
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
    virtual void PrintInfo(const int display_width, const std::vector<std::string> &input_names, std::ostream &outp=std::cout) const {
        const int column_width = int(display_width / 3.0) - 1;
        outp << "|" << std::setfill(' ') << std::left << std::setw(display_width - 1)
                << "Min-max scaling" 
                << "|" << std::endl;
        outp << "+" << std::setfill('-') << std::setw(display_width)
                << std::right << "+" << std::endl;
        outp << std::setfill(' ');
        outp << "|" << std::left << std::setw(column_width)
                << "Variable:";
        outp << "|" << std::left << std::setw(column_width) << "min"
                << "|" << std::left << std::setw(column_width) << "max"
                << "|" << std::endl;      
        outp << "+" << std::setfill('-') << std::setw(display_width)
                << std::right << "+" << std::endl;
        outp << std::setfill(' ');

        /*--- Hidden layer information ---*/
        for (auto iInput = 0u; iInput < vals_min.size(); iInput++)
        outp << "|" << std::left << std::setw(column_width)
                    << std::to_string(iInput + 1) + ": " + input_names[iInput]
                    << "|" << std::right << std::setw(column_width)
                    << vals_min[iInput] << "|" << std::right
                    << std::setw(column_width) << vals_max[iInput] << "|"
                    << std::endl;
    };
};
}