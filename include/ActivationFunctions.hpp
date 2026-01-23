/*!
* \file ActivationFunctions.hpp
* \brief Activation functions supported by MLPCpp
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
#include <vector>
#include <iostream>
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

class ActivationFunctionBase{
    
    protected:
        std::string name, tag;
        mlpdouble input;
        mlpdouble output;
        mlpdouble Jacobian{0};
        mlpdouble Hessian{0};
        bool calc_gradient {false};
        bool calc_gradient_2 {false};
    public:
    
    mlpdouble GetOutput() const {return output;}
    mlpdouble GetJacobian() const {return Jacobian;}
    mlpdouble GetHessian() const {return Hessian;}
    std::string GetName() const {return name;}
    std::string GetTag() const {return tag;}
    ActivationFunctionBase() = default;
    virtual mlpdouble operator() (const mlpdouble x,const bool calc_Jacobian=false, const bool calc_Hessian=false)=0;
    
    ~ActivationFunctionBase() = default;
};

class Lin final: public ActivationFunctionBase {
    public:
        Lin() {name="Linear", tag="linear";}
        mlpdouble operator() (const mlpdouble x, const bool calc_Jacobian=false, const bool calc_Hessian=false) override {
            output = x;
            if (calc_Jacobian)
                Jacobian = 1.0;
            if (calc_Hessian)
                Hessian = 0.0;
            return output;
        }
};

class Elu final: public ActivationFunctionBase {
    public:
        Elu() {name="Elu", tag="elu";}
        mlpdouble operator() (const mlpdouble x,const bool calc_Jacobian=false, const bool calc_Hessian=false) override {
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

class Sigmoid final: public ActivationFunctionBase {
    public:
        Sigmoid() {name="Sigmoid", tag="sigmoid";}
        virtual mlpdouble operator() (const mlpdouble x,const bool calc_Jacobian=false, const bool calc_Hessian=false) {
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

class Exponential final: public ActivationFunctionBase {
    public:
    Exponential() {name="Exponential", tag="exponential";}
    virtual mlpdouble operator() (const mlpdouble x, const bool calc_Jacobian=false, const bool calc_Hessian=false) {
        output = exp(x);
        if (calc_Jacobian) Jacobian = output;
        if (calc_Hessian) Hessian = output;
        return output;
    }
};

class Relu final: public ActivationFunctionBase {
    public:
    Relu() {name="ReLu", tag="relu";}
    virtual mlpdouble operator() (const mlpdouble x, const bool calc_Jacobian=false, const bool calc_Hessian=false) {
        if (x > 0) {
            output = x;
            if (calc_Jacobian) 
                Jacobian = 1.0;
        }else{
            output = 0.0;
            if (calc_Jacobian)
                Jacobian = 0.0;
        }
        if (calc_Hessian)
            Hessian = 0.0;
        return output;
    }
};

class Swish final: public ActivationFunctionBase {
    public:
    Swish() {name="Swish", tag="swish"; }
    virtual mlpdouble operator() (const mlpdouble x, const bool calc_Jacobian=false, const bool calc_Hessian=false) {
        const mlpdouble exp_x = exp(x);
        output = x * exp_x/ (1 + exp_x);
        if (calc_Jacobian) {
            Jacobian = exp_x * (x + exp_x + 1) / pow(exp_x + 1, 2);
            if (calc_Hessian)
                Hessian = exp_x * (-exp_x * (x - 2) + x + 2) / pow(exp_x + 1, 3);
        }
        return output;
    }
};

class Tanh final: public ActivationFunctionBase {
    public:
    Tanh() {name="Tanh", tag="tanh";}
    virtual mlpdouble operator() (const mlpdouble x, const bool calc_Jacobian=false, const bool calc_Hessian=false) {
        const mlpdouble tnh = tanh(x);
        output = tnh;
        if (calc_Jacobian){
            Jacobian = pow(cosh(x), -2);
            if (calc_Hessian){
                Hessian = -2*tnh*Jacobian;
            }
        }
        return output;
    }
};



class SeLu final: public ActivationFunctionBase {
    private:
        const mlpdouble lambda {1.05070098};
        const mlpdouble alpha {1.67326324};
    public:
    SeLu() {name="SeLu", tag="selu";}
    virtual mlpdouble operator() (const mlpdouble x, const bool calc_Jacobian=false, const bool calc_Hessian=false) {
        if (x > 0) {
            output = lambda * x;
            if (calc_Jacobian){
                Jacobian = lambda;
                if (calc_Hessian){
                    Hessian = 0.0;
                }
            }
        } else {
            const mlpdouble exp_x = exp(x);
            output = lambda * alpha * (exp_x - 1);
            if (calc_Jacobian){
                Jacobian = output + lambda * alpha;
                if ( calc_Hessian) {
                    Hessian = Jacobian;
                }
            }
        }
        return output;
    }
};

class GeLu final: public ActivationFunctionBase {
    private:
    const mlpdouble gelu_c{0.5*sqrt(2)};
    public:
    GeLu() {name="GeLu", tag="gelu";}
    virtual mlpdouble operator() (const mlpdouble x, const bool calc_Jacobian=false, const bool calc_Hessian=false) {
        output = 0.5 * x * (1 + erf(x / sqrt(2)));
        if (calc_Jacobian) {
            Jacobian = 0.5 + 0.5*sqrt(2/M_PI) * exp(-0.5*pow(x,2)) * x + 0.5 * erf(x/sqrt(2));
            if (calc_Hessian)
                Hessian = exp(-0.5*pow(x,2))*(sqrt(2/M_PI) - 0.5*sqrt(2/M_PI)*pow(x,2));
        }
        return output;
    }
};