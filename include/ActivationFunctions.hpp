
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
        std::string name;
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
    ActivationFunctionBase(){}
    virtual mlpdouble call (mlpdouble x,bool calc_Jacobian=false, bool calc_Hessian=false)=0;
    
    
};

class Lin final: public ActivationFunctionBase {
    public:
        Lin() {name="Linear";}
        mlpdouble call (mlpdouble x, bool calc_Jacobian=false, bool calc_Hessian=false) override {
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
        Elu() {name="Elu";}
        mlpdouble call (mlpdouble x,bool calc_Jacobian=false, bool calc_Hessian=false) override {
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
        Sigmoid() {name="Sigmoid";}
        virtual mlpdouble call (mlpdouble x,bool calc_Jacobian=false, bool calc_Hessian=false) {
            mlpdouble exp_x = exp(-x);
            output = 1.0 / (1 + exp_x);
            if (calc_Jacobian) {
                Jacobian = exp_x / pow(exp_x + 1, 2);
                if (calc_Hessian) {
                    exp_x = exp(x);
                    Hessian = -(exp_x * (exp_x - 1)) / pow(exp_x + 1, 3);
                }
            }
            return output;
        }

};

class Exponential final: public ActivationFunctionBase {
    public:
    Exponential() {name="Exponential";}
    virtual mlpdouble call (mlpdouble x, bool calc_Jacobian=false, bool calc_Hessian=false) {
        output = exp(x);
        if (calc_Jacobian) Jacobian = output;
        if (calc_Hessian) Hessian = output;
        return output;
    }
};

class Relu final: public ActivationFunctionBase {
    public:
    Relu() {name="ReLu";}
    virtual mlpdouble call (mlpdouble x, bool calc_Jacobian=false, bool calc_Hessian=false) {
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
    Swish() {name="Swish";}
    virtual mlpdouble call (mlpdouble x, bool calc_Jacobian=false, bool calc_Hessian=false) {
        output = x / (1 + exp(-x));
        if (calc_Jacobian) {
            mlpdouble exp_x = exp(x);
            Jacobian = exp_x * (x + exp_x + 1) / pow(exp_x + 1, 2);
            if (calc_Hessian){
                Hessian = exp_x*(-exp_x * (x - 2) + x + 2) / pow(exp_x + 1,3);
            }
        }
        return output;
    }
};

class Tanh final: public ActivationFunctionBase {
    public:
    Tanh() {name="Tanh";}
    virtual mlpdouble call (mlpdouble x, bool calc_Jacobian=false, bool calc_Hessian=false) {
        mlpdouble tnh = tanh(x);
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
    SeLu() {name="SeLu";}
    virtual mlpdouble call (mlpdouble x, bool calc_Jacobian=false, bool calc_Hessian=false) {
        if (x > 0) {
            output = lambda * x;
            if (calc_Jacobian){
                Jacobian = lambda;
                if (calc_Hessian){
                    Hessian = 0.0;
                }
            }
        } else {
            mlpdouble exp_x = exp(x);
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
    GeLu() {name="GeLu";}
    virtual mlpdouble call (mlpdouble x, bool calc_Jacobian=false, bool calc_Hessian=false) {
        output = 0.5*x*(1 + erf(x / sqrt(2)));
        if (calc_Jacobian) {
            mlpdouble exp_x = exp(-gelu_c * x);
            Jacobian = exp_x * (gelu_c * x + exp_x + 1) / pow(exp_x + 1, 2);
            if (calc_Hessian)
            Hessian = x * ((5.79361 * pow(exp_x, 2) / pow(exp_x + 1, 3)) -
                                    (2.8968 * exp_x / pow(exp_x + 1, 2))) +
                        3.404 * exp_x / pow(exp_x + 1, 2);
        }
        return output;
    }
};

