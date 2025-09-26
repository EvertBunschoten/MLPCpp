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

class Linear final: public ActivationFunctionBase {
    public:
        Linear() {name="Linear";}
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

ActivationFunctionBase* DefineActivationFunction (std::string name_activation_function) {
    ENUM_ACTIVATION_FUNCTION i_phi = activation_function_map[name_activation_function];
    ActivationFunctionBase * function_out;
    switch (i_phi)
    {
    case ENUM_ACTIVATION_FUNCTION::LINEAR:
        function_out = new Linear();
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
        function_out = new Linear();
        break;
    }
    return function_out;
}