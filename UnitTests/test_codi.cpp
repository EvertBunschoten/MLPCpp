#include <iostream>
#include <vector>
#include <cmath>
#include "codi.hpp"
#include "variable_def.hpp"

// Finite difference helper
mlpdouble finite_difference(
    const std::vector<mlpdouble>& p,
    int idx,
    mlpdouble eps = 1e-6)
{
    std::vector<mlpdouble> p1 = p;
    std::vector<mlpdouble> p2 = p;

    p1[idx] += eps;
    p2[idx] -= eps;

    auto forward = [](const std::vector<mlpdouble>& w) -> mlpdouble {

        mlpdouble x1 = 1.0;
        mlpdouble x2 = 2.0;

        mlpdouble w11 = w[0];
        mlpdouble w12 = w[1];
        mlpdouble w21 = w[2];
        mlpdouble w22 = w[3];
        mlpdouble v1  = w[4];
        mlpdouble v2  = w[5];
        mlpdouble b2  = w[6];

        // 2-layer NN (NONLINEAR = important!)
        mlpdouble h1 = std::tanh(w11 * x1 + w12 * x2);
        mlpdouble h2 = std::tanh(w21 * x1 + w22 * x2);

        mlpdouble y = v1 * h1 + v2 * h2 + b2;

        mlpdouble L = y * y;
        return L;
    };

    return (forward(p1) - forward(p2)) / (2.0 * eps);
}

int main()
{
    // Inputs
    codi::RealReverse x1 = 1.0;
    codi::RealReverse x2 = 2.0;

    codi::RealReverse w11 = 0.1, w12 = -0.2;
    codi::RealReverse w21 = 0.4, w22 = 0.3;

    codi::RealReverse v1  = 0.2, v2  = -0.5;
    codi::RealReverse b2  = 0.1;

    codi::RealReverse::Tape& tape = codi::RealReverse::getTape();

    // Tape setup
    tape.setActive();

    tape.registerInput(w11);
    tape.registerInput(w12);
    tape.registerInput(w21);
    tape.registerInput(w22);
    tape.registerInput(v1);
    tape.registerInput(v2);
    tape.registerInput(b2);

    // Forward pass (NN)
    codi::RealReverse h1 = codi::tanh(w11 * x1 + w12 * x2);
    codi::RealReverse h2 = codi::tanh(w21 * x1 + w22 * x2);

    codi::RealReverse y  = v1 * h1 + v2 * h2 + b2;

    codi::RealReverse L  = y * y;   // loss

    tape.registerOutput(L);

    // Reverse pass (AD)
    tape.setPassive();
    L.setGradient(1.0);
    tape.evaluate();

    // Print AD gradients
    std::cout << "=== AD Gradients ===\n";
    std::cout << "dL/dw11 = " << w11.getGradient() << "\n";
    std::cout << "dL/dw12 = " << w12.getGradient() << "\n";
    std::cout << "dL/dw21 = " << w21.getGradient() << "\n";
    std::cout << "dL/dw22 = " << w22.getGradient() << "\n";
    std::cout << "dL/dv1  = " << v1.getGradient()  << "\n";
    std::cout << "dL/dv2  = " << v2.getGradient()  << "\n";
    std::cout << "dL/db2  = " << b2.getGradient()  << "\n";

    // Finite difference check
    std::vector<double> p = {
        0.1, -0.2,
        0.4,  0.3,
        0.2, -0.5,
        0.1
    };

    std::cout << "\n=== Finite Difference ===\n";
    std::cout << "dw11 FD = " << finite_difference(p, 0) << "\n";
    std::cout << "dw12 FD = " << finite_difference(p, 1) << "\n";
    std::cout << "dw21 FD = " << finite_difference(p, 2) << "\n";
    std::cout << "dw22 FD = " << finite_difference(p, 3) << "\n";
    std::cout << "dv1  FD = " << finite_difference(p, 4) << "\n";
    std::cout << "dv2  FD = " << finite_difference(p, 5) << "\n";
    std::cout << "db2  FD = " << finite_difference(p, 6) << "\n";

    return 0;
}