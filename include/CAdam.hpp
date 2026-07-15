#pragma once

#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include "variable_def.hpp"


class CAdam {
public:
    CAdam(
        mlpdouble learning_rate = 1e-3,
        mlpdouble beta1 = 0.9,
        mlpdouble beta2 = 0.999,
        mlpdouble epsilon = 1e-8
    )
    : lr(learning_rate),
      beta1(beta1),
      beta2(beta2),
      eps(epsilon),
      timeStep(0)
    {
        if (lr < 0.0) {
            throw std::invalid_argument("CAdam: learning_rate must be >= 0");
        }
        
        if (beta1 < 0.0 || beta1 >= 1.0) {
            throw std::invalid_argument("CAdam: beta1 must be in the range [0, 1)");
        }
        
        if (beta2 < 0.0 || beta2 >= 1.0) {
            throw std::invalid_argument("CAdam: beta2 must be in the range [0, 1)");
        }
        
        if (eps <= 0.0) {
            throw std::invalid_argument("CAdam: epsilon must be > 0");
        }
    }

    // Initialization
    void initialize(std::size_t size) {
        firstMoment_.assign(size, 0);
        secondMoment_.assign(size, 0);
        timeStep = 0;
    }

    // Optimization step
    void step(std::vector<mlpdouble>& params,
              const std::vector<mlpdouble>& grads)
    {
        if (params.size() != grads.size()) {
            throw std::runtime_error("CAdam: params and grads size mismatch");
        }

        if (firstMoment_.size() != params.size()) {
            initialize(params.size());
        }

        timeStep++;

        for (std::size_t i = 0; i < params.size(); ++i) {
            // First moment
            firstMoment_[i] = beta1 * firstMoment_[i] + (1.0 - beta1) * grads[i];

            // Second moment
            secondMoment_[i] = beta2 * secondMoment_[i] + (1.0 - beta2) * grads[i] * grads[i];

            // Bias correction
            mlpdouble firstMoment_hat = firstMoment_[i] / (1.0 - std::pow(beta1, timeStep));
            mlpdouble secondMoment_hat = secondMoment_[i] / (1.0 - std::pow(beta2, timeStep));

            // Update
            params[i] -= lr * firstMoment_hat / (std::sqrt(secondMoment_hat) + eps);
        }
    }

    // Reset
    void reset() {
        firstMoment_.clear();
        secondMoment_.clear();
        timeStep = 0;
    }


    // Accessors
    mlpdouble getLearningRate() const { return lr; }
    mlpdouble getBeta1() const { return beta1; }
    mlpdouble getBeta2() const { return beta2; }
    mlpdouble getEpsilon() const { return eps; }
    std::size_t getTimeStep() const { return timeStep; }

    const std::vector<mlpdouble>& getFirstMoment() const { return firstMoment_; }
    const std::vector<mlpdouble>& getSecondMoment() const { return secondMoment_; }

    std::size_t getNumberOfVariables() const { return firstMoment_.size(); }

    

private:
    mlpdouble lr;
    mlpdouble beta1;
    mlpdouble beta2;
    mlpdouble eps;

    std::vector<mlpdouble> firstMoment_;   // first moment
    std::vector<mlpdouble> secondMoment_;   // second moment
    std::size_t timeStep;              // timestep
};