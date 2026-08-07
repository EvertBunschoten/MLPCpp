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
      t(0)
    {}

    // Initialization
    void initialize(std::size_t size) {
        m.assign(size, 0);
        v.assign(size, 0);
        t = 0;
    }

    // Optimization step
    void step(std::vector<mlpdouble>& params,
              const std::vector<mlpdouble>& grads)
    {
        if (params.size() != grads.size()) {
            throw std::runtime_error("CAdam: params and grads size mismatch");
        }

        if (m.size() != params.size()) {
            initialize(params.size());
        }

        t++;

        for (std::size_t i = 0; i < params.size(); ++i) {
            // First moment
            m[i] = beta1 * m[i] + (1.0 - beta1) * grads[i];

            // Second moment
            v[i] = beta2 * v[i] + (1.0 - beta2) * grads[i] * grads[i];

            // Bias correction
            mlpdouble m_hat = m[i] / (1.0 - std::pow(beta1, t));
            mlpdouble v_hat = v[i] / (1.0 - std::pow(beta2, t));

            // Update
            params[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
        }
    }

    // Reset
    void reset() {
        m.clear();
        v.clear();
        t = 0;
    }


    // Accessors
    mlpdouble getLearningRate() const { return lr; }
    mlpdouble getBeta1() const { return beta1; }
    mlpdouble getBeta2() const { return beta2; }
    mlpdouble getEpsilon() const { return eps; }
    std::size_t getTimeStep() const { return t; }

    const std::vector<mlpdouble>& getFirstMoment() const { return m; }
    const std::vector<mlpdouble>& getSecondMoment() const { return v; }

    std::size_t getStateSize() const { return m.size(); }

    

private:
    mlpdouble lr;
    mlpdouble beta1;
    mlpdouble beta2;
    mlpdouble eps;

    std::vector<mlpdouble> m;   // first moment
    std::vector<mlpdouble> v;   // second moment
    std::size_t t;              // timestep
};