/*!
 * \file CAdam.hpp
 * \brief Adam optimizer used by the MLPToolbox trainers.
 *
 */

#pragma once

#include "variable_def.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

class CAdam {
private:
  mlpdouble learning_rate_;
  mlpdouble beta1_;
  mlpdouble beta2_;
  mlpdouble epsilon_;

  std::vector<mlpdouble> first_moment_;
  std::vector<mlpdouble> second_moment_;
  std::size_t timestep_;

public:
  CAdam(mlpdouble learning_rate = 1e-3, mlpdouble beta1 = 0.9,
        mlpdouble beta2 = 0.999, mlpdouble epsilon = 1e-8)
      : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2),
        epsilon_(epsilon), timestep_(0) {}

  void Initialize(std::size_t state_size) {
    first_moment_.assign(state_size, mlpdouble(0.0));
    second_moment_.assign(state_size, mlpdouble(0.0));
    timestep_ = 0;
  }

  void OptimizationStep(std::vector<mlpdouble> &parameters,
                        const std::vector<mlpdouble> &gradients) {
    if (parameters.size() != gradients.size()) {
      throw std::runtime_error(
          "CAdam::OptimizationStep: parameter and gradient size mismatch.");
    }

    if (first_moment_.size() != parameters.size()) {
      Initialize(parameters.size());
    }

    ++timestep_;

    const auto first_moment_bias_correction =
        mlpdouble(1.0) - std::pow(beta1_, timestep_);
    const auto second_moment_bias_correction =
        mlpdouble(1.0) - std::pow(beta2_, timestep_);

    for (auto i = std::size_t{0}; i < parameters.size(); ++i) {
      first_moment_[i] =
          beta1_ * first_moment_[i] + (mlpdouble(1.0) - beta1_) * gradients[i];
      second_moment_[i] =
          beta2_ * second_moment_[i] +
          (mlpdouble(1.0) - beta2_) * gradients[i] * gradients[i];

      const auto corrected_first_moment =
          first_moment_[i] / first_moment_bias_correction;
      const auto corrected_second_moment =
          second_moment_[i] / second_moment_bias_correction;

      parameters[i] -= learning_rate_ * corrected_first_moment /
                       (std::sqrt(corrected_second_moment) + epsilon_);
    }
  }

  void Reset() {
    first_moment_.clear();
    second_moment_.clear();
    timestep_ = 0;
  }

  mlpdouble GetLearningRate() const { return learning_rate_; }
  mlpdouble GetBeta1() const { return beta1_; }
  mlpdouble GetBeta2() const { return beta2_; }
  mlpdouble GetEpsilon() const { return epsilon_; }
  std::size_t GetTimeStep() const { return timestep_; }

  const std::vector<mlpdouble> &GetFirstMoment() const { return first_moment_; }
  const std::vector<mlpdouble> &GetSecondMoment() const {
    return second_moment_;
  }

  std::size_t GetStateSize() const { return first_moment_.size(); }
};
