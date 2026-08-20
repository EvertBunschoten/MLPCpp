/*
 * ============================================================
 *  CGradientAnnealer — Learning-Rate Annealing for PINNs
 * ============================================================
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

//  AnnealerConfig
struct AnnealerConfig {

  // EMA decay coefficient
  double alpha = 0.9;

  double lambda_init = 1.0;

  double lambda_min = 1e-4;
  double lambda_max = 1e+4;

  std::size_t n_data_terms = 1;
};

//  GradStats  — pre-computed statistics for ONE loss term.
struct GradStats {
  double max_abs;
  double mean_abs;

  // compute stats from a flat gradient vector.
  static GradStats from_grads(const std::vector<double> &g) {
    if (g.empty())
      return {0.0, 0.0};
    double mx = 0.0, sum = 0.0;
    for (double gi : g) {
      double ag = std::abs(gi);
      if (ag > mx)
        mx = ag;
      sum += ag;
    }
    return {mx, sum / static_cast<double>(g.size())};
  }
};

//  CGradientAnnealer
class CGradientAnnealer {
public:
  // Constructor

  explicit CGradientAnnealer(AnnealerConfig cfg = {})
      : cfg_(cfg), lambda_(cfg.n_data_terms, cfg.lambda_init),
        lambda_hat_(cfg.n_data_terms, cfg.lambda_init), step_(0) {
    if (cfg.n_data_terms == 0)
      throw std::invalid_argument(
          "CGradientAnnealer: n_data_terms must be >= 1");
    if (cfg.alpha <= 0.0 || cfg.alpha >= 1.0)
      throw std::invalid_argument("CGradientAnnealer: alpha must be in (0,1)");
  }

  // Non-copyable (owns EMA state), movable.
  CGradientAnnealer(const CGradientAnnealer &) = delete;
  CGradientAnnealer &operator=(const CGradientAnnealer &) = delete;
  CGradientAnnealer(CGradientAnnealer &&) = default;
  CGradientAnnealer &operator=(CGradientAnnealer &&) = default;

  ~CGradientAnnealer() = default;

  void update(const GradStats &grad_ref,
              const std::vector<GradStats> &grad_data) {
    assert(grad_data.size() == cfg_.n_data_terms);

    ++step_;

    const double ref_max = grad_ref.max_abs;

    for (std::size_t i = 0; i < cfg_.n_data_terms; ++i) {
      const double data_mean = grad_data[i].mean_abs;

      if (data_mean < 1e-15) {
        lambda_hat_[i] = cfg_.lambda_max;
      } else {
        lambda_hat_[i] = ref_max / data_mean;
      }

      // EMA update
      lambda_[i] =
          (1.0 - cfg_.alpha) * lambda_[i] + cfg_.alpha * lambda_hat_[i];

      // Hard clamp
      lambda_[i] = std::clamp(lambda_[i], cfg_.lambda_min, cfg_.lambda_max);
    }
  }

  // Accessors
  double get_lambda(std::size_t i) const {
    assert(i < cfg_.n_data_terms);
    return lambda_[i];
  }

  const std::vector<double> &lambdas() const noexcept { return lambda_; }

  double get_lambda_hat(std::size_t i) const {
    assert(i < cfg_.n_data_terms);
    return lambda_hat_[i];
  }

  std::size_t step() const noexcept { return step_; }
  std::size_t n_data_terms() const noexcept { return cfg_.n_data_terms; }
  const AnnealerConfig &config() const noexcept { return cfg_; }

  // Reset EMA state
  void reset() noexcept {
    step_ = 0;
    std::fill(lambda_.begin(), lambda_.end(), cfg_.lambda_init);
    std::fill(lambda_hat_.begin(), lambda_hat_.end(), cfg_.lambda_init);
  }

private:
  AnnealerConfig cfg_;
  std::vector<double> lambda_;
  std::vector<double> lambda_hat_;
  std::size_t step_;
};
