#pragma once
#include "CBaseLoss.hpp"
#include <stdexcept>

namespace MLPToolbox {

class CMeanSquaredErrorLoss : public CBaseLoss { // FIX 1: Added 'public'
public:
  CMeanSquaredErrorLoss() : CBaseLoss("data loss") {}

  mlpdouble
  Evaluate(const std::vector<PredictionResult> &predictions,
           const std::vector<std::vector<mlpdouble>> &ref_data) override {
    if (predictions.size() != ref_data.size()) {
      throw std::runtime_error(
          "CDataLoss: predictions and ref_data size mismatch");
    }

    if (predictions.empty()) {
      last_loss_value_ = mlpdouble(0.0);
      return mlpdouble(0.0);
    }

    mlpdouble mse = 0.0;
    const size_t N = predictions.size();
    const size_t n_outputs = predictions[0].outputs.size();

    for (size_t i = 0; i < N; ++i) {
      if (predictions[i].outputs.size() != n_outputs) {
        throw std::runtime_error(
            "CDataLoss: inconsistent number of outputs in predictions");
      }
      if (ref_data[i].size() != n_outputs) {
        throw std::runtime_error(
            "CDataLoss: ref_data[i] size does not match number of outputs");
      }
    }

    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < n_outputs; ++j) {
        mlpdouble diff = predictions[i].outputs[j] - ref_data[i][j];
        mse += diff * diff;
      }
    }

    mlpdouble loss = mse / static_cast<mlpdouble>(N * n_outputs);

    last_loss_value_ = loss;
    return loss;
  }
};

} // namespace MLPToolbox
