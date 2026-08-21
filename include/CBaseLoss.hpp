/*!
 * \file CBaseLoss.hpp
 * \brief Abstract base class for all PINN loss functions.
 *
 */

#pragma once

#include "variable_def.hpp"
#include <cmath>
#include <string>
#include <vector>

namespace MLPToolbox {

template <typename T> inline double to_double(const T &x) {
  return static_cast<double>(x);
}

#ifdef MLP_CUSTOM_TYPE
template <>
inline double to_double<codi::RealReverse>(const codi::RealReverse &x) {
  return x.getValue();
}
#endif

struct PredictionResult {
  std::vector<mlpdouble> inputs;
  std::vector<mlpdouble> outputs;

  mlpdouble **jacobian{nullptr};
  mlpdouble ***hessian{nullptr};
};

class CBaseLoss {
public:
  explicit CBaseLoss(const std::string &name)
      : name_(name), last_loss_value_(0.0) {}

  virtual ~CBaseLoss() = default;

  CBaseLoss(const CBaseLoss &) = delete;
  CBaseLoss &operator=(const CBaseLoss &) = delete;

  /*!
   * \brief Evaluate the loss at a batch of collocation points.
   *
   * \param preds     - Network predictions at each point (outputs + Jacobians +
   * Hessians) \param ref_data  - Reference labels at each point (used by data
   * loss, empty for physics) \returns         - Scalar loss value (on the CoDi
   * tape when mlpdouble=RealReverse)
   */
  virtual mlpdouble
  Evaluate(const std::vector<PredictionResult> &preds,
           const std::vector<std::vector<mlpdouble>> &ref_data) = 0;

  std::string GetName() const { return name_; }
  mlpdouble GetLastLossValue() const { return last_loss_value_; }

protected:
  std::string name_;
  mlpdouble last_loss_value_{mlpdouble(0.0)};
};

} // namespace MLPToolbox
