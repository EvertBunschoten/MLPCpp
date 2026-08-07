/*!
 * \file CBaseLoss.hpp
 * \brief Abstract base class for all PINN loss functions.
 *
 */

#pragma once

#include <vector>
#include <string>
#include <cmath>
#include "variable_def.hpp"

namespace MLPToolbox {   

template<typename T>
inline double to_double(const T& x) {
    return static_cast<double>(x);
}

#ifdef MLP_CUSTOM_TYPE
template<>
inline double to_double<codi::RealReverse>(const codi::RealReverse& x) {
    return x.getValue();
}
#endif

// ============================================================================
// PredictionResult
// ============================================================================
struct PredictionResult {
    std::vector<mlpdouble> inputs;
    std::vector<mlpdouble> outputs;

    // Non-owning views into external derivative buffers.
    //
    // Layout: input-major (matches CNeuralNetwork internal storage)
    //   jacobian[input_index][output_index]     = d(output_o)/d(input_i)
    //   hessian[input_i][input_j][output_index] = d²(output_o)/d(input_i)d(input_j)
    //
    // Argument order in accessors matches this layout:
    //   Jac(input, output)              -> jacobian[input][output]
    //   Hess(input_i, input_j, output)  -> hessian[input_i][input_j][output]
    //
    // LIFETIME: These pointers are valid only as long as the provider's
    // buffer remains unchanged. For CPointDerivatives, they are invalidated
    // by the next Fill() call. Do NOT copy PredictionResult and use
    // derivatives after Fill() is called again.
    mlpdouble**  jacobian{nullptr};
    mlpdouble*** hessian{nullptr};
};

//  CBaseLoss
class CBaseLoss {
public:
    explicit CBaseLoss(const std::string& name)
        : name_(name)
        , last_loss_value_(0.0)   // BUG 3 FIX: was last_loss_value (missing _)
    {}

    virtual ~CBaseLoss() = default;

    CBaseLoss(const CBaseLoss&)            = delete;
    CBaseLoss& operator=(const CBaseLoss&) = delete;

    /*!
     * \brief Evaluate the loss at a batch of collocation points.
     *
     * \param preds     - Network predictions at each point (outputs + Jacobians + Hessians)
     * \param ref_data  - Reference labels at each point (used by data loss, empty for physics)
     * \returns         - Scalar loss value (on the CoDi tape when mlpdouble=RealReverse)
     */
    virtual mlpdouble Evaluate(
        const std::vector<PredictionResult>&       preds,
        const std::vector<std::vector<mlpdouble>>& ref_data
    ) = 0;

    std::string GetName()          const { return name_; }
    double      GetLastLossValue() const { return last_loss_value_; }

protected:
    std::string name_;
    double      last_loss_value_;
};

}  // namespace MLPToolbox