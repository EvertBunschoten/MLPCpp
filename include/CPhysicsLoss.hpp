#pragma once
/*!
 * \file CPhysicsLoss.hpp
 * \brief Physics-informed loss for PINN training.
 *
 * Provides a flexible residual-based loss that can depend on network
 * inputs, outputs, first derivatives (Jacobian) and second derivatives
 * (Hessian).  Residuals are supplied by the user as callable objects.
 *
 * Key design points
 * -----------------
 * - Equations are declared by *name* (not by hard-coded indices).
 *   The mapping from equation-local indices onto the full network is
 *   resolved once at construction time.
 * - EvaluateOne() returns the *un-normalised* sum of weighted residual
 *   squares for a single collocation point.  The trainer is responsible
 *   for the final 1/(N·N_eq) averaging when streaming points.
 * - Evaluate() is the batch interface that performs that averaging.
 * - PhysicsData holds optional auxiliary variables (source terms,
 *   material properties, \ldots) that a residual may need.
 * - PhysicsState gives the residual a safe, readable API to query
 *   inputs, outputs and derivatives.
 */

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "CBaseLoss.hpp"
#include "variable_def.hpp"

namespace MLPToolbox {

// Forward declarations needed before ResidualFunction is defined.
class PhysicsData;
class PhysicsState;

/*!
 * \brief Signature of a user-supplied residual.
 *
 * The residual receives a PhysicsState (network quantities at one
 * collocation point) and optional PhysicsData (auxiliary variables).
 * It must return a scalar residual value; the loss squares and weights
 * that value internally.
 */
using ResidualFunction = std::function<mlpdouble(const PhysicsState&, const PhysicsData&)>;

// ============================================================================
// PhysicsData
// ============================================================================
/*!
 * \brief Container for optional auxiliary physics variables at one point.
 *
 * Typical use-cases: forcing terms, material coefficients, boundary
 * values that are not network inputs/outputs.  Variables are addressed
 * either by integer index or by name.
 *
 * Construction validates that names are unique and non-empty and that
 * the number of names matches the number of values.
 */
class PhysicsData {
public:
    /*!
     * \param[in] values  Numeric values of the auxiliary variables.
     * \param[in] names   Corresponding unique names (same length as values).
     * \throw std::invalid_argument on size mismatch, empty name or duplicate name.
     */
    PhysicsData(std::vector<mlpdouble> values, std::vector<std::string> names)
        : values_(std::move(values)), names_(std::move(names)) {
        if (names_.size() != values_.size()) {
            throw std::invalid_argument(
                "PhysicsData: names and values must have identical sizes. " +
                std::string("names=") + std::to_string(names_.size()) +
                ", values=" + std::to_string(values_.size()));
        }
        for (std::size_t i = 0; i < names_.size(); ++i) {
            if (names_[i].empty()) {
                throw std::invalid_argument("PhysicsData: physics variable name cannot be empty.");
            }
            if (name_to_index_.find(names_[i]) != name_to_index_.end()) {
                throw std::invalid_argument("PhysicsData: duplicate variable name '" + names_[i] + "'.");
            }
            name_to_index_.emplace(names_[i], i);
        }
    }

    bool Empty() const noexcept { return values_.empty(); }
    std::size_t Size() const noexcept { return values_.size(); }
    bool Has(const std::string& name) const noexcept { return name_to_index_.find(name) != name_to_index_.end(); }

    /*! \brief Access by zero-based index. */
    mlpdouble At(std::size_t index) const {
        if (index >= values_.size()) {
            throw std::out_of_range("PhysicsData: index " + std::to_string(index) + " is out of range.");
        }
        return values_[index];
    }

    /*! \brief Access by variable name. */
    mlpdouble At(const std::string& name) const {
        const auto it = name_to_index_.find(name);
        if (it == name_to_index_.end()) {
            throw std::out_of_range("PhysicsData: variable '" + name + "' not found.");
        }
        return values_[it->second];
    }

    // Convenience aliases used by residual writers.
    mlpdouble Ref(std::size_t index) const { return At(index); }
    mlpdouble Ref(const std::string& name) const { return At(name); }

    const std::vector<std::string>& Names() const noexcept { return names_; }
    const std::vector<mlpdouble>& Values() const noexcept { return values_; }

private:
    std::vector<mlpdouble> values_;
    std::vector<std::string> names_;
    std::unordered_map<std::string, std::size_t> name_to_index_;
};

// ============================================================================
// PhysicsState
// ============================================================================
/*!
 * \brief Read-only view of network quantities at a single collocation point.
 *
 * Provides two levels of access:
 *   - Network-level: In(), Out(), Jac(), Hess() use full-network indices.
 *   - Equation-level: EquationIn(), EquationOut(), EquationJac(), EquationHess()
 *     use the (usually smaller) index sets declared by a CPhysicsEquation.
 *
 * Jacobian layout expected from PredictionResult (input-major):
 *   jacobian[input_i][output_o]
 * Hessian layout:
 *   hessian[input_i][input_j][output_o]
 *
 * The object does not own the PredictionResult; the caller must keep it
 * alive for the lifetime of the PhysicsState.
 */
class PhysicsState {
public:
    /*!
     * \param[in] pred                     Network prediction (inputs, outputs, optional derivatives).
     * \param[in] equation_input_indices   Mapping from equation-local input indices to network indices.
     * \param[in] equation_output_indices  Mapping from equation-local output indices to network indices.
     * \param[in] n_network_inputs         Expected number of network inputs (for validation).
     * \param[in] n_network_outputs        Expected number of network outputs (for validation).
     */
    PhysicsState(const PredictionResult& pred,
                 const std::vector<std::size_t>& equation_input_indices,
                 const std::vector<std::size_t>& equation_output_indices,
                 std::size_t n_network_inputs,
                 std::size_t n_network_outputs)
        : pred_(pred), equation_input_indices_(equation_input_indices),
          equation_output_indices_(equation_output_indices),
          n_network_inputs_(n_network_inputs),
          n_network_outputs_(n_network_outputs) {
        if (pred_.inputs.size() != n_network_inputs_) {
            throw std::invalid_argument(
                "PhysicsState: PredictionResult input dimension mismatch. " +
                std::string("expected=") + std::to_string(n_network_inputs_) +
                ", got=" + std::to_string(pred_.inputs.size()));
        }
        if (pred_.outputs.size() != n_network_outputs_) {
            throw std::invalid_argument(
                "PhysicsState: PredictionResult output dimension mismatch. " +
                std::string("expected=") + std::to_string(n_network_outputs_) +
                ", got=" + std::to_string(pred_.outputs.size()));
        }
    }

    // ---- Network-level accessors -------------------------------------------

    /*! \brief Network input by full-network index. */
    mlpdouble In(std::size_t network_input_index) const {
        if (network_input_index >= pred_.inputs.size()) {
            throw std::out_of_range("PhysicsState::In: input index out of range.");
        }
        return pred_.inputs[network_input_index];
    }

    /*! \brief Network output by full-network index. */
    mlpdouble Out(std::size_t network_output_index) const {
        if (network_output_index >= pred_.outputs.size()) {
            throw std::out_of_range("PhysicsState::Out: output index out of range.");
        }
        return pred_.outputs[network_output_index];
    }

    /*!
     * \brief First derivative \partial output / \partial input (full-network indices).
     * \note Requires that PredictionResult::jacobian is non-null.
     */
    mlpdouble Jac(std::size_t input_index, std::size_t output_index) const {
        if (input_index >= pred_.inputs.size())  throw std::out_of_range("PhysicsState::Jac: input index out of range.");
        if (output_index >= pred_.outputs.size()) throw std::out_of_range("PhysicsState::Jac: output index out of range.");
        if (pred_.jacobian == nullptr)            throw std::runtime_error("PhysicsState::Jac: Jacobian was not provided by PredictionResult.");
        return pred_.jacobian[input_index][output_index];
    }

    /*!
     * \brief Second derivative \partial^{2} output / (\partial input_i \partial input_j).
     * \note Requires that PredictionResult::hessian is non-null.
     */
    mlpdouble Hess(std::size_t input_i, std::size_t input_j, std::size_t output_index) const {
        if (input_i >= pred_.inputs.size() || input_j >= pred_.inputs.size()) throw std::out_of_range("PhysicsState::Hess: input index out of range.");
        if (output_index >= pred_.outputs.size())                             throw std::out_of_range("PhysicsState::Hess: output index out of range.");
        if (pred_.hessian == nullptr)                                         throw std::runtime_error("PhysicsState::Hess: Hessian was not provided by PredictionResult.");
        return pred_.hessian[input_i][input_j][output_index];
    }

    // ---- Equation-level accessors (preferred for residual writers) ---------

    /*! \brief Input that belongs to the current equation (equation-local index). */
    mlpdouble EquationIn(std::size_t equation_input_index) const {
        if (equation_input_index >= equation_input_indices_.size()) throw std::out_of_range("PhysicsState::EquationIn: index out of range.");
        return In(equation_input_indices_[equation_input_index]);
    }

    /*! \brief Output that belongs to the current equation (equation-local index). */
    mlpdouble EquationOut(std::size_t equation_output_index) const {
        if (equation_output_index >= equation_output_indices_.size()) throw std::out_of_range("PhysicsState::EquationOut: index out of range.");
        return Out(equation_output_indices_[equation_output_index]);
    }

    /*!
     * \brief Jacobian entry restricted to the variables declared by the equation.
     *
     * Example: for residual “dy/du = 0” with input_names={"u"}, output_names={"y"}
     *          EquationJac(0,0) returns dy/du.
     */
    mlpdouble EquationJac(std::size_t equation_input_index, std::size_t equation_output_index) const {
        if (equation_input_index >= equation_input_indices_.size())  throw std::out_of_range("PhysicsState::EquationJac: input index out of range.");
        if (equation_output_index >= equation_output_indices_.size()) throw std::out_of_range("PhysicsState::EquationJac: output index out of range.");
        return Jac(equation_input_indices_[equation_input_index], equation_output_indices_[equation_output_index]);
    }

    /*! \brief Hessian entry restricted to the variables declared by the equation. */
    mlpdouble EquationHess(std::size_t equation_input_i, std::size_t equation_input_j, std::size_t equation_output_index) const {
        if (equation_input_i >= equation_input_indices_.size() || equation_input_j >= equation_input_indices_.size()) throw std::out_of_range("PhysicsState::EquationHess: input index out of range.");
        if (equation_output_index >= equation_output_indices_.size()) throw std::out_of_range("PhysicsState::EquationHess: output index out of range.");
        return Hess(equation_input_indices_[equation_input_i], equation_input_indices_[equation_input_j], equation_output_indices_[equation_output_index]);
    }

    std::size_t NumEquationInputs()  const noexcept { return equation_input_indices_.size(); }
    std::size_t NumEquationOutputs() const noexcept { return equation_output_indices_.size(); }
    std::size_t NumNetworkInputs()   const noexcept { return n_network_inputs_; }
    std::size_t NumNetworkOutputs()  const noexcept { return n_network_outputs_; }

private:
    const PredictionResult& pred_;
    const std::vector<std::size_t>& equation_input_indices_;
    const std::vector<std::size_t>& equation_output_indices_;
    std::size_t n_network_inputs_;
    std::size_t n_network_outputs_;
};

// ============================================================================
// CPhysicsEquation
// ============================================================================
/*!
 * \brief Description of a single residual equation.
 *
 * Members
 * -------
 * name                 Unique identifier (used only for diagnostics).
 * input_names          Subset of network input names that the residual depends on.
 * output_names         Subset of network output names that the residual depends on.
 * weight               Multiplicative factor applied to residual^{2} (default 1).
 * requires_jacobian    Must be true if the residual reads any first derivative.
 * requires_hessian     Must be true if the residual reads any second derivative.
 * residual             User-supplied callable that returns the residual value.
 *
 * The trainer inspects requires_jacobian / requires_hessian to decide
 * which derivatives to request from the network at evaluation time.
 */
struct CPhysicsEquation {
    std::string name;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    double weight{1.0};
    bool requires_jacobian{false};
    bool requires_hessian{false};
    ResidualFunction residual;
};

// ============================================================================
// CPhysicsLoss
// ============================================================================
/*!
 * \brief Aggregates one or more CPhysicsEquation objects into a loss term.
 *
 * Construction performs all static validation (unique names, unknown
 * variable references, missing callbacks, negative weights) and resolves
 * the name\to index maps used by PhysicsState.
 *
 * Evaluation contract
 * -------------------
 * - EvaluateOne(pred [, physics_data])
 *     Returns the *un-normalised* sum
 *         sum_e  weight_e · residual_e(pred)^{2}
 *     for a single collocation point.  The caller (normally the trainer)
 *     is responsible for averaging over points and equations.
 *
 * - Evaluate(preds, physics_data)
 *     Batch interface: calls EvaluateOne for every point and returns the
 *     normalised mean
 *         (1/(N · N_eq)) · sum_p sum_e weight_e · residual_e^{2}.
 *
 * The two interfaces therefore differ by the normalisation factor; the
 * streaming trainer must use EvaluateOne exclusively.
 */
class CPhysicsLoss : public CBaseLoss {
public:
    /*!
     * \param[in] name                   Unique loss identifier.
     * \param[in] network_input_names    Ordered list of network input names (must match the network).
     * \param[in] network_output_names   Ordered list of network output names (must match the network).
     * \param[in] physics_variable_names Names of optional auxiliary variables (may be empty).
     * \param[in] equations              Non-empty list of residual equations.
     */
    CPhysicsLoss(std::string name,
                 std::vector<std::string> network_input_names,
                 std::vector<std::string> network_output_names,
                 std::vector<std::string> physics_variable_names,
                 std::vector<CPhysicsEquation> equations)
        : CBaseLoss(name),
          network_input_names_(std::move(network_input_names)),
          network_output_names_(std::move(network_output_names)),
          physics_variable_names_(std::move(physics_variable_names)),
          equations_(std::move(equations)) {
        if (name_.empty()) throw std::invalid_argument("CPhysicsLoss: loss name cannot be empty.");
        if (equations_.empty()) throw std::invalid_argument("CPhysicsLoss: at least one equation is required.");

        ValidateNetworkNames();
        ValidatePhysicsVariableNames();
        ResolveEquationIndices();
        ValidateEquationCallbacks();
        ValidateEquationWeights();
        DetermineDerivativeRequirements();
    }

    std::size_t NumEquations() const noexcept { return equations_.size(); }
    std::size_t NumPhysicsVariables() const noexcept { return physics_variable_names_.size(); }
    bool RequiresJacobian() const noexcept { return requires_jacobian_; }
    bool RequiresHessian() const noexcept { return requires_hessian_; }

    const std::vector<std::string>& GetNetworkInputNames() const noexcept { return network_input_names_; }
    const std::vector<std::string>& GetNetworkOutputNames() const noexcept { return network_output_names_; }
    const std::vector<std::string>& GetPhysicsVariableNames() const noexcept { return physics_variable_names_; }

    const CPhysicsEquation& GetEquation(std::size_t index) const {
        if (index >= equations_.size()) throw std::out_of_range("CPhysicsLoss::GetEquation: index out of range.");
        return equations_[index];
    }

    /*!
     * \brief Un-normalised physics loss for a single collocation point.
     *
     * \param[in] pred          Network prediction (must contain Jacobian/Hessian
     *                          when any equation requests them).
     * \param[in] physics_data  Optional auxiliary values; size must equal
     *                          NumPhysicsVariables() (may be empty).
     * \return sum_e weight_e · residual_e^{2}   (no division by N or N_eq)
     */
    mlpdouble EvaluateOne(const PredictionResult& pred,
                           const std::vector<mlpdouble>& physics_data = {}) const {
        ValidatePrediction(pred);
        if (physics_data.size() != physics_variable_names_.size()) {
            throw std::invalid_argument(
                "CPhysicsLoss '" + name_ + "': physics data size mismatch. Expected " +
                std::to_string(physics_variable_names_.size()) + ", got " +
                std::to_string(physics_data.size()));
        }

        PhysicsData data(physics_data, physics_variable_names_);
        mlpdouble raw_loss = mlpdouble(0.0);
        for (std::size_t e = 0; e < equations_.size(); ++e) {
            PhysicsState state(pred,
                               equation_input_indices_[e],
                               equation_output_indices_[e],
                               network_input_names_.size(),
                               network_output_names_.size());
            const mlpdouble residual = equations_[e].residual(state, data);
            const double w = equations_[e].weight;
            raw_loss += mlpdouble(w) * residual * residual;
        }
        return raw_loss;
    }

    /*!
     * \brief Normalised batch physics loss.
     *
     * Calls EvaluateOne for every prediction and returns the mean
     * residual^{2} over points and equations:
     *     (1/(N · N_eq)) · sum_p EvaluateOne(preds[p], \ldots)
     *
     * Prefer EvaluateOne when the trainer streams collocation points one
     * at a time; the trainer itself performs the equivalent normalisation.
     */
    mlpdouble Evaluate(const std::vector<PredictionResult>& preds,
                       const std::vector<std::vector<mlpdouble>>& physics_data) override {
        const std::size_t N = preds.size();
        if (N == 0) {
            last_loss_value_ = 0.0;
            return mlpdouble(0.0);
        }
        if (!physics_variable_names_.empty() && physics_data.size() != N) {
            throw std::invalid_argument(
                "CPhysicsLoss '" + name_ + "': physics_data must contain exactly one row per prediction.");
        }
        if (physics_variable_names_.empty() && !physics_data.empty() && physics_data.size() != N) {
            throw std::invalid_argument(
                "CPhysicsLoss '" + name_ + "': supplied physics_data has wrong number of rows.");
        }

        mlpdouble raw_total = mlpdouble(0.0);
        for (std::size_t p = 0; p < N; ++p) {
            const std::vector<mlpdouble> empty_data;
            const auto& point_data = physics_data.empty() ? empty_data : physics_data[p];
            raw_total += EvaluateOne(preds[p], point_data);
        }

        // Average over both points and equations so that the loss magnitude
        // is independent of the number of residual equations.
        const double denominator = static_cast<double>(N) * static_cast<double>(equations_.size());
        const mlpdouble normalized = raw_total / mlpdouble(denominator);
        last_loss_value_ = to_double(normalized);
        return normalized;
    }

private:
    // ---- Construction-time validation --------------------------------------

    void ValidateNetworkNames() {
        ValidateUniqueNames(network_input_names_, "network input");
        ValidateUniqueNames(network_output_names_, "network output");
    }

    void ValidateUniqueNames(const std::vector<std::string>& names, const std::string& context) const {
        std::unordered_set<std::string> seen;
        for (const auto& name : names) {
            if (name.empty()) {
                throw std::invalid_argument(
                    "CPhysicsLoss '" + name_ + "': " + context + " name cannot be empty.");
            }
            if (seen.find(name) != seen.end()) {
                throw std::invalid_argument(
                    "CPhysicsLoss '" + name_ + "': duplicate " + context + " name '" + name + "'.");
            }
            seen.insert(name);
        }
    }

    void ValidatePhysicsVariableNames() const {
        std::unordered_set<std::string> seen;
        for (const auto& name : physics_variable_names_) {
            if (name.empty()) {
                throw std::invalid_argument(
                    "CPhysicsLoss '" + name_ + "': physics variable name cannot be empty.");
            }
            if (seen.find(name) != seen.end()) {
                throw std::invalid_argument(
                    "CPhysicsLoss '" + name_ + "': duplicate physics variable name '" + name + "'.");
            }
            seen.insert(name);
        }
    }

    /*!
     * \brief Resolve every equation’s input/output names onto network indices.
     *
     * After this call, equation_input_indices_[e][i] is the network index
     * that corresponds to the i-th name in equations_[e].input_names
     * (likewise for outputs).  Unknown names raise an exception.
     */
    void ResolveEquationIndices() {
        equation_input_indices_.resize(equations_.size());
        equation_output_indices_.resize(equations_.size());

        for (std::size_t e = 0; e < equations_.size(); ++e) {
            const auto& eq = equations_[e];
            if (eq.name.empty()) {
                throw std::invalid_argument("CPhysicsLoss '" + name_ + "': equation name cannot be empty.");
            }

            ValidateUniqueNames(eq.input_names, "input for equation '" + eq.name + "'");
            ValidateUniqueNames(eq.output_names, "output for equation '" + eq.name + "'");

            std::vector<std::size_t> input_indices;
            input_indices.reserve(eq.input_names.size());
            for (const auto& input_name : eq.input_names) {
                auto it = std::find(network_input_names_.begin(), network_input_names_.end(), input_name);
                if (it == network_input_names_.end()) {
                    throw std::invalid_argument(
                        "CPhysicsLoss '" + name_ + "': equation '" + eq.name +
                        "' references unknown network input '" + input_name + "'.");
                }
                input_indices.push_back(static_cast<std::size_t>(std::distance(network_input_names_.begin(), it)));
            }

            std::vector<std::size_t> output_indices;
            output_indices.reserve(eq.output_names.size());
            for (const auto& output_name : eq.output_names) {
                auto it = std::find(network_output_names_.begin(), network_output_names_.end(), output_name);
                if (it == network_output_names_.end()) {
                    throw std::invalid_argument(
                        "CPhysicsLoss '" + name_ + "': equation '" + eq.name +
                        "' references unknown network output '" + output_name + "'.");
                }
                output_indices.push_back(static_cast<std::size_t>(std::distance(network_output_names_.begin(), it)));
            }

            equation_input_indices_[e] = std::move(input_indices);
            equation_output_indices_[e] = std::move(output_indices);
        }
    }

    void ValidateEquationCallbacks() const {
        for (const auto& eq : equations_) {
            if (!eq.residual) {
                throw std::invalid_argument(
                    "CPhysicsLoss '" + name_ + "': equation '" + eq.name + "' has no residual callback.");
            }
        }
    }

    void ValidateEquationWeights() const {
        for (const auto& eq : equations_) {
            if (eq.weight < 0.0) {
                throw std::invalid_argument(
                    "CPhysicsLoss '" + name_ + "': equation '" + eq.name +
                    "' has negative weight (" + std::to_string(eq.weight) + ").");
            }
        }
    }

    /*! Aggregate the derivative requirements of all equations. */
    void DetermineDerivativeRequirements() {
        requires_jacobian_ = false;
        requires_hessian_ = false;
        for (const auto& eq : equations_) {
            requires_jacobian_ = requires_jacobian_ || eq.requires_jacobian;
            requires_hessian_ = requires_hessian_ || eq.requires_hessian;
        }
    }

    /*! Runtime check that a PredictionResult supplies everything the loss needs. */
    void ValidatePrediction(const PredictionResult& pred) const {
        if (pred.inputs.size() != network_input_names_.size()) {
            throw std::invalid_argument(
                "CPhysicsLoss '" + name_ + "': PredictionResult input dimension mismatch. Expected " +
                std::to_string(network_input_names_.size()) + ", got " + std::to_string(pred.inputs.size()));
        }
        if (pred.outputs.size() != network_output_names_.size()) {
            throw std::invalid_argument(
                "CPhysicsLoss '" + name_ + "': PredictionResult output dimension mismatch. Expected " +
                std::to_string(network_output_names_.size()) + ", got " + std::to_string(pred.outputs.size()));
        }
        if (requires_jacobian_ && pred.jacobian == nullptr) {
            throw std::invalid_argument(
                "CPhysicsLoss '" + name_ + "': Jacobian is required but PredictionResult does not contain one.");
        }
        if (requires_hessian_ && pred.hessian == nullptr) {
            throw std::invalid_argument(
                "CPhysicsLoss '" + name_ + "': Hessian is required but PredictionResult does not contain one.");
        }
    }

    // ---- Data members ------------------------------------------------------
    std::vector<std::string> network_input_names_;
    std::vector<std::string> network_output_names_;
    std::vector<std::string> physics_variable_names_;
    std::vector<CPhysicsEquation> equations_;

    // Per-equation mappings: equation-local index → network index.
    std::vector<std::vector<std::size_t>> equation_input_indices_;
    std::vector<std::vector<std::size_t>> equation_output_indices_;

    bool requires_jacobian_{false};
    bool requires_hessian_{false};
};

} // namespace MLPToolbox
