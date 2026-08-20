/*!
 * \file CMLPTrainer.hpp
 * \brief PINN trainer with streaming physics evaluation and reverse-mode AD.
 *
 * This trainer matches the current CPhysicsLoss contract:
 *
 *   - PredictionResult owns inputs/outputs
 *   - jacobian/hessian are non-owning views
 *   - derivative layout is input-major:
 *       jacobian[input][output]
 *       hessian[input_i][input_j][output]
 *   - CPhysicsLoss::EvaluateOne() returns a raw per-point loss
 *   - CPhysicsLoss::Evaluate() performs canonical normalization
 *
 * Physics training mode:
 *   For each collocation point (or mini-batch of points):
 *     Predict(point, jac, hess)
 *     copy derivatives into reusable storage
 *     EvaluateOne() for every physics loss on that set
 *
 * Reference-data mode:
 *   Standard mini-batch supervised learning. Supports multiple reference
 *   data terms (M > 1) for generic annealing.
 *
 * The trainer keeps one point worth of derivative data in memory at a time.
 *
 * Derivative accessor convention (VERIFIED):
 *   CNeuralNetwork stores derivatives in input-major layout internally:
 *     output_Jacobian[iInput][iOutput]
 *     output_Hessian[iInput][jInput][iOutput]
 *
 *   The public accessors present arguments in output-first order:
 *     GetJacobian(iOutput, iInput)  -> output_Jacobian[iInput][iOutput]
 *     GetHessian(iOutput, iInput, jInput) -> output_Hessian[iInput][jInput][iOutput]
 *
 *   CPointDerivatives::Fill() calls these with (o, i) / (o, i, j) and
 *   stores results in input-major buffers.
 *
 * Physics data lifecycle:
 *   Physics data may be provided before or after Build(), but training
 *   fails at the first TrainStep() if a physics loss declares physics
 *   variables (NumPhysicsVariables() > 0) and no data has been set.
 *   Build() validates structure only; the runtime evaluation is the
 *   enforcement point for missing data.
 *
 * Convergence:
 *   Configurable via ConvergenceMetric enum.  Default is Weighted, which
 *   checks loss_total (L_phys + sum lambda_k * L_ref_k).  When the
 *   annealer is off, loss_total still differs from loss_raw if any
 *   equation weight is not 1.0.
 *
 * Empty collocation sets:
 *   Allowed.  A set with zero points contributes zero loss and zero
 *   gradients.  The set is skipped during streaming evaluation.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <mpi.h>

#include "CNeuralNetwork.hpp"
#include "CAdam.hpp"
#include "CGradientAnnealer.hpp"
#include "CBaseLoss.hpp"
#include "CPhysicsLoss.hpp"
#include "variable_def.hpp"

namespace MLPToolbox {

// ============================================================================
// Terminology
// ============================================================================
// - "Collocation point": a coordinate (x, y, ...) sampled inside the domain
//   or on its boundary where a PDE residual or boundary condition is
//   evaluated during training. Standard term from numerical PDE methods
//   (the "collocation method"), used throughout the PINN literature this
//   trainer implements (Raissi et al. 2019; Wang, Teng & Perdikaris 2020,
//   "Understanding and mitigating gradient pathologies in PINNs").
// - "Collocation set": a named, reusable group of collocation points (e.g.
//   "interior", "top_bc", "wall_bc") registered via SetCollocationPoints().
//   One or more losses (physics or boundary) can be attached to the same
//   set; the trainer mini-batches over each set's points independently.
// - "Lr" / "physics loss": a PDE-residual term, registered via
//   AddPhysicsLoss(). Always summed unweighted into the aggregate residual.
// - "Li" / "data-like term": anything the network is fit TO rather than a
//   PDE it must satisfy — supervised reference data (AddReferenceLoss()) or
//   boundary/initial conditions (AddBoundaryLoss()). Each Li term can get
//   its own adaptively-annealed weight (lambda_i), per Algorithm 1 of the
//   Wang/Teng/Perdikaris paper above.
// - "Gradient annealing": the adaptive-lambda scheme from that paper, which
//   rescales each Li term's contribution to the training gradient based on
//   how its gradient magnitude compares to the aggregate Lr gradient, to
//   counteract the "vanishing Li gradient" pathology the paper documents.
// ============================================================================


// ============================================================================
// Trainer configuration
// ============================================================================

enum class ConvergenceMetric {
    Raw,       // Unweighted: L_ref + sum L_phys
    Weighted   // Annealed:   L_phys + sum lambda_k * L_ref_k
};

struct TrainerConfig {
    std::size_t max_epochs{1000};
    std::size_t batch_size{32};          // 0 = full reference batch
    std::size_t physics_batch_size{0};   // 0 = full physics batch (all points)
    double      conv_tol_abs{1e-8};
    double      conv_tol_rel{1e-6};
    ConvergenceMetric conv_metric{ConvergenceMetric::Weighted};
    double      max_grad_norm{0.0};      // 0 = no clipping
    bool        use_annealer{true};
    bool        verbose{false};
    std::size_t log_every{100};          // 0 = disabled
    bool        shuffle_per_epoch{true};
    std::size_t annealer_update_freq{10};
};

struct TrainStepResult {
    double loss_ref{0.0};      // Sum of all reference losses
    std::vector<double> loss_phys;
    std::vector<double> loss_bcs;   // NEW: per-term boundary-condition losses
    double loss_total{0.0};    // weighted: L_phys + sum lambda_k * L_ref_k
    double loss_raw{0.0};      // unweighted: L_ref + sum L_phys
    std::vector<double> lambdas;
};

// ============================================================================
// CPointDerivatives
// ============================================================================

class CPointDerivatives {
public:
    CPointDerivatives(std::size_t n_in, std::size_t n_out)
        : n_in_(n_in),
          n_out_(n_out),
          jac_flat_(n_in * n_out),
          jac_rows_(n_in, nullptr),
          hess_flat_(n_in * n_in * n_out),
          hess_rows_(n_in * n_in, nullptr),
          hess_planes_(n_in, nullptr)
    {
        if (n_in_ == 0 || n_out_ == 0) {
            throw std::invalid_argument(
                "CPointDerivatives: network dimensions must be positive.");
        }

        for (std::size_t i = 0; i < n_in_; ++i) {
            jac_rows_[i] = &jac_flat_[i * n_out_];
        }

        for (std::size_t i = 0; i < n_in_; ++i) {
            hess_planes_[i] = &hess_rows_[i * n_in_];
            for (std::size_t j = 0; j < n_in_; ++j) {
                hess_rows_[i * n_in_ + j] = &hess_flat_[(i * n_in_ + j) * n_out_];
            }
        }
    }

    // Modified to accept a pre-allocated PredictionResult to prevent heap allocations
    void Fill(CNeuralNetwork& net,
              const std::vector<mlpdouble>& x,
              PredictionResult& pr,
              bool eval_jac,
              bool eval_hess)
    {
        if (x.size() != n_in_) {
            throw std::invalid_argument(
                "CPointDerivatives::Fill: input dimension mismatch.");
        }

        const bool compute_jac = eval_jac || eval_hess;

        net.Predict(x, compute_jac, eval_hess);

        pr.inputs = x;
        if (pr.outputs.size() != n_out_) {
            pr.outputs.resize(n_out_);
        }

        for (std::size_t o = 0; o < n_out_; ++o) {
            pr.outputs[o] = net.GetOutput(o);
        }

        pr.jacobian = nullptr;
        pr.hessian  = nullptr;

        if (compute_jac) {
            for (std::size_t i = 0; i < n_in_; ++i) {
                for (std::size_t o = 0; o < n_out_; ++o) {
                    jac_flat_[i * n_out_ + o] = net.GetJacobian(o, i);
                }
            }
            pr.jacobian = jac_rows_.data();
        }

        if (eval_hess) {
            for (std::size_t i = 0; i < n_in_; ++i) {
                for (std::size_t j = 0; j < n_in_; ++j) {
                    for (std::size_t o = 0; o < n_out_; ++o) {
                        hess_flat_[(i * n_in_ + j) * n_out_ + o] =
                            net.GetHessian(o, i, j);
                    }
                }
            }
            pr.hessian = hess_planes_.data();
        }
    }

private:
    std::size_t n_in_;
    std::size_t n_out_;

    std::vector<mlpdouble>   jac_flat_;
    std::vector<mlpdouble*>  jac_rows_;

    std::vector<mlpdouble>   hess_flat_;
    std::vector<mlpdouble*>  hess_rows_;
    std::vector<mlpdouble**> hess_planes_;
};

// ============================================================================
// CMLPTrainer
// ============================================================================

class CMLPTrainer {
public:
    CMLPTrainer(CNeuralNetwork& net,
                CAdam adam,
                AnnealerConfig annealer_cfg,
                TrainerConfig cfg = {})
        : net_(net),
          adam_(std::move(adam)),
          annealer_cfg_(annealer_cfg),
          cfg_(cfg),
          point_storage_(net.GetnInputs(), net.GetnOutputs()),
          rng_(std::random_device{}())
    {
        const std::size_t n_w = net_.GetWeightsBiases().size();
        if (n_w == 0) {
            throw std::invalid_argument(
                "CMLPTrainer: network has no trainable parameters.");
        }

        if (cfg_.conv_tol_abs < 0.0 || cfg_.conv_tol_rel < 0.0) {
            throw std::invalid_argument(
                "CMLPTrainer: convergence tolerances must be non-negative.");
        }

        if (cfg_.max_grad_norm < 0.0) {
            throw std::invalid_argument(
                "CMLPTrainer: max_grad_norm must be non-negative.");
        }

        adam_.initialize(n_w);
    }

    // ------------------------------------------------------------------------
    // Reference losses (Supports M > 1 for generic annealing)
    // ------------------------------------------------------------------------

    void AddReferenceLoss(std::shared_ptr<CBaseLoss> loss) {
        if (!loss) {
            throw std::invalid_argument(
                "CMLPTrainer: reference loss cannot be null.");
        }
        if (built_) {
            throw std::runtime_error(
                "CMLPTrainer: cannot add reference loss after Build().");
        }
        ref_losses_.push_back(std::move(loss));
    }

    void ClearReferenceLosses() {
        if (built_) {
            throw std::runtime_error(
                "CMLPTrainer: cannot clear reference losses after Build().");
        }
        ref_losses_.clear();
    }

    // --------------------------------------------------------------------
    // Boundary Losses (L_i evaluated on collocation sets)
    // --------------------------------------------------------------------
    void AddBoundaryLoss(std::shared_ptr<CPhysicsLoss> loss, const std::string& collocation_set_name) {
        if (!loss) throw std::invalid_argument("CMLPTrainer: boundary loss cannot be null.");
        if (coll_sets_.find(collocation_set_name) == coll_sets_.end()) throw std::invalid_argument("CMLPTrainer: collocation set '" + collocation_set_name + "' does not exist.");
        
        bcs_losses_.push_back(std::move(loss));
        bcs_set_name_.push_back(collocation_set_name);
    }

    // ------------------------------------------------------------------------
    // Training data
    // ------------------------------------------------------------------------

    void SetTrainingData(const std::vector<std::vector<mlpdouble>>& inputs,
                         const std::vector<std::vector<mlpdouble>>& targets)
    {
        if (built_) {
            throw std::runtime_error(
                "CMLPTrainer: cannot modify training data after Build().");
        }

        if (inputs.size() != targets.size()) {
            throw std::invalid_argument(
                "CMLPTrainer: input/target sample count mismatch.");
        }

        const std::size_t n_in  = net_.GetnInputs();
        const std::size_t n_out = net_.GetnOutputs();

        for (std::size_t i = 0; i < inputs.size(); ++i) {
            if (inputs[i].size() != n_in) {
                throw std::invalid_argument(
                    "CMLPTrainer: input dimension mismatch at sample " +
                    std::to_string(i) + ".");
            }
            if (targets[i].size() != n_out) {
                throw std::invalid_argument(
                    "CMLPTrainer: target dimension mismatch at sample " +
                    std::to_string(i) + ".");
            }
        }

        train_inputs_ = inputs;
        train_targets_ = targets;

        total_samples_ = inputs.size();
        indices_.resize(total_samples_);
        std::iota(indices_.begin(), indices_.end(), std::size_t{0});

        batch_cursor_ = 0;
        epoch_started_ = false;
    }

    // ------------------------------------------------------------------------
    // Collocation sets
    // ------------------------------------------------------------------------

    void SetCollocationPoints(const std::string& set_name,
                              const std::vector<std::vector<mlpdouble>>& points)
    {
        if (built_) {
            throw std::runtime_error(
                "CMLPTrainer: cannot modify collocation sets after Build().");
        }

        if (set_name.empty()) {
            throw std::invalid_argument(
                "CMLPTrainer: collocation set name cannot be empty.");
        }

        const std::size_t n_in = net_.GetnInputs();
        for (std::size_t p = 0; p < points.size(); ++p) {
            if (points[p].size() != n_in) {
                throw std::invalid_argument(
                    "CMLPTrainer: collocation point dimension mismatch in set '" +
                    set_name + "' at point " + std::to_string(p) + ".");
            }
        }

        const bool is_new = (coll_sets_.find(set_name) == coll_sets_.end());
        coll_sets_[set_name] = points;
        if (is_new) {
            set_order_.push_back(set_name);
            
            coll_indices_[set_name].resize(points.size());
            std::iota(coll_indices_[set_name].begin(), coll_indices_[set_name].end(), std::size_t{0});
            coll_batch_cursor_[set_name] = 0;
            coll_epoch_started_[set_name] = false;
        }
    }

    void SetPhysicsCollocationPoints(const std::vector<std::vector<mlpdouble>>& points) {
        SetCollocationPoints("default", points);
    }

    // ------------------------------------------------------------------------
    // Physics data for a registered loss
    // ------------------------------------------------------------------------

    void SetPhysicsData(const std::string& loss_name,
                        const std::vector<std::vector<mlpdouble>>& data)
    {
        const std::size_t idx = FindPhysicsLossIndex(loss_name);
        const std::string& set_name = phys_set_name_[idx];

        auto set_it = coll_sets_.find(set_name);
        if (set_it == coll_sets_.end()) {
            throw std::runtime_error(
                "CMLPTrainer: collocation set '" + set_name + "' not found.");
        }

        const std::size_t n_points = set_it->second.size();
        const std::size_t n_phys   = phys_losses_[idx]->NumPhysicsVariables();

        if (n_phys == 0 && !data.empty()) {
            throw std::invalid_argument(
                "CMLPTrainer: loss '" + loss_name +
                "' has no physics variables but data was provided.");
        }

        if (!data.empty()) {
            if (data.size() != n_points) {
                throw std::invalid_argument(
                    "CMLPTrainer: physics data row count mismatch for loss '" +
                    loss_name + "'. Expected " +
                    std::to_string(n_points) + ", got " +
                    std::to_string(data.size()) + ".");
            }

            for (std::size_t p = 0; p < data.size(); ++p) {
                if (data[p].size() != n_phys) {
                    throw std::invalid_argument(
                        "CMLPTrainer: physics data dimension mismatch for loss '" +
                        loss_name + "' at point " + std::to_string(p) +
                        ". Expected " + std::to_string(n_phys) +
                        ", got " + std::to_string(data[p].size()) + ".");
                }
            }
        }

        phys_data_[idx] = data;
    }

    std::vector<std::vector<mlpdouble>> MakeZeroPhysicsData(const std::string& loss_name) const {
        const std::size_t idx = FindPhysicsLossIndex(loss_name);
        const std::string& set_name = phys_set_name_[idx];
        const std::size_t n_points = coll_sets_.at(set_name).size();
        const std::size_t n_phys   = phys_losses_[idx]->NumPhysicsVariables();

        if (n_phys == 0) {
            return {};
        }

        return std::vector<std::vector<mlpdouble>>(
            n_points, std::vector<mlpdouble>(n_phys, mlpdouble(0.0)));
    }

    // ------------------------------------------------------------------------
    // Physics losses
    // ------------------------------------------------------------------------

    void AddPhysicsLoss(std::shared_ptr<CPhysicsLoss> loss,
                        const std::string& collocation_set_name,
                        std::vector<std::vector<mlpdouble>> per_point_data = {})
    {
        if (built_) {
            throw std::runtime_error(
                "CMLPTrainer: cannot add physics loss after Build().");
        }

        if (!loss) {
            throw std::invalid_argument(
                "CMLPTrainer: physics loss cannot be null.");
        }

        if (collocation_set_name.empty()) {
            throw std::invalid_argument(
                "CMLPTrainer: collocation set name cannot be empty.");
        }

        if (coll_sets_.find(collocation_set_name) == coll_sets_.end()) {
            throw std::invalid_argument(
                "CMLPTrainer: collocation set '" + collocation_set_name +
                "' does not exist.");
        }

        if (!HasSameSignature(net_.GetInputVars(), loss->GetNetworkInputNames())) {
            throw std::invalid_argument(
                "CMLPTrainer: physics loss '" + loss->GetName() +
                "' input names do not match the network input names.");
        }

        if (!HasSameSignature(net_.GetOutputVars(), loss->GetNetworkOutputNames())) {
            throw std::invalid_argument(
                "CMLPTrainer: physics loss '" + loss->GetName() +
                "' output names do not match the network output names.");
        }

        for (const auto& existing : phys_losses_) {
            if (existing->GetName() == loss->GetName()) {
                throw std::invalid_argument(
                    "CMLPTrainer: duplicate physics loss name '" +
                    loss->GetName() + "'.");
            }
        }

        const std::size_t n_phys = loss->NumPhysicsVariables();

        if (n_phys == 0 && !per_point_data.empty()) {
            throw std::invalid_argument(
                "CMLPTrainer: loss '" + loss->GetName() +
                "' has no physics variables but data was provided.");
        }

        if (!per_point_data.empty()) {
            const auto& points = coll_sets_.at(collocation_set_name);

            if (per_point_data.size() != points.size()) {
                throw std::invalid_argument(
                    "CMLPTrainer: physics data row count mismatch for loss '" +
                    loss->GetName() + "'.");
            }

            for (std::size_t p = 0; p < per_point_data.size(); ++p) {
                if (per_point_data[p].size() != n_phys) {
                    throw std::invalid_argument(
                        "CMLPTrainer: physics data dimension mismatch for loss '" +
                        loss->GetName() + "' at point " + std::to_string(p) + ".");
                }
            }
        }

        phys_losses_.push_back(std::move(loss));
        phys_set_name_.push_back(collocation_set_name);
        phys_data_.push_back(std::move(per_point_data));
    }

    // ------------------------------------------------------------------------
    // Build
    // ------------------------------------------------------------------------

    void Build() {
        if (built_) {
            return;
        }

        if (!HasUniqueNames(net_.GetInputVars())) {
            throw std::invalid_argument(
                "CMLPTrainer: network input names must be unique.");
        }

        if (!HasUniqueNames(net_.GetOutputVars())) {
            throw std::invalid_argument(
                "CMLPTrainer: network output names must be unique.");
        }

        const bool has_ref_objective = !ref_losses_.empty() && (total_samples_ > 0);
        const bool has_phys_objective = !phys_losses_.empty();

        if (!has_ref_objective && !has_phys_objective) {
            throw std::runtime_error(
                "CMLPTrainer: no active training objective.");
        }

        losses_by_set_.clear();
        set_needs_jac_.clear();
        set_needs_hess_.clear();
        active_set_order_.clear();
        empty_sets_.clear();

        for (std::size_t k = 0; k < phys_losses_.size(); ++k) {
            if (!phys_losses_[k]) {
                throw std::logic_error("CMLPTrainer: null physics loss.");
            }

            const std::string& set_name = phys_set_name_[k];
            auto set_it = coll_sets_.find(set_name);
            if (set_it == coll_sets_.end()) {
                throw std::runtime_error(
                    "CMLPTrainer: physics loss '" +
                    phys_losses_[k]->GetName() +
                    "' references missing collocation set '" +
                    set_name + "'.");
            }

            const std::size_t n_points = set_it->second.size();
            const std::size_t n_phys   = phys_losses_[k]->NumPhysicsVariables();

            if (phys_losses_[k]->NumEquations() == 0) {
                throw std::runtime_error(
                    "CMLPTrainer: physics loss '" + phys_losses_[k]->GetName() +
                    "' has zero equations.");
            }

            if (n_points == 0) {
                empty_sets_.insert(set_name);
                if (cfg_.verbose) {
                    std::cout << "CMLPTrainer: collocation set '" << set_name
                              << "' is empty — loss '" << phys_losses_[k]->GetName()
                              << "' will contribute zero.\n";
                }
                continue;
            }

            if (!phys_data_[k].empty()) {
                if (phys_data_[k].size() != n_points) {
                    throw std::runtime_error(
                        "CMLPTrainer: physics data row count mismatch for loss '" +
                        phys_losses_[k]->GetName() + "'. Expected " +
                        std::to_string(n_points) + ", got " +
                        std::to_string(phys_data_[k].size()) + ".");
                }

                for (std::size_t p = 0; p < phys_data_[k].size(); ++p) {
                    if (phys_data_[k][p].size() != n_phys) {
                        throw std::runtime_error(
                            "CMLPTrainer: physics data dimension mismatch for loss '" +
                            phys_losses_[k]->GetName() + "' at point " +
                            std::to_string(p) + ". Expected " +
                            std::to_string(n_phys) + ", got " +
                            std::to_string(phys_data_[k][p].size()) + ".");
                    }
                }
            }

            losses_by_set_[set_name].push_back(k);
            set_needs_jac_[set_name] = set_needs_jac_[set_name] ||
                                       phys_losses_[k]->RequiresJacobian();
            set_needs_hess_[set_name] = set_needs_hess_[set_name] ||
                                        phys_losses_[k]->RequiresHessian();
        }

        for (const auto& set_name : set_order_) {
            if (losses_by_set_.find(set_name) != losses_by_set_.end() &&
                empty_sets_.find(set_name) == empty_sets_.end())
            {
                active_set_order_.push_back(set_name);
            }
        }


        bcs_losses_by_set_.clear();
        for (std::size_t k = 0; k < bcs_losses_.size(); ++k) {
            const std::string& set_name = bcs_set_name_[k];
            bcs_losses_by_set_[set_name].push_back(k);
            
            // Add to active set order if it has points
            if (coll_sets_.find(set_name) != coll_sets_.end() && !coll_sets_.at(set_name).empty()) {
                if (std::find(active_set_order_.begin(), active_set_order_.end(), set_name) == active_set_order_.end()) {
                    active_set_order_.push_back(set_name);
                }
                // FIX: BC sets must be registered in the needs_jac/hess maps so .at() doesn't throw
                if (set_needs_jac_.find(set_name) == set_needs_jac_.end()) set_needs_jac_[set_name] = false;
                if (set_needs_hess_.find(set_name) == set_needs_hess_.end()) set_needs_hess_[set_name] = false;
            }
        }

        annealer_.reset();
        const bool has_data_like_terms = has_ref_objective || !bcs_losses_.empty();
        if (cfg_.use_annealer && has_data_like_terms && has_phys_objective) {
            AnnealerConfig a = annealer_cfg_;
            // M data terms includes BCs now! Works with reference data,
            // boundary losses, or both — a pure-PINN run with only
            // AddBoundaryLoss() terms and no SetTrainingData() must still
            // get an annealer if BCs are present.
            a.n_data_terms = ref_losses_.size() + bcs_losses_.size(); 
            annealer_ = std::make_unique<CGradientAnnealer>(a);
        }

        built_ = true;
    }

    // ------------------------------------------------------------------------
    // Train one step
    //
    // Decomposed per the Single Responsibility Principle: each stage of a
    // training step (sensitivity setup, tape/weight registration, loss
    // evaluation, normalization, gradient computation, clipping, the Adam
    // update, and result bookkeeping) lives in its own private method below.
    // This method is intentionally just the orchestration of those stages.
    // ------------------------------------------------------------------------

    TrainStepResult TrainStep() {
        if (!built_) {
            Build();
        }

        using Tape = typename mlpdouble::Tape;
        Tape& tape = mlpdouble::getTape();

        TrainStepState st;
        st.n_phys   = phys_losses_.size();
        st.n_ref    = ref_losses_.size();
        st.have_ref = (st.n_ref > 0) && (total_samples_ > 0);

        if (st.have_ref) {
            NextReferenceBatch(); // Updates batch_x_ and batch_y_
        }

        RegisterWeightsOnTape(tape, st);
        InitializeSensitivities(st);

        EvaluateReferenceLoss(tape, st);
        EvaluatePhysicsAndBoundaryLosses(st);
        NormalizeAndRegisterLosses(tape, st);

        // Only pay for the full (n_ref + n_bcs + 1)-sweep annealed path on
        // steps that actually refresh lambda; otherwise reuse the cached
        // lambdas in a single combined sweep.
        const bool is_anneal_update_step =
            (cfg_.annealer_update_freq > 0) && (step_ % cfg_.annealer_update_freq == 0);
        const bool have_data_like_terms = st.have_ref || !st.L_bcs.empty();
        const bool need_separate_grads =
            (annealer_ && have_data_like_terms && st.n_phys > 0 && is_anneal_update_step);

        st.lambdas.assign(st.n_ref + bcs_losses_.size(), 1.0);
        if (annealer_) {
            for (std::size_t i = 0; i < st.n_ref + bcs_losses_.size(); ++i) {
                st.lambdas[i] = annealer_->get_lambda(i);
            }
        }

        if (!need_separate_grads) {
            ComputeFastPathGradient(tape, st);
        } else {
            ComputeAnnealedPathGradient(tape, st);
        }

        tape.reset();

        ApplyGradientClipping(st.n_w);
        ApplyAdamUpdate(st);

        TrainStepResult result = BuildTrainStepResult(st);
        last_result_ = result;
        ++step_;

        if (cfg_.verbose && cfg_.log_every > 0 && (step_ % cfg_.log_every == 0)) {
            LogStep(result);
        }

        return result;
    }

    // ------------------------------------------------------------------------
    // Train one epoch
    // ------------------------------------------------------------------------

    void TrainEpoch() {
        const bool has_ref = (total_samples_ > 0 && !ref_losses_.empty());
        const bool has_phys = !active_set_order_.empty();

        if (!has_ref && !has_phys) return;

        PrepareEpoch();
        PreparePhysicsEpoch();

        std::size_t n_batches = 1;
        if (has_ref) {
            const std::size_t batch_size =
                cfg_.batch_size > 0 ? std::min(cfg_.batch_size, total_samples_) : total_samples_;
            n_batches = (total_samples_ + batch_size - 1) / batch_size;
        } else if (has_phys) {
            const std::string& first_set = active_set_order_[0];
            const std::size_t n_points = coll_sets_.at(first_set).size();
            const std::size_t phys_bs = cfg_.physics_batch_size > 0 ? 
                                        std::min(cfg_.physics_batch_size, n_points) : n_points;
            n_batches = (n_points + phys_bs - 1) / phys_bs;
            if (n_batches == 0) n_batches = 1;
        }

        // n_batches above is computed from THIS rank's local
        // data/collocation-set size, which can differ across ranks (e.g.
        // reference-data partitioning that distributes a remainder to the
        // last rank). If ranks disagree on n_batches, they call TrainStep()
        // — and therefore MPI_Allreduce() — a different number of times in
        // the same epoch, which is undefined behavior for MPI collectives
        // (can hang, or silently pair up gradients from different logical
        // steps/epochs across ranks). Force agreement via a collective MAX;
        // NextPhysicsBatch()/NextReferenceBatch() both wrap-and-reshuffle
        // when exhausted, so a rank with fewer local samples than the
        // agreed n_batches safely reuses (reshuffled) data rather than
        // erroring.
        if (use_mpi_ && mpi_size_ > 1) {
            unsigned long long n_batches_ull = static_cast<unsigned long long>(n_batches);
            MPI_Allreduce(MPI_IN_PLACE, &n_batches_ull, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
            n_batches = static_cast<std::size_t>(n_batches_ull);
        }

        // Reset epoch sums
        epoch_loss_sum_        = 0.0;
        epoch_loss_total_sum_  = 0.0;
        epoch_loss_ref_sum_    = 0.0;
        std::fill(epoch_loss_phys_sum_.begin(), epoch_loss_phys_sum_.end(), 0.0);
        std::fill(epoch_loss_bcs_sum_.begin(), epoch_loss_bcs_sum_.end(), 0.0);
        std::fill(epoch_loss_lambda_sum_.begin(), epoch_loss_lambda_sum_.end(), 0.0);
        epoch_loss_count_      = 0;

        for (std::size_t b = 0; b < n_batches; ++b) {
            const TrainStepResult res = TrainStep();
            epoch_loss_sum_       += res.loss_raw;
            epoch_loss_total_sum_ += res.loss_total;
            epoch_loss_ref_sum_   += res.loss_ref;
            
            if (epoch_loss_phys_sum_.size() < res.loss_phys.size()) epoch_loss_phys_sum_.resize(res.loss_phys.size(), 0.0);
            for (std::size_t k = 0; k < res.loss_phys.size(); ++k) epoch_loss_phys_sum_[k] += res.loss_phys[k];

            if (epoch_loss_bcs_sum_.size() < res.loss_bcs.size()) epoch_loss_bcs_sum_.resize(res.loss_bcs.size(), 0.0);
            for (std::size_t k = 0; k < res.loss_bcs.size(); ++k) epoch_loss_bcs_sum_[k] += res.loss_bcs[k];

            if (epoch_loss_lambda_sum_.size() < res.lambdas.size()) epoch_loss_lambda_sum_.resize(res.lambdas.size(), 0.0);
            for (std::size_t k = 0; k < res.lambdas.size(); ++k) epoch_loss_lambda_sum_[k] += res.lambdas[k];

            ++epoch_loss_count_;
        }

        EndEpoch();
        EndPhysicsEpoch();
    }
    // ------------------------------------------------------------------------
    // Full training
    // ------------------------------------------------------------------------

    void Train() {
        if (!built_) {
            Build();
        }

        double previous_loss = std::numeric_limits<double>::quiet_NaN();

        for (std::size_t epoch = 0; epoch < cfg_.max_epochs; ++epoch) {
            TrainEpoch();

            const double current_loss =
                (epoch_loss_count_ > 0)
                    ? (cfg_.conv_metric == ConvergenceMetric::Weighted
                           ? epoch_loss_total_sum_ / static_cast<double>(epoch_loss_count_)
                           : epoch_loss_sum_ / static_cast<double>(epoch_loss_count_))
                    : last_result_.loss_raw;

            if (cfg_.verbose) {
                const char* metric_str =
                    cfg_.conv_metric == ConvergenceMetric::Weighted
                        ? "weighted" : "raw";

                std::cout
                    << "Epoch " << (epoch + 1) << "/" << cfg_.max_epochs
                    << " | " << metric_str << "_loss=" << current_loss
                    << " | loss_total=" << last_result_.loss_total
                    << " | loss_raw=" << last_result_.loss_raw
                    << "\n";
            }

            if (std::isfinite(previous_loss)) {
                const double abs_delta =
                    std::abs(current_loss - previous_loss);

                const double rel_delta =
                    abs_delta / std::max(1.0, std::abs(previous_loss));

                if (abs_delta <= cfg_.conv_tol_abs ||
                    rel_delta <= cfg_.conv_tol_rel) {
                    if (cfg_.verbose) {
                        std::cout << "Converged at epoch " << (epoch + 1) << "\n";
                    }
                    break;
                }
            }

            previous_loss = current_loss;
        }
    }

    // ------------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------------

    const TrainStepResult& GetLastResult() const noexcept {
        return last_result_;
    }

    double GetEpochAverageLoss() const noexcept {
        if (epoch_loss_count_ > 0) {
            return epoch_loss_sum_ / static_cast<double>(epoch_loss_count_);
        }
        return last_result_.loss_raw;
    }

    std::size_t GetStep() const noexcept {
        return step_;
    }

    double GetEpochAverageLossRef() const noexcept {
        if (epoch_loss_count_ > 0) return epoch_loss_ref_sum_ / static_cast<double>(epoch_loss_count_);
        return last_result_.loss_ref;
    }

    double GetEpochAverageLossPhys(std::size_t k = 0) const noexcept {
        if (epoch_loss_count_ > 0 && k < epoch_loss_phys_sum_.size()) {
            return epoch_loss_phys_sum_[k] / static_cast<double>(epoch_loss_count_);
        }
        return (k < last_result_.loss_phys.size()) ? last_result_.loss_phys[k] : 0.0;
    }

    double GetEpochAverageLossBC(std::size_t k = 0) const noexcept {
        if (epoch_loss_count_ > 0 && k < epoch_loss_bcs_sum_.size()) {
            return epoch_loss_bcs_sum_[k] / static_cast<double>(epoch_loss_count_);
        }
        return (k < last_result_.loss_bcs.size()) ? last_result_.loss_bcs[k] : 0.0;
    }

    double GetEpochAverageLambda(std::size_t k = 0) const noexcept {
        if (epoch_loss_count_ > 0 && k < epoch_loss_lambda_sum_.size()) {
            return epoch_loss_lambda_sum_[k] / static_cast<double>(epoch_loss_count_);
        }
        return (k < last_result_.lambdas.size()) ? last_result_.lambdas[k] : 1.0;
    }

    double GetEpochAverageLossTotal() const noexcept {
        if (epoch_loss_count_ > 0) return epoch_loss_total_sum_ / static_cast<double>(epoch_loss_count_);
        return last_result_.loss_total;
    }

    // MPI Enable
    void EnableMPI() {
        use_mpi_ = true;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_);
    }


private:
    // --------------------------------------------------------------------
    // TrainStep() decomposition
    // --------------------------------------------------------------------

    // Transient per-TrainStep() state, grouped into one struct so the
    // helper methods below don't need long, error-prone parameter lists.
    struct TrainStepState {
        std::size_t n_w{0};
        std::size_t n_ref{0};
        std::size_t n_phys{0};
        bool have_ref{false};

        std::vector<mlpdouble> weights;
        std::vector<mlpdouble> L_ref_vec;
        std::vector<mlpdouble> L_phys;
        std::vector<std::size_t> n_points_seen;
        std::vector<mlpdouble> L_bcs;
        std::vector<std::size_t> n_bc_points_seen;
        std::vector<double> lambdas;
    };

    // Activate the AD tape and register the network's current weights as
    // tape inputs. Populates st.weights / st.n_w; must run before any other
    // TrainStep() helper (they all depend on st.n_w).
    void RegisterWeightsOnTape(typename mlpdouble::Tape& tape, TrainStepState& st) {
        tape.reset();
        tape.setActive();

        st.weights = net_.GetWeightsBiases();
        st.n_w = st.weights.size();

        if (st.n_w == 0) {
            tape.reset();
            throw std::runtime_error(
                "CMLPTrainer: network has no trainable parameters.");
        }

        for (auto& w : st.weights) {
            tape.registerInput(w);
        }
        net_.SetWeightsBiases(st.weights);
    }

    // Zero out all per-step gradient accumulation buffers and size the AD
    // scratch buffers later used for the Adam update.
    void InitializeSensitivities(TrainStepState& st) {
        grad_total_.assign(st.n_w, 0.0);
        if (grad_per_data_term_.size() < st.n_ref) grad_per_data_term_.resize(st.n_ref);
        for (std::size_t i = 0; i < st.n_ref; ++i) grad_per_data_term_[i].assign(st.n_w, 0.0);

        clean_weights_ad_.resize(st.n_w);
        g_total_ad_.resize(st.n_w);
    }

    // Evaluate all M reference/data terms on the current mini-batch
    // (batch_x_/batch_y_, already populated by NextReferenceBatch()).
    void EvaluateReferenceLoss(typename mlpdouble::Tape& tape, TrainStepState& st) {
        st.L_ref_vec.assign(st.n_ref, mlpdouble(0.0));
        if (!st.have_ref) return;

        if (batch_preds_.size() < batch_x_.size()) {
            batch_preds_.resize(batch_x_.size());
        }
        for (std::size_t i = 0; i < batch_x_.size(); ++i) {
            if (batch_preds_[i].outputs.size() != net_.GetnOutputs()) {
                batch_preds_[i].outputs.resize(net_.GetnOutputs());
            }
            net_.Predict(batch_x_[i], false, false);
            batch_preds_[i].inputs = batch_x_[i];
            for (std::size_t o = 0; o < net_.GetnOutputs(); ++o) {
                batch_preds_[i].outputs[o] = net_.GetOutput(o);
            }
        }
        batch_preds_.resize(batch_x_.size());

        for (std::size_t i = 0; i < st.n_ref; ++i) {
            st.L_ref_vec[i] = ref_losses_[i]->Evaluate(batch_preds_, batch_y_);
            tape.registerOutput(st.L_ref_vec[i]);
        }
    }

    // Stream through every active collocation set's current mini-batch,
    // accumulating raw (un-normalized) physics (Lr) and boundary (Li)
    // losses on the same active tape.
    void EvaluatePhysicsAndBoundaryLosses(TrainStepState& st) {
        st.L_phys.assign(st.n_phys, mlpdouble(0.0));
        st.n_points_seen.assign(st.n_phys, 0);
        st.L_bcs.assign(bcs_losses_.size(), mlpdouble(0.0));
        st.n_bc_points_seen.assign(bcs_losses_.size(), 0);

        typename mlpdouble::Tape& tape = mlpdouble::getTape();

        if (current_pred_.outputs.size() != net_.GetnOutputs()) {
            current_pred_.outputs.resize(net_.GetnOutputs());
        }

        for (const auto& set_name : active_set_order_) {
            const auto& points = coll_sets_.at(set_name);

            const bool need_jac  = set_needs_jac_.at(set_name);
            const bool need_hess = set_needs_hess_.at(set_name);
            const bool eval_jac  = need_jac || need_hess;
            const bool eval_hess = need_hess;

            NextPhysicsBatch(set_name); // Updates batch_indices_

            for (std::size_t idx : batch_indices_) {
                point_storage_.Fill(net_, points[idx], current_pred_, eval_jac, eval_hess);

                // Evaluate PDEs (L_r)
                auto phys_it = losses_by_set_.find(set_name);
                if (phys_it != losses_by_set_.end()) {
                    for (const std::size_t k : phys_it->second) {
                        const std::size_t n_phys_vars = phys_losses_[k]->NumPhysicsVariables();
                        const auto& data = phys_data_[k];

                        if (n_phys_vars > 0 && data.empty()) {
                            tape.reset();
                            throw std::runtime_error("CMLPTrainer: missing physics data.");
                        }

                        if (data.empty()) {
                            st.L_phys[k] += phys_losses_[k]->EvaluateOne(current_pred_);
                        } else {
                            st.L_phys[k] += phys_losses_[k]->EvaluateOne(current_pred_, data[idx]);
                        }
                        ++st.n_points_seen[k];
                    }
                }

                // Evaluate BCs (L_i) on the same active tape!
                auto bcs_it = bcs_losses_by_set_.find(set_name);
                if (bcs_it != bcs_losses_by_set_.end()) {
                    for (const std::size_t k : bcs_it->second) {
                        st.L_bcs[k] += bcs_losses_[k]->EvaluateOne(current_pred_);
                        ++st.n_bc_points_seen[k];
                    }
                }
            }
        }
    }

    // Normalize accumulated physics/boundary losses by point (and, for
    // physics losses, equation) count, register them as tape outputs, and
    // deactivate the tape. Must run after EvaluatePhysicsAndBoundaryLosses()
    // and before any reverse sweep.
    void NormalizeAndRegisterLosses(typename mlpdouble::Tape& tape, TrainStepState& st) {
        for (std::size_t k = 0; k < st.n_phys; ++k) {
            if (st.n_points_seen[k] == 0) {
                st.L_phys[k] = mlpdouble(0.0);
                tape.registerOutput(st.L_phys[k]);
                continue;
            }
            const double denom = static_cast<double>(st.n_points_seen[k]) *
                                  static_cast<double>(phys_losses_[k]->NumEquations());
            if (denom <= 0.0) throw std::runtime_error("CMLPTrainer: invalid normalization.");
            st.L_phys[k] = st.L_phys[k] / mlpdouble(denom);
            tape.registerOutput(st.L_phys[k]);
        }

        for (std::size_t k = 0; k < bcs_losses_.size(); ++k) {
            if (st.n_bc_points_seen[k] == 0) {
                st.L_bcs[k] = mlpdouble(0.0);
                tape.registerOutput(st.L_bcs[k]);
                continue;
            }
            st.L_bcs[k] = st.L_bcs[k] / mlpdouble(st.n_bc_points_seen[k]);
            tape.registerOutput(st.L_bcs[k]);
        }

        tape.setPassive();
    }

    // Single combined reverse sweep, using the annealer's cached lambda
    // weights (or 1.0 if no annealer). Used on every step except periodic
    // annealer-update steps.
    void ComputeFastPathGradient(typename mlpdouble::Tape& tape, TrainStepState& st) {
        auto zero_gradients = [&st]() {
            for (auto& w : st.weights) w.setGradient(0.0);
        };
        auto read_gradients = [&](std::vector<double>& g) {
            for (std::size_t i = 0; i < st.n_w; ++i) {
                g[i] = to_double(st.weights[i].getGradient());
            }
        };

        zero_gradients();
        tape.clearAdjoints();

        for (std::size_t i = 0; i < st.n_ref; ++i) st.L_ref_vec[i].setGradient(st.lambdas[i]);
        for (std::size_t k = 0; k < st.L_bcs.size(); ++k) st.L_bcs[k].setGradient(st.lambdas[st.n_ref + k]);
        for (std::size_t k = 0; k < st.n_phys; ++k) st.L_phys[k].setGradient(1.0);

        tape.evaluate();
        read_gradients(grad_total_);

        if (use_mpi_ && mpi_size_ > 1) {
            MPI_Allreduce(MPI_IN_PLACE, grad_total_.data(), st.n_w, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            for (double& g : grad_total_) g /= mpi_size_;
        }
    }

    // (n_ref + n_bcs + 1)-sweep gradient computation that also refreshes the
    // annealer's per-term lambda estimates (Paper Algorithm 1): one combined
    // sweep for the aggregate residual (Lr) — only the aggregate is ever
    // consumed, so a single combined sweep is correct and sufficient there —
    // plus one sweep per individual data-like term (Li), since each needs
    // its OWN gradient for GradStats, not a shared combined one.
    void ComputeAnnealedPathGradient(typename mlpdouble::Tape& tape, TrainStepState& st) {
        auto zero_gradients = [&st]() {
            for (auto& w : st.weights) w.setGradient(0.0);
        };
        auto read_gradients = [&](std::vector<double>& g) {
            for (std::size_t i = 0; i < st.n_w; ++i) {
                g[i] = to_double(st.weights[i].getGradient());
            }
        };

        // Combined residual gradient (Lr)
        zero_gradients();
        tape.clearAdjoints();
        for (std::size_t i = 0; i < st.n_ref; ++i) st.L_ref_vec[i].setGradient(0.0);
        for (auto& lb : st.L_bcs) lb.setGradient(0.0);
        for (std::size_t k = 0; k < st.n_phys; ++k) st.L_phys[k].setGradient(1.0);
        tape.evaluate();
        read_gradients(grad_total_);

        // One sweep per individual data-like term
        grad_per_data_term_.resize(st.n_ref + st.L_bcs.size());
        for (std::size_t i = 0; i < st.n_ref + st.L_bcs.size(); ++i) {
            if (grad_per_data_term_[i].size() < st.n_w) grad_per_data_term_[i].assign(st.n_w, 0.0);
        }
        for (std::size_t i = 0; i < st.n_ref; ++i) {
            zero_gradients();
            tape.clearAdjoints();
            for (std::size_t j = 0; j < st.n_ref; ++j) st.L_ref_vec[j].setGradient(j == i ? 1.0 : 0.0);
            for (auto& lb : st.L_bcs) lb.setGradient(0.0);
            for (auto& lp : st.L_phys) lp.setGradient(0.0);
            tape.evaluate();
            read_gradients(grad_per_data_term_[i]);
        }
        for (std::size_t k = 0; k < st.L_bcs.size(); ++k) {
            zero_gradients();
            tape.clearAdjoints();
            for (auto& lr : st.L_ref_vec) lr.setGradient(0.0);
            for (std::size_t j = 0; j < st.L_bcs.size(); ++j) st.L_bcs[j].setGradient(j == k ? 1.0 : 0.0);
            for (auto& lp : st.L_phys) lp.setGradient(0.0);
            tape.evaluate();
            read_gradients(grad_per_data_term_[st.n_ref + k]);
        }

        if (use_mpi_ && mpi_size_ > 1) {
            MPI_Allreduce(MPI_IN_PLACE, grad_total_.data(), st.n_w, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            for (double& g : grad_total_) g /= mpi_size_;
            for (std::size_t i = 0; i < st.n_ref + st.L_bcs.size(); ++i) {
                MPI_Allreduce(MPI_IN_PLACE, grad_per_data_term_[i].data(), st.n_w, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                for (double& g : grad_per_data_term_[i]) g /= mpi_size_;
            }
        }

        const GradStats base_stats = GradStats::from_grads(grad_total_);
        std::vector<GradStats> data_stats_vec(st.n_ref + st.L_bcs.size());
        for (std::size_t i = 0; i < data_stats_vec.size(); ++i) {
            data_stats_vec[i] = GradStats::from_grads(grad_per_data_term_[i]);
        }
        annealer_->update(base_stats, data_stats_vec);

        for (std::size_t i = 0; i < st.n_ref + st.L_bcs.size(); ++i) {
            st.lambdas[i] = annealer_->get_lambda(i);
        }

        // Fold the weighted data-term gradients into grad_total_, which
        // already holds the unweighted residual aggregate. The ref loop is
        // naturally a no-op when n_ref==0 (pure PINN) — must NOT be gated
        // behind st.have_ref, or the BC contributions get silently dropped.
        for (std::size_t i = 0; i < st.n_w; ++i) {
            for (std::size_t r = 0; r < st.n_ref; ++r) grad_total_[i] += st.lambdas[r] * grad_per_data_term_[r][i];
            for (std::size_t k = 0; k < st.L_bcs.size(); ++k) {
                grad_total_[i] += st.lambdas[st.n_ref + k] * grad_per_data_term_[st.n_ref + k][i];
            }
        }
    }

    void ApplyGradientClipping(std::size_t n_w) {
        if (cfg_.max_grad_norm <= 0.0) return;
        double norm_sq = 0.0;
        for (std::size_t i = 0; i < n_w; ++i) norm_sq += grad_total_[i] * grad_total_[i];
        const double norm = std::sqrt(norm_sq);
        if (norm > cfg_.max_grad_norm) {
            const double scale = cfg_.max_grad_norm / norm;
            for (std::size_t i = 0; i < n_w; ++i) grad_total_[i] *= scale;
        }
    }

    void ApplyAdamUpdate(TrainStepState& st) {
        for (std::size_t i = 0; i < st.n_w; ++i) {
            clean_weights_ad_[i] = mlpdouble(to_double(st.weights[i]));
            g_total_ad_[i]       = mlpdouble(grad_total_[i]);
        }
        adam_.step(clean_weights_ad_, g_total_ad_);
        net_.SetWeightsBiases(clean_weights_ad_);
    }

    TrainStepResult BuildTrainStepResult(const TrainStepState& st) const {
        TrainStepResult result;
        result.lambdas = st.lambdas;
        result.loss_phys.resize(st.n_phys);
        result.loss_bcs.resize(st.L_bcs.size());

        double total_ref_loss = 0.0;
        for (std::size_t i = 0; i < st.n_ref; ++i) total_ref_loss += to_double(st.L_ref_vec[i]);
        result.loss_ref = st.have_ref ? total_ref_loss : 0.0;

        for (std::size_t k = 0; k < st.n_phys; ++k) {
            result.loss_phys[k] = to_double(st.L_phys[k]);
        }

        result.loss_raw = result.loss_ref;
        result.loss_total = 0.0;
        for (std::size_t k = 0; k < st.n_phys; ++k) {
            result.loss_raw   += result.loss_phys[k];
            result.loss_total += result.loss_phys[k];
        }

        if (st.have_ref) {
            for (std::size_t r = 0; r < st.n_ref; ++r) {
                result.loss_total += st.lambdas[r] * to_double(st.L_ref_vec[r]);
            }
        }
        for (std::size_t k = 0; k < st.L_bcs.size(); ++k) {
            result.loss_bcs[k]  = to_double(st.L_bcs[k]);
            result.loss_raw    += result.loss_bcs[k];
            result.loss_total  += st.lambdas[st.n_ref + k] * result.loss_bcs[k];
        }

        return result;
    }

    // --------------------------------------------------------------------
    // General helpers
    // --------------------------------------------------------------------

    static bool HasSameSignature(const std::vector<std::string>& a,
                                 const std::vector<std::string>& b)
    {
        return a == b;
    }

    static bool HasUniqueNames(const std::vector<std::string>& names)
    {
        std::unordered_set<std::string> seen;
        for (const auto& n : names) {
            if (seen.find(n) != seen.end()) {
                return false;
            }
            seen.insert(n);
        }
        return true;
    }

    std::size_t FindPhysicsLossIndex(const std::string& name) const {
        for (std::size_t i = 0; i < phys_losses_.size(); ++i) {
            if (phys_losses_[i]->GetName() == name) {
                return i;
            }
        }
        throw std::runtime_error(
            "CMLPTrainer: no physics loss named '" + name + "'.");
    }

    void PrepareEpoch() {
        if (cfg_.shuffle_per_epoch && total_samples_ > 0) {
            std::shuffle(indices_.begin(), indices_.end(), rng_);
        }
        batch_cursor_ = 0;
        epoch_started_ = true;
    }

    void EndEpoch() {
        batch_cursor_ = 0;
        epoch_started_ = false;
    }

    void PreparePhysicsEpoch() {
        for (auto& pair : coll_epoch_started_) {
            pair.second = false;
        }
    }

    void EndPhysicsEpoch() {
        for (auto& pair : coll_epoch_started_) {
            pair.second = false;
        }
        for (auto& pair : coll_batch_cursor_) {
            pair.second = 0;
        }
    }

    void NextPhysicsBatch(const std::string& set_name) {
        batch_indices_.clear();
        auto it = coll_sets_.find(set_name);
        if (it == coll_sets_.end() || it->second.empty()) return;

        const std::size_t n_points = it->second.size();
        const std::size_t phys_bs = cfg_.physics_batch_size > 0 ? 
                                    std::min(cfg_.physics_batch_size, n_points) : n_points;

        if (!coll_epoch_started_.at(set_name)) {
            if (cfg_.shuffle_per_epoch && n_points > 0) {
                std::shuffle(coll_indices_[set_name].begin(), coll_indices_[set_name].end(), rng_);
            }
            coll_batch_cursor_[set_name] = 0;
            coll_epoch_started_[set_name] = true;
        }

        const std::size_t remaining = n_points - coll_batch_cursor_[set_name];
        const std::size_t actual = std::min(phys_bs, remaining);

        batch_indices_.resize(actual);
        for (std::size_t i = 0; i < actual; ++i) {
            batch_indices_[i] = coll_indices_[set_name][coll_batch_cursor_[set_name]++];
        }

        if (coll_batch_cursor_[set_name] >= n_points) {
            coll_epoch_started_[set_name] = false; 
        }
    }

    void NextReferenceBatch() {
        batch_x_.clear();
        batch_y_.clear();

        if (total_samples_ == 0) return;
        if (!epoch_started_) PrepareEpoch();

        const std::size_t batch_size =
            cfg_.batch_size > 0 ? std::min(cfg_.batch_size, total_samples_) : total_samples_;

        // Wrap around and reshuffle if exhausted mid-epoch, mirroring
        // NextPhysicsBatch(). Required for correct MPI operation: when
        // n_batches is synchronized across ranks (see TrainEpoch()), a rank
        // with fewer local reference samples than another rank must still
        // be able to supply `n_batches` batches without throwing.
        if (batch_cursor_ >= total_samples_) {
            if (cfg_.shuffle_per_epoch) {
                std::shuffle(indices_.begin(), indices_.end(), rng_);
            }
            batch_cursor_ = 0;
        }

        const std::size_t remaining = total_samples_ - batch_cursor_;
        const std::size_t actual = std::min(batch_size, remaining);

        batch_x_.resize(actual);
        batch_y_.resize(actual);
        for (std::size_t i = 0; i < actual; ++i) {
            const std::size_t idx = indices_[batch_cursor_++];
            batch_x_[i] = train_inputs_[idx];
            batch_y_[i] = train_targets_[idx];
        }
    }

    void LogStep(const TrainStepResult& r) const {
        std::cout
            << "Step " << step_
            << " | L_ref=" << r.loss_ref
            << " | L_raw=" << r.loss_raw
            << " | L_total=" << r.loss_total;

        for (std::size_t k = 0; k < r.loss_phys.size(); ++k) {
            std::cout
                << " | phys[" << k << "]=" << r.loss_phys[k];
        }
        for (std::size_t i = 0; i < r.lambdas.size(); ++i) {
            std::cout << " (lambda_ref[" << i << "]=" << r.lambdas[i] << ")";
        }

        std::cout << "\n";
    }

private:
    // ------------------------------------------------------------------------
    // Members
    // ------------------------------------------------------------------------
    CNeuralNetwork& net_;
    CAdam adam_;
    AnnealerConfig annealer_cfg_;
    std::unique_ptr<CGradientAnnealer> annealer_;
    TrainerConfig cfg_;
    bool built_{false};

    std::vector<std::shared_ptr<CBaseLoss>> ref_losses_;

    std::vector<std::shared_ptr<CPhysicsLoss>> phys_losses_;
    std::vector<std::string> phys_set_name_;
    std::vector<std::vector<std::vector<mlpdouble>>> phys_data_;

    std::vector<std::shared_ptr<CPhysicsLoss>> bcs_losses_;
    std::vector<std::string> bcs_set_name_;
    std::unordered_map<std::string, std::vector<std::size_t>> bcs_losses_by_set_;

    std::unordered_map<std::string, std::vector<std::vector<mlpdouble>>> coll_sets_;
    std::vector<std::string> set_order_;
    std::unordered_set<std::string> empty_sets_;

    std::unordered_map<std::string, std::vector<std::size_t>> coll_indices_;
    std::unordered_map<std::string, std::size_t> coll_batch_cursor_;
    std::unordered_map<std::string, bool> coll_epoch_started_;

    std::unordered_map<std::string, std::vector<std::size_t>> losses_by_set_;
    std::unordered_map<std::string, bool> set_needs_jac_;
    std::unordered_map<std::string, bool> set_needs_hess_;
    std::vector<std::string> active_set_order_;

    CPointDerivatives point_storage_;

    std::vector<std::vector<mlpdouble>> train_inputs_;
    std::vector<std::vector<mlpdouble>> train_targets_;
    std::size_t total_samples_{0};
    std::vector<std::size_t> indices_;
    std::size_t batch_cursor_{0};
    bool epoch_started_{false};

    std::mt19937 rng_;

    std::size_t step_{0};
    TrainStepResult last_result_;
    double epoch_loss_sum_{0.0};
    double epoch_loss_total_sum_{0.0};
    std::size_t epoch_loss_count_{0};
    
    double epoch_loss_ref_sum_{0.0};
    std::vector<double> epoch_loss_phys_sum_;
    std::vector<double> epoch_loss_bcs_sum_;
    std::vector<double> epoch_loss_lambda_sum_;

    // Pre-allocated buffers to prevent heap allocations inside TrainStep
    std::vector<std::vector<mlpdouble>> batch_x_;
    std::vector<std::vector<mlpdouble>> batch_y_;
    std::vector<std::size_t> batch_indices_;
    std::vector<PredictionResult> batch_preds_;
    PredictionResult current_pred_;
    std::vector<double> grad_total_;
    std::vector<std::vector<double>> grad_per_data_term_;
    std::vector<mlpdouble> clean_weights_ad_;
    std::vector<mlpdouble> g_total_ad_;

    // MPI
    bool use_mpi_{false};
    int mpi_rank_{0};
    int mpi_size_{1};
};

} // namespace MLPToolbox