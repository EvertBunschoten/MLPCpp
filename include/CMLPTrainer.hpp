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
 *   - CPhysicsLoss::EvaluateSingleSample() returns a raw per-point loss
 *   - CPhysicsLoss::Evaluate() performs canonical normalization
 *
 * Physics training mode:
 *   For each collocation point (or mini-batch of points):
 *     Predict(point, jac, hess)
 *     copy derivatives into reusable storage
 *     EvaluateSingleSample() for every physics loss on that set
 *
 * Data-fitting mode:
 *   Standard mini-batch supervised learning on registered fitting
 *   losses. Supports multiple fitting terms (M > 1) for generic
 *   annealing.
 *
 * The trainer keeps one point worth of derivative data in memory at a
 * time.
 *
 * Derivative accessor convention (VERIFIED):
 *   CNeuralNetwork stores derivatives in input-major layout internally:
 *     output_Jacobian[iInput][iOutput]
 *     output_Hessian[iInput][jInput][iOutput]
 *
 *   The public accessors present arguments in output-first order:
 *     GetJacobian(iOutput, iInput)  -> output_Jacobian[iInput][iOutput]
 *     GetHessian(iOutput, iInput, jInput) ->
 * Output_Hessian[iInput][jInput][iOutput]
 *
 *   CPointDerivatives::Fill() calls these with (o, i) / (o, i, j) and
 *   stores results in input-major buffers.
 *
 * Physics data lifecycle:
 *   Physics data may be provided before or after
 *   FinalizeConfiguration(), but training fails at the first TrainStep()
 *   if a physics loss declares physics variables
 *   (NumPhysicsVariables() > 0) and no data has been set.
 *   FinalizeConfiguration() validates structure only; the runtime
 *   evaluation is the enforcement point for missing data.
 *
 * Convergence:
 *   Configurable via ConvergenceMetric enum.  Default is Weighted, which
 *   checks loss_total (L_phys + sum lambda_k * L_fit_k).  When the
 *   annealer is off, loss_total still differs from loss_raw if any
 *   equation weight is not 1.0.  A NaN or infinite loss aborts training
 *   with an error.
 *
 * Empty collocation sets:
 *   Allowed.  A set with zero points contributes zero loss and zero
 *   gradients.  The set is skipped during streaming evaluation.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
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

#include "CAdam.hpp"
#include "CBaseLoss.hpp"
#include "CGradientAnnealer.hpp"
#include "CNeuralNetwork.hpp"
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
//   PDE it must satisfy — supervised fitting data (AddFittingLoss()) or
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
  Raw,     // Unweighted: L_fit + sum L_phys
  Weighted // Annealed:   L_phys + sum lambda_k * L_fit_k
};

struct TrainerConfig {
  std::size_t max_epochs{1000};
  std::size_t batch_size{32};
  std::size_t physics_batch_size{0}; // 0 = full physics batch (all points)
  double conv_tol_abs{1e-8};
  double conv_tol_rel{1e-6};
  ConvergenceMetric conv_metric{ConvergenceMetric::Weighted};
  double max_grad_norm{0.0}; // 0 = no clipping
  bool use_annealer{true};
  bool verbose{false};
  std::size_t log_every{100}; // 0 = disabled
  bool shuffle_per_epoch{true};
  std::size_t annealer_update_freq{10};
};

struct TrainStepResult {
  double loss_ref{0.0};
  std::vector<double> loss_phys;
  std::vector<double> loss_bcs;
  double loss_total{0.0};
  double loss_raw{0.0};
  std::vector<double> lambdas;
};


class CPointDerivatives {
public:
  CPointDerivatives(std::size_t n_in, std::size_t n_out)
      : n_in_(n_in), n_out_(n_out), jac_flat_(n_in * n_out),
        jac_rows_(n_in, nullptr), hess_flat_(n_in * n_in * n_out),
        hess_rows_(n_in * n_in, nullptr), hess_planes_(n_in, nullptr) {
    if (n_in_ == 0 || n_out_ == 0) {
      throw std::invalid_argument(
          "CPointDerivatives: network dimensions must be positive.");
    }

    for (auto i = 0; i < n_in_; ++i) {
      jac_rows_[i] = &jac_flat_[i * n_out_];
    }

    for (auto i = 0; i < n_in_; ++i) {
      hess_planes_[i] = &hess_rows_[i * n_in_];
      for (auto j = 0; j < n_in_; ++j) {
        hess_rows_[i * n_in_ + j] = &hess_flat_[(i * n_in_ + j) * n_out_];
      }
    }
  }

  void Fill(CNeuralNetwork &net, const std::vector<mlpdouble> &x,
            PredictionResult &pr, bool eval_jac, bool eval_hess) {
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

    for (auto o = 0; o < n_out_; ++o) {
      pr.outputs[o] = net.GetOutput(o);
    }

    pr.jacobian = nullptr;
    pr.hessian = nullptr;

    if (compute_jac) {
      for (auto i = 0; i < n_in_; ++i) {
        for (auto o = 0; o < n_out_; ++o) {
          jac_flat_[i * n_out_ + o] = net.GetJacobian(o, i);
        }
      }
      pr.jacobian = jac_rows_.data();
    }

    if (eval_hess) {
      for (auto i = 0; i < n_in_; ++i) {
        for (auto j = 0; j < n_in_; ++j) {
          for (auto o = 0; o < n_out_; ++o) {
            hess_flat_[(i * n_in_ + j) * n_out_ + o] = net.GetHessian(o, i, j);
          }
        }
      }
      pr.hessian = hess_planes_.data();
    }
  }

private:
  std::size_t n_in_;
  std::size_t n_out_;

  std::vector<mlpdouble> jac_flat_;
  std::vector<mlpdouble *> jac_rows_;

  std::vector<mlpdouble> hess_flat_;
  std::vector<mlpdouble *> hess_rows_;
  std::vector<mlpdouble **> hess_planes_;
};


class CMLPTrainer {
public:
  CMLPTrainer(CNeuralNetwork &net, CAdam adam, AnnealerConfig annealer_cfg,
              TrainerConfig cfg = {})
      : net_(net), adam_(std::move(adam)), annealer_cfg_(annealer_cfg),
        cfg_(cfg), point_storage_(net.GetnInputs(), net.GetnOutputs()),
        rng_(std::random_device{}()) {
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

    adam_.Initialize(n_w);
  }

  void AddFittingLoss(std::shared_ptr<CBaseLoss> loss) {
    if (!loss) {
      throw std::invalid_argument("CMLPTrainer: fitting loss cannot be null.");
    }
    if (finalized_) {
      throw std::runtime_error("CMLPTrainer: cannot add fitting loss after "
                               "FinalizeConfiguration().");
    }
    fitting_losses_.push_back(std::move(loss));
  }

  void ClearFittingLosses() {
    if (finalized_) {
      throw std::runtime_error("CMLPTrainer: cannot clear fitting losses after "
                               "FinalizeConfiguration().");
    }
    fitting_losses_.clear();
  }

  void AddBoundaryLoss(std::shared_ptr<CPhysicsLoss> loss,
                       const std::string &collocation_set_name) {
    if (!loss) {
      throw std::invalid_argument("CMLPTrainer: boundary loss cannot be null.");
    }
    if (finalized_) {
      throw std::runtime_error("CMLPTrainer: cannot add boundary loss after "
                               "FinalizeConfiguration().");
    }
    if (coll_sets_.find(collocation_set_name) == coll_sets_.end()) {
      throw std::invalid_argument("CMLPTrainer: collocation set '" +
                                  collocation_set_name + "' does not exist.");
    }

    bcs_losses_.push_back(std::move(loss));
    bcs_set_name_.push_back(collocation_set_name);
  }

  void SetTrainingData(const std::vector<std::vector<mlpdouble>> &inputs,
                       const std::vector<std::vector<mlpdouble>> &targets) {
    if (finalized_) {
      throw std::runtime_error("CMLPTrainer: cannot modify training data after "
                               "FinalizeConfiguration().");
    }

    if (inputs.size() != targets.size()) {
      throw std::invalid_argument(
          "CMLPTrainer: input/target sample count mismatch.");
    }

    const std::size_t n_in = net_.GetnInputs();
    const std::size_t n_out = net_.GetnOutputs();

    for (auto i = 0; i < inputs.size(); ++i) {
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

  void SetCollocationPoints(const std::string &set_name,
                            const std::vector<std::vector<mlpdouble>> &points) {
    if (finalized_) {
      throw std::runtime_error(
          "CMLPTrainer: cannot modify collocation sets after "
          "FinalizeConfiguration().");
    }

    if (set_name.empty()) {
      throw std::invalid_argument(
          "CMLPTrainer: collocation set name cannot be empty.");
    }

    const std::size_t n_in = net_.GetnInputs();
    for (auto p = 0; p < points.size(); ++p) {
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
    }

    coll_indices_[set_name].resize(points.size());
    std::iota(coll_indices_[set_name].begin(), coll_indices_[set_name].end(),
              std::size_t{0});
    coll_batch_cursor_[set_name] = 0;
    coll_epoch_started_[set_name] = false;
  }

  void SetPhysicsCollocationPoints(
      const std::vector<std::vector<mlpdouble>> &points) {
    SetCollocationPoints("default", points);
  }


  void SetPhysicsData(const std::string &loss_name,
                      const std::vector<std::vector<mlpdouble>> &data) {
    const std::size_t idx = FindPhysicsLossIndex(loss_name);
    const std::string &set_name = phys_set_name_[idx];

    auto set_it = coll_sets_.find(set_name);
    if (set_it == coll_sets_.end()) {
      throw std::runtime_error("CMLPTrainer: collocation set '" + set_name +
                               "' not found.");
    }

    const std::size_t n_points = set_it->second.size();
    const std::size_t n_phys = phys_losses_[idx]->NumPhysicsVariables();

    if (n_phys == 0 && !data.empty()) {
      throw std::invalid_argument(
          "CMLPTrainer: loss '" + loss_name +
          "' has no physics variables but data was provided.");
    }

    if (!data.empty()) {
      if (data.size() != n_points) {
        throw std::invalid_argument(
            "CMLPTrainer: physics data row count mismatch for loss '" +
            loss_name + "'. Expected " + std::to_string(n_points) + ", got " +
            std::to_string(data.size()) + ".");
      }

      for (auto p = 0; p < data.size(); ++p) {
        if (data[p].size() != n_phys) {
          throw std::invalid_argument(
              "CMLPTrainer: physics data dimension mismatch for loss '" +
              loss_name + "' at point " + std::to_string(p) + ". Expected " +
              std::to_string(n_phys) + ", got " +
              std::to_string(data[p].size()) + ".");
        }
      }
    }

    phys_data_[idx] = data;
  }

  std::vector<std::vector<mlpdouble>>
  MakeZeroPhysicsData(const std::string &loss_name) const {
    const std::size_t idx = FindPhysicsLossIndex(loss_name);
    const std::string &set_name = phys_set_name_[idx];
    const std::size_t n_points = coll_sets_.at(set_name).size();
    const std::size_t n_phys = phys_losses_[idx]->NumPhysicsVariables();

    if (n_phys == 0) {
      return {};
    }

    return std::vector<std::vector<mlpdouble>>(
        n_points, std::vector<mlpdouble>(n_phys, mlpdouble(0.0)));
  }

  void AddPhysicsLoss(std::shared_ptr<CPhysicsLoss> loss,
                      const std::string &collocation_set_name,
                      std::vector<std::vector<mlpdouble>> per_point_data = {}) {
    if (finalized_) {
      throw std::runtime_error("CMLPTrainer: cannot add physics loss after "
                               "FinalizeConfiguration().");
    }

    if (!loss) {
      throw std::invalid_argument("CMLPTrainer: physics loss cannot be null.");
    }

    if (collocation_set_name.empty()) {
      throw std::invalid_argument(
          "CMLPTrainer: collocation set name cannot be empty.");
    }

    if (coll_sets_.find(collocation_set_name) == coll_sets_.end()) {
      throw std::invalid_argument("CMLPTrainer: collocation set '" +
                                  collocation_set_name + "' does not exist.");
    }

    if (!HasSameSignature(net_.GetInputVars(), loss->GetNetworkInputNames())) {
      throw std::invalid_argument(
          "CMLPTrainer: physics loss '" + loss->GetName() +
          "' input names do not match the network input names.");
    }

    if (!HasSameSignature(net_.GetOutputVars(),
                          loss->GetNetworkOutputNames())) {
      throw std::invalid_argument(
          "CMLPTrainer: physics loss '" + loss->GetName() +
          "' output names do not match the network output names.");
    }

    for (const auto &existing : phys_losses_) {
      if (existing->GetName() == loss->GetName()) {
        throw std::invalid_argument(
            "CMLPTrainer: duplicate physics loss name '" + loss->GetName() +
            "'.");
      }
    }

    const std::size_t n_phys = loss->NumPhysicsVariables();

    if (n_phys == 0 && !per_point_data.empty()) {
      throw std::invalid_argument(
          "CMLPTrainer: loss '" + loss->GetName() +
          "' has no physics variables but data was provided.");
    }

    if (!per_point_data.empty()) {
      const auto &points = coll_sets_.at(collocation_set_name);

      if (per_point_data.size() != points.size()) {
        throw std::invalid_argument(
            "CMLPTrainer: physics data row count mismatch for loss '" +
            loss->GetName() + "'.");
      }

      for (auto p = 0; p < per_point_data.size(); ++p) {
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


  void SetHistoryFile(const std::string &filename) {
    history_filename_ = filename;
  }

  void FinalizeConfiguration() {
    if (finalized_) {
      return;
    }

    ValidateNetworkVariableNames();
    ValidateActiveObjective();
    RegisterPhysicsLosses();
    BuildActiveSetOrder();
    RegisterBoundaryLosses();
    CreateAnnealer();

    finalized_ = true;
  }


  TrainStepResult TrainStep() {
    if (!finalized_) {
      FinalizeConfiguration();
    }

    using Tape = typename mlpdouble::Tape;
    Tape &tape = mlpdouble::getTape();

    TrainStepState st;
    st.n_phys = phys_losses_.size();
    st.n_ref = fitting_losses_.size();
    st.have_ref = (st.n_ref > 0) && (total_samples_ > 0);

    if (st.have_ref) {
      NextDataFittingBatch();
    }

    RegisterWeightsOnTape(tape, st);
    InitializeSensitivities(st);

    EvaluateFittingLoss(tape, st);
    EvaluatePhysicsAndBoundaryLosses(st);
    NormalizeAndRegisterLosses(tape, st);

    InitializeStepLambdas(st);
    ComputeStepGradients(tape, st);

    tape.reset();

    ApplyGradientClipping(st.n_w);
    ApplyAdamUpdate(st);

    TrainStepResult result = AssembleStepResult(st);
    last_result_ = result;
    ++step_;

    if (cfg_.verbose && cfg_.log_every > 0 && (step_ % cfg_.log_every == 0)) {
      LogStep(result);
    }

    return result;
  }


  void TrainEpoch() {
    if (!finalized_) {
      FinalizeConfiguration();
    }

    if (!EpochHasWork()) {
      return;
    }

    PrepareEpoch();
    PreparePhysicsEpoch();

    const std::size_t n_batches = ComputeEpochBatchCount();
    ResetEpochStatistics();

    for (auto b = 0; b < n_batches; ++b) {
      AccumulateEpochStatistics(TrainStep());
    }

    EndEpoch();
    EndPhysicsEpoch();
  }

  void Train() {
    if (!finalized_) {
      FinalizeConfiguration();
    }

    std::ofstream history_file = OpenHistoryFile();

    double previous_loss = std::numeric_limits<double>::quiet_NaN();

    for (auto epoch = 0; epoch < cfg_.max_epochs; ++epoch) {
      TrainEpoch();

      const double current_loss = ComputeConvergenceLoss();

      WriteHistoryRow(history_file, epoch);

      if (!std::isfinite(current_loss)) {
        throw std::runtime_error(
            "CMLPTrainer: training diverged — the loss became NaN or "
            "infinite at epoch " +
            std::to_string(epoch + 1) + ".");
      }

      if (cfg_.verbose) {
        LogEpoch(epoch, current_loss);
      }

      if (HasConverged(current_loss, previous_loss)) {
        if (cfg_.verbose) {
          std::cout << "Converged at epoch " << (epoch + 1) << "\n";
        }
        break;
      }

      previous_loss = current_loss;
    }
  }


  const TrainStepResult &GetLastResult() const noexcept { return last_result_; }

  double GetEpochAverageLoss() const noexcept {
    if (epoch_loss_count_ > 0) {
      return epoch_loss_sum_ / static_cast<double>(epoch_loss_count_);
    }
    return last_result_.loss_raw;
  }

  std::size_t GetStep() const noexcept { return step_; }

  double GetEpochAverageLossRef() const noexcept {
    if (epoch_loss_count_ > 0)
      return epoch_loss_ref_sum_ / static_cast<double>(epoch_loss_count_);
    return last_result_.loss_ref;
  }

  double GetEpochAverageLossPhys(std::size_t k = 0) const noexcept {
    if (epoch_loss_count_ > 0 && k < epoch_loss_phys_sum_.size()) {
      return epoch_loss_phys_sum_[k] / static_cast<double>(epoch_loss_count_);
    }
    return (k < last_result_.loss_phys.size()) ? last_result_.loss_phys[k]
                                               : 0.0;
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
    if (epoch_loss_count_ > 0)
      return epoch_loss_total_sum_ / static_cast<double>(epoch_loss_count_);
    return last_result_.loss_total;
  }

private:

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

  void ValidateNetworkVariableNames() const {
    if (!HasUniqueNames(net_.GetInputVars())) {
      throw std::invalid_argument(
          "CMLPTrainer: network input names must be unique.");
    }
    if (!HasUniqueNames(net_.GetOutputVars())) {
      throw std::invalid_argument(
          "CMLPTrainer: network output names must be unique.");
    }
  }

  void ValidateActiveObjective() const {
    const bool has_fitting_objective =
        !fitting_losses_.empty() && (total_samples_ > 0);
    const bool has_phys_objective = !phys_losses_.empty();
    if (!has_fitting_objective && !has_phys_objective) {
      throw std::runtime_error("CMLPTrainer: no active training objective.");
    }
  }

  void RegisterPhysicsLosses() {
    losses_by_set_.clear();
    set_needs_jac_.clear();
    set_needs_hess_.clear();
    empty_sets_.clear();

    for (auto k = 0; k < phys_losses_.size(); ++k) {
      ValidatePhysicsLossStructure(k);
      RegisterPhysicsLoss(k);
    }
  }

  void ValidatePhysicsLossStructure(std::size_t k) const {
    if (!phys_losses_[k]) {
      throw std::logic_error("CMLPTrainer: null physics loss.");
    }
    if (phys_losses_[k]->NumEquations() == 0) {
      throw std::runtime_error("CMLPTrainer: physics loss '" +
                               phys_losses_[k]->GetName() +
                               "' has zero equations.");
    }
    if (coll_sets_.find(phys_set_name_[k]) == coll_sets_.end()) {
      throw std::runtime_error(
          "CMLPTrainer: physics loss '" + phys_losses_[k]->GetName() +
          "' references missing collocation set '" + phys_set_name_[k] + "'.");
    }
  }

  void RegisterPhysicsLoss(std::size_t k) {
    const std::string &set_name = phys_set_name_[k];

    if (coll_sets_.at(set_name).empty()) {
      empty_sets_.insert(set_name);
      if (cfg_.verbose) {
        std::cout << "CMLPTrainer: collocation set '" << set_name
                  << "' is empty — loss '" << phys_losses_[k]->GetName()
                  << "' will contribute zero.\n";
      }
      return;
    }

    ValidatePhysicsLossData(k);

    losses_by_set_[set_name].push_back(k);
    set_needs_jac_[set_name] =
        set_needs_jac_[set_name] || phys_losses_[k]->RequiresJacobian();
    set_needs_hess_[set_name] =
        set_needs_hess_[set_name] || phys_losses_[k]->RequiresHessian();
  }

  void ValidatePhysicsLossData(std::size_t k) const {
    const auto &data = phys_data_[k];
    if (data.empty()) {
      return;
    }

    const std::size_t n_points = coll_sets_.at(phys_set_name_[k]).size();
    const std::size_t n_phys = phys_losses_[k]->NumPhysicsVariables();

    if (data.size() != n_points) {
      throw std::runtime_error(
          "CMLPTrainer: physics data row count mismatch for loss '" +
          phys_losses_[k]->GetName() + "'. Expected " +
          std::to_string(n_points) + ", got " + std::to_string(data.size()) +
          ".");
    }
    for (auto p = 0; p < data.size(); ++p) {
      if (data[p].size() != n_phys) {
        throw std::runtime_error(
            "CMLPTrainer: physics data dimension mismatch for loss '" +
            phys_losses_[k]->GetName() + "' at point " + std::to_string(p) +
            ". Expected " + std::to_string(n_phys) + ", got " +
            std::to_string(data[p].size()) + ".");
      }
    }
  }

  void BuildActiveSetOrder() {
    active_set_order_.clear();
    for (const auto &set_name : set_order_) {
      if (losses_by_set_.find(set_name) != losses_by_set_.end() &&
          empty_sets_.find(set_name) == empty_sets_.end()) {
        active_set_order_.push_back(set_name);
      }
    }
  }

  void RegisterBoundaryLosses() {
    bcs_losses_by_set_.clear();
    for (auto k = 0; k < bcs_losses_.size(); ++k) {
      const std::string &set_name = bcs_set_name_[k];
      bcs_losses_by_set_[set_name].push_back(k);

      if (coll_sets_.find(set_name) == coll_sets_.end() ||
          coll_sets_.at(set_name).empty()) {
        continue;
      }
      if (std::find(active_set_order_.begin(), active_set_order_.end(),
                    set_name) == active_set_order_.end()) {
        active_set_order_.push_back(set_name);
      }
      if (set_needs_jac_.find(set_name) == set_needs_jac_.end()) {
        set_needs_jac_[set_name] = false;
      }
      if (set_needs_hess_.find(set_name) == set_needs_hess_.end()) {
        set_needs_hess_[set_name] = false;
      }
      set_needs_jac_[set_name] =
          set_needs_jac_[set_name] || bcs_losses_[k]->RequiresJacobian();
      set_needs_hess_[set_name] =
          set_needs_hess_[set_name] || bcs_losses_[k]->RequiresHessian();
    }
  }

  void CreateAnnealer() {
    annealer_.reset();
    const bool has_data_like_terms =
        (!fitting_losses_.empty() && (total_samples_ > 0)) ||
        !bcs_losses_.empty();
    if (cfg_.use_annealer && has_data_like_terms && !phys_losses_.empty()) {
      AnnealerConfig a = annealer_cfg_;
      a.n_data_terms = fitting_losses_.size() + bcs_losses_.size();
      annealer_ = std::make_unique<CGradientAnnealer>(a);
    }
  }


  void RegisterWeightsOnTape(typename mlpdouble::Tape &tape,
                             TrainStepState &st) {
    tape.reset();
    tape.setActive();

    st.weights = net_.GetWeightsBiases();
    st.n_w = st.weights.size();

    if (st.n_w == 0) {
      tape.reset();
      throw std::runtime_error(
          "CMLPTrainer: network has no trainable parameters.");
    }

    for (auto &w : st.weights) {
      tape.registerInput(w);
    }
    net_.SetWeightsBiases(st.weights);
  }

  // Zero out all per-step gradient accumulation buffers and size the AD
  // scratch buffers later used for the Adam update.
  void InitializeSensitivities(TrainStepState &st) {
    grad_total_.assign(st.n_w, 0.0);
    if (grad_per_data_term_.size() < st.n_ref)
      grad_per_data_term_.resize(st.n_ref);
    for (auto i = 0; i < st.n_ref; ++i)
      grad_per_data_term_[i].assign(st.n_w, 0.0);

    clean_weights_ad_.resize(st.n_w);
    g_total_ad_.resize(st.n_w);
  }

  void EvaluateFittingLoss(typename mlpdouble::Tape &tape, TrainStepState &st) {
    st.L_ref_vec.assign(st.n_ref, mlpdouble(0.0));
    if (!st.have_ref)
      return;

    if (batch_preds_.size() < batch_x_.size()) {
      batch_preds_.resize(batch_x_.size());
    }
    for (auto i = 0; i < batch_x_.size(); ++i) {
      if (batch_preds_[i].outputs.size() != net_.GetnOutputs()) {
        batch_preds_[i].outputs.resize(net_.GetnOutputs());
      }
      net_.Predict(batch_x_[i], false, false);
      batch_preds_[i].inputs = batch_x_[i];
      for (auto o = 0; o < net_.GetnOutputs(); ++o) {
        batch_preds_[i].outputs[o] = net_.GetOutput(o);
      }
    }
    batch_preds_.resize(batch_x_.size());

    for (auto i = 0; i < st.n_ref; ++i) {
      st.L_ref_vec[i] = fitting_losses_[i]->Evaluate(batch_preds_, batch_y_);
      tape.registerOutput(st.L_ref_vec[i]);
    }
  }

  void EvaluatePhysicsAndBoundaryLosses(TrainStepState &st) {
    st.L_phys.assign(st.n_phys, mlpdouble(0.0));
    st.n_points_seen.assign(st.n_phys, 0);
    st.L_bcs.assign(bcs_losses_.size(), mlpdouble(0.0));
    st.n_bc_points_seen.assign(bcs_losses_.size(), 0);

    typename mlpdouble::Tape &tape = mlpdouble::getTape();

    if (current_pred_.outputs.size() != net_.GetnOutputs()) {
      current_pred_.outputs.resize(net_.GetnOutputs());
    }

    for (const auto &set_name : active_set_order_) {
      const auto &points = coll_sets_.at(set_name);

      const bool need_jac = set_needs_jac_.at(set_name);
      const bool need_hess = set_needs_hess_.at(set_name);
      const bool eval_jac = need_jac || need_hess;
      const bool eval_hess = need_hess;

      NextPhysicsBatch(set_name);

      for (auto idx : batch_indices_) {
        point_storage_.Fill(net_, points[idx], current_pred_, eval_jac,
                            eval_hess);

        auto phys_it = losses_by_set_.find(set_name);
        if (phys_it != losses_by_set_.end()) {
          for (const std::size_t k : phys_it->second) {
            const std::size_t n_phys_vars =
                phys_losses_[k]->NumPhysicsVariables();
            const auto &data = phys_data_[k];

            if (n_phys_vars > 0 && data.empty()) {
              tape.reset();
              throw std::runtime_error("CMLPTrainer: missing physics data.");
            }

            if (data.empty()) {
              st.L_phys[k] +=
                  phys_losses_[k]->EvaluateSingleSample(current_pred_);
            } else {
              st.L_phys[k] += phys_losses_[k]->EvaluateSingleSample(
                  current_pred_, data[idx]);
            }
            ++st.n_points_seen[k];
          }
        }

        auto bcs_it = bcs_losses_by_set_.find(set_name);
        if (bcs_it != bcs_losses_by_set_.end()) {
          for (const std::size_t k : bcs_it->second) {
            st.L_bcs[k] += bcs_losses_[k]->EvaluateSingleSample(current_pred_);
            ++st.n_bc_points_seen[k];
          }
        }
      }
    }
  }

  void NormalizeAndRegisterLosses(typename mlpdouble::Tape &tape,
                                  TrainStepState &st) {
    for (auto k = 0; k < st.n_phys; ++k) {
      if (st.n_points_seen[k] == 0) {
        st.L_phys[k] = mlpdouble(0.0);
        tape.registerOutput(st.L_phys[k]);
        continue;
      }
      const double denom = static_cast<double>(st.n_points_seen[k]) *
                           static_cast<double>(phys_losses_[k]->NumEquations());
      if (denom <= 0.0)
        throw std::runtime_error("CMLPTrainer: invalid normalization.");
      st.L_phys[k] = st.L_phys[k] / denom;
      tape.registerOutput(st.L_phys[k]);
    }

    for (auto k = 0; k < bcs_losses_.size(); ++k) {
      if (st.n_bc_points_seen[k] == 0) {
        st.L_bcs[k] = mlpdouble(0.0);
        tape.registerOutput(st.L_bcs[k]);
        continue;
      }
      st.L_bcs[k] = st.L_bcs[k] / st.n_bc_points_seen[k];
      tape.registerOutput(st.L_bcs[k]);
    }

    tape.setPassive();
  }

  void InitializeStepLambdas(TrainStepState &st) {
    st.lambdas.assign(st.n_ref + bcs_losses_.size(), 1.0);
    if (annealer_) {
      for (auto i = 0; i < st.n_ref + bcs_losses_.size(); ++i) {
        st.lambdas[i] = annealer_->get_lambda(i);
      }
    }
  }

  bool NeedsAnnealerRefresh(const TrainStepState &st) const {
    const bool is_anneal_update_step = (cfg_.annealer_update_freq > 0) &&
                                       (step_ % cfg_.annealer_update_freq == 0);
    const bool has_data_like_terms = st.have_ref || !st.L_bcs.empty();
    return annealer_ && has_data_like_terms && st.n_phys > 0 &&
           is_anneal_update_step;
  }

  void ComputeStepGradients(typename mlpdouble::Tape &tape,
                            TrainStepState &st) {
    if (NeedsAnnealerRefresh(st)) {
      ComputeAnnealedPathGradient(tape, st);
    } else {
      ComputeFastPathGradient(tape, st);
    }
  }

  void ComputeFastPathGradient(typename mlpdouble::Tape &tape,
                               TrainStepState &st) {
    auto zero_gradients = [&st]() {
      for (auto &w : st.weights)
        w.setGradient(0.0);
    };
    auto read_gradients = [&](std::vector<double> &g) {
      for (auto i = 0; i < st.n_w; ++i) {
        g[i] = to_double(st.weights[i].getGradient());
      }
    };

    zero_gradients();
    tape.clearAdjoints();

    for (auto i = 0; i < st.n_ref; ++i)
      st.L_ref_vec[i].setGradient(st.lambdas[i]);
    for (auto k = 0; k < st.L_bcs.size(); ++k)
      st.L_bcs[k].setGradient(st.lambdas[st.n_ref + k]);
    for (auto k = 0; k < st.n_phys; ++k)
      st.L_phys[k].setGradient(1.0);

    tape.evaluate();
    read_gradients(grad_total_);
  }

  void ComputeAnnealedPathGradient(typename mlpdouble::Tape &tape,
                                   TrainStepState &st) {
    auto zero_gradients = [&st]() {
      for (auto &w : st.weights)
        w.setGradient(0.0);
    };
    auto read_gradients = [&](std::vector<double> &g) {
      for (auto i = 0; i < st.n_w; ++i) {
        g[i] = to_double(st.weights[i].getGradient());
      }
    };

    // Combined residual gradient (Lr)
    zero_gradients();
    tape.clearAdjoints();
    for (auto i = 0; i < st.n_ref; ++i)
      st.L_ref_vec[i].setGradient(0.0);
    for (auto &lb : st.L_bcs)
      lb.setGradient(0.0);
    for (auto k = 0; k < st.n_phys; ++k)
      st.L_phys[k].setGradient(1.0);
    tape.evaluate();
    read_gradients(grad_total_);

    // One sweep per individual data-like term
    grad_per_data_term_.resize(st.n_ref + st.L_bcs.size());
    for (auto i = 0; i < st.n_ref + st.L_bcs.size(); ++i) {
      if (grad_per_data_term_[i].size() < st.n_w)
        grad_per_data_term_[i].assign(st.n_w, 0.0);
    }
    for (auto i = 0; i < st.n_ref; ++i) {
      zero_gradients();
      tape.clearAdjoints();
      for (auto j = 0; j < st.n_ref; ++j)
        st.L_ref_vec[j].setGradient(j == i ? 1.0 : 0.0);
      for (auto &lb : st.L_bcs)
        lb.setGradient(0.0);
      for (auto &lp : st.L_phys)
        lp.setGradient(0.0);
      tape.evaluate();
      read_gradients(grad_per_data_term_[i]);
    }
    for (auto k = 0; k < st.L_bcs.size(); ++k) {
      zero_gradients();
      tape.clearAdjoints();
      for (auto &lr : st.L_ref_vec)
        lr.setGradient(0.0);
      for (auto j = 0; j < st.L_bcs.size(); ++j)
        st.L_bcs[j].setGradient(j == k ? 1.0 : 0.0);
      for (auto &lp : st.L_phys)
        lp.setGradient(0.0);
      tape.evaluate();
      read_gradients(grad_per_data_term_[st.n_ref + k]);
    }

    const SingleLossGradientStats base_stats =
        SingleLossGradientStats::from_grads(grad_total_);
    std::vector<SingleLossGradientStats> data_stats_vec(st.n_ref +
                                                        st.L_bcs.size());
    for (auto i = 0; i < data_stats_vec.size(); ++i) {
      data_stats_vec[i] =
          SingleLossGradientStats::from_grads(grad_per_data_term_[i]);
    }
    annealer_->update(base_stats, data_stats_vec);

    for (auto i = 0; i < st.n_ref + st.L_bcs.size(); ++i) {
      st.lambdas[i] = annealer_->get_lambda(i);
    }

    for (auto i = 0; i < st.n_w; ++i) {
      for (auto r = 0; r < st.n_ref; ++r)
        grad_total_[i] += st.lambdas[r] * grad_per_data_term_[r][i];
      for (auto k = 0; k < st.L_bcs.size(); ++k) {
        grad_total_[i] +=
            st.lambdas[st.n_ref + k] * grad_per_data_term_[st.n_ref + k][i];
      }
    }
  }

  void ApplyGradientClipping(std::size_t n_w) {
    if (cfg_.max_grad_norm <= 0.0)
      return;
    double norm_sq = 0.0;
    for (auto i = 0; i < n_w; ++i)
      norm_sq += grad_total_[i] * grad_total_[i];
    const double norm = std::sqrt(norm_sq);
    if (norm > cfg_.max_grad_norm) {
      const double scale = cfg_.max_grad_norm / norm;
      for (auto i = 0; i < n_w; ++i)
        grad_total_[i] *= scale;
    }
  }

  void ApplyAdamUpdate(TrainStepState &st) {
    for (auto i = 0; i < st.n_w; ++i) {
      clean_weights_ad_[i] = mlpdouble(to_double(st.weights[i]));
      g_total_ad_[i] = mlpdouble(grad_total_[i]);
    }
    adam_.OptimizationStep(clean_weights_ad_, g_total_ad_);
    net_.SetWeightsBiases(clean_weights_ad_);
  }

  TrainStepResult AssembleStepResult(const TrainStepState &st) const {
    TrainStepResult result;
    result.lambdas = st.lambdas;
    result.loss_phys.resize(st.n_phys);
    result.loss_bcs.resize(st.L_bcs.size());

    double total_fitting_loss = 0.0;
    for (auto i = 0; i < st.n_ref; ++i)
      total_fitting_loss += to_double(st.L_ref_vec[i]);
    result.loss_ref = st.have_ref ? total_fitting_loss : 0.0;

    for (auto k = 0; k < st.n_phys; ++k) {
      result.loss_phys[k] = to_double(st.L_phys[k]);
    }

    result.loss_raw = result.loss_ref;
    result.loss_total = 0.0;
    for (auto k = 0; k < st.n_phys; ++k) {
      result.loss_raw += result.loss_phys[k];
      result.loss_total += result.loss_phys[k];
    }

    if (st.have_ref) {
      for (auto r = 0; r < st.n_ref; ++r) {
        result.loss_total += st.lambdas[r] * to_double(st.L_ref_vec[r]);
      }
    }
    for (auto k = 0; k < st.L_bcs.size(); ++k) {
      result.loss_bcs[k] = to_double(st.L_bcs[k]);
      result.loss_raw += result.loss_bcs[k];
      result.loss_total += st.lambdas[st.n_ref + k] * result.loss_bcs[k];
    }

    return result;
  }

  bool EpochHasWork() const {
    const bool has_ref = (total_samples_ > 0) && !fitting_losses_.empty();
    const bool has_phys = !active_set_order_.empty();
    return has_ref || has_phys;
  }

  std::size_t ComputeEpochBatchCount() const {
    std::size_t n_batches = 1;

    if ((total_samples_ > 0) && !fitting_losses_.empty()) {
      const std::size_t batch_size =
          cfg_.batch_size > 0 ? std::min(cfg_.batch_size, total_samples_)
                              : total_samples_;
      n_batches = (total_samples_ + batch_size - 1) / batch_size;
    } else if (!active_set_order_.empty()) {
      const std::string &first_set = active_set_order_.front();
      const std::size_t n_points = coll_sets_.at(first_set).size();
      if (n_points > 0) {
        const std::size_t phys_bs =
            cfg_.physics_batch_size > 0
                ? std::min(cfg_.physics_batch_size, n_points)
                : n_points;
        n_batches = (n_points + phys_bs - 1) / phys_bs;
      }
    }

    if (n_batches == 0) {
      n_batches = 1;
    }
    return n_batches;
  }

  void ResetEpochStatistics() {
    epoch_loss_sum_ = 0.0;
    epoch_loss_total_sum_ = 0.0;
    epoch_loss_ref_sum_ = 0.0;
    std::fill(epoch_loss_phys_sum_.begin(), epoch_loss_phys_sum_.end(), 0.0);
    std::fill(epoch_loss_bcs_sum_.begin(), epoch_loss_bcs_sum_.end(), 0.0);
    std::fill(epoch_loss_lambda_sum_.begin(), epoch_loss_lambda_sum_.end(),
              0.0);
    epoch_loss_count_ = 0;
  }

  void AccumulateEpochStatistics(const TrainStepResult &res) {
    epoch_loss_sum_ += res.loss_raw;
    epoch_loss_total_sum_ += res.loss_total;
    epoch_loss_ref_sum_ += res.loss_ref;

    if (epoch_loss_phys_sum_.size() < res.loss_phys.size())
      epoch_loss_phys_sum_.resize(res.loss_phys.size(), 0.0);
    for (auto k = 0; k < res.loss_phys.size(); ++k)
      epoch_loss_phys_sum_[k] += res.loss_phys[k];

    if (epoch_loss_bcs_sum_.size() < res.loss_bcs.size())
      epoch_loss_bcs_sum_.resize(res.loss_bcs.size(), 0.0);
    for (auto k = 0; k < res.loss_bcs.size(); ++k)
      epoch_loss_bcs_sum_[k] += res.loss_bcs[k];

    if (epoch_loss_lambda_sum_.size() < res.lambdas.size())
      epoch_loss_lambda_sum_.resize(res.lambdas.size(), 0.0);
    for (auto k = 0; k < res.lambdas.size(); ++k)
      epoch_loss_lambda_sum_[k] += res.lambdas[k];

    ++epoch_loss_count_;
  }



  double ComputeConvergenceLoss() const {
    if (epoch_loss_count_ == 0) {
      return last_result_.loss_raw;
    }
    if (cfg_.conv_metric == ConvergenceMetric::Weighted) {
      return epoch_loss_total_sum_ / static_cast<double>(epoch_loss_count_);
    }
    return epoch_loss_sum_ / static_cast<double>(epoch_loss_count_);
  }

  bool HasConverged(double current_loss, double previous_loss) const {
    if (!std::isfinite(previous_loss)) {
      return false;
    }
    const double abs_delta = std::abs(current_loss - previous_loss);
    const double rel_delta = abs_delta / std::max(1.0, std::abs(previous_loss));
    return abs_delta <= cfg_.conv_tol_abs || rel_delta <= cfg_.conv_tol_rel;
  }


  void NextDataFittingBatch() {
    batch_x_.clear();
    batch_y_.clear();
    if (total_samples_ == 0)
      return;

    epoch_started_ = true;

    const std::size_t batch_size =
        cfg_.batch_size > 0 ? std::min(cfg_.batch_size, total_samples_)
                            : total_samples_;

    for (auto b = 0; b < batch_size; ++b) {
      if (batch_cursor_ >= total_samples_) {
        if (cfg_.shuffle_per_epoch && total_samples_ > 1)
          std::shuffle(indices_.begin(), indices_.end(), rng_);
        batch_cursor_ = 0;
      }
      const std::size_t idx = indices_[batch_cursor_++];
      batch_x_.push_back(train_inputs_[idx]);
      batch_y_.push_back(train_targets_[idx]);
    }
  }


  void NextPhysicsBatch(const std::string &set_name) {
    batch_indices_.clear();

    auto it = coll_sets_.find(set_name);
    if (it == coll_sets_.end() || it->second.empty())
      return;

    auto &indices = coll_indices_.at(set_name);
    auto &cursor = coll_batch_cursor_[set_name];
    coll_epoch_started_[set_name] = true;

    if (cursor >= indices.size()) {
      if (cfg_.shuffle_per_epoch && indices.size() > 1)
        std::shuffle(indices.begin(), indices.end(), rng_);
      cursor = 0;
    }

    const std::size_t batch_size =
        cfg_.physics_batch_size > 0
            ? std::min(cfg_.physics_batch_size, indices.size())
            : indices.size();

    for (auto b = 0; b < batch_size && cursor < indices.size(); ++b) {
      batch_indices_.push_back(indices[cursor++]);
    }
  }

  std::ofstream OpenHistoryFile() const {
    std::ofstream file;
    if (history_filename_.empty()) {
      return file;
    }

    file.open(history_filename_);
    if (!file.is_open()) {
      throw std::runtime_error("CMLPTrainer: cannot open history file '" +
                               history_filename_ + "'.");
    }

    file << "epoch,loss_data,loss_phys,loss_bc,loss_total,loss_raw";
    for (auto i = 0; i < fitting_losses_.size() + bcs_losses_.size(); ++i) {
      file << ",lambda_" << i;
    }
    file << "\n";
    return file;
  }

  void WriteHistoryRow(std::ofstream &file, std::size_t epoch) const {
    if (!file.is_open()) {
      return;
    }

    double physics_loss_sum = 0.0;
    for (auto k = 0; k < phys_losses_.size(); ++k) {
      physics_loss_sum += GetEpochAverageLossPhys(k);
    }
    double bcs_loss_sum = 0.0;
    for (auto k = 0; k < bcs_losses_.size(); ++k) {
      bcs_loss_sum += GetEpochAverageLossBC(k);
    }

    file << (epoch + 1) << "," << std::scientific << std::setprecision(8)
         << GetEpochAverageLossRef() << "," << physics_loss_sum << ","
         << bcs_loss_sum << "," << GetEpochAverageLossTotal() << ","
         << GetEpochAverageLoss();
    for (auto i = 0; i < fitting_losses_.size() + bcs_losses_.size(); ++i) {
      file << "," << GetEpochAverageLambda(i);
    }
    file << "\n";
    file.flush();
  }

  void LogEpoch(std::size_t epoch, double current_loss) const {
    const char *metric_str =
        cfg_.conv_metric == ConvergenceMetric::Weighted ? "weighted" : "raw";

    std::cout << "Epoch " << std::setw(5) << (epoch + 1) << "/" << std::setw(5)
              << cfg_.max_epochs << " | " << std::setw(8) << metric_str
              << "_loss=" << std::scientific << std::setprecision(6)
              << std::setw(14) << current_loss
              << " | loss_total=" << std::setw(14) << last_result_.loss_total
              << " | loss_raw=" << std::setw(14) << last_result_.loss_raw
              << "\n";
  }

  void LogStep(const TrainStepResult &res) const {
    std::cout << "step " << std::setw(6) << step_ << std::scientific
              << std::setprecision(6) << " | loss_total=" << std::setw(14)
              << res.loss_total << " | loss_raw=" << std::setw(14)
              << res.loss_raw << " | loss_ref=" << std::setw(14)
              << res.loss_ref;
    for (auto k = 0; k < res.loss_phys.size(); ++k)
      std::cout << " | L_phys[" << k << "]=" << std::setw(14)
                << res.loss_phys[k];
    for (auto k = 0; k < res.loss_bcs.size(); ++k)
      std::cout << " | L_bc[" << k << "]=" << std::setw(14) << res.loss_bcs[k];
    for (auto k = 0; k < res.lambdas.size(); ++k)
      std::cout << " | lambda[" << k << "]=" << std::setw(14) << res.lambdas[k];
    std::cout << "\n";
  }

 
  void PrepareEpoch() {
    if (cfg_.shuffle_per_epoch && total_samples_ > 1) {
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
    for (auto &pair : coll_epoch_started_) {
      pair.second = false;
    }
    if (cfg_.shuffle_per_epoch) {
      for (auto &pair : coll_indices_) {
        if (pair.second.size() > 1)
          std::shuffle(pair.second.begin(), pair.second.end(), rng_);
      }
    }
  }

  void EndPhysicsEpoch() {
    for (auto &pair : coll_epoch_started_) {
      pair.second = false;
    }
    for (auto &pair : coll_batch_cursor_) {
      pair.second = 0;
    }
  }

 
  static bool HasSameSignature(const std::vector<std::string> &a,
                               const std::vector<std::string> &b) {
    return a == b;
  }

  static bool HasUniqueNames(const std::vector<std::string> &names) {
    std::unordered_set<std::string> seen;
    for (const auto &n : names) {
      if (seen.find(n) != seen.end()) {
        return false;
      }
      seen.insert(n);
    }
    return true;
  }

  std::size_t FindPhysicsLossIndex(const std::string &name) const {
    for (auto i = 0; i < phys_losses_.size(); ++i) {
      if (phys_losses_[i]->GetName() == name) {
        return i;
      }
    }
    throw std::runtime_error("CMLPTrainer: no physics loss named '" + name +
                             "'.");
  }


  CNeuralNetwork &net_;
  CAdam adam_;
  AnnealerConfig annealer_cfg_;
  TrainerConfig cfg_;
  CPointDerivatives point_storage_;
  std::mt19937_64 rng_;

  std::vector<std::shared_ptr<CBaseLoss>> fitting_losses_;
  std::vector<std::shared_ptr<CPhysicsLoss>> phys_losses_;
  std::vector<std::string> phys_set_name_;
  std::vector<std::vector<std::vector<mlpdouble>>> phys_data_;
  std::vector<std::shared_ptr<CPhysicsLoss>> bcs_losses_;
  std::vector<std::string> bcs_set_name_;

  std::vector<std::vector<mlpdouble>> train_inputs_;
  std::vector<std::vector<mlpdouble>> train_targets_;
  std::size_t total_samples_{0};
  std::vector<std::size_t> indices_;
  std::size_t batch_cursor_{0};
  bool epoch_started_{false};
  std::vector<std::vector<mlpdouble>> batch_x_;
  std::vector<std::vector<mlpdouble>> batch_y_;
  std::vector<PredictionResult> batch_preds_;

  std::unordered_map<std::string, std::vector<std::vector<mlpdouble>>>
      coll_sets_;
  std::vector<std::string> set_order_;
  std::unordered_map<std::string, std::vector<std::size_t>> coll_indices_;
  std::unordered_map<std::string, std::size_t> coll_batch_cursor_;
  std::unordered_map<std::string, bool> coll_epoch_started_;
  std::vector<std::size_t> batch_indices_;
  PredictionResult current_pred_;

  bool finalized_{false};
  std::unordered_map<std::string, std::vector<std::size_t>> losses_by_set_;
  std::unordered_map<std::string, std::vector<std::size_t>> bcs_losses_by_set_;
  std::unordered_map<std::string, bool> set_needs_jac_;
  std::unordered_map<std::string, bool> set_needs_hess_;
  std::vector<std::string> active_set_order_;
  std::unordered_set<std::string> empty_sets_;
  std::unique_ptr<CGradientAnnealer> annealer_;

  std::vector<double> grad_total_;
  std::vector<std::vector<double>> grad_per_data_term_;
  std::vector<mlpdouble> clean_weights_ad_;
  std::vector<mlpdouble> g_total_ad_;

  std::size_t step_{0};
  TrainStepResult last_result_{};
  double epoch_loss_sum_{0.0};
  double epoch_loss_total_sum_{0.0};
  double epoch_loss_ref_sum_{0.0};
  std::vector<double> epoch_loss_phys_sum_;
  std::vector<double> epoch_loss_bcs_sum_;
  std::vector<double> epoch_loss_lambda_sum_;
  std::size_t epoch_loss_count_{0};
  std::string history_filename_;
};

} // namespace MLPToolbox
