/*!
 * \file demo_case.cpp
 * \brief Integration test: train CNeuralNetwork on reference data with
 *        a physics loss using the CMLPTrainer and CPhysicsLoss APIs.
 */

#define MLP_CUSTOM_TYPE codi::RealReverse
#include "codi.hpp"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "CAdam.hpp"
#include "CGradientAnnealer.hpp"
#include "CMLPTrainer.hpp"
#include "CMeanSquaredErrorLoss.hpp"
#include "CNeuralNetwork.hpp"
#include "CPhysicsLoss.hpp"
#include "variable_def.hpp"

using namespace MLPToolbox;

// ============================================================================
// CSV reader
// ============================================================================
static std::vector<std::vector<double>> readCSV(const std::string &filename) {
  std::vector<std::vector<double>> data;
  std::ifstream file(filename);

  if (!file.is_open()) {
    std::cerr << "ERROR: Cannot open " << filename << "\n";
    return data;
  }

  std::string line;
  if (!std::getline(file, line))
    return data; // Skip header

  while (std::getline(file, line)) {
    if (line.empty())
      continue;
    std::stringstream ss(line);
    std::vector<double> row;
    std::string cell;
    while (std::getline(ss, cell, '\t')) {
      if (!cell.empty())
        row.push_back(std::stod(cell));
    }
    if (row.size() == 3)
      data.push_back(row);
  }
  return data;
}

// ============================================================================
// Main
// ============================================================================
int main() {
  // =========================================================================
  // 1. Load reference data
  // =========================================================================
  std::cout << "Loading reference data...\n";
  const auto rawData = readCSV("reference_data.csv");
  if (rawData.empty()) {
    std::cerr << "ERROR: Empty dataset. Run test_problem.py first.\n";
    return 1;
  }
  const std::size_t N = rawData.size();
  std::cout << "Loaded " << N << " data points.\n";

  // =========================================================================
  // 2. Compute data ranges
  // =========================================================================
  double u_min = 1e9, u_max = -1e9;
  double v_min = 1e9, v_max = -1e9;
  double y_min = 1e9, y_max = -1e9;

  std::vector<double> u_raw(N), v_raw(N), y_raw(N);
  for (std::size_t i = 0; i < N; ++i) {
    u_raw[i] = rawData[i][0];
    v_raw[i] = rawData[i][1];
    y_raw[i] = rawData[i][2];

    u_min = std::min(u_min, u_raw[i]);
    u_max = std::max(u_max, u_raw[i]);
    v_min = std::min(v_min, v_raw[i]);
    v_max = std::max(v_max, v_raw[i]);
    y_min = std::min(y_min, y_raw[i]);
    y_max = std::max(y_max, y_raw[i]);
  }

  assert(u_max > u_min && v_max > v_min && y_max > y_min);

  // =========================================================================
  // 3. Convert dataset to mlpdouble
  // =========================================================================
  std::vector<std::vector<mlpdouble>> X(N, std::vector<mlpdouble>(2));
  std::vector<std::vector<mlpdouble>> Y(N, std::vector<mlpdouble>(1));
  for (std::size_t i = 0; i < N; ++i) {
    X[i][0] = mlpdouble(u_raw[i]);
    X[i][1] = mlpdouble(v_raw[i]);
    Y[i][0] = mlpdouble(y_raw[i]);
  }

  // =========================================================================
  // 4. Build neural network
  // =========================================================================
  std::vector<std::size_t> architecture = {2, 16, 16, 1};
  CNeuralNetwork net(architecture);

  for (std::size_t iLayer = 1; iLayer < net.GetnLayers() - 1; ++iLayer) {
    net.SetActivationFunction(iLayer, "tanh");
  }

  net.SetInputRegularization("minmax");
  net.SetInputNorm(0, static_cast<mlpdouble>(u_min),
                   static_cast<mlpdouble>(u_max));
  net.SetInputNorm(1, static_cast<mlpdouble>(v_min),
                   static_cast<mlpdouble>(v_max));

  net.SetOutputRegularization("minmax");
  net.SetOutputNorm(0, static_cast<mlpdouble>(y_min),
                    static_cast<mlpdouble>(y_max));

  net.SetInputName(0, "u");
  net.SetInputName(1, "v");
  net.SetOutputName(0, "y");

  net.RandomWeights();
  net.DisplayNetwork();

  // =========================================================================
  // 5. Create physics loss
  // =========================================================================
  CPhysicsEquation eq;
  eq.name = "dy_du_eq";
  eq.input_names = {"u"};
  eq.output_names = {"y"};
  eq.requires_jacobian = true;
  eq.requires_hessian = false;
  eq.residual = [](const PhysicsState &state,
                   const PhysicsData & /*data*/) -> mlpdouble {
    return state.Jac(0, 0);
  };
  auto physics_loss = std::make_shared<CPhysicsLoss>(
      "dy_du_zero", net.GetInputVars(), net.GetOutputVars(),
      std::vector<std::string>{}, std::vector<CPhysicsEquation>{eq});

  // =========================================================================
  // 6. Create optimizer
  // =========================================================================
  CAdam optimizer(1e-3, 0.9, 0.999, 1e-8);

  // =========================================================================
  // 7. Configure gradient annealing
  // =========================================================================
  AnnealerConfig anneal_cfg;
  anneal_cfg.n_data_terms = 1;
  anneal_cfg.alpha = 0.9;
  anneal_cfg.lambda_init = 1.0;
  anneal_cfg.lambda_min = 1e-4;
  anneal_cfg.lambda_max = 1e20;

  // =========================================================================
  // 8. Configure trainer
  // =========================================================================
  TrainerConfig trainer_cfg;
  trainer_cfg.max_epochs = 200;
  trainer_cfg.batch_size = 32;
  trainer_cfg.physics_batch_size = 32;
  trainer_cfg.conv_tol_abs = 1e-8;
  trainer_cfg.conv_tol_rel = 1e-6;
  trainer_cfg.use_annealer = false; // Set to false for pure data comparison
  trainer_cfg.verbose = true;       // the trainer owns epoch logging
  trainer_cfg.log_every = 10;
  trainer_cfg.shuffle_per_epoch = true;

  // =========================================================================
  // 9. Construct trainer, register data and losses
  // =========================================================================
  CMLPTrainer trainer(net, std::move(optimizer), anneal_cfg, trainer_cfg);

  trainer.SetTrainingData(X, Y);
  trainer.AddFittingLoss(std::make_shared<CMeanSquaredErrorLoss>());

  // =========================================================================
  // 10. Register physics collocation set and loss (ENABLED for PINN)
  // =========================================================================
  std::vector<std::vector<mlpdouble>> physics_points;
  const int Ncoll = 200;
  for (int i = 0; i < Ncoll; ++i) {
    double t = static_cast<double>(i) / (Ncoll - 1);
    physics_points.push_back({mlpdouble(u_min + t * (u_max - u_min)),
                              mlpdouble(v_min + t * (v_max - v_min))});
  }
  trainer.SetCollocationPoints("colloc", physics_points);
  trainer.AddPhysicsLoss(physics_loss, "colloc");

  trainer.FinalizeConfiguration();

  // =========================================================================
  // 11. Train (the trainer writes the per-epoch history CSV itself)
  // =========================================================================
  std::cout << "\nStarting training (" << trainer_cfg.max_epochs
            << " epochs)...\n";
  std::cout << "Physics mini-batch size: " << trainer_cfg.physics_batch_size
            << "\n\n";

  trainer.SetHistoryFile("pinn_training_history.csv");
  trainer.Train();

  // =========================================================================
  // 12. Final training result
  // =========================================================================
  const TrainStepResult &final_result = trainer.GetLastResult();
  const double final_phys =
      final_result.loss_phys.empty() ? 0.0 : final_result.loss_phys[0];

  std::cout << "\n======================================\n";
  std::cout << "Training complete.\n";
  std::cout << "Final fitting loss:   " << std::scientific
            << final_result.loss_ref << "\n";
  std::cout << "Final physics loss:   " << final_phys << "\n";
  std::cout << "Final total loss:     " << final_result.loss_total << "\n";
  std::cout << "======================================\n";

  // =========================================================================
  // 13. Save trained network
  // =========================================================================
  net.WriteNeuralNetwork("trained_model_ad.mlp");
  std::cout << "Model saved to trained_model_ad.mlp\n";
  std::cout << "History saved to pinn_training_history.csv\n";

  return 0;
}