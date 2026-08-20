/*!
 * \file cavity_trainer.cpp
 * \brief Standalone MPI Hybrid PINN trainer for 2D Lid-Driven Cavity flow.
 * Implements Paper Algorithm 1: PDEs as L_r, Data+BCs as L_i.
 */

#define MLP_CUSTOM_TYPE codi::RealReverse
#include "codi.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <mpi.h>

#include "CNeuralNetwork.hpp"
#include "CAdam.hpp"
#include "CGradientAnnealer.hpp"
#include "CMLPTrainer.hpp"
#include "CPhysicsLoss.hpp"
#include "CMeanSquaredErrorLoss.hpp"
#include "variable_def.hpp"

using namespace MLPToolbox;

// Helper to read SU2 CSV and convert to non-dimensional primitives
bool ReadSU2CSV_NonDim(const std::string& csv_filename,
                       std::vector<std::vector<mlpdouble>>& X,
                       std::vector<std::vector<mlpdouble>>& Y,
                       double U_ref, double p_ref) {
    std::ifstream file(csv_filename);
    if (!file.is_open()) return false;

    std::string line;
    std::getline(file, line); // Skip header

    std::stringstream header_ss(line);
    std::string col_name;
    std::vector<std::string> headers;
    while (std::getline(header_ss, col_name, ',')) {
        if (!col_name.empty() && col_name.front() == '"') col_name.erase(0, 1);
        if (!col_name.empty() && col_name.back() == '"') col_name.pop_back();
        headers.push_back(col_name);
    }

    int idx_x = -1, idx_y = -1, idx_rho = -1, idx_rhou = -1, idx_rhov = -1, idx_E = -1;
    for (size_t i = 0; i < headers.size(); ++i) {
        if (headers[i] == "x") idx_x = i;
        if (headers[i] == "y") idx_y = i;
        if (headers[i] == "Density") idx_rho = i;
        if (headers[i] == "Momentum_x") idx_rhou = i;
        if (headers[i] == "Momentum_y") idx_rhov = i;
        if (headers[i] == "Energy") idx_E = i;
    }

    if (idx_x == -1 || idx_y == -1 || idx_rho == -1 || idx_rhou == -1 || idx_rhov == -1 || idx_E == -1) {
        std::cerr << "ERROR: CSV header missing required conservative fields.\n";
        return false;
    }

    const double gamma = 1.4;

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string val;
        std::vector<double> row_data;
        while (std::getline(ss, val, ',')) {
            try { row_data.push_back(std::stod(val)); } catch (...) { row_data.push_back(0.0); }
        }
        
        if (row_data.size() > std::max({idx_x, idx_y, idx_rho, idx_rhou, idx_rhov, idx_E})) {
            X.push_back({mlpdouble(row_data[idx_x]), mlpdouble(row_data[idx_y])});
            
            double rho = std::max(row_data[idx_rho], 1e-8);
            double u_dim = row_data[idx_rhou] / rho;
            double v_dim = row_data[idx_rhov] / rho;
            double p_dim = (gamma - 1.0) * (row_data[idx_E] - 0.5 * rho * (u_dim*u_dim + v_dim*v_dim));
            
            // Non-dimensionalize targets for the neural network
            Y.push_back({mlpdouble(u_dim / U_ref), mlpdouble(v_dim / U_ref), mlpdouble(p_dim / p_ref)});
        }
    }
    return !X.empty();
}

int main(int argc, char** argv) {
    // =========================================================================
    // 1. MPI Initialization
    // =========================================================================
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        std::cout << "\n======================================\n";
        std::cout << "Starting Hybrid PINN Lid-Driven Cavity (Paper Algorithm 1)...\n";
        std::cout << "Running with " << size << " MPI ranks.\n";
        std::cout << "======================================\n";
    }

    // =========================================================================
    // 2. Physical Reference Scales
    // =========================================================================
    const double U_lid = 33.179;
    const double L_ref = 1.0;
    const double rho_ref = 5.04175e-04;

    // Use ideal gas static pressure for absolute pressure scaling
    const double R_gas = 287.058;
    const double T_ref = 288.15;
    const double p_ref = rho_ref * R_gas * T_ref;

    // Coefficient required in the non-dimensional momentum equation
    // because pressure is scaled by rho*R*T instead of rho*U^2
    const double C_p = (R_gas * T_ref) / (U_lid * U_lid);

    // Reynolds number
    const double Re = 1000.0;

    // =========================================================================
    // 3. Load reference data (Non-Dimensionalized)
    // =========================================================================
    std::vector<std::vector<mlpdouble>> X, Y;
    if (!ReadSU2CSV_NonDim("restart_flow.csv", X, Y, U_lid, p_ref)) {
        if (rank == 0) std::cerr << "ERROR: Failed to load dataset. Did SU2 run first?\n";
        MPI_Finalize();
        return 1;
    }
    const std::size_t N = X.size();
    if (rank == 0) std::cout << "Loaded " << N << " data points.\n";

    // =========================================================================
    // 4. Build neural network (2 inputs -> 32x32x32 hidden -> 3 outputs)
    // =========================================================================
    std::vector<std::size_t> architecture = {2, 32, 32, 32, 3};
    CNeuralNetwork net(architecture);
    for (std::size_t iLayer = 1; iLayer < net.GetnLayers() - 1; ++iLayer) {
        net.SetActivationFunction(iLayer, "tanh");
    }
    // Explicitly ensure output layer is linear
    net.SetActivationFunction(net.GetnLayers() - 1, "linear");

    // Non-dimensional inputs and outputs are O(1)
    net.SetInputRegularization("minmax");
    net.SetInputNorm(0, 0.0, 1.0);
    net.SetInputNorm(1, 0.0, 1.0);

    net.SetOutputRegularization("minmax");
    net.SetOutputNorm(0, -1.0, 1.0);
    net.SetOutputNorm(1, -1.0, 1.0);
    net.SetOutputNorm(2, -1.0, 1.0);

    net.SetInputName(0, "x"); net.SetInputName(1, "y");
    net.SetOutputName(0, "u"); net.SetOutputName(1, "v"); net.SetOutputName(2, "p");

    if (rank == 0) net.RandomWeights();
    
    std::vector<mlpdouble> weights = net.GetWeightsBiases();
    std::vector<double> w_double(weights.size());
    for(size_t i=0; i<weights.size(); ++i) w_double[i] = to_double(weights[i]);
    MPI_Bcast(w_double.data(), w_double.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for(size_t i=0; i<weights.size(); ++i) weights[i] = mlpdouble(w_double[i]);
    net.SetWeightsBiases(weights);

    if (rank == 0) net.DisplayNetwork();

    // =========================================================================
    // 5. Create Physics Losses (L_r) - Clean Non-Dimensional + SDF Down-weighting
    // =========================================================================
    // OPTIMIZATION: Untraced SDF calculation (returns standard double)
    auto sdf_weight = [](const PhysicsState& s) -> double {
        double x = to_double(s.In(0));
        double y = to_double(s.In(1));
        return std::min({x, 1.0 - x, y, 1.0 - y});
    };

    // Eq 1: Continuity: du/dx + dv/dy = 0
    CPhysicsEquation eq_cont;
    eq_cont.name = "continuity";
    eq_cont.input_names = {"x", "y"};
    eq_cont.output_names = {"u", "v"};
    eq_cont.requires_jacobian = true;
    eq_cont.residual = [sdf_weight](const PhysicsState& s, const PhysicsData&) -> mlpdouble {
        mlpdouble res = s.Jac(0, 0) + s.Jac(1, 1);
        return res * mlpdouble(sdf_weight(s));
    };
    auto loss_cont = std::make_shared<CPhysicsLoss>("loss_continuity", net.GetInputVars(), net.GetOutputVars(), std::vector<std::string>{}, std::vector<CPhysicsEquation>{eq_cont});

    // Eq 2: X-Momentum: u*du/dx + v*du/dy + C_p*dp/dx - (1/Re)*(d2u/dx2 + d2u/dy2) = 0
    CPhysicsEquation eq_xmom;
    eq_xmom.name = "x_momentum";
    eq_xmom.input_names = {"x", "y"};
    eq_xmom.output_names = {"u", "v", "p"};
    eq_xmom.requires_jacobian = true;
    eq_xmom.requires_hessian = true;
    eq_xmom.residual = [Re, C_p, sdf_weight](const PhysicsState& s, const PhysicsData&) -> mlpdouble {
        mlpdouble u = s.Out(0);
        mlpdouble v = s.Out(1);
        mlpdouble dudx = s.Jac(0, 0);
        mlpdouble dudy = s.Jac(0, 1);
        mlpdouble dpdx = s.Jac(2, 0);
        mlpdouble d2udx2 = s.Hess(0, 0, 0);
        mlpdouble d2udy2 = s.Hess(0, 1, 1);
        mlpdouble res = u * dudx + v * dudy + C_p * dpdx - (1.0 / Re) * (d2udx2 + d2udy2); // <-- Added C_p
        return res * mlpdouble(sdf_weight(s));
    };
    auto loss_xmom = std::make_shared<CPhysicsLoss>("loss_x_momentum", net.GetInputVars(), net.GetOutputVars(), std::vector<std::string>{}, std::vector<CPhysicsEquation>{eq_xmom});

    // Eq 3: Y-Momentum: u*dv/dx + v*dv/dy + C_p*dp/dy - (1/Re)*(d2v/dx2 + d2v/dy2) = 0
    CPhysicsEquation eq_ymom;
    eq_ymom.name = "y_momentum";
    eq_ymom.input_names = {"x", "y"};
    eq_ymom.output_names = {"u", "v", "p"};
    eq_ymom.requires_jacobian = true;
    eq_ymom.requires_hessian = true;
    eq_ymom.residual = [Re, C_p, sdf_weight](const PhysicsState& s, const PhysicsData&) -> mlpdouble {
        mlpdouble u = s.Out(0);
        mlpdouble v = s.Out(1);
        mlpdouble dvdx = s.Jac(1, 0);
        mlpdouble dvdy = s.Jac(1, 1);
        mlpdouble dpdy = s.Jac(2, 1);
        mlpdouble d2vdx2 = s.Hess(1, 0, 0);
        mlpdouble d2vdy2 = s.Hess(1, 1, 1);
        mlpdouble res = u * dvdx + v * dvdy + C_p * dpdy - (1.0 / Re) * (d2vdx2 + d2vdy2); // <-- Added C_p
        return res * mlpdouble(sdf_weight(s));
    };
    auto loss_ymom = std::make_shared<CPhysicsLoss>("loss_y_momentum", net.GetInputVars(), net.GetOutputVars(), std::vector<std::string>{}, std::vector<CPhysicsEquation>{eq_ymom});
    // =========================================================================
    // 6. Create Boundary Condition Losses (L_i) - with Corner Smoothing
    // =========================================================================
    Top wall (y=1): u=1, v=0
    CPhysicsEquation eq_top_u;
    eq_top_u.name = "bc_top_u";
    eq_top_u.input_names = {"x", "y"};
    eq_top_u.output_names = {"u"};
    eq_top_u.residual = [](const PhysicsState& s, const PhysicsData&) -> mlpdouble {
        double x = to_double(s.In(0));
        double target_u = 1.0;
        if (x < 0.1) target_u = x / 0.1;
        else if (x > 0.9) target_u = (1.0 - x) / 0.1;
        return s.Out(0) - mlpdouble(target_u);
    };
    auto loss_top_u = std::make_shared<CPhysicsLoss>("loss_top_u", net.GetInputVars(), net.GetOutputVars(), std::vector<std::string>{}, std::vector<CPhysicsEquation>{eq_top_u});

    CPhysicsEquation eq_top_v;
    eq_top_v.name = "bc_top_v";
    eq_top_v.input_names = {"x", "y"};
    eq_top_v.output_names = {"v"};
    eq_top_v.residual = [](const PhysicsState& s, const PhysicsData&) -> mlpdouble {
        return s.Out(0) - 0.0;
    };
    auto loss_top_v = std::make_shared<CPhysicsLoss>("loss_top_v", net.GetInputVars(), net.GetOutputVars(), std::vector<std::string>{}, std::vector<CPhysicsEquation>{eq_top_v});

    // Other walls (x=0, x=1, y=0): u=0, v=0
    CPhysicsEquation eq_wall_u;
    eq_wall_u.name = "bc_wall_u";
    eq_wall_u.input_names = {"x", "y"};
    eq_wall_u.output_names = {"u"};
    eq_wall_u.residual = [](const PhysicsState& s, const PhysicsData&) -> mlpdouble {
        return s.Out(0) - 0.0;
    };
    auto loss_wall_u = std::make_shared<CPhysicsLoss>("loss_wall_u", net.GetInputVars(), net.GetOutputVars(), std::vector<std::string>{}, std::vector<CPhysicsEquation>{eq_wall_u});

    CPhysicsEquation eq_wall_v;
    eq_wall_v.name = "bc_wall_v";
    eq_wall_v.input_names = {"x", "y"};
    eq_wall_v.output_names = {"v"};
    eq_wall_v.residual = [](const PhysicsState& s, const PhysicsData&) -> mlpdouble {
        return s.Out(0) - 0.0;
    };
    auto loss_wall_v = std::make_shared<CPhysicsLoss>("loss_wall_v", net.GetInputVars(), net.GetOutputVars(), std::vector<std::string>{}, std::vector<CPhysicsEquation>{eq_wall_v});

    // =========================================================================
    // 7. Create optimizer and trainer
    // =========================================================================
    CAdam optimizer(1e-3, 0.9, 0.999, 1e-8);
    AnnealerConfig anneal_cfg;
    anneal_cfg.n_data_terms = 1; // Overridden in Build() to ref_losses_.size() + bcs_losses_.size()
    anneal_cfg.alpha = 0.9; 
    anneal_cfg.lambda_max = 1e10;

    TrainerConfig trainer_cfg;
    trainer_cfg.max_epochs         = 400;
    trainer_cfg.batch_size         = 128;    
    trainer_cfg.physics_batch_size = 128;    
    trainer_cfg.use_annealer       = true;
    trainer_cfg.verbose            = false;  
    trainer_cfg.log_every          = 10;
    trainer_cfg.shuffle_per_epoch  = true;
    trainer_cfg.annealer_update_freq = 10;

    CMLPTrainer trainer(net, std::move(optimizer), anneal_cfg, trainer_cfg);

    // =========================================================================
    // 8. Partition Data among MPI Ranks
    // =========================================================================
    std::size_t points_per_rank = N / size;
    std::size_t start_idx = rank * points_per_rank;
    std::size_t end_idx = (rank == size - 1) ? N : start_idx + points_per_rank;
    std::size_t local_N = end_idx - start_idx;

    std::vector<std::vector<mlpdouble>> X_local(local_N), Y_local(local_N);
    for (std::size_t i = 0; i < local_N; ++i) {
        X_local[i] = X[start_idx + i];
        Y_local[i] = Y[start_idx + i];
    }

    trainer.SetTrainingData(X_local, Y_local);
    
    // Register Data as L_i (Reference Losses)
    trainer.AddReferenceLoss(std::make_shared<CMeanSquaredErrorLoss>()); // SU2 Data

    // Generate Collocation Points (Interior + Boundaries)
    std::mt19937 gen(42 + rank); 
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::vector<std::vector<mlpdouble>> physics_points;
    std::size_t local_coll = (2000 + size - 1) / size;
    for (std::size_t i = 0; i < local_coll; ++i) {
        physics_points.push_back({mlpdouble(dist(gen)), mlpdouble(dist(gen))});
    }
    trainer.SetCollocationPoints("interior", physics_points);

    std::vector<std::vector<mlpdouble>> top_bc_pts, wall_bc_pts;
    std::size_t local_bc = (1000 + size - 1) / size;
    for (std::size_t i = 0; i < local_bc; ++i) {
        double t = dist(gen);
        top_bc_pts.push_back({mlpdouble(t), mlpdouble(1.0)});
        wall_bc_pts.push_back({mlpdouble(0.0), mlpdouble(t)});
        wall_bc_pts.push_back({mlpdouble(1.0), mlpdouble(t)});
        wall_bc_pts.push_back({mlpdouble(t), mlpdouble(0.0)});
    }
    trainer.SetCollocationPoints("top_bc", top_bc_pts);
    trainer.SetCollocationPoints("wall_bc", wall_bc_pts);

    // Register PDEs as L_r (Physics Losses), Data+BCs as L_i
    trainer.AddPhysicsLoss(loss_cont, "interior");
    trainer.AddPhysicsLoss(loss_xmom, "interior");
    trainer.AddPhysicsLoss(loss_ymom, "interior");
    trainer.AddBoundaryLoss(loss_top_u, "top_bc");
    trainer.AddBoundaryLoss(loss_top_v, "top_bc");
    trainer.AddBoundaryLoss(loss_wall_u, "wall_bc");
    trainer.AddBoundaryLoss(loss_wall_v, "wall_bc");

    trainer.EnableMPI();
    trainer.Build();

    // =========================================================================
    // 9. Train & Log Custom History
    // =========================================================================
    if (rank == 0) {
    std::cout << "\nStarting training (" << trainer_cfg.max_epochs << " epochs)...\n";
    std::cout << std::left
              << std::setw(8)  << "Epoch"
              << std::setw(14) << "L_Data"
              << std::setw(14) << "L_Phys"
              << std::setw(14) << "L_BC"
              << std::setw(12) << "Lambda"
              << std::setw(14) << "L_Total"
              << "\n";
    std::cout << std::string(82, '-') << "\n";
    }

    std::ofstream history_file;
    if (rank == 0) {
        history_file.open("hybrid_pinn_history.csv");
        history_file << "epoch,loss_data,loss_phys,loss_bc,lambda,loss_total\n";
    }

    for (std::size_t epoch = 0; epoch < trainer_cfg.max_epochs; ++epoch) {
        trainer.TrainEpoch();
        
        // L_i (Reference) losses
        const double avg_data_loss = trainer.GetEpochAverageLossRef();
        
        // L_r (Physics) losses (PDEs only, size is 3)
        double avg_physics_loss = 0.0;
        for (size_t k = 0; k < 3; ++k) {
            avg_physics_loss += trainer.GetEpochAverageLossPhys(k);
        }
        
        // L_i (Boundary) losses (size is 4)
        double avg_bc_loss = 0.0;
        for (size_t k = 0; k < 4; ++k) {
            avg_bc_loss += trainer.GetEpochAverageLossBC(k);
        }

        const double avg_lambda    = trainer.GetEpochAverageLambda(0);
        const double avg_loss_total = trainer.GetEpochAverageLossTotal();

        if (rank == 0) {
            history_file << (epoch + 1) << ","
                        << std::scientific << std::setprecision(8) << avg_data_loss << ","
                        << avg_physics_loss << ","
                        << avg_bc_loss << ","
                        << std::fixed << std::setprecision(8) << avg_lambda << ","
                        << std::scientific << std::setprecision(8) << avg_loss_total << "\n";

            if (epoch % 10 == 0 || epoch == trainer_cfg.max_epochs - 1) {
                std::cout << std::left
                        << std::setw(8)  << (epoch + 1)
                        << std::setw(14) << std::scientific << std::setprecision(4) << avg_data_loss
                        << std::setw(14) << avg_physics_loss
                        << std::setw(14) << avg_bc_loss
                        << std::setw(12) << std::fixed << std::setprecision(2) << avg_lambda
                        << std::setw(14) << std::scientific << std::setprecision(4) << avg_loss_total
                        << "\n";
            }
        }
    }

    if (rank == 0) {
        history_file.close();
        net.WriteNeuralNetwork("hybrid_pinn_model.mlp");
        std::cout << "\nTraining complete. Model saved to hybrid_pinn_model.mlp\n";
    }

    // =========================================================================
    // 10. Output Predictions for Visualization (Convert to Dimensional)
    // =========================================================================
    if (rank == 0) {
        std::ofstream pred_file("hybrid_pinn_predictions.csv");
        pred_file << "x,y,u,v,p\n";
        for (int i = 0; i <= 100; ++i) {
            for (int j = 0; j <= 100; ++j) {
                double x = static_cast<double>(i) / 100.0;
                double y = static_cast<double>(j) / 100.0;
                std::vector<mlpdouble> input = {mlpdouble(x), mlpdouble(y)};
                
                net.Predict(input, false, false);
                
                // Convert non-dimensional outputs back to dimensional
                double u_nd = to_double(net.GetOutput(0));
                double v_nd = to_double(net.GetOutput(1));
                double p_nd = to_double(net.GetOutput(2));
                
                double u_dim = u_nd * U_lid;
                double v_dim = v_nd * U_lid;
                double p_dim = p_nd * p_ref;
                
                pred_file << std::scientific << std::setprecision(8) 
                          << x << "," << y << "," << u_dim << "," << v_dim << "," << p_dim << "\n";
            }
        }
        pred_file.close();
        std::cout << "Predictions saved to hybrid_pinn_predictions.csv\n";
    }

    MPI_Finalize();
    return 0;
}