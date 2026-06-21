/*!
 * \file demo_case.cpp
 * \brief Integration test: train CNeuralNetwork on spiral data with a
 *        physics loss using CoDiPack reverse-mode AD.
 *
 * Physics loss: enforce dy/du = 0 at the origin (u=0, v=0),
 * i.e. the network's first input-Jacobian should be zero there.
 * L_phys = (dy/du)^2
 */

#define MLP_CUSTOM_TYPE codi::RealReverse
#include "codi.hpp"

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <iomanip>

#include "CNeuralNetwork.hpp"
#include "CAdam.hpp"
#include "CGradientAnnealer.hpp"
#include "variable_def.hpp"   // mlpdouble = codi::RealReverse (macro defined above)

using namespace MLPToolbox;
using Tape = codi::RealReverse::Tape;


//  CSV reader (returns raw double)
static std::vector<std::vector<double>> readCSV(const std::string& filename)
{
    std::vector<std::vector<double>> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "ERROR: Cannot open " << filename << "\n";
        std::exit(1);
    }
    std::string line;
    std::getline(file, line);           // skip header
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> row;
        std::string cell;
        while (std::getline(ss, cell, '\t'))
            row.push_back(std::stod(cell));
        if (row.size() == 3) data.push_back(row);
    }
    return data;
}


int main() {
    // Load data as plain double
    std::cout << "Loading reference data...\n";
    auto rawData = readCSV("reference_data.csv");
    if (rawData.empty()) { std::cerr << "ERROR: empty dataset\n"; return 1; }

    const std::size_t N = rawData.size();
    std::cout << "Loaded " << N << " data points\n";

    double u_min =  1e9, u_max = -1e9;
    double v_min =  1e9, v_max = -1e9;
    double y_min =  1e9, y_max = -1e9;

    std::vector<double> u_raw(N), v_raw(N), y_raw(N);
    for (std::size_t i = 0; i < N; ++i) {
        u_raw[i] = rawData[i][0];
        v_raw[i] = rawData[i][1];
        y_raw[i] = rawData[i][2];
        u_min = std::min(u_min, u_raw[i]); u_max = std::max(u_max, u_raw[i]);
        v_min = std::min(v_min, v_raw[i]); v_max = std::max(v_max, v_raw[i]);
        y_min = std::min(y_min, y_raw[i]); y_max = std::max(y_max, y_raw[i]);
    }

    assert(u_max > u_min && "u range is degenerate");
    assert(v_max > v_min && "v range is degenerate");
    assert(y_max > y_min && "y range is degenerate");

    std::cout << "u range: [" << u_min << ", " << u_max << "]\n";
    std::cout << "v range: [" << v_min << ", " << v_max << "]\n";
    std::cout << "y range: [" << y_min << ", " << y_max << "]\n\n";

    // build mlpdouble dataset (codi::RealReverse, passive values)
    std::vector<std::vector<mlpdouble>> X(N, std::vector<mlpdouble>(2));
    std::vector<mlpdouble> y(N);
    for (std::size_t i = 0; i < N; i++){
        X[i][0] = mlpdouble(u_raw[i]);
        X[i][1] = mlpdouble(v_raw[i]);
        y[i] = mlpdouble(y_raw[i]);
    }

    // build NN
    std::vector<size_t> arch = {2, 16, 16, 1};
    CNeuralNetwork net(arch);
    net.SetActivationFunction("tanh");

    net.SetInputRegularization("minmax");
    net.SetInputNorm(0, static_cast<mlpdouble>(u_min), static_cast<mlpdouble>(u_max));
    net.SetInputNorm(1, static_cast<mlpdouble>(v_min), static_cast<mlpdouble>(v_max));

    net.SetOutputRegularization("minmax");
    net.SetOutputNorm(0, static_cast<mlpdouble>(y_min), static_cast<mlpdouble>(y_max));

    net.SetInputName(0, "u");
    net.SetInputName(1, "v");
    net.SetOutputName(0, "y");

    net.RandomWeights();
    net.DisplayNetwork();

    // Optimizer and Annealer
    CAdam optimizer(1e-3, 0.9, 0.999, 1e-8);

    AnnealerConfig anneal_cfg;
    anneal_cfg.n_data_terms = 1;   // one data-fit term (MSE)
    anneal_cfg.alpha        = 0.9;
    anneal_cfg.lambda_init  = 1.0;
    anneal_cfg.lambda_min   = 1e-4;
    anneal_cfg.lambda_max   = 5.0;
    CGradientAnnealer annealer(anneal_cfg);


    // computing loss
    // 1. data loss (mse)
    // w_reg: weights that are already on the tape
    auto compute_data_loss = [&](const std::vector<mlpdouble>& w_reg) -> mlpdouble {
        net.SetWeightsBiases(w_reg);

        mlpdouble mse = mlpdouble(0.0);
        for (std::size_t i = 0; i < N; i++){
            net.Predict(X[i]);
            mlpdouble error = net.GetOutput(0) - y[i];
            mse += (error * error);
        }

        return mse / mlpdouble(static_cast<double>(N));
    };


    auto compute_phys_loss = [&](const std::vector<mlpdouble>& w_reg) -> mlpdouble {
        net.SetWeightsBiases(w_reg);

        std::vector<mlpdouble> x_col = {
            mlpdouble(0.0),
            mlpdouble(0.0)
        };

        net.Predict(x_col, true, false);
        mlpdouble dydu = net.GetJacobian(0, 0);
        return dydu * dydu;
    };


    // training loop
    const unsigned int epochs = 50;
    std::cout << "Starting training (" << epochs << " epochs)...\n\n";
    std::cout << std::left
              << std::setw(8)  << "Epoch"
              << std::setw(16) << "L_data"
              << std::setw(16) << "L_phys"
              << std::setw(12) << "lambda"
              << std::setw(12) << "L_total"
              << "\n"
              << std::string(64, '-') << "\n";

    Tape& tape = codi::RealReverse::getTape();
    for (unsigned int epoch = 0; epoch < epochs; epoch++){
        std::vector<double> g_data, g_phys;
        double val_data_loss, val_phys_loss;

        // data loss
        {
            tape.reset();
            tape.setActive();

            std::vector<mlpdouble> w = net.GetWeightsBiases();
            for(auto& wi: w) tape.registerInput(wi);
            
            mlpdouble L_data = compute_data_loss(w);
            tape.registerOutput(L_data);
            tape.setPassive();

            val_data_loss = L_data.getValue();

            L_data.setGradient(1.0);
            tape.evaluate();

            g_data.resize(w.size());
            for (std::size_t i = 0; i < w.size(); i++)
                g_data[i] = w[i].getGradient();
            
            tape.reset();

        }

        // physics loss

        {

            tape.setActive();
            std::vector<mlpdouble> w = net.GetWeightsBiases();
            for (auto& wi: w) tape.registerInput(wi);

            mlpdouble L_phys = compute_phys_loss(w);
            tape.registerOutput(L_phys);
            tape.setPassive();

            val_phys_loss = L_phys.getValue();

            L_phys.setGradient(1.0);
            tape.evaluate();

            g_phys.resize(w.size());
            for (std::size_t i = 0; i < w.size(); i++)
                g_phys[i] = w[i].getGradient();

            tape.reset();

        }

        // anealer update step
        GradStats stats_phys = GradStats::from_grads(g_phys);
        GradStats stats_data = GradStats::from_grads(g_data);

        annealer.update(stats_phys, {stats_data});
        double lambda = annealer.get_lambda(0);

        const std::size_t n = g_data.size();
        std::vector<mlpdouble> g_total(n);

        for (std::size_t i = 0; i < n; i++)
            g_total[i] = mlpdouble(g_phys[i] + lambda * g_data[i]);
        
        std::vector<mlpdouble> w_curr = net.GetWeightsBiases();
        optimizer.step(w_curr, g_total);
        net.SetWeightsBiases(w_curr);

        if (epoch % 10 == 0) {
            double L_total = val_phys_loss + lambda * val_data_loss;
            double lr = static_cast<double>(optimizer.getLearningRate().getValue());
            std::cout << std::left
                      << std::setw(8)  << epoch
                      << std::setw(16) << std::scientific << std::setprecision(4)
                                       << val_data_loss
                      << std::setw(16) << val_phys_loss
                      << std::setw(12) << std::fixed << std::setprecision(4) << lambda
                      << std::setw(12) << std::scientific << L_total
                      << "\n";
        }
    }

    std::cout << "\nTraining complete.\n";

    {
        tape.reset();
        tape.setActive();
        auto w = net.GetWeightsBiases();
        for (auto& wi : w) tape.registerInput(wi);
        mlpdouble L_final = mlpdouble(0.0);
        for (std::size_t i = 0; i < N; ++i) {
            net.SetWeightsBiases(w);
            net.Predict(X[i]);
            mlpdouble diff = net.GetOutput(0) - y[i];
            L_final = L_final + diff * diff;
        }
        L_final = L_final / mlpdouble(static_cast<double>(N));
        tape.reset();
        std::cout << "Final data loss (MSE): " << L_final.getValue() << "\n";
    }

    net.WriteNeuralNetwork("trained_model_ad.mlp");
    std::cout << "Model saved to trained_model_ad.mlp\n";

    return 0;

}