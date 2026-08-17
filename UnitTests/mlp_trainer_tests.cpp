#define MLP_CUSTOM_TYPE codi::RealReverse
#include "codi.hpp"

#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>
#include <memory>

#include "CPhysicsLoss.hpp"
#include "CBaseLoss.hpp"
#include "CMeanSquaredErrorLoss.hpp"
#include "CMLPTrainer.hpp"
#include "CAdam.hpp"
#include "CGradientAnnealer.hpp"
#include "CNeuralNetwork.hpp"
#include "variable_def.hpp"

using namespace MLPToolbox;

// ============================================================
// Helpers
// ============================================================

#define REQUIRE_EQUAL_TOL(a, b, tol) \
    REQUIRE(to_double(a) == Catch::Approx(to_double(b)).margin(tol))

// ============================================================
// CMLPTrainer Unit Tests
// ============================================================

TEST_CASE("CMLPTrainer constructor validation", "[CMLPTrainer]") {
    std::vector<std::size_t> arch = {1, 4, 1};
    CNeuralNetwork net(arch);
    net.SetInputName(0, "x");
    net.SetOutputName(0, "y");
    net.RandomWeights();

    CAdam adam(1e-3);
    AnnealerConfig annealer_cfg;
    annealer_cfg.n_data_terms = 1;

    TrainerConfig cfg_neg_abs;
    cfg_neg_abs.conv_tol_abs = -1e-8;
    REQUIRE_THROWS_AS(CMLPTrainer(net, adam, annealer_cfg, cfg_neg_abs), std::invalid_argument);

    TrainerConfig cfg_neg_rel;
    cfg_neg_rel.conv_tol_rel = -1e-6;
    REQUIRE_THROWS_AS(CMLPTrainer(net, adam, annealer_cfg, cfg_neg_rel), std::invalid_argument);

    TrainerConfig cfg_neg_grad;
    cfg_neg_grad.max_grad_norm = -1.0;
    REQUIRE_THROWS_AS(CMLPTrainer(net, adam, annealer_cfg, cfg_neg_grad), std::invalid_argument);
}

TEST_CASE("CMLPTrainer reference loss management", "[CMLPTrainer]") {
    std::vector<std::size_t> arch = {1, 2, 1};
    CNeuralNetwork net(arch);
    net.SetInputName(0, "x");
    net.SetOutputName(0, "y");
    net.RandomWeights();

    CAdam adam(1e-3);
    AnnealerConfig annealer_cfg;
    annealer_cfg.n_data_terms = 1;
    TrainerConfig cfg;

    CMLPTrainer trainer(net, adam, annealer_cfg, cfg);

    auto custom_loss = std::make_shared<CMeanSquaredErrorLoss>();
    REQUIRE_NOTHROW(trainer.AddReferenceLoss(custom_loss));
    REQUIRE_THROWS_AS(trainer.AddReferenceLoss(nullptr), std::invalid_argument);

    std::vector<std::vector<mlpdouble>> X = {{0.0}};
    std::vector<std::vector<mlpdouble>> Y = {{0.0}};
    trainer.SetTrainingData(X, Y);

    trainer.Build();

    REQUIRE_THROWS_AS(trainer.AddReferenceLoss(custom_loss), std::runtime_error);
    REQUIRE_THROWS_AS(trainer.ClearReferenceLosses(), std::runtime_error);
}

TEST_CASE("CMLPTrainer training data validation", "[CMLPTrainer]") {
    std::vector<std::size_t> arch = {1, 2, 1};
    CNeuralNetwork net(arch);
    net.SetInputName(0, "x");
    net.SetOutputName(0, "y");
    net.RandomWeights();

    CAdam adam(1e-3);
    AnnealerConfig annealer_cfg;
    annealer_cfg.n_data_terms = 1;
    TrainerConfig cfg;

    CMLPTrainer trainer(net, adam, annealer_cfg, cfg);

    std::vector<std::vector<mlpdouble>> X = {{0.0}, {1.0}};
    std::vector<std::vector<mlpdouble>> Y_bad_size = {{0.0}};
    std::vector<std::vector<mlpdouble>> Y_bad_dim = {{0.0, 1.0}, {1.0, 2.0}};

    REQUIRE_THROWS_AS(trainer.SetTrainingData(X, Y_bad_size), std::invalid_argument);
    REQUIRE_THROWS_AS(trainer.SetTrainingData(X, Y_bad_dim), std::invalid_argument);

    std::vector<std::vector<mlpdouble>> Y = {{0.0}, {1.0}};
    REQUIRE_NOTHROW(trainer.SetTrainingData(X, Y));

    trainer.AddReferenceLoss(std::make_shared<CMeanSquaredErrorLoss>());
    trainer.Build();
    REQUIRE_THROWS_AS(trainer.SetTrainingData(X, Y), std::runtime_error);
}

TEST_CASE("CMLPTrainer collocation points validation", "[CMLPTrainer]") {
    std::vector<std::size_t> arch = {1, 2, 1};
    CNeuralNetwork net(arch);
    net.SetInputName(0, "x");
    net.SetOutputName(0, "y");
    net.RandomWeights();

    CAdam adam(1e-3);
    AnnealerConfig annealer_cfg;
    annealer_cfg.n_data_terms = 1;
    TrainerConfig cfg;

    CMLPTrainer trainer(net, adam, annealer_cfg, cfg);

    REQUIRE_THROWS_AS(trainer.SetCollocationPoints("", {{0.5}}), std::invalid_argument);
    REQUIRE_THROWS_AS(trainer.SetCollocationPoints("test", {{0.5, 0.5}}), std::invalid_argument);

    REQUIRE_NOTHROW(trainer.SetCollocationPoints("test", {{0.5}}));
    
    std::vector<std::vector<mlpdouble>> X = {{0.0}};
    std::vector<std::vector<mlpdouble>> Y = {{0.0}};
    trainer.SetTrainingData(X, Y);
    trainer.AddReferenceLoss(std::make_shared<CMeanSquaredErrorLoss>());

    trainer.Build();
    REQUIRE_THROWS_AS(trainer.SetCollocationPoints("test2", {{0.5}}), std::runtime_error);
}

TEST_CASE("CMLPTrainer physics loss validation", "[CMLPTrainer]") {
    std::vector<std::size_t> arch = {1, 2, 1};
    CNeuralNetwork net(arch);
    net.SetInputName(0, "x");
    net.SetOutputName(0, "y");
    net.RandomWeights();

    CAdam adam(1e-3);
    AnnealerConfig annealer_cfg;
    annealer_cfg.n_data_terms = 1;
    TrainerConfig cfg;

    CMLPTrainer trainer(net, adam, annealer_cfg, cfg);
    trainer.SetCollocationPoints("colloc", {{0.5}});

    CPhysicsEquation eq;
    eq.name = "eq";
    eq.input_names = {"x"};
    eq.output_names = {"y"};
    eq.residual = [](const PhysicsState&, const PhysicsData&) { return mlpdouble(0.0); };

    auto phys_loss = std::make_shared<CPhysicsLoss>(
        "phys", net.GetInputVars(), net.GetOutputVars(), std::vector<std::string>{}, std::vector<CPhysicsEquation>{eq}
    );

    REQUIRE_THROWS_AS(trainer.AddPhysicsLoss(nullptr, "colloc"), std::invalid_argument);
    REQUIRE_THROWS_AS(trainer.AddPhysicsLoss(phys_loss, ""), std::invalid_argument);
    REQUIRE_THROWS_AS(trainer.AddPhysicsLoss(phys_loss, "non_existent"), std::invalid_argument);

    REQUIRE_NOTHROW(trainer.AddPhysicsLoss(phys_loss, "colloc"));
    REQUIRE_THROWS_AS(trainer.AddPhysicsLoss(phys_loss, "colloc"), std::invalid_argument);

    trainer.Build();
    REQUIRE_THROWS_AS(trainer.AddPhysicsLoss(phys_loss, "colloc"), std::runtime_error);
}

TEST_CASE("CMLPTrainer physics data validation", "[CMLPTrainer]") {
    std::vector<std::size_t> arch = {1, 2, 1};
    CNeuralNetwork net(arch);
    net.SetInputName(0, "x");
    net.SetOutputName(0, "y");
    net.RandomWeights();

    CAdam adam(1e-3);
    AnnealerConfig annealer_cfg;
    annealer_cfg.n_data_terms = 1;
    TrainerConfig cfg;

    CMLPTrainer trainer(net, adam, annealer_cfg, cfg);
    trainer.SetCollocationPoints("colloc", {{0.5}, {1.0}});

    CPhysicsEquation eq;
    eq.name = "eq";
    eq.input_names = {"x"};
    eq.output_names = {"y"};
    eq.residual = [](const PhysicsState&, const PhysicsData&) { return mlpdouble(0.0); };

    auto phys_loss = std::make_shared<CPhysicsLoss>(
        "phys", net.GetInputVars(), net.GetOutputVars(), std::vector<std::string>{"src"}, std::vector<CPhysicsEquation>{eq}
    );

    trainer.AddPhysicsLoss(phys_loss, "colloc");

    std::vector<std::vector<mlpdouble>> valid_data = {{1.0}, {2.0}};
    REQUIRE_NOTHROW(trainer.SetPhysicsData("phys", valid_data));

    std::vector<std::vector<mlpdouble>> bad_rows = {{1.0}};
    REQUIRE_THROWS_AS(trainer.SetPhysicsData("phys", bad_rows), std::invalid_argument);

    std::vector<std::vector<mlpdouble>> bad_size = {{1.0, 2.0}, {3.0, 4.0}};
    REQUIRE_THROWS_AS(trainer.SetPhysicsData("phys", bad_size), std::invalid_argument);

    REQUIRE_THROWS_AS(trainer.SetPhysicsData("non_existent", valid_data), std::runtime_error);
}

TEST_CASE("CMLPTrainer build validation", "[CMLPTrainer]") {
    std::vector<std::size_t> arch = {1, 2, 1};
    CNeuralNetwork net(arch);
    net.SetInputName(0, "x");
    net.SetOutputName(0, "y");
    net.RandomWeights();

    CAdam adam(1e-3);
    AnnealerConfig annealer_cfg;
    annealer_cfg.n_data_terms = 1;
    TrainerConfig cfg;

    CMLPTrainer trainer(net, adam, annealer_cfg, cfg);

    REQUIRE_THROWS_AS(trainer.Build(), std::runtime_error);

    std::vector<std::vector<mlpdouble>> X = {{0.0}};
    std::vector<std::vector<mlpdouble>> Y = {{0.0}};
    trainer.SetTrainingData(X, Y);
    trainer.AddReferenceLoss(std::make_shared<CMeanSquaredErrorLoss>());
    REQUIRE_NOTHROW(trainer.Build());
}

TEST_CASE("CMLPTrainer gradient clipping", "[CMLPTrainer]") {
    std::vector<std::size_t> arch = {1, 4, 1};
    CNeuralNetwork net(arch);
    net.SetInputName(0, "x");
    net.SetOutputName(0, "y");
    net.RandomWeights();

    CAdam adam(1e-3, 0.9, 0.999, 1e-8);
    AnnealerConfig annealer_cfg;
    annealer_cfg.n_data_terms = 1;
    TrainerConfig cfg;
    cfg.max_epochs = 1;
    cfg.batch_size = 2;
    cfg.max_grad_norm = 1e-6; 
    cfg.verbose = false;

    CMLPTrainer trainer(net, adam, annealer_cfg, cfg);

    std::vector<std::vector<mlpdouble>> X = {{0.0}, {1.0}};
    std::vector<std::vector<mlpdouble>> Y = {{0.0}, {1.0}};
    trainer.SetTrainingData(X, Y);
    trainer.AddReferenceLoss(std::make_shared<CMeanSquaredErrorLoss>());
    
    trainer.SetCollocationPoints("colloc", {{0.5}});

    CPhysicsEquation eq;
    eq.name = "dy_dx";
    eq.input_names = {"x"};
    eq.output_names = {"y"};
    eq.requires_jacobian = true;
    eq.residual = [](const PhysicsState& s, const PhysicsData&) -> mlpdouble {
        return s.Jac(0, 0); 
    };

    auto phys_loss = std::make_shared<CPhysicsLoss>(
        "dy_dx", net.GetInputVars(), net.GetOutputVars(), std::vector<std::string>{}, std::vector<CPhysicsEquation>{eq}
    );

    trainer.AddPhysicsLoss(phys_loss, "colloc");
    trainer.Build();
    
    REQUIRE_NOTHROW(trainer.TrainStep());
}

TEST_CASE("CMLPTrainer convergence early stopping", "[CMLPTrainer]") {
    std::vector<std::size_t> arch = {1, 2, 1};
    CNeuralNetwork net(arch);
    net.SetInputName(0, "x");
    net.SetOutputName(0, "y");
    net.RandomWeights();

    CAdam adam(1e-3);
    AnnealerConfig annealer_cfg;
    annealer_cfg.n_data_terms = 1;
    TrainerConfig cfg;
    cfg.max_epochs = 100;
    cfg.conv_tol_abs = 1e-2; 
    cfg.verbose = false;

    CMLPTrainer trainer(net, adam, annealer_cfg, cfg);

    std::vector<std::vector<mlpdouble>> X = {{0.0}, {1.0}};
    std::vector<std::vector<mlpdouble>> Y = {{0.0}, {1.0}};
    trainer.SetTrainingData(X, Y);
    trainer.AddReferenceLoss(std::make_shared<CMeanSquaredErrorLoss>());

    trainer.Build();
    trainer.Train();

    REQUIRE(trainer.GetStep() < 100);
}

TEST_CASE("CMLPTrainer empty collocation set handling", "[CMLPTrainer]") {
    std::vector<std::size_t> arch = {1, 2, 1};
    CNeuralNetwork net(arch);
    net.SetInputName(0, "x");
    net.SetOutputName(0, "y");
    net.RandomWeights();

    CAdam adam(1e-3);
    AnnealerConfig annealer_cfg;
    annealer_cfg.n_data_terms = 1;
    TrainerConfig cfg;
    cfg.max_epochs = 1;
    cfg.verbose = false;

    CMLPTrainer trainer(net, adam, annealer_cfg, cfg);

    std::vector<std::vector<mlpdouble>> X = {{0.0}, {1.0}};
    std::vector<std::vector<mlpdouble>> Y = {{0.0}, {1.0}};
    trainer.SetTrainingData(X, Y);
    trainer.AddReferenceLoss(std::make_shared<CMeanSquaredErrorLoss>());

    trainer.SetCollocationPoints("empty_colloc", {});

    CPhysicsEquation eq;
    eq.name = "eq";
    eq.input_names = {"x"};
    eq.output_names = {"y"};
    eq.residual = [](const PhysicsState&, const PhysicsData&) { return mlpdouble(0.0); };

    auto phys_loss = std::make_shared<CPhysicsLoss>(
        "phys", net.GetInputVars(), net.GetOutputVars(), std::vector<std::string>{}, std::vector<CPhysicsEquation>{eq}
    );

    trainer.AddPhysicsLoss(phys_loss, "empty_colloc");
    trainer.Build();

    REQUIRE_NOTHROW(trainer.TrainStep());
    REQUIRE(trainer.GetLastResult().loss_phys[0] == 0.0);
}

TEST_CASE("CMLPTrainer multiple physics losses", "[CMLPTrainer]") {
    std::vector<std::size_t> arch = {1, 4, 1};
    CNeuralNetwork net(arch);
    net.SetInputName(0, "x");
    net.SetOutputName(0, "y");
    net.RandomWeights();

    CAdam adam(1e-3, 0.9, 0.999, 1e-8);
    AnnealerConfig annealer_cfg;
    annealer_cfg.n_data_terms = 1; // Trainer will override this based on added ref losses
    TrainerConfig cfg;
    cfg.max_epochs = 2;
    cfg.batch_size = 2;
    cfg.verbose = false;
    cfg.use_annealer = true;

    CMLPTrainer trainer(net, adam, annealer_cfg, cfg);

    std::vector<std::vector<mlpdouble>> X = {{0.0}, {1.0}};
    std::vector<std::vector<mlpdouble>> Y = {{0.0}, {1.0}};
    trainer.SetTrainingData(X, Y);
    trainer.AddReferenceLoss(std::make_shared<CMeanSquaredErrorLoss>());

    trainer.SetCollocationPoints("colloc1", {{0.5}});
    trainer.SetCollocationPoints("colloc2", {{0.8}});

    CPhysicsEquation eq1;
    eq1.name = "eq1";
    eq1.input_names = {"x"};
    eq1.output_names = {"y"};
    eq1.requires_jacobian = true;
    eq1.residual = [](const PhysicsState& s, const PhysicsData&) -> mlpdouble {
        return s.Jac(0, 0); 
    };

    CPhysicsEquation eq2;
    eq2.name = "eq2";
    eq2.input_names = {"x"};
    eq2.output_names = {"y"};
    eq2.requires_jacobian = true;
    eq2.residual = [](const PhysicsState& s, const PhysicsData&) -> mlpdouble {
        return s.Jac(0, 0) - 1.0; 
    };

    auto phys_loss1 = std::make_shared<CPhysicsLoss>(
        "phys1", net.GetInputVars(), net.GetOutputVars(), std::vector<std::string>{}, std::vector<CPhysicsEquation>{eq1}
    );

    auto phys_loss2 = std::make_shared<CPhysicsLoss>(
        "phys2", net.GetInputVars(), net.GetOutputVars(), std::vector<std::string>{}, std::vector<CPhysicsEquation>{eq2}
    );

    trainer.AddPhysicsLoss(phys_loss1, "colloc1");
    trainer.AddPhysicsLoss(phys_loss2, "colloc2");
    trainer.Build();
    trainer.TrainEpoch();

    const auto& res = trainer.GetLastResult();
    REQUIRE(res.loss_phys.size() == 2);
    REQUIRE(res.lambdas.size() == 1); // 1 reference data term
    REQUIRE(trainer.GetStep() == 1);
}

TEST_CASE("CMLPTrainer multiple reference losses", "[CMLPTrainer]") {
    std::vector<std::size_t> arch = {1, 4, 2}; // 2 outputs
    CNeuralNetwork net(arch);
    net.SetInputName(0, "x");
    net.SetOutputName(0, "y1");
    net.SetOutputName(1, "y2");
    net.RandomWeights();

    CAdam adam(1e-3, 0.9, 0.999, 1e-8);
    AnnealerConfig annealer_cfg;
    TrainerConfig cfg;
    cfg.max_epochs = 1;
    cfg.batch_size = 2;
    cfg.verbose = false;
    cfg.use_annealer = true;

    CMLPTrainer trainer(net, adam, annealer_cfg, cfg);

    // Add two separate reference losses
    trainer.AddReferenceLoss(std::make_shared<CMeanSquaredErrorLoss>());
    trainer.AddReferenceLoss(std::make_shared<CMeanSquaredErrorLoss>());

    std::vector<std::vector<mlpdouble>> X = {{0.0}, {1.0}};
    std::vector<std::vector<mlpdouble>> Y = {{0.0, 0.0}, {1.0, 1.0}};
    trainer.SetTrainingData(X, Y);

    trainer.SetCollocationPoints("colloc", {{0.5}});

    CPhysicsEquation eq;
    eq.name = "dy1_dx";
    eq.input_names = {"x"};
    eq.output_names = {"y1"};
    eq.requires_jacobian = true;
    eq.residual = [](const PhysicsState& s, const PhysicsData&) -> mlpdouble {
        return s.Jac(0, 0); 
    };

    auto phys_loss = std::make_shared<CPhysicsLoss>(
        "dy1_dx", net.GetInputVars(), net.GetOutputVars(), std::vector<std::string>{}, std::vector<CPhysicsEquation>{eq}
    );

    trainer.AddPhysicsLoss(phys_loss, "colloc");
    trainer.Build();
    trainer.TrainStep();

    const auto& res = trainer.GetLastResult();
    REQUIRE(res.lambdas.size() == 2); // M=2 data terms
}

// ============================================================
// Physics Mini-Batch Tests
// ============================================================

TEST_CASE("CMLPTrainer physics mini-batch limits points per step", "[CMLPTrainer]") {
    std::vector<std::size_t> arch = {1, 4, 1};
    CNeuralNetwork net(arch);
    net.SetInputName(0, "x");
    net.SetOutputName(0, "y");
    net.RandomWeights();

    CAdam adam(1e-3);
    AnnealerConfig annealer_cfg;
    annealer_cfg.n_data_terms = 1;
    TrainerConfig cfg;
    cfg.max_epochs = 1;
    cfg.batch_size = 2;
    cfg.physics_batch_size = 3; // Sample exactly 3 physics points per step
    cfg.verbose = false;

    CMLPTrainer trainer(net, adam, annealer_cfg, cfg);

    std::vector<std::vector<mlpdouble>> X = {{0.0}, {1.0}};
    std::vector<std::vector<mlpdouble>> Y = {{0.0}, {1.0}};
    trainer.SetTrainingData(X, Y);
    trainer.AddReferenceLoss(std::make_shared<CMeanSquaredErrorLoss>());

    // Create 10 collocation points
    std::vector<std::vector<mlpdouble>> colloc;
    for(int i = 0; i < 10; ++i) {
        colloc.push_back({static_cast<double>(i) / 10.0});
    }
    trainer.SetCollocationPoints("colloc", colloc);

    int call_count = 0;
    CPhysicsEquation eq;
    eq.name = "eq";
    eq.input_names = {"x"};
    eq.output_names = {"y"};
    eq.requires_jacobian = false;
    eq.residual = [&call_count](const PhysicsState&, const PhysicsData&) -> mlpdouble {
        ++call_count;
        return mlpdouble(1.0);
    };

    auto phys_loss = std::make_shared<CPhysicsLoss>(
        "phys", net.GetInputVars(), net.GetOutputVars(), std::vector<std::string>{}, std::vector<CPhysicsEquation>{eq}
    );

    trainer.AddPhysicsLoss(phys_loss, "colloc");
    trainer.Build();
    
    call_count = 0;
    trainer.TrainStep();
    
    // With physics_batch_size = 3, the residual should be called exactly 3 times.
    REQUIRE(call_count == 3);
}

TEST_CASE("CMLPTrainer physics full batch when size is 0", "[CMLPTrainer]") {
    std::vector<std::size_t> arch = {1, 4, 1};
    CNeuralNetwork net(arch);
    net.SetInputName(0, "x");
    net.SetOutputName(0, "y");
    net.RandomWeights();

    CAdam adam(1e-3);
    AnnealerConfig annealer_cfg;
    annealer_cfg.n_data_terms = 1;
    TrainerConfig cfg;
    cfg.max_epochs = 1;
    cfg.batch_size = 2;
    cfg.physics_batch_size = 0; // 0 means full batch
    cfg.verbose = false;

    CMLPTrainer trainer(net, adam, annealer_cfg, cfg);

    std::vector<std::vector<mlpdouble>> X = {{0.0}, {1.0}};
    std::vector<std::vector<mlpdouble>> Y = {{0.0}, {1.0}};
    trainer.SetTrainingData(X, Y);
    trainer.AddReferenceLoss(std::make_shared<CMeanSquaredErrorLoss>());

    std::vector<std::vector<mlpdouble>> colloc;
    for(int i = 0; i < 10; ++i) {
        colloc.push_back({static_cast<double>(i) / 10.0});
    }
    trainer.SetCollocationPoints("colloc", colloc);

    int call_count = 0;
    CPhysicsEquation eq;
    eq.name = "eq";
    eq.input_names = {"x"};
    eq.output_names = {"y"};
    eq.requires_jacobian = false;
    eq.residual = [&call_count](const PhysicsState&, const PhysicsData&) -> mlpdouble {
        ++call_count;
        return mlpdouble(1.0);
    };

    auto phys_loss = std::make_shared<CPhysicsLoss>(
        "phys", net.GetInputVars(), net.GetOutputVars(), std::vector<std::string>{}, std::vector<CPhysicsEquation>{eq}
    );

    trainer.AddPhysicsLoss(phys_loss, "colloc");
    trainer.Build();
    
    call_count = 0;
    trainer.TrainStep();
    
    // With physics_batch_size = 0, all 10 points should be evaluated.
    REQUIRE(call_count == 10);
}

TEST_CASE("CMLPTrainer basic construction and training", "[CMLPTrainer]") {
    std::vector<std::size_t> arch = {1, 4, 1};
    CNeuralNetwork net(arch);
    net.SetInputName(0, "x");
    net.SetOutputName(0, "y");
    net.RandomWeights();

    CAdam adam(1e-3, 0.9, 0.999, 1e-8);
    AnnealerConfig annealer_cfg;
    annealer_cfg.n_data_terms = 1;
    TrainerConfig trainer_cfg;
    trainer_cfg.max_epochs = 2;
    trainer_cfg.batch_size = 2;
    trainer_cfg.verbose = false;

    CMLPTrainer trainer(net, adam, annealer_cfg, trainer_cfg);

    std::vector<std::vector<mlpdouble>> X = {{0.0}, {1.0}};
    std::vector<std::vector<mlpdouble>> Y = {{0.0}, {1.0}};
    trainer.SetTrainingData(X, Y);
    trainer.AddReferenceLoss(std::make_shared<CMeanSquaredErrorLoss>());
    trainer.SetCollocationPoints("colloc", {{0.5}});

    CPhysicsEquation eq;
    eq.name = "dy_dx";
    eq.input_names = {"x"};
    eq.output_names = {"y"};
    eq.requires_jacobian = true;
    eq.residual = [](const PhysicsState& s, const PhysicsData&) -> mlpdouble {
        return s.Jac(0, 0); 
    };

    auto phys_loss = std::make_shared<CPhysicsLoss>(
        "dy_dx", 
        net.GetInputVars(), 
        net.GetOutputVars(), 
        std::vector<std::string>{}, 
        std::vector<CPhysicsEquation>{eq}
    );

    trainer.AddPhysicsLoss(phys_loss, "colloc");
    trainer.Build();
    trainer.TrainEpoch();

    const auto& res = trainer.GetLastResult();
    REQUIRE(res.loss_phys.size() == 1);
    REQUIRE(trainer.GetStep() == 1);
}

TEST_CASE("CMLPTrainer missing physics data throws at runtime", "[CMLPTrainer]") {
    std::vector<std::size_t> arch = {1, 2, 1};
    CNeuralNetwork net(arch);
    net.SetInputName(0, "x");
    net.SetOutputName(0, "y");
    net.RandomWeights();

    CAdam adam(1e-3);
    AnnealerConfig annealer_cfg;
    annealer_cfg.n_data_terms = 1;
    TrainerConfig trainer_cfg;
    trainer_cfg.max_epochs = 1;
    trainer_cfg.batch_size = 1;
    trainer_cfg.verbose = false;

    CMLPTrainer trainer(net, adam, annealer_cfg, trainer_cfg);
    trainer.SetCollocationPoints("colloc", {{0.5}});

    CPhysicsEquation eq;
    eq.name = "dy_dx_source";
    eq.input_names = {"x"};
    eq.output_names = {"y"};
    eq.requires_jacobian = true;
    eq.residual = [](const PhysicsState& s, const PhysicsData& data) -> mlpdouble {
        return s.Jac(0, 0) - data.Ref("source");
    };

    auto phys_loss = std::make_shared<CPhysicsLoss>(
        "dy_dx_source", 
        net.GetInputVars(), 
        net.GetOutputVars(), 
        std::vector<std::string>{"source"}, 
        std::vector<CPhysicsEquation>{eq}
    );

    trainer.AddPhysicsLoss(phys_loss, "colloc");
    trainer.Build();

    REQUIRE_THROWS_AS(trainer.TrainStep(), std::runtime_error);
}

TEST_CASE("CMLPTrainer MakeZeroPhysicsData helper", "[CMLPTrainer]") {
    std::vector<std::size_t> arch = {1, 2, 1};
    CNeuralNetwork net(arch);
    net.SetInputName(0, "x");
    net.SetOutputName(0, "y");
    net.RandomWeights();

    CAdam adam(1e-3);
    AnnealerConfig annealer_cfg;
    annealer_cfg.n_data_terms = 1;
    TrainerConfig trainer_cfg;
    trainer_cfg.max_epochs = 1;
    trainer_cfg.verbose = false;

    CMLPTrainer trainer(net, adam, annealer_cfg, trainer_cfg);
    trainer.SetCollocationPoints("colloc", {{0.5}, {1.0}});

    CPhysicsEquation eq;
    eq.name = "dy_dx_source";
    eq.input_names = {"x"};
    eq.output_names = {"y"};
    eq.requires_jacobian = true;
    eq.residual = [](const PhysicsState& s, const PhysicsData& data) -> mlpdouble {
        return s.Jac(0, 0) - data.Ref("source");
    };

    auto phys_loss = std::make_shared<CPhysicsLoss>(
        "dy_dx_source", 
        net.GetInputVars(), 
        net.GetOutputVars(), 
        std::vector<std::string>{"source"}, 
        std::vector<CPhysicsEquation>{eq}
    );

    trainer.AddPhysicsLoss(phys_loss, "colloc");
    
    auto zero_data = trainer.MakeZeroPhysicsData("dy_dx_source");
    REQUIRE(zero_data.size() == 2);
    REQUIRE(zero_data[0].size() == 1);
    REQUIRE_EQUAL_TOL(zero_data[0][0], 0.0, 1e-12);

    trainer.SetPhysicsData("dy_dx_source", zero_data);
    trainer.Build();

    REQUIRE_NOTHROW(trainer.TrainStep());
}