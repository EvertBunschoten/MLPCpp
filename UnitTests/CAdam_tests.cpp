#define CATCH_CONFIG_MAIN

#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <string>

#include "CAdam.hpp"
#include "variable_def.hpp"
#include "testing_macros.hpp"

// ============================================================
//  Unit Tests
// ============================================================

TEST_CASE("CAdam default constructor", "[CAdam]") {
    CAdam adam;
    REQUIRE_EQUAL_TOL(adam.getLearningRate(), 1e-3, 1e-12);
    REQUIRE_EQUAL_TOL(adam.getBeta1(), 0.9, 1e-12);
    REQUIRE_EQUAL_TOL(adam.getBeta2(), 0.999, 1e-12);
    REQUIRE_EQUAL_TOL(adam.getEpsilon(), 1e-8, 1e-12);
    REQUIRE(adam.getTimeStep() == 0);
    REQUIRE(adam.getNumberOfVariables() == 0);
}

TEST_CASE("CAdam custom constructor", "[CAdam]") {
    CAdam adam(0.01, 0.8, 0.99, 1e-7);
    REQUIRE_EQUAL_TOL(adam.getLearningRate(), 0.01, 1e-12);
    REQUIRE_EQUAL_TOL(adam.getBeta1(), 0.8, 1e-12);
    REQUIRE_EQUAL_TOL(adam.getBeta2(), 0.99, 1e-12);
    REQUIRE_EQUAL_TOL(adam.getEpsilon(), 1e-7, 1e-12);
}

TEST_CASE("CAdam constructor invalid learning rate", "[CAdam]") {
    REQUIRE_THROWS_AS(CAdam(-1e-3, 0.9, 0.999, 1e-8), std::invalid_argument);
}

TEST_CASE("CAdam constructor invalid beta1", "[CAdam]") {
    REQUIRE_THROWS_AS(CAdam(1e-3, 1.0, 0.999, 1e-8), std::invalid_argument);
    REQUIRE_THROWS_AS(CAdam(1e-3, -0.1, 0.999, 1e-8), std::invalid_argument);
}

TEST_CASE("CAdam constructor invalid beta2", "[CAdam]") {
    REQUIRE_THROWS_AS(CAdam(1e-3, 0.9, 1.0, 1e-8), std::invalid_argument);
    REQUIRE_THROWS_AS(CAdam(1e-3, 0.9, -0.1, 1e-8), std::invalid_argument);
}

TEST_CASE("CAdam constructor invalid epsilon", "[CAdam]") {
    REQUIRE_THROWS_AS(CAdam(1e-3, 0.9, 0.999, 0.0), std::invalid_argument);
    REQUIRE_THROWS_AS(CAdam(1e-3, 0.9, 0.999, -1e-8), std::invalid_argument);
}

TEST_CASE("CAdam initialize", "[CAdam]") {
    CAdam adam;
    adam.initialize(10);
    REQUIRE(adam.getNumberOfVariables() == 10);
    REQUIRE(adam.getTimeStep() == 0);

    const auto& m = adam.getFirstMoment();
    const auto& v = adam.getSecondMoment();

    for (std::size_t i = 0; i < 10; i++) {
        REQUIRE_EQUAL_TOL(m[i], 0.0, 1e-12);
        REQUIRE_EQUAL_TOL(v[i], 0.0, 1e-12);
    }
}

TEST_CASE("CAdam auto initialize", "[CAdam]") {
    CAdam adam;
    std::vector<mlpdouble> params = {1.0, 2.0, 3.0};
    std::vector<mlpdouble> grads = {0.1, 0.2, 0.3};

    adam.step(params, grads);

    REQUIRE(adam.getNumberOfVariables() == 3);
    REQUIRE(adam.getTimeStep() == 1);
}

TEST_CASE("CAdam step size mismatch throws", "[CAdam]") {
    CAdam adam;
    std::vector<mlpdouble> params = {1.0, 2.0};
    std::vector<mlpdouble> grads = {0.5};

    REQUIRE_THROWS_AS(adam.step(params, grads), std::runtime_error);
}

TEST_CASE("CAdam single step", "[CAdam]") {
    const mlpdouble lr = 1e-3, b1 = 0.9, b2 = 0.999, eps = 1e-8;
    CAdam adam(lr, b1, b2, eps);

    std::vector<mlpdouble> params = {1.0};
    std::vector<mlpdouble> grads = {0.5};
    adam.step(params, grads);

    mlpdouble first_moment = (1 - b1) * 0.5;
    mlpdouble second_moment = (1 - b2) * 0.5 * 0.5;

    mlpdouble first_moment_h = first_moment / (1 - b1);
    mlpdouble second_moment_h = second_moment / (1 - b2);

    mlpdouble expected_p = 1 - lr * first_moment_h / (std::sqrt(static_cast<double>(second_moment_h)) + eps);

    REQUIRE_EQUAL_TOL(adam.getFirstMoment()[0], first_moment, 1e-12);
    REQUIRE_EQUAL_TOL(adam.getSecondMoment()[0], second_moment, 1e-12);
    REQUIRE_EQUAL_TOL(params[0], expected_p, 1e-6);
    REQUIRE(adam.getTimeStep() == 1);
}

TEST_CASE("CAdam multiple steps", "[CAdam]") {
    const mlpdouble lr = 1e-3, b1 = 0.9, b2 = 0.999, eps = 1e-8;
    CAdam adam(lr, b1, b2, eps);

    std::vector<mlpdouble> params = {1.0, -1.0};
    
    // Step 1
    std::vector<mlpdouble> grads1 = {0.5, -0.2};
    adam.step(params, grads1);

    mlpdouble m1_0 = (1.0 - b1) * grads1[0]; 
    mlpdouble v1_0 = (1.0 - b2) * grads1[0] * grads1[0]; 
    mlpdouble mh1_0 = m1_0 / (1.0 - b1); 
    mlpdouble vh1_0 = v1_0 / (1.0 - b2); 
    mlpdouble expected_p1_0 = 1.0 - lr * mh1_0 / (std::sqrt(static_cast<double>(vh1_0)) + eps);

    mlpdouble m1_1 = (1.0 - b1) * grads1[1]; 
    mlpdouble v1_1 = (1.0 - b2) * grads1[1] * grads1[1]; 
    mlpdouble mh1_1 = m1_1 / (1.0 - b1); 
    mlpdouble vh1_1 = v1_1 / (1.0 - b2); 
    mlpdouble expected_p1_1 = -1.0 - lr * mh1_1 / (std::sqrt(static_cast<double>(vh1_1)) + eps);

    REQUIRE_EQUAL_TOL(adam.getFirstMoment()[0], m1_0, 1e-12);
    REQUIRE_EQUAL_TOL(adam.getSecondMoment()[0], v1_0, 1e-12);
    REQUIRE_EQUAL_TOL(params[0], expected_p1_0, 1e-6);
    
    REQUIRE_EQUAL_TOL(adam.getFirstMoment()[1], m1_1, 1e-12);
    REQUIRE_EQUAL_TOL(adam.getSecondMoment()[1], v1_1, 1e-12);
    REQUIRE_EQUAL_TOL(params[1], expected_p1_1, 1e-6);
    
    REQUIRE(adam.getTimeStep() == 1);

    // step2
    std::vector<mlpdouble> grads2 = {0.3, 0.4};
    adam.step(params, grads2);

    mlpdouble m2_0 = b1 * m1_0 + (1.0 - b1) * grads2[0]; 
    mlpdouble v2_0 = b2 * v1_0 + (1.0 - b2) * grads2[0] * grads2[0]; 
    mlpdouble mh2_0 = m2_0 / (1.0 - b1 * b1); 
    mlpdouble vh2_0 = v2_0 / (1.0 - b2 * b2); 
    mlpdouble expected_p2_0 = expected_p1_0 - lr * mh2_0 / (std::sqrt(static_cast<double>(vh2_0)) + eps);

    mlpdouble m2_1 = b1 * m1_1 + (1.0 - b1) * grads2[1]; 
    mlpdouble v2_1 = b2 * v1_1 + (1.0 - b2) * grads2[1] * grads2[1]; 
    mlpdouble mh2_1 = m2_1 / (1.0 - b1 * b1); 
    mlpdouble vh2_1 = v2_1 / (1.0 - b2 * b2); 
    mlpdouble expected_p2_1 = expected_p1_1 - lr * mh2_1 / (std::sqrt(static_cast<double>(vh2_1)) + eps);

    REQUIRE_EQUAL_TOL(adam.getFirstMoment()[0], m2_0, 1e-12);
    REQUIRE_EQUAL_TOL(adam.getSecondMoment()[0], v2_0, 1e-12);
    REQUIRE_EQUAL_TOL(params[0], expected_p2_0, 1e-6);
    
    REQUIRE_EQUAL_TOL(adam.getFirstMoment()[1], m2_1, 1e-12);
    REQUIRE_EQUAL_TOL(adam.getSecondMoment()[1], v2_1, 1e-12);
    REQUIRE_EQUAL_TOL(params[1], expected_p2_1, 1e-6);

    REQUIRE(adam.getTimeStep() == 2);
}

TEST_CASE("CAdam zero gradients", "[CAdam]") {
    CAdam adam;
    std::vector<mlpdouble> params = {1.0, 2.0};
    adam.step(params, {0.0, 0.0});

    REQUIRE_EQUAL_TOL(adam.getFirstMoment()[0], 0.0, 1e-12);
    REQUIRE_EQUAL_TOL(adam.getSecondMoment()[0], 0.0, 1e-12);
    REQUIRE_EQUAL_TOL(params[0], 1.0, 1e-12);
    REQUIRE_EQUAL_TOL(params[1], 2.0, 1e-12);
}

TEST_CASE("CAdam negative gradients", "[CAdam]") {
    CAdam adam(0.01);
    std::vector<mlpdouble> params = {1.0};
    adam.step(params, {-0.5});

    REQUIRE(static_cast<double>(params[0]) > 1.0);
    REQUIRE(static_cast<double>(adam.getFirstMoment()[0]) < 0.0);
}

TEST_CASE("CAdam reset", "[CAdam]") {
    CAdam adam;
    std::vector<mlpdouble> params = {1.0}, grads = {0.5};
    adam.step(params, grads);
    adam.step(params, grads);
    REQUIRE(adam.getTimeStep() == 2);

    adam.reset();
    REQUIRE(adam.getTimeStep() == 0);
    REQUIRE(adam.getNumberOfVariables() == 0);
    REQUIRE(adam.getFirstMoment().empty());
    REQUIRE(adam.getSecondMoment().empty());
}

TEST_CASE("CAdam convergence quadratic", "[CAdam]") {
    // Minimize f(x) = x^2, gradient is 2x
    CAdam adam(0.1);
    std::vector<mlpdouble> params = {5.0};

    for (int i = 0; i < 200; ++i) {
        std::vector<mlpdouble> g = {2.0 * params[0]};
        adam.step(params, g);
    }
    
    REQUIRE(std::abs(static_cast<double>(params[0])) < 0.05);
}

TEST_CASE("CAdam determinism", "[CAdam]") {
    auto run = []() {
        CAdam adam(1e-3, 0.9, 0.999, 1e-8);
        std::vector<mlpdouble> params = {1.0, -1.0};
        std::vector<std::vector<mlpdouble>> grads = {
            { 0.5, -0.3}, { 0.2, -0.1},
            {-0.4,  0.6}, { 0.1,  0.1}
        };
        for (auto& grad : grads) adam.step(params, grad);
        return params;
    };

    auto r1 = run(), r2 = run();
    REQUIRE_EQUAL_TOL(r1[0], r2[0], 1e-12);
    REQUIRE_EQUAL_TOL(r1[1], r2[1], 1e-12);
}