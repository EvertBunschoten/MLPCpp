#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <string>

#include "CAdam.hpp"
#include "variable_def.hpp"

// some helpers 
static int g_passed = 0;
static int g_failed = 0;
static int g_total  = 0;

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "  FAILED: " << (msg) \
                      << "  [" << __FILE__ << ":" << __LINE__ << "]" << std::endl; \
            return false; \
        } \
    } while (0)


#define TEST_ASSERT_NEAR(a, b, tol, msg) \
    do { \
        double _a = static_cast<double>(a); \
        double _b = static_cast<double>(b); \
        if (std::abs(_a - _b) > static_cast<double>(tol)) { \
            std::cerr << "  FAILED: " << (msg) \
                      << "  (expected " << _b << ", got " << _a << ")" \
                      << "  [" << __FILE__ << ":" << __LINE__ << "]" << std::endl; \
            return false; \
        } \
    } while (0)

#define RUN_TEST(fn) \
    do { \
        std::cout << "  " << #fn << " ... " << std::flush; \
        ++g_total; \
        if (fn()) { std::cout << "PASSED" << std::endl; ++g_passed; } \
        else      { std::cout << "FAILED" << std::endl; ++g_failed; } \
    } while (0)

//-------------
bool test_default_constructor() {
    CAdam adam;
    TEST_ASSERT_NEAR(adam.getLearningRate(), 1e-3, 1e-12, "Default Learning Rate");
    TEST_ASSERT_NEAR(adam.getBeta1(), 0.9, 1e-12, "Default beta1");
    TEST_ASSERT_NEAR(adam.getBeta2(), 0.999, 1e-12, "Default beta2");
    TEST_ASSERT_NEAR(adam.getEpsilon(), 1e-8,  1e-12, "Default epsilon");
    TEST_ASSERT(adam.getTimeStep() == 0, "Default timestep should be 0");
    TEST_ASSERT(adam.getStateSize() == 0, "Default state size should be 0");
    return true;
}


bool test_custom_constructor() {
    CAdam adam(0.01, 0.8, 0.99, 1e-7);
    TEST_ASSERT_NEAR(adam.getLearningRate(), 0.01,  1e-12, "Custom lr");
    TEST_ASSERT_NEAR(adam.getBeta1(), 0.8, 1e-12, "Custom beta1");
    TEST_ASSERT_NEAR(adam.getBeta2(), 0.99, 1e-12, "Custom beta2");
    TEST_ASSERT_NEAR(adam.getEpsilon(), 1e-7, 1e-12, "Custom epsilon");
    return true;
}

bool test_initialize() {
    CAdam adam;
    adam.initialize(10);
    TEST_ASSERT(adam.getStateSize() == 10, "State size after init");
    TEST_ASSERT(adam.getTimeStep()  == 0,  "Timestep should be 0 after init");

    const auto& m = adam.getFirstMoment();
    const auto& v = adam.getSecondMoment();

    for (std::size_t i = 0; i < 10; i++) {
        TEST_ASSERT_NEAR(m[i], 0.0, 1e-12, "m should be zero initially");
        TEST_ASSERT_NEAR(v[i], 0.0, 1e-12, "v should be zero initially");
    }

    return true;
}


bool test_auto_initialize() {
    CAdam adam;
    std::vector<mlpdouble> params = {1.0, 2.0, 3.0};
    std::vector<mlpdouble> grads = {0.1, 0.2, 0.3};

    adam.step(params, grads);

    TEST_ASSERT(adam.getStateSize() == 3, "Should auto init to size 3");
    TEST_ASSERT(adam.getTimeStep() == 1, "Timestep should be 1");

    return true;
}


bool test_step_size_mismatch_throws() {
    CAdam adam;
    std::vector<mlpdouble> params = {1.0, 2.0};
    std::vector<mlpdouble> grads = {0.5};

    bool threw = false;
    try { adam.step(params, grads); }
    catch (const std::runtime_error&) { threw = true; }

    TEST_ASSERT(threw, "Should throw std::runtime_error on size mismatch");
    return true;
}

bool test_single_step() {
    const mlpdouble lr = 1e-3, b1 = 0.9, b2 = 0.999, eps = 1e-8;
    CAdam adam(lr, b1, b2, eps);

    std::vector<mlpdouble> params = {1.0};
    std::vector<mlpdouble> grads = {0.5};
    adam.step(params, grads);

    mlpdouble m = (1 - b1) * 0.5;
    mlpdouble v = (1 - b2) * 0.5 * 0.5;

    mlpdouble mh = m / (1 - b1);
    mlpdouble vh = v / (1 - b2);

    mlpdouble expected_p = 1 - lr * mh / (std::sqrt(static_cast<double>(vh)) + eps);

    TEST_ASSERT_NEAR(adam.getFirstMoment()[0],  m,  1e-12, "m after step 1");
    TEST_ASSERT_NEAR(adam.getSecondMoment()[0], v,  1e-12, "v after step 1");
    TEST_ASSERT_NEAR(params[0], expected_p, 1e-6, "param after step 1");
    TEST_ASSERT(adam.getTimeStep() == 1, "timestep == 1");
    return true;
}

bool test_multiple_steps() {
    const mlpdouble lr = 1e-3, b1 = 0.9, b2 = 0.999, eps = 1e-8;
    CAdam adam(lr, b1, b2, eps);

    std::vector<mlpdouble> params = {1.0, -1.0};
    
    // Step 1
    std::vector<mlpdouble> grads1 = {0.5, -0.2};
    adam.step(params, grads1);

    mlpdouble m1_0 = (1.0 - b1) * grads1[0]; // 0.1 * 0.5 = 0.05
    mlpdouble v1_0 = (1.0 - b2) * grads1[0] * grads1[0]; // 0.001 * 0.25 = 0.00025
    mlpdouble mh1_0 = m1_0 / (1.0 - b1); // 0.5
    mlpdouble vh1_0 = v1_0 / (1.0 - b2); // 0.25
    mlpdouble expected_p1_0 = 1.0 - lr * mh1_0 / (std::sqrt(static_cast<double>(vh1_0)) + eps);

    mlpdouble m1_1 = (1.0 - b1) * grads1[1]; // 0.1 * -0.2 = -0.02
    mlpdouble v1_1 = (1.0 - b2) * grads1[1] * grads1[1]; // 0.001 * 0.04 = 0.00004
    mlpdouble mh1_1 = m1_1 / (1.0 - b1); // -0.2
    mlpdouble vh1_1 = v1_1 / (1.0 - b2); // 0.04
    mlpdouble expected_p1_1 = -1.0 - lr * mh1_1 / (std::sqrt(static_cast<double>(vh1_1)) + eps);

    // Verify Step 1 State
    TEST_ASSERT_NEAR(adam.getFirstMoment()[0], m1_0, 1e-12, "m[0] after step 1");
    TEST_ASSERT_NEAR(adam.getSecondMoment()[0], v1_0, 1e-12, "v[0] after step 1");
    TEST_ASSERT_NEAR(params[0], expected_p1_0, 1e-6, "params[0] after step 1");
    
    TEST_ASSERT_NEAR(adam.getFirstMoment()[1], m1_1, 1e-12, "m[1] after step 1");
    TEST_ASSERT_NEAR(adam.getSecondMoment()[1], v1_1, 1e-12, "v[1] after step 1");
    TEST_ASSERT_NEAR(params[1], expected_p1_1, 1e-6, "params[1] after step 1");
    
    TEST_ASSERT(adam.getTimeStep() == 1, "timestep == 1 after step 1");

    // step2
    std::vector<mlpdouble> grads2 = {0.3, 0.4};
    adam.step(params, grads2);

    mlpdouble m2_0 = b1 * m1_0 + (1.0 - b1) * grads2[0]; // 0.9 * 0.05 + 0.1 * 0.3 = 0.075
    mlpdouble v2_0 = b2 * v1_0 + (1.0 - b2) * grads2[0] * grads2[0]; // 0.999 * 0.00025 + 0.001 * 0.09 = 0.00033975
    mlpdouble mh2_0 = m2_0 / (1.0 - b1 * b1); // 0.075 / 0.19
    mlpdouble vh2_0 = v2_0 / (1.0 - b2 * b2); // 0.00033975 / 0.001999
    mlpdouble expected_p2_0 = expected_p1_0 - lr * mh2_0 / (std::sqrt(static_cast<double>(vh2_0)) + eps);

    mlpdouble m2_1 = b1 * m1_1 + (1.0 - b1) * grads2[1]; // 0.9 * -0.02 + 0.1 * 0.4 = 0.022
    mlpdouble v2_1 = b2 * v1_1 + (1.0 - b2) * grads2[1] * grads2[1]; // 0.999 * 0.00004 + 0.001 * 0.16 = 0.00019996
    mlpdouble mh2_1 = m2_1 / (1.0 - b1 * b1); // 0.022 / 0.19
    mlpdouble vh2_1 = v2_1 / (1.0 - b2 * b2); // 0.00019996 / 0.001999
    mlpdouble expected_p2_1 = expected_p1_1 - lr * mh2_1 / (std::sqrt(static_cast<double>(vh2_1)) + eps);

    // Verify Step 2 State
    TEST_ASSERT_NEAR(adam.getFirstMoment()[0], m2_0, 1e-12, "m[0] after step 2");
    TEST_ASSERT_NEAR(adam.getSecondMoment()[0], v2_0, 1e-12, "v[0] after step 2");
    TEST_ASSERT_NEAR(params[0], expected_p2_0, 1e-6, "params[0] after step 2");
    
    TEST_ASSERT_NEAR(adam.getFirstMoment()[1], m2_1, 1e-12, "m[1] after step 2");
    TEST_ASSERT_NEAR(adam.getSecondMoment()[1], v2_1, 1e-12, "v[1] after step 2");
    TEST_ASSERT_NEAR(params[1], expected_p2_1, 1e-6, "params[1] after step 2");

    TEST_ASSERT(adam.getTimeStep() == 2, "timestep == 2 after step 2");
    
    return true;
}

bool test_zero_gradients() {
    CAdam adam;
    std::vector<mlpdouble> p = {1.0, 2.0};
    adam.step(p, {0.0, 0.0});

    TEST_ASSERT_NEAR(adam.getFirstMoment()[0],  0.0, 1e-12, "m with zero grad");
    TEST_ASSERT_NEAR(adam.getSecondMoment()[0], 0.0, 1e-12, "v with zero grad");
    TEST_ASSERT_NEAR(p[0], 1.0, 1e-12, "p[0] unchanged");
    TEST_ASSERT_NEAR(p[1], 2.0, 1e-12, "p[1] unchanged");
    return true;
}

bool test_negative_gradients() {
    CAdam adam(0.01);
    std::vector<mlpdouble> p = {1.0};
    adam.step(p, {-0.5});

    TEST_ASSERT(static_cast<double>(p[0]) > 1.0, "Param should increase with negative gradient");
    TEST_ASSERT(static_cast<double>(adam.getFirstMoment()[0]) < 0.0, "m should be negative");
    return true;
}

bool test_reset() {
    CAdam adam;
    std::vector<mlpdouble> p = {1.0}, g = {0.5};
    adam.step(p, g);
    adam.step(p, g);
    TEST_ASSERT(adam.getTimeStep() == 2, "t should be 2");

    adam.reset();
    TEST_ASSERT(adam.getTimeStep() == 0, "t should be 0 after reset");
    TEST_ASSERT(adam.getStateSize() == 0, "state size 0 after reset");
    TEST_ASSERT(adam.getFirstMoment().empty(),  "m empty after reset");
    TEST_ASSERT(adam.getSecondMoment().empty(), "v empty after reset");
    return true;
}

bool test_convergence_quadratic() {
    // Minimize f(x) = x^2, gradient is 2x
    CAdam adam(0.1);
    std::vector<mlpdouble> p = {5.0};

    for (int i = 0; i < 200; ++i) {
        std::vector<mlpdouble> g = {2.0 * p[0]};
        adam.step(p, g);
    }
    
    TEST_ASSERT(std::abs(static_cast<double>(p[0])) < 0.05,
                "Should converge near 0 (got " + std::to_string(static_cast<double>(p[0])) + ")");
    return true;
}

bool test_determinism() {
    auto run = []() {
        CAdam adam(1e-3, 0.9, 0.999, 1e-8);
        std::vector<mlpdouble> p = {1.0, -1.0};
        std::vector<std::vector<mlpdouble>> gs = {
            { 0.5, -0.3}, { 0.2, -0.1},
            {-0.4,  0.6}, { 0.1,  0.1}
        };
        for (auto& g : gs) adam.step(p, g);
        return p;
    };

    auto r1 = run(), r2 = run();
    TEST_ASSERT_NEAR(r1[0], r2[0], 1e-12, "Determinism p[0]");
    TEST_ASSERT_NEAR(r1[1], r2[1], 1e-12, "Determinism p[1]");
    return true;
}


// main
int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  CAdam Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    RUN_TEST(test_default_constructor);
    RUN_TEST(test_custom_constructor);
    RUN_TEST(test_initialize);
    RUN_TEST(test_auto_initialize);
    RUN_TEST(test_step_size_mismatch_throws);
    RUN_TEST(test_single_step);
    RUN_TEST(test_multiple_steps);
    RUN_TEST(test_zero_gradients);
    RUN_TEST(test_negative_gradients);
    RUN_TEST(test_reset);
    RUN_TEST(test_convergence_quadratic);
    RUN_TEST(test_determinism);

    std::cout << "========================================" << std::endl;
    std::cout << "  Results: " << g_passed << " / " << g_total << " passed";
    if (g_failed > 0) std::cout << "  (" << g_failed << " FAILED)";
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;

    return g_failed > 0 ? 1 : 0;
}