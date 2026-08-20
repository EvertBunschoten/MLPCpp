#define CATCH_CONFIG_MAIN
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "CGradientAnnealer.hpp"
#include "variable_def.hpp"

// GradStats Tests
TEST_CASE("GradStats handles empty vectors", "[GradStats]") {
  std::vector<mlpdouble> g;
  GradStats stats = GradStats::from_grads(g);
  CHECK(stats.max_abs == Catch::Approx(0.0));
  CHECK(stats.mean_abs == Catch::Approx(0.0));
}

TEST_CASE("GradStats handles single element", "[GradStats]") {
  std::vector<mlpdouble> g = {-5.0};
  GradStats stats = GradStats::from_grads(g);
  CHECK(stats.max_abs == Catch::Approx(5.0));
  CHECK(stats.mean_abs == Catch::Approx(5.0));
}

TEST_CASE("GradStats handles multiple elements", "[GradStats]") {
  std::vector<mlpdouble> g = {1.0, -2.0, 3.0, -4.0};
  GradStats stats = GradStats::from_grads(g);
  CHECK(stats.max_abs == Catch::Approx(4.0));
  CHECK(stats.mean_abs == Catch::Approx((1.0 + 2.0 + 3.0 + 4.0) / 4.0));
}

// Constructor Tests
TEST_CASE("CGradientAnnealer constructor validates config",
          "[CGradientAnnealer]") {
  SECTION("Valid config does not throw") {
    AnnealerConfig cfg;
    cfg.n_data_terms = 2;
    cfg.alpha = 0.9;
    REQUIRE_NOTHROW(CGradientAnnealer{cfg});
  }

  SECTION("Zero data terms throws invalid_argument") {
    AnnealerConfig cfg;
    cfg.n_data_terms = 0;
    cfg.alpha = 0.9;
    REQUIRE_THROWS_AS(CGradientAnnealer{cfg}, std::invalid_argument);
  }

  SECTION("Alpha = 0.0 throws invalid_argument") {
    AnnealerConfig cfg;
    cfg.n_data_terms = 1;
    cfg.alpha = 0.0;
    REQUIRE_THROWS_AS(CGradientAnnealer{cfg}, std::invalid_argument);
  }

  SECTION("Alpha = 1.0 throws invalid_argument") {
    AnnealerConfig cfg;
    cfg.n_data_terms = 1;
    cfg.alpha = 1.0;
    REQUIRE_THROWS_AS(CGradientAnnealer{cfg}, std::invalid_argument);
  }
}

TEST_CASE("CGradientAnnealer basic update logic", "[CGradientAnnealer]") {
  AnnealerConfig cfg;
  cfg.n_data_terms = 1;
  cfg.alpha = 0.5;
  cfg.lambda_init = 1.0;
  cfg.lambda_min = 0.1;
  cfg.lambda_max = 10.0;

  CGradientAnnealer annealer(cfg);

  GradStats ref{2.0, 1.0};
  GradStats data{1.0, 0.5};

  annealer.update(ref, {data});

  // lambda_hat = 2.0 / 0.5 = 4.0
  CHECK(annealer.get_lambda_hat(0) == Catch::Approx(4.0));
  // lambda = (1 - 0.5) * 1.0 + 0.5 * 4.0 = 2.5
  CHECK(annealer.get_lambda(0) == Catch::Approx(2.5));
  CHECK(annealer.step() == 1);
}

TEST_CASE("CGradientAnnealer handles zero data mean", "[CGradientAnnealer]") {
  AnnealerConfig cfg;
  cfg.n_data_terms = 1;
  cfg.alpha = 0.9;
  cfg.lambda_init = 1.0;
  cfg.lambda_min = 0.1;
  cfg.lambda_max = 10.0;

  CGradientAnnealer annealer(cfg);

  GradStats ref{2.0, 1.0};
  GradStats data{1.0, 1e-16}; // mean_abs < 1e-15 threshold

  annealer.update(ref, {data});

  CHECK(annealer.get_lambda_hat(0) == Catch::Approx(cfg.lambda_max));
  // lambda = 0.1 * 1.0 + 0.9 * 10.0 = 90.1
  CHECK(annealer.get_lambda(0) == Catch::Approx(9.1));
}

TEST_CASE("CGradientAnnealer enforces clamping", "[CGradientAnnealer]") {
  AnnealerConfig cfg;
  cfg.n_data_terms = 1;
  cfg.alpha = 0.9;
  cfg.lambda_init = 1.0;
  cfg.lambda_min = 0.5;
  cfg.lambda_max = 2.0;

  CGradientAnnealer annealer(cfg);

  SECTION("Upper clamp") {
    GradStats ref{10.0, 1.0};
    GradStats data{1.0, 1.0};
    annealer.update(ref, {data});
    // lambda = 0.1 * 1.0 + 0.9 * 10.0 = 9.1 -> clamped to 2.0
    CHECK(annealer.get_lambda(0) == Catch::Approx(2.0));
  }

  SECTION("Lower clamp") {
    annealer.reset();
    GradStats ref{0.01, 0.0};
    GradStats data{1.0, 1.0};
    annealer.update(ref, {data});
    // lambda = 0.1 * 1.0 + 0.9 * 0.01 = 0.109 -> clamped to 0.5
    CHECK(annealer.get_lambda(0) == Catch::Approx(0.5));
  }
}

TEST_CASE("CGradientAnnealer handles multiple terms and EMA accumulation",
          "[CGradientAnnealer]") {
  AnnealerConfig cfg;
  cfg.n_data_terms = 2;
  cfg.alpha = 0.5;
  cfg.lambda_init = 1.0;
  cfg.lambda_max = 100.0;

  CGradientAnnealer annealer(cfg);

  GradStats ref{4.0, 2.0};
  GradStats data1{2.0, 1.0}; // lambda_hat = 4.0
  GradStats data2{1.0, 2.0}; // lambda_hat = 2.0

  // step1
  annealer.update(ref, {data1, data2});
  CHECK(annealer.get_lambda(0) == Catch::Approx(2.5));
  CHECK(annealer.get_lambda(1) == Catch::Approx(1.5));

  // step2
  annealer.update(ref, {data1, data2});
  CHECK(annealer.get_lambda(0) == Catch::Approx(3.25));
  CHECK(annealer.get_lambda(1) == Catch::Approx(1.75));
  CHECK(annealer.step() == 2);
}

TEST_CASE("CGradientAnnealer reset and accessors", "[CGradientAnnealer]") {
  AnnealerConfig cfg;
  cfg.n_data_terms = 3;
  cfg.alpha = 0.8;
  cfg.lambda_init = 2.0;

  CGradientAnnealer annealer(cfg);

  CHECK(annealer.n_data_terms() == 3);
  CHECK(annealer.step() == 0);
  CHECK(annealer.config().alpha == Catch::Approx(0.8));

  auto lambdas = annealer.lambdas();
  REQUIRE(lambdas.size() == 3);
  for (double l : lambdas) {
    CHECK(l == Catch::Approx(2.0));
  }

  // Update and then reset
  GradStats ref{2.0, 1.0};
  GradStats data{1.0, 0.5};
  annealer.update(ref, {data, data, data});

  CHECK(annealer.step() == 1);

  annealer.reset();
  CHECK(annealer.step() == 0);
  for (std::size_t i = 0; i < 3; ++i) {
    CHECK(annealer.get_lambda(i) == Catch::Approx(cfg.lambda_init));
    CHECK(annealer.get_lambda_hat(i) == Catch::Approx(cfg.lambda_init));
  }
}
