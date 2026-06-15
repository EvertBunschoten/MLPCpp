#define CATCH_CONFIG_MAIN 

#include "catch.hpp"

#include <vector>
#include <stdexcept>
#include <cmath>

#include "CGradientAnnealer.hpp" 
#include "variable_def.hpp"


// GradStats Tests
TEST_CASE("GradStats handles empty vectors", "[GradStats]") {
    std::vector<mlpdouble> g;
    GradStats stats = GradStats::from_grads(g);
    CHECK(stats.max_abs == Approx(0.0));
    CHECK(stats.mean_abs == Approx(0.0));
}

TEST_CASE("GradStats handles single element", "[GradStats]") {
    std::vector<mlpdouble> g = {-5.0};
    GradStats stats = GradStats::from_grads(g);
    CHECK(stats.max_abs == Approx(5.0));
    CHECK(stats.mean_abs == Approx(5.0));
}

TEST_CASE("GradStats handles multiple elements", "[GradStats]") {
    std::vector<mlpdouble> g = {1.0, -2.0, 3.0, -4.0};
    GradStats stats = GradStats::from_grads(g);
    CHECK(stats.max_abs == Approx(4.0));
    CHECK(stats.mean_abs == Approx((1.0 + 2.0 + 3.0 + 4.0) / 4.0));
}

// Constructor Tests
TEST_CASE("CGradientAnnealer constructor validates config", "[CGradientAnnealer]") {
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
