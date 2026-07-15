#pragma once
#include "catch.hpp"

#define REQUIRE_EQUAL_TOL(a, b, tol) \
    REQUIRE(static_cast<double>(a) == Approx(static_cast<double>(b)).margin(tol))
