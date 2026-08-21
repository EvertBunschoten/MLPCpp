#define MLP_CUSTOM_TYPE codi::RealReverse
#include "codi.hpp"

#define CATCH_CONFIG_MAIN
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "CBaseLoss.hpp"
#include "CPhysicsLoss.hpp"

using namespace MLPToolbox;

// ============================================================
// Helpers
// ============================================================

#define REQUIRE_EQUAL_TOL(a, b, tol)                                           \
  REQUIRE(to_double(a) == Catch::Approx(to_double(b)).margin(tol))

// Layout: jacobian[input][output]
static mlpdouble **AllocateJacobian(std::size_t n_inputs,
                                    std::size_t n_outputs) {
  mlpdouble **jac = new mlpdouble *[n_inputs];
  for (std::size_t i = 0; i < n_inputs; ++i) {
    jac[i] = new mlpdouble[n_outputs];
    for (std::size_t o = 0; o < n_outputs; ++o) {
      jac[i][o] = 0.0;
    }
  }
  return jac;
}

// Layout: hessian[input_i][input_j][output]
static mlpdouble ***AllocateHessian(std::size_t n_inputs,
                                    std::size_t n_outputs) {
  mlpdouble ***hess = new mlpdouble **[n_inputs];
  for (std::size_t i = 0; i < n_inputs; ++i) {
    hess[i] = new mlpdouble *[n_inputs];
    for (std::size_t j = 0; j < n_inputs; ++j) {
      hess[i][j] = new mlpdouble[n_outputs];
      for (std::size_t o = 0; o < n_outputs; ++o) {
        hess[i][j][o] = 0.0;
      }
    }
  }
  return hess;
}

static void FreeJacobian(mlpdouble **jac, std::size_t n_inputs) {
  if (!jac)
    return;
  for (std::size_t i = 0; i < n_inputs; ++i)
    delete[] jac[i];
  delete[] jac;
}

static void FreeHessian(mlpdouble ***hess, std::size_t n_inputs) {
  if (!hess)
    return;
  for (std::size_t i = 0; i < n_inputs; ++i) {
    for (std::size_t j = 0; j < n_inputs; ++j)
      delete[] hess[i][j];
    delete[] hess[i];
  }
  delete[] hess;
}

static PredictionResult MakePrediction(std::vector<mlpdouble> inputs,
                                       std::vector<mlpdouble> outputs,
                                       mlpdouble **jac = nullptr,
                                       mlpdouble ***hess = nullptr) {
  PredictionResult pred;
  pred.inputs = std::move(inputs);
  pred.outputs = std::move(outputs);
  pred.jacobian = jac;
  pred.hessian = hess;
  return pred;
}

// ============================================================
// CPhysicsLoss Unit Tests
// ============================================================

TEST_CASE("CPhysicsLoss empty equations throws", "[CPhysicsLoss]") {
  std::vector<std::string> net_in = {"x", "t"};
  std::vector<std::string> net_out = {"u", "p"};
  REQUIRE_THROWS_AS(CPhysicsLoss("test", net_in, net_out, {}, {}),
                    std::invalid_argument);
}

TEST_CASE("CPhysicsLoss sparsity flags", "[CPhysicsLoss]") {
  std::vector<std::string> net_in = {"x", "t"};
  std::vector<std::string> net_out = {"u", "p"};

  CPhysicsEquation eq_algebraic;
  eq_algebraic.name = "alg";
  eq_algebraic.input_names = {"x"};
  eq_algebraic.output_names = {"u"};
  eq_algebraic.requires_jacobian = false;
  eq_algebraic.requires_hessian = false;
  eq_algebraic.residual = [](const PhysicsState &, const PhysicsData &) {
    return mlpdouble(1.0);
  };

  CPhysicsLoss loss_algebraic("algebraic", net_in, net_out, {}, {eq_algebraic});
  REQUIRE(loss_algebraic.RequiresJacobian() == false);
  REQUIRE(loss_algebraic.RequiresHessian() == false);

  CPhysicsEquation eq_jac;
  eq_jac.name = "jac_eq";
  eq_jac.input_names = {"x"};
  eq_jac.output_names = {"u"};
  eq_jac.requires_jacobian = true;
  eq_jac.requires_hessian = false;
  eq_jac.residual = [](const PhysicsState &s, const PhysicsData &) {
    return s.Jac(0, 0);
  };

  CPhysicsLoss loss_jac("jac_only", net_in, net_out, {}, {eq_jac});
  REQUIRE(loss_jac.RequiresJacobian() == true);
  REQUIRE(loss_jac.RequiresHessian() == false);

  CPhysicsEquation eq_hess;
  eq_hess.name = "hess_eq";
  eq_hess.input_names = {"x"};
  eq_hess.output_names = {"u"};
  eq_hess.requires_jacobian = true;
  eq_hess.requires_hessian = true;
  eq_hess.residual = [](const PhysicsState &s, const PhysicsData &) {
    return s.Hess(0, 0, 0);
  };

  CPhysicsLoss loss_hess("hess_full", net_in, net_out, {}, {eq_hess});
  REQUIRE(loss_hess.RequiresJacobian() == true);
  REQUIRE(loss_hess.RequiresHessian() == true);
}

TEST_CASE("CPhysicsLoss missing network variable throws", "[CPhysicsLoss]") {
  std::vector<std::string> net_in = {"x", "t"};
  std::vector<std::string> net_out = {"u", "p"};

  CPhysicsEquation eq;
  eq.name = "bad_input";
  eq.input_names = {"z"};
  eq.output_names = {"u"};
  eq.residual = [](const PhysicsState &, const PhysicsData &) {
    return mlpdouble(0.0);
  };
  REQUIRE_THROWS_AS(CPhysicsLoss("test", net_in, net_out, {}, {eq}),
                    std::invalid_argument);

  eq.name = "bad_output";
  eq.input_names = {"x"};
  eq.output_names = {"v"};
  REQUIRE_THROWS_AS(CPhysicsLoss("test", net_in, net_out, {}, {eq}),
                    std::invalid_argument);
}

TEST_CASE("CPhysicsLoss duplicate physics names throws", "[CPhysicsLoss]") {
  std::vector<std::string> net_in = {"x"};
  std::vector<std::string> net_out = {"u"};
  CPhysicsEquation eq;
  eq.name = "dup_phys";
  eq.input_names = {"x"};
  eq.output_names = {"u"};
  eq.residual = [](const PhysicsState &, const PhysicsData &) {
    return mlpdouble(0.0);
  };
  REQUIRE_THROWS_AS(
      CPhysicsLoss("dup_physics", net_in, net_out, {"fx", "fx"}, {eq}),
      std::invalid_argument);
}

TEST_CASE("CPhysicsLoss correct MSE with per-equation residuals",
          "[CPhysicsLoss]") {
  std::vector<std::string> net_in = {"x"};
  std::vector<std::string> net_out = {"u"};
  int eq1_calls = 0;
  int eq2_calls = 0;

  CPhysicsEquation eq1;
  eq1.name = "eq1";
  eq1.input_names = {"x"};
  eq1.output_names = {"u"};
  eq1.residual = [&eq1_calls](const PhysicsState &, const PhysicsData &) {
    ++eq1_calls;
    return eq1_calls == 1 ? mlpdouble(2.0) : mlpdouble(1.0);
  };

  CPhysicsEquation eq2;
  eq2.name = "eq2";
  eq2.input_names = {"x"};
  eq2.output_names = {"u"};
  eq2.residual = [&eq2_calls](const PhysicsState &, const PhysicsData &) {
    ++eq2_calls;
    return eq2_calls == 1 ? mlpdouble(3.0) : mlpdouble(4.0);
  };

  CPhysicsLoss loss("math_test", net_in, net_out, {}, {eq1, eq2});
  PredictionResult pred1 = MakePrediction({1.0}, {1.0});
  PredictionResult pred2 = MakePrediction({2.0}, {2.0});

  mlpdouble result = loss.Evaluate({pred1, pred2}, {});
  REQUIRE_EQUAL_TOL(result, 7.5, 1e-12);
  REQUIRE_EQUAL_TOL(loss.GetLastLossValue(), 7.5, 1e-12);
}

TEST_CASE("CPhysicsLoss empty predictions", "[CPhysicsLoss]") {
  std::vector<std::string> net_in = {"x"};
  std::vector<std::string> net_out = {"u"};
  CPhysicsEquation eq;
  eq.name = "empty_test";
  eq.input_names = {"x"};
  eq.output_names = {"u"};
  eq.residual = [](const PhysicsState &, const PhysicsData &) {
    return mlpdouble(1.0);
  };

  CPhysicsLoss loss("empty_test", net_in, net_out, {}, {eq});
  mlpdouble result = loss.Evaluate({}, {});
  REQUIRE_EQUAL_TOL(result, 0.0, 1e-12);
  REQUIRE_EQUAL_TOL(loss.GetLastLossValue(), 0.0, 1e-12);
}

TEST_CASE("PhysicsState accessor mapping", "[CPhysicsLoss]") {
  std::vector<std::string> net_in = {"x", "y", "t"};
  std::vector<std::string> net_out = {"u", "v", "p"};
  std::vector<mlpdouble> captured_in, captured_out, captured_jac, captured_hess;

  CPhysicsEquation eq;
  eq.name = "accessor_test";
  eq.input_names = {"t", "x", "y"};
  eq.output_names = {"p", "u"};
  eq.requires_jacobian = true;
  eq.requires_hessian = true;
  eq.residual = [&](const PhysicsState &s, const PhysicsData &) {
    captured_in.push_back(s.In(0));
    captured_in.push_back(s.In(1));
    captured_in.push_back(s.In(2));
    captured_out.push_back(s.Out(0));
    captured_out.push_back(s.Out(1));
    captured_jac.push_back(s.Jac(2, 0));
    captured_jac.push_back(s.Jac(0, 1));
    captured_hess.push_back(s.Hess(2, 0, 0));
    return mlpdouble(0.0);
  };

  CPhysicsLoss loss("accessor_test", net_in, net_out, {}, {eq});
  mlpdouble **jac = AllocateJacobian(3, 3);
  mlpdouble ***hess = AllocateHessian(3, 3);

  jac[1][2] = 3.0;
  jac[2][0] = 5.0;
  hess[1][2][2] = 4.0;

  PredictionResult pred =
      MakePrediction({10.0, 20.0, 30.0}, {100.0, 200.0, 300.0}, jac, hess);
  loss.Evaluate({pred}, {});

  REQUIRE_EQUAL_TOL(captured_in[0], 30.0, 1e-12);
  REQUIRE_EQUAL_TOL(captured_in[1], 10.0, 1e-12);
  REQUIRE_EQUAL_TOL(captured_in[2], 20.0, 1e-12);
  REQUIRE_EQUAL_TOL(captured_out[0], 300.0, 1e-12);
  REQUIRE_EQUAL_TOL(captured_out[1], 100.0, 1e-12);
  REQUIRE_EQUAL_TOL(captured_jac[0], 3.0, 1e-12);
  REQUIRE_EQUAL_TOL(captured_jac[1], 5.0, 1e-12);
  REQUIRE_EQUAL_TOL(captured_hess[0], 4.0, 1e-12);

  FreeJacobian(jac, 3);
  FreeHessian(hess, 3);
}

TEST_CASE("CPhysicsLoss passes per-point physics data", "[CPhysicsLoss]") {
  std::vector<std::string> net_in = {"x", "t"};
  std::vector<std::string> net_out = {"u"};
  std::vector<mlpdouble> captured_source;

  CPhysicsEquation eq;
  eq.name = "rhs";
  eq.input_names = {"x", "t"};
  eq.output_names = {"u"};
  eq.requires_jacobian = true;
  eq.residual = [&](const PhysicsState &s, const PhysicsData &data) {
    captured_source.push_back(data.Ref("source"));
    return s.Jac(1, 0) - data.Ref("source");
  };

  CPhysicsLoss loss("rhs_test", net_in, net_out, {"source"}, {eq});
  mlpdouble **jac1 = AllocateJacobian(2, 1);
  jac1[1][0] = 10.0;
  PredictionResult pred1 = MakePrediction({1.0, 0.0}, {2.0}, jac1);

  mlpdouble **jac2 = AllocateJacobian(2, 1);
  jac2[1][0] = 20.0;
  PredictionResult pred2 = MakePrediction({2.0, 1.0}, {3.0}, jac2);

  std::vector<std::vector<mlpdouble>> physics_data = {{10.0}, {20.0}};
  mlpdouble result = loss.Evaluate({pred1, pred2}, physics_data);

  REQUIRE_EQUAL_TOL(result, 0.0, 1e-12);
  REQUIRE(captured_source.size() == 2);
  REQUIRE_EQUAL_TOL(captured_source[0], 10.0, 1e-12);
  REQUIRE_EQUAL_TOL(captured_source[1], 20.0, 1e-12);

  FreeJacobian(jac1, 2);
  FreeJacobian(jac2, 2);
}

TEST_CASE("CPhysicsLoss rejects physics data row count mismatch",
          "[CPhysicsLoss]") {
  std::vector<std::string> net_in = {"x"};
  std::vector<std::string> net_out = {"u"};
  CPhysicsEquation eq;
  eq.name = "rows";
  eq.input_names = {"x"};
  eq.output_names = {"u"};
  eq.residual = [](const PhysicsState &, const PhysicsData &) {
    return mlpdouble(0.0);
  };

  CPhysicsLoss loss("physics_rows", net_in, net_out, {"source"}, {eq});
  PredictionResult p1 = MakePrediction({0.0}, {0.0});
  PredictionResult p2 = MakePrediction({1.0}, {1.0});

  REQUIRE_THROWS_AS(loss.Evaluate({p1, p2}, {{0.0}}), std::invalid_argument);
}

TEST_CASE("CPhysicsLoss rejects physics data row size mismatch",
          "[CPhysicsLoss]") {
  std::vector<std::string> net_in = {"x"};
  std::vector<std::string> net_out = {"u"};
  CPhysicsEquation eq;
  eq.name = "size";
  eq.input_names = {"x"};
  eq.output_names = {"u"};
  eq.residual = [](const PhysicsState &, const PhysicsData &) {
    return mlpdouble(0.0);
  };

  CPhysicsLoss loss("physics_size", net_in, net_out, {"fx", "fy"}, {eq});
  PredictionResult pred = MakePrediction({1.0}, {1.0});

  REQUIRE_THROWS_AS(loss.Evaluate({pred}, {{1.0}}), std::invalid_argument);
}

TEST_CASE("CPhysicsLoss Jacobian nullptr throws", "[CPhysicsLoss]") {
  std::vector<std::string> net_in = {"x"};
  std::vector<std::string> net_out = {"u"};
  CPhysicsEquation eq;
  eq.name = "jac_null";
  eq.input_names = {"x"};
  eq.output_names = {"u"};
  eq.requires_jacobian = true;
  eq.residual = [](const PhysicsState &s, const PhysicsData &) {
    return s.Jac(0, 0);
  };

  CPhysicsLoss loss("jac_null", net_in, net_out, {}, {eq});
  PredictionResult pred = MakePrediction({1.0}, {1.0}, nullptr, nullptr);

  REQUIRE_THROWS_AS(loss.Evaluate({pred}, {}), std::invalid_argument);
}

TEST_CASE("CPhysicsLoss Hessian nullptr throws", "[CPhysicsLoss]") {
  std::vector<std::string> net_in = {"x"};
  std::vector<std::string> net_out = {"u"};
  CPhysicsEquation eq;
  eq.name = "hess_null";
  eq.input_names = {"x"};
  eq.output_names = {"u"};
  eq.requires_jacobian = true;
  eq.requires_hessian = true;
  eq.residual = [](const PhysicsState &s, const PhysicsData &) {
    return s.Hess(0, 0, 0);
  };

  CPhysicsLoss loss("hess_null", net_in, net_out, {}, {eq});
  mlpdouble **jac = AllocateJacobian(1, 1);
  PredictionResult pred = MakePrediction({1.0}, {1.0}, jac, nullptr);

  REQUIRE_THROWS_AS(loss.Evaluate({pred}, {}), std::invalid_argument);

  FreeJacobian(jac, 1);
}

TEST_CASE("CPhysicsLoss multiple source terms via separate equations",
          "[CPhysicsLoss]") {
  std::vector<std::string> net_in = {"x", "y", "t"};
  std::vector<std::string> net_out = {"u", "v"};

  CPhysicsEquation eq1;
  eq1.name = "u_rhs";
  eq1.input_names = {"x", "y", "t"};
  eq1.output_names = {"u"};
  eq1.requires_jacobian = true;
  eq1.residual = [](const PhysicsState &s, const PhysicsData &data) {
    return s.Jac(2, 0) - data.Ref("fx");
  };

  CPhysicsEquation eq2;
  eq2.name = "v_rhs";
  eq2.input_names = {"x", "y", "t"};
  eq2.output_names = {"v"};
  eq2.requires_jacobian = true;
  eq2.residual = [](const PhysicsState &s, const PhysicsData &data) {
    return s.Jac(2, 0) - data.Ref("fy");
  };

  CPhysicsLoss loss("multiple_rhs", net_in, net_out, {"fx", "fy"}, {eq1, eq2});
  mlpdouble **jac = AllocateJacobian(3, 2);
  jac[2][0] = 5.0;
  jac[2][1] = 7.0;

  PredictionResult pred = MakePrediction({1.0, 2.0, 3.0}, {10.0, 20.0}, jac);
  std::vector<std::vector<mlpdouble>> physics_data = {{5.0, 7.0}};

  mlpdouble result = loss.Evaluate({pred}, physics_data);
  REQUIRE_EQUAL_TOL(result, 0.0, 1e-12);
  FreeJacobian(jac, 3);
}

TEST_CASE("PhysicsData accessor mapping", "[CPhysicsLoss]") {
  std::vector<std::string> net_in = {"x"};
  std::vector<std::string> net_out = {"u"};
  std::vector<mlpdouble> captured;

  CPhysicsEquation eq;
  eq.name = "data_accessor";
  eq.input_names = {"x"};
  eq.output_names = {"u"};
  eq.residual = [&](const PhysicsState &, const PhysicsData &data) {
    captured.push_back(data.Ref("rho"));
    captured.push_back(data.Ref("mu"));
    return mlpdouble(0.0);
  };

  CPhysicsLoss loss("physics_data_accessor", net_in, net_out, {"rho", "mu"},
                    {eq});
  PredictionResult pred = MakePrediction({1.0}, {2.0});
  loss.Evaluate({pred}, {{1000.0, 0.001}});

  REQUIRE(captured.size() == 2);
  REQUIRE_EQUAL_TOL(captured[0], 1000.0, 1e-12);
  REQUIRE_EQUAL_TOL(captured[1], 0.001, 1e-12);
}

TEST_CASE("PhysicsData missing variable throws", "[CPhysicsLoss]") {
  std::vector<std::string> net_in = {"x"};
  std::vector<std::string> net_out = {"u"};
  CPhysicsEquation eq;
  eq.name = "missing_var";
  eq.input_names = {"x"};
  eq.output_names = {"u"};
  eq.residual = [](const PhysicsState &, const PhysicsData &data) {
    return data.Ref("does_not_exist");
  };

  CPhysicsLoss loss("missing_physics", net_in, net_out, {}, {eq});
  PredictionResult pred = MakePrediction({1.0}, {1.0});

  REQUIRE_THROWS_AS(loss.Evaluate({pred}, {}), std::out_of_range);
}

TEST_CASE("PhysicsData Has detects configured variables", "[CPhysicsLoss]") {
  std::vector<std::string> net_in = {"x"};
  std::vector<std::string> net_out = {"u"};
  CPhysicsEquation eq;
  eq.name = "has_test";
  eq.input_names = {"x"};
  eq.output_names = {"u"};
  eq.residual = [](const PhysicsState &, const PhysicsData &data) {
    REQUIRE(data.Has("fx"));
    REQUIRE(data.Has("fy"));
    REQUIRE_FALSE(data.Has("fz"));
    return mlpdouble(0.0);
  };

  CPhysicsLoss loss("has_test", net_in, net_out, {"fx", "fy"}, {eq});
  PredictionResult pred = MakePrediction({1.0}, {1.0});
  loss.Evaluate({pred}, {{1.0, 2.0}});
}

TEST_CASE("PhysicsState index-based access", "[CPhysicsLoss]") {
  std::vector<std::string> net_in = {"x", "y"};
  std::vector<std::string> net_out = {"u", "v"};
  CPhysicsEquation eq;
  eq.name = "index_access";
  eq.input_names = {"x", "y"};
  eq.output_names = {"u", "v"};
  eq.residual = [](const PhysicsState &s, const PhysicsData &) {
    REQUIRE_EQUAL_TOL(s.In(0), 10.0, 1e-12);
    REQUIRE_EQUAL_TOL(s.In(1), 20.0, 1e-12);
    REQUIRE_EQUAL_TOL(s.Out(0), 100.0, 1e-12);
    REQUIRE_EQUAL_TOL(s.Out(1), 200.0, 1e-12);
    return mlpdouble(0.0);
  };

  CPhysicsLoss loss("index_access", net_in, net_out, {}, {eq});
  PredictionResult pred = MakePrediction({10.0, 20.0}, {100.0, 200.0});
  loss.Evaluate({pred}, {});
}

TEST_CASE("PhysicsState bounds checking", "[CPhysicsLoss]") {
  std::vector<std::string> net_in = {"x"};
  std::vector<std::string> net_out = {"u"};
  CPhysicsEquation eq;
  eq.name = "bounds";
  eq.input_names = {"x"};
  eq.output_names = {"u"};
  eq.residual = [](const PhysicsState &s, const PhysicsData &) {
    REQUIRE_THROWS_AS(s.In(1), std::out_of_range);
    REQUIRE_THROWS_AS(s.Out(1), std::out_of_range);
    return mlpdouble(0.0);
  };

  CPhysicsLoss loss("bounds_test", net_in, net_out, {}, {eq});
  PredictionResult pred = MakePrediction({1.0}, {2.0});
  loss.Evaluate({pred}, {});
}

TEST_CASE("CPhysicsLoss EvaluateSingleSample returns raw loss",
          "[CPhysicsLoss]") {
  std::vector<std::string> net_in = {"x"};
  std::vector<std::string> net_out = {"u"};
  CPhysicsEquation eq1;
  eq1.name = "r1";
  eq1.input_names = {"x"};
  eq1.output_names = {"u"};
  eq1.residual = [](const PhysicsState &, const PhysicsData &) {
    return mlpdouble(3.0);
  };

  CPhysicsEquation eq2;
  eq2.name = "r2";
  eq2.input_names = {"x"};
  eq2.output_names = {"u"};
  eq2.residual = [](const PhysicsState &, const PhysicsData &) {
    return mlpdouble(4.0);
  };

  CPhysicsLoss loss("raw_test", net_in, net_out, {}, {eq1, eq2});
  PredictionResult pred = MakePrediction({1.0}, {1.0});

  mlpdouble raw = loss.EvaluateSingleSample(pred);
  REQUIRE_EQUAL_TOL(raw, 25.0, 1e-12);

  mlpdouble normalized = loss.Evaluate({pred}, {});
  REQUIRE_EQUAL_TOL(normalized, 12.5, 1e-12);
}

TEST_CASE("CPhysicsLoss per-equation weights", "[CPhysicsLoss]") {
  std::vector<std::string> net_in = {"x"};
  std::vector<std::string> net_out = {"u"};
  CPhysicsEquation eq1;
  eq1.name = "weighted";
  eq1.input_names = {"x"};
  eq1.output_names = {"u"};
  eq1.weight = 2.0;
  eq1.residual = [](const PhysicsState &, const PhysicsData &) {
    return mlpdouble(3.0);
  };

  CPhysicsEquation eq2;
  eq2.name = "unweighted";
  eq2.input_names = {"x"};
  eq2.output_names = {"u"};
  eq2.weight = 1.0;
  eq2.residual = [](const PhysicsState &, const PhysicsData &) {
    return mlpdouble(4.0);
  };

  CPhysicsLoss loss("weight_test", net_in, net_out, {}, {eq1, eq2});
  PredictionResult pred = MakePrediction({1.0}, {1.0});

  mlpdouble raw = loss.EvaluateSingleSample(pred);
  REQUIRE_EQUAL_TOL(raw, 34.0, 1e-12);

  mlpdouble normalized = loss.Evaluate({pred}, {});
  REQUIRE_EQUAL_TOL(normalized, 17.0, 1e-12);
}

TEST_CASE("CPhysicsLoss negative weight throws", "[CPhysicsLoss]") {
  std::vector<std::string> net_in = {"x"};
  std::vector<std::string> net_out = {"u"};
  CPhysicsEquation eq;
  eq.name = "neg_weight";
  eq.input_names = {"x"};
  eq.output_names = {"u"};
  eq.weight = -1.0;
  eq.residual = [](const PhysicsState &, const PhysicsData &) {
    return mlpdouble(0.0);
  };

  REQUIRE_THROWS_AS(CPhysicsLoss("neg", net_in, net_out, {}, {eq}),
                    std::invalid_argument);
}

TEST_CASE("CPhysicsLoss duplicate network input names throws",
          "[CPhysicsLoss]") {
  std::vector<std::string> dup_in = {"x", "x"};
  std::vector<std::string> net_out = {"u"};
  CPhysicsEquation eq;
  eq.name = "dup_net";
  eq.input_names = {"x"};
  eq.output_names = {"u"};
  eq.residual = [](const PhysicsState &, const PhysicsData &) {
    return mlpdouble(0.0);
  };

  REQUIRE_THROWS_AS(CPhysicsLoss("dup_net_in", dup_in, net_out, {}, {eq}),
                    std::invalid_argument);
}

TEST_CASE("CPhysicsLoss duplicate network output names throws",
          "[CPhysicsLoss]") {
  std::vector<std::string> net_in = {"x"};
  std::vector<std::string> dup_out = {"u", "u"};
  CPhysicsEquation eq;
  eq.name = "dup_net";
  eq.input_names = {"x"};
  eq.output_names = {"u"};
  eq.residual = [](const PhysicsState &, const PhysicsData &) {
    return mlpdouble(0.0);
  };

  REQUIRE_THROWS_AS(CPhysicsLoss("dup_net_out", net_in, dup_out, {}, {eq}),
                    std::invalid_argument);
}

TEST_CASE("CPhysicsLoss empty equation name throws", "[CPhysicsLoss]") {
  std::vector<std::string> net_in = {"x"};
  std::vector<std::string> net_out = {"u"};
  CPhysicsEquation eq;
  eq.name = "";
  eq.input_names = {"x"};
  eq.output_names = {"u"};
  eq.residual = [](const PhysicsState &, const PhysicsData &) {
    return mlpdouble(0.0);
  };

  REQUIRE_THROWS_AS(CPhysicsLoss("empty_eq_name", net_in, net_out, {}, {eq}),
                    std::invalid_argument);
}

TEST_CASE("CPhysicsEquation duplicate input names throws", "[CPhysicsLoss]") {
  std::vector<std::string> net_in = {"x", "y"};
  std::vector<std::string> net_out = {"u"};
  CPhysicsEquation eq;
  eq.name = "dup_eq_in";
  eq.input_names = {"x", "x"};
  eq.output_names = {"u"};
  eq.residual = [](const PhysicsState &, const PhysicsData &) {
    return mlpdouble(0.0);
  };

  REQUIRE_THROWS_AS(CPhysicsLoss("dup_eq_in", net_in, net_out, {}, {eq}),
                    std::invalid_argument);
}

TEST_CASE("CPhysicsEquation duplicate output names throws", "[CPhysicsLoss]") {
  std::vector<std::string> net_in = {"x"};
  std::vector<std::string> net_out = {"u", "v"};
  CPhysicsEquation eq;
  eq.name = "dup_eq_out";
  eq.input_names = {"x"};
  eq.output_names = {"u", "u"};
  eq.residual = [](const PhysicsState &, const PhysicsData &) {
    return mlpdouble(0.0);
  };

  REQUIRE_THROWS_AS(CPhysicsLoss("dup_eq_out", net_in, net_out, {}, {eq}),
                    std::invalid_argument);
}

TEST_CASE("CPhysicsLoss equations with different subsets", "[CPhysicsLoss]") {
  std::vector<std::string> net_in = {"x", "y", "t"};
  std::vector<std::string> net_out = {"u", "v"};

  CPhysicsEquation eq1;
  eq1.name = "u_eq";
  eq1.input_names = {"x", "t"};
  eq1.output_names = {"u"};
  eq1.requires_jacobian = true;
  eq1.residual = [](const PhysicsState &s, const PhysicsData &) {
    return s.Jac(1, 0) - mlpdouble(5.0);
  };

  CPhysicsEquation eq2;
  eq2.name = "v_eq";
  eq2.input_names = {"y", "t"};
  eq2.output_names = {"v"};
  eq2.requires_jacobian = true;
  eq2.residual = [](const PhysicsState &s, const PhysicsData &) {
    return s.Jac(1, 0) - mlpdouble(7.0);
  };

  CPhysicsLoss loss("subset_test", net_in, net_out, {}, {eq1, eq2});
  mlpdouble **jac = AllocateJacobian(3, 2);
  jac[2][0] = 5.0;
  jac[2][1] = 7.0;

  PredictionResult pred = MakePrediction({1.0, 2.0, 3.0}, {10.0, 20.0}, jac);
  mlpdouble result = loss.Evaluate({pred}, {});
  REQUIRE_EQUAL_TOL(result, 0.0, 1e-12);
  FreeJacobian(jac, 3);
}

TEST_CASE("CPhysicsLoss empty physics variable name throws", "[CPhysicsLoss]") {
  std::vector<std::string> net_in = {"x"};
  std::vector<std::string> net_out = {"u"};
  CPhysicsEquation eq;
  eq.name = "empty_phys";
  eq.input_names = {"x"};
  eq.output_names = {"u"};
  eq.residual = [](const PhysicsState &, const PhysicsData &) {
    return mlpdouble(0.0);
  };

  REQUIRE_THROWS_AS(
      CPhysicsLoss("empty_phys", net_in, net_out, {"", "fx"}, {eq}),
      std::invalid_argument);
}

TEST_CASE("CPhysicsLoss PredictionResult dimension mismatch throws",
          "[CPhysicsLoss]") {
  std::vector<std::string> net_in = {"x", "y"};
  std::vector<std::string> net_out = {"u"};
  CPhysicsEquation eq;
  eq.name = "dim";
  eq.input_names = {"x", "y"};
  eq.output_names = {"u"};
  eq.residual = [](const PhysicsState &, const PhysicsData &) {
    return mlpdouble(0.0);
  };

  CPhysicsLoss loss("dim_test", net_in, net_out, {}, {eq});

  PredictionResult bad_in = MakePrediction({1.0}, {1.0});
  REQUIRE_THROWS_AS(loss.EvaluateSingleSample(bad_in), std::invalid_argument);

  PredictionResult bad_out = MakePrediction({1.0, 2.0}, {1.0, 2.0});
  REQUIRE_THROWS_AS(loss.EvaluateSingleSample(bad_out), std::invalid_argument);
}

TEST_CASE("Jac/Hess throws when pointer is null regardless of flags",
          "[CPhysicsLoss]") {
  std::vector<std::string> net_in = {"x"};
  std::vector<std::string> net_out = {"u"};

  CPhysicsEquation eq_bad;
  eq_bad.name = "undeclared_jac";
  eq_bad.input_names = {"x"};
  eq_bad.output_names = {"u"};
  eq_bad.requires_jacobian = false;
  eq_bad.residual = [](const PhysicsState &s, const PhysicsData &) {
    return s.Jac(0, 0);
  };

  CPhysicsLoss loss("undeclared", net_in, net_out, {}, {eq_bad});
  PredictionResult pred = MakePrediction({1.0}, {1.0});

  REQUIRE_THROWS_AS(loss.EvaluateSingleSample(pred), std::runtime_error);
}
