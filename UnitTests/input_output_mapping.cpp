#include "../include/CLookUp_ANN.hpp"
#include "test_subjects.hpp"

#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

TEST_CASE("Different queries, same network", "[CIOMap]") {
    /*! \brief Different queries with the same network should return the same values. */

    /* Create two identical networks with different input and output variable names. */
    std::vector<std::string> input_names_1 = {"a","b","c"}, output_names_1 = {"x", "z"};
    MLPToolbox::CNeuralNetwork * mlp_1 = CreateRandomNetwork(input_names_1, output_names_1);

    MLPToolbox::CNeuralNetwork * mlp_2 = new MLPToolbox::CNeuralNetwork(*mlp_1);
    mlp_2->SetInputName(0, "d");
    mlp_2->SetInputName(1, "e");
    mlp_2->SetInputName(2, "f");
    mlp_2->SetOutputName(0, "y");
    mlp_2->SetOutputName(1, "q");

    /* Add networks to collection */
    MLPToolbox::CLookUp_ANN mlp_collection;
    mlp_collection.AddNetwork(mlp_1);
    mlp_collection.AddNetwork(mlp_2);

    /* Define two queries for the two sets of inputs-outputs that refer to the same variables. */
    double val_in_1, val_in_2, val_in_3, val_out_1, val_out_2;
    MLPToolbox::CIOMap query_1, query_2;
    query_1.AddQueryInput("a", &val_in_1);
    query_1.AddQueryInput("b", &val_in_2);
    query_1.AddQueryInput("c", &val_in_3);
    query_1.AddQueryOutput("x", &val_out_1);
    
    query_2.AddQueryInput("f", &val_in_3);
    query_2.AddQueryInput("e", &val_in_2);
    query_2.AddQueryInput("d", &val_in_1);
    query_2.AddQueryOutput("y", &val_out_2);

    mlp_collection.PairVariableswithMLPs(query_1);
    mlp_collection.PairVariableswithMLPs(query_2);
    
    /* Evaluate the output of the two networks */
    std::random_device rd;  
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    bool inside{false};
    /* Only compare output when input are within network input range */
    while (!inside) {
        val_in_1 = dis(gen);
        val_in_2 = dis(gen);
        val_in_3 = dis(gen);
        bool inside_1 = mlp_collection.Predict(query_1);
        bool inside_2 = mlp_collection.Predict(query_2);
        inside = (inside_1 && inside_2);
    }  

    REQUIRE(val_out_1 == val_out_2);

    delete mlp_1;
    delete mlp_2;
}

TEST_CASE("Same networks in query", "[CIOMap]") {
    /*! \brief A query with two identical networks should produce equal outputs. */

    /* Create two identical networks with different input and output variable names. */
    std::vector<std::string> input_names_1 = {"a","b","c"}, output_names_1 = {"x", "z"};
    MLPToolbox::CNeuralNetwork * mlp_1 = CreateRandomNetwork(input_names_1, output_names_1);

    MLPToolbox::CNeuralNetwork * mlp_2 = new MLPToolbox::CNeuralNetwork(*mlp_1);
    mlp_2->SetInputName(0, "d");
    mlp_2->SetInputName(1, "e");
    mlp_2->SetInputName(2, "f");
    mlp_2->SetOutputName(0, "y");
    mlp_2->SetOutputName(1, "q");

    /* Add networks to collection */
    MLPToolbox::CLookUp_ANN mlp_collection;
    mlp_collection.AddNetwork(mlp_1);
    mlp_collection.AddNetwork(mlp_2);

    /* Define two queries for the two sets of inputs-outputs that refer to the same variables. */
    double val_in_1, val_in_2, val_in_3, val_out_1, val_out_2;
    MLPToolbox::CIOMap query_1, query_2;
    query_1.AddQueryInput("a", &val_in_1);
    query_1.AddQueryInput("b", &val_in_2);
    query_1.AddQueryInput("c", &val_in_3);
    query_1.AddQueryOutput("x", &val_out_1);
    
    query_2.AddQueryInput("f", &val_in_3);
    query_2.AddQueryInput("e", &val_in_2);
    query_2.AddQueryInput("d", &val_in_1);
    query_2.AddQueryOutput("y", &val_out_2);

    mlp_collection.PairVariableswithMLPs(query_1);
    mlp_collection.PairVariableswithMLPs(query_2);
    
    /* Evaluate the output of the two networks */
    std::random_device rd;  
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    bool inside{false};
    /* Only compare output when input are within network input range */
    while (!inside) {
        val_in_1 = dis(gen);
        val_in_2 = dis(gen);
        val_in_3 = dis(gen);
        bool inside_1 = mlp_collection.Predict(query_1);
        bool inside_2 = mlp_collection.Predict(query_2);
        inside = (inside_1 && inside_2);
    }  

    REQUIRE(val_out_1==val_out_2);
    delete mlp_1;
    delete mlp_2;
}

TEST_CASE("Query with multiple networks", "[CIOMap]") {
    /*! \brief Single query for networks with multiple networks and input/output variables. */

    /* Create two identical networks with different input and output variables. */
    std::vector<std::string> input_names_1 = {"a","b"}, output_names_1 = {"x", "y"};
    MLPToolbox::CNeuralNetwork * mlp_1 = CreateRandomNetwork(input_names_1, output_names_1);
    MLPToolbox::CNeuralNetwork * mlp_2 = new MLPToolbox::CNeuralNetwork(*mlp_1);
    mlp_2->SetInputName(0, "c");
    mlp_2->SetInputName(1, "d");
    mlp_2->SetOutputName(0, "z");
    mlp_2->SetOutputName(1, "q");

    MLPToolbox::CLookUp_ANN mlp_collection;
    mlp_collection.AddNetwork(mlp_1);
    mlp_collection.AddNetwork(mlp_2);

    /* Link network inputs and outputs to the same variables. */
    double val_in_1, val_in_2, val_out_1, val_out_2;
    MLPToolbox::CIOMap query;
    query.AddQueryInput("a", &val_in_1);
    query.AddQueryInput("b", &val_in_2);
    query.AddQueryInput("c", &val_in_1);
    query.AddQueryInput("d", &val_in_2);
    query.AddQueryOutput("x", &val_out_1);
    query.AddQueryOutput("y", &val_out_2);
    query.AddQueryOutput("z", &val_out_1);
    query.AddQueryOutput("q", &val_out_2);

    mlp_collection.PairVariableswithMLPs(query);

    /* Evaluate the output of the two networks */
    std::random_device rd;  
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    bool inside{false};
    /* Only compare output when input are within network input range */
    while (!inside) {
        val_in_1 = dis(gen);
        val_in_2 = dis(gen);
        inside = mlp_collection.Predict(query);
    }  
    REQUIRE(val_out_1==mlp_1->GetOutput(0));
    REQUIRE(val_out_2==mlp_1->GetOutput(1));
    REQUIRE(val_out_1==mlp_2->GetOutput(0));
    REQUIRE(val_out_2==mlp_2->GetOutput(1));
    delete mlp_1;
    delete mlp_2;
}

TEST_CASE("Null queries", "[CIOMap]") {
    /*! \brief Queries to NULL or None should always return zero. */

    std::vector<std::string> input_names_1 = {"a","b"}, output_names_1 = {"x", "y"};
    MLPToolbox::CNeuralNetwork * mlp_1 = CreateRandomNetwork(input_names_1, output_names_1);
    MLPToolbox::CLookUp_ANN mlp_collection;
    mlp_collection.AddNetwork(mlp_1);

    double val_in_1, val_in_2, val_out_1{1.0}, val_out_2{1.0}, val_out_3{1.0}, val_out_4{1.0}, val_out_5{1.0};
    MLPToolbox::CIOMap query_1;
    query_1.AddQueryInput("a", &val_in_1);
    query_1.AddQueryInput("b", &val_in_2);
    query_1.AddQueryOutput("null", &val_out_1);
    query_1.AddQueryOutput("y", &val_out_2);
    query_1.AddQueryOutput("NULL", &val_out_3);
    query_1.AddQueryOutput("none", &val_out_4);
    query_1.AddQueryOutput("NoNe", &val_out_5);
    mlp_collection.PairVariableswithMLPs(query_1);
    /* Evaluate the output of the two networks */
    std::random_device rd;  
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    bool inside{false};
    /* Only compare output when input are within network input range */
    while (!inside) {
        val_in_1 = dis(gen);
        val_in_2 = dis(gen);
        inside = mlp_collection.Predict(query_1);
    }  
    REQUIRE(val_out_1==0.0);
    REQUIRE(val_out_2==mlp_1->GetOutput(1));
    REQUIRE(val_out_3==0.0);
    REQUIRE(val_out_4==0.0);
    REQUIRE(val_out_5==0.0);
    delete mlp_1;
}

TEST_CASE("Jacobian and Hessian queries", "[CIOMap]") {
    std::vector<std::string> input_names_1 = {"a","b"}, output_names_1 = {"x", "y"};
    std::vector<std::string> input_names_2 = {"c","d"}, output_names_2 = {"z", "q"};
    MLPToolbox::CNeuralNetwork * mlp_1 = CreateRandomNetwork(input_names_1, output_names_1);
    MLPToolbox::CNeuralNetwork * mlp_2 = CreateRandomNetwork(input_names_2, output_names_2);
    
    MLPToolbox::CLookUp_ANN mlp_collection;
    mlp_collection.AddNetwork(mlp_1);
    mlp_collection.AddNetwork(mlp_2);

    double val_a, val_b, val_c, val_d;
    double val_x, val_y, val_z, val_q;
    double val_dxda, val_dzdc, val_d2ydb2, val_d2qdcdd;

    MLPToolbox::CIOMap derivative_query;
    derivative_query.AddQueryInput("a", &val_a);
    derivative_query.AddQueryInput("b", &val_b);
    derivative_query.AddQueryInput("c", &val_c);
    derivative_query.AddQueryInput("d", &val_d);

    derivative_query.AddQueryOutput("x", &val_x);
    derivative_query.AddQueryOutput("y", &val_y);
    derivative_query.AddQueryOutput("z", &val_z);
    derivative_query.AddQueryOutput("q", &val_q);

    derivative_query.AddQueryJacobian("x", "a", &val_dxda);
    derivative_query.AddQueryJacobian("z", "c", &val_dzdc);

    derivative_query.AddQueryHessian("y","b","b", &val_d2ydb2);
    derivative_query.AddQueryHessian("q","c","d", &val_d2qdcdd);
    
    mlp_collection.PairVariableswithMLPs(derivative_query);

    auto vals_input = RandomInputs(4);
    val_a = vals_input[0];
    val_b = vals_input[1];
    val_c = vals_input[2];
    val_d = vals_input[3];

    mlp_collection.Predict(derivative_query);

    REQUIRE(val_dxda==mlp_1->GetJacobian(0, 0));
    REQUIRE(val_dzdc==mlp_2->GetJacobian(0, 0));
    REQUIRE(val_d2ydb2==mlp_1->GetHessian(1, 1, 1));
    REQUIRE(val_d2qdcdd==mlp_2->GetHessian(1, 0, 1));

    delete mlp_1;
    delete mlp_2;
}

TEST_CASE("Input-output accessors through vectors", "[CIOMap]") {
    std::vector<std::string> input_names_1 = {"a","b"}, output_names_1 = {"x", "y"},
                             input_names_2 = {"b","a"}, output_names_2 = {"z"};
    MLPToolbox::CNeuralNetwork * mlp_1 = CreateRandomNetwork(input_names_1, output_names_1);
    MLPToolbox::CNeuralNetwork * mlp_2 = CreateRandomNetwork(input_names_2, output_names_2);

    MLPToolbox::CLookUp_ANN mlp_collection;
    mlp_collection.AddNetwork(mlp_1);
    mlp_collection.AddNetwork(mlp_2);

    /* Define query variables*/
    MLPToolbox::CIOMap query_memberwise, query_vector;

    double val_a, val_b, val_x_m, val_y_m, val_z_m, val_null_m, val_x_v, val_y_v, val_z_v,val_null_v;
    val_a = 0.2;
    val_b = 0.8;

    /* Specify query input and output variables through vectors */
    std::vector<std::string> input_vec = {"a", "b"};
    std::vector<std::string> output_vec = {"x","y","z","null"};
    std::vector<double*> refs_out_vec = {&val_x_v, &val_y_v, &val_z_v, &val_null_v};
    query_vector.SetQueryInput(input_vec);
    query_vector.SetQueryOutput(output_vec);

    /* Specify query variables member-wise. */
    query_memberwise.AddQueryInput("a", &val_a);
    query_memberwise.AddQueryInput("b", &val_b);
    query_memberwise.AddQueryOutput("x", &val_x_m);
    query_memberwise.AddQueryOutput("y", &val_y_m);
    query_memberwise.AddQueryOutput("z", &val_z_m);
    query_memberwise.AddQueryOutput("null", &val_null_m);
    
    mlp_collection.PairVariableswithMLPs(query_memberwise);
    mlp_collection.PairVariableswithMLPs(query_vector);
    
    /* Evaluate network output */
    std::vector<double> vals_in_vec = {val_a, val_b};
    bool inside_m = mlp_collection.Predict(query_memberwise);
    bool inside_v = mlp_collection.Predict(query_vector, vals_in_vec, refs_out_vec);

    REQUIRE(inside_m==inside_v);
    REQUIRE(val_x_v==val_x_m);
    REQUIRE(val_y_v==val_y_m);
    REQUIRE(val_z_v==val_z_m);

    delete mlp_1;
    delete mlp_2;
}

TEST_CASE("Ill-defined queries for look-up", "[CIOMap]") {
    /*! \brief impossible queries should return errors */
    std::vector<std::string> input_names_1 = {"a","b"}, output_names_1 = {"x", "y"};
    MLPToolbox::CNeuralNetwork * mlp_1 = CreateRandomNetwork(input_names_1, output_names_1);
    MLPToolbox::CLookUp_ANN mlp_collection;
    mlp_collection.AddNetwork(mlp_1);

    {
    MLPToolbox::CIOMap superfluous_outputs;
    superfluous_outputs.AddQueryInput("a");
    superfluous_outputs.AddQueryInput("b");
    superfluous_outputs.AddQueryOutput("x");
    superfluous_outputs.AddQueryOutput("y");
    superfluous_outputs.AddQueryOutput("z");
    REQUIRE_THROWS_AS(mlp_collection.PairVariableswithMLPs(superfluous_outputs), MLPToolbox::QueryOutputNotFoundException);
    }

    {
    MLPToolbox::CIOMap superfluous_inputs;
    superfluous_inputs.AddQueryInput("a");
    superfluous_inputs.AddQueryInput("b");
    superfluous_inputs.AddQueryInput("c");
    superfluous_inputs.AddQueryOutput("x");
    superfluous_inputs.AddQueryOutput("y");
    REQUIRE_THROWS_AS(mlp_collection.PairVariableswithMLPs(superfluous_inputs), MLPToolbox::QueryInputNotFoundException);
    }

    {
    MLPToolbox::CIOMap duplicate_inputs;
    duplicate_inputs.AddQueryInput("a");
    duplicate_inputs.AddQueryInput("b");
    duplicate_inputs.AddQueryInput("b");
    duplicate_inputs.AddQueryOutput("x");
    REQUIRE_THROWS_AS(mlp_collection.PairVariableswithMLPs(duplicate_inputs), MLPToolbox::DuplicatesInQueryException);
    }

    {
    MLPToolbox::CIOMap duplicate_outputs;
    duplicate_outputs.AddQueryInput("a");
    duplicate_outputs.AddQueryInput("b");
    duplicate_outputs.AddQueryOutput("x");
    duplicate_outputs.AddQueryOutput("x");
    duplicate_outputs.AddQueryOutput("y");
    REQUIRE_THROWS_AS(mlp_collection.PairVariableswithMLPs(duplicate_outputs), MLPToolbox::DuplicatesInQueryException);
    }

    {
    MLPToolbox::CIOMap shared_variables;
    shared_variables.AddQueryInput("a");
    shared_variables.AddQueryInput("b");
    shared_variables.AddQueryOutput("a");
    shared_variables.AddQueryOutput("y");
    REQUIRE_THROWS_AS(mlp_collection.PairVariableswithMLPs(shared_variables), MLPToolbox::SharedInputsOutputsException);
    }

    {
    MLPToolbox::CIOMap insufficient_jacobian;
    insufficient_jacobian.AddQueryInput("a");
    insufficient_jacobian.AddQueryInput("b");
    insufficient_jacobian.AddQueryOutput("y");
    insufficient_jacobian.AddQueryJacobian("x", "a", nullptr);
    REQUIRE_THROWS_AS(mlp_collection.PairVariableswithMLPs(insufficient_jacobian), MLPToolbox::InsufficientJacobianException);
    }
    {
    MLPToolbox::CIOMap insufficient_hessian;
    insufficient_hessian.AddQueryInput("a");
    insufficient_hessian.AddQueryInput("b");
    insufficient_hessian.AddQueryOutput("y");
    insufficient_hessian.AddQueryHessian("x", "a","b", nullptr);
    REQUIRE_THROWS_AS(mlp_collection.PairVariableswithMLPs(insufficient_hessian), MLPToolbox::InsufficientHessianException);
    }
    delete mlp_1;
}

TEST_CASE("Ill-defined queries for Jacobians and Hessians", "[CIOMap]") {
    std::vector<std::string> input_names_1 = {"a","b"}, output_names_1 = {"x", "y"};
    MLPToolbox::CNeuralNetwork * mlp_1 = CreateRandomNetwork(input_names_1, output_names_1);
    std::vector<std::string> input_names_2 = {"c","d"}, output_names_2 = {"z", "q"};
    MLPToolbox::CNeuralNetwork * mlp_2 = CreateRandomNetwork(input_names_2, output_names_2);

    MLPToolbox::CLookUp_ANN mlp_collection;
    mlp_collection.AddNetwork(mlp_1);
    mlp_collection.AddNetwork(mlp_2);


    {
    MLPToolbox::CIOMap insufficient_jacobian;
    insufficient_jacobian.AddQueryInput("a");
    insufficient_jacobian.AddQueryInput("b");
    insufficient_jacobian.AddQueryOutput("y");
    insufficient_jacobian.AddQueryJacobian("x", "a", nullptr);
    REQUIRE_THROWS_AS(mlp_collection.PairVariableswithMLPs(insufficient_jacobian), MLPToolbox::InsufficientJacobianException);
    }
    {
    MLPToolbox::CIOMap insufficient_hessian;
    insufficient_hessian.AddQueryInput("a");
    insufficient_hessian.AddQueryInput("b");
    insufficient_hessian.AddQueryOutput("y");
    insufficient_hessian.AddQueryHessian("x", "a","b", nullptr);
    REQUIRE_THROWS_AS(mlp_collection.PairVariableswithMLPs(insufficient_hessian), MLPToolbox::InsufficientHessianException);
    }

    {
        MLPToolbox::CIOMap illdefined_jacobian;
        illdefined_jacobian.AddQueryInput("a");
        illdefined_jacobian.AddQueryInput("b");
        illdefined_jacobian.AddQueryInput("c");
        illdefined_jacobian.AddQueryInput("d");
        illdefined_jacobian.AddQueryOutput("x");
        illdefined_jacobian.AddQueryOutput("z");
        illdefined_jacobian.AddQueryJacobian("z", "a", nullptr);
        REQUIRE_THROWS_AS(mlp_collection.PairVariableswithMLPs(illdefined_jacobian), MLPToolbox::JacobianNotSupportedException);
    }

    {
        MLPToolbox::CIOMap illdefined_hessian;
        illdefined_hessian.AddQueryInput("a");
        illdefined_hessian.AddQueryInput("b");
        illdefined_hessian.AddQueryInput("c");
        illdefined_hessian.AddQueryInput("d");
        illdefined_hessian.AddQueryOutput("x");
        illdefined_hessian.AddQueryOutput("z");
        illdefined_hessian.AddQueryHessian("z", "c","b", nullptr);
        REQUIRE_THROWS_AS(mlp_collection.PairVariableswithMLPs(illdefined_hessian), MLPToolbox::HessianNotSupportedException);
    }
    delete mlp_1;
    delete mlp_2;
}