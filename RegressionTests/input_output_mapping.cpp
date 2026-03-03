#include "../include/CLookUp_ANN.hpp"
#include "unit_test.hpp"

bool InputOutputMapping::DifferentInputsDifferentOutputs() {
    /*! \brief Different queries with the same network should return the same values. */

    /* Create two identical networks with different input and output variable names. */
    MLPToolbox::CNeuralNetwork * mlp_1 = CreateRandomNetwork();
    mlp_1->SetInputName(0, "a");
    mlp_1->SetInputName(1, "b");
    mlp_1->SetInputName(2, "c");
    mlp_1->SetOutputName(0, "x");

    MLPToolbox::CNeuralNetwork * mlp_2 = new MLPToolbox::CNeuralNetwork(*mlp_1);
    mlp_2->SetInputName(0, "d");
    mlp_2->SetInputName(1, "e");
    mlp_2->SetInputName(2, "f");
    mlp_2->SetOutputName(0, "y");

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
    val_in_1 = 0.4;
    val_in_2 = 0.7;
    val_in_3 = 0.2;
    mlp_collection.Predict(query_1);
    mlp_collection.Predict(query_2);

    bool passed_test = (val_out_1 == val_out_2);

    delete mlp_1;
    delete mlp_2;
    return passed_test;
}


bool InputOutputMapping::SameInputsDifferentOutputs() {
    /*! \brief Different queries with the same network should return the same values. */

    /* Create two identical networks with different input and output variable names. */
    MLPToolbox::CNeuralNetwork * mlp_1 = CreateRandomNetwork();
    mlp_1->SetInputName(0, "a");
    mlp_1->SetInputName(1, "b");
    mlp_1->SetInputName(2, "c");
    mlp_1->SetOutputName(0, "x");

    MLPToolbox::CNeuralNetwork * mlp_2 = new MLPToolbox::CNeuralNetwork(*mlp_1);
    mlp_2->SetInputName(0, "a");
    mlp_2->SetInputName(1, "b");
    mlp_2->SetInputName(2, "c");
    mlp_2->SetOutputName(0, "y");

    /* Add networks to collection */
    MLPToolbox::CLookUp_ANN mlp_collection;
    mlp_collection.AddNetwork(mlp_1);
    mlp_collection.AddNetwork(mlp_2);

    /* Define two queries for the two sets of inputs-outputs that refer to the same variables. */
    double val_in_1, val_in_2, val_in_3, val_out_1, val_out_2;
    MLPToolbox::CIOMap query;
    query.AddQueryInput("a", &val_in_1);
    query.AddQueryInput("b", &val_in_2);
    query.AddQueryInput("c", &val_in_3);
    query.AddQueryOutput("x", &val_out_1);
    query.AddQueryOutput("y", &val_out_2);

    mlp_collection.PairVariableswithMLPs(query);
    
    /* Evaluate the output of the two networks */
    val_in_1 = 0.4;
    val_in_2 = 0.7;
    val_in_3 = 0.2;
    mlp_collection.Predict(query);

    bool passed_test = (val_out_1 == val_out_2);

    delete mlp_1;
    delete mlp_2;
    return passed_test;
}



bool InputOutputMapping::RunTest() {
    bool test_multiple_inputs = DifferentInputsDifferentOutputs();
    bool test_multiple_outputs = SameInputsDifferentOutputs();
    passed = (test_multiple_inputs && test_multiple_outputs);
    return passed;
};