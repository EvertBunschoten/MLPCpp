#include "../include/CLookUp_ANN.hpp"
#include "unit_test.hpp"

bool InputOutputMapping::DifferentInputsDifferentOutputs() {
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
    bool passed_test = (val_out_1 == val_out_2);

    /* Update summary */
    summary << "Network inputs: " << std::endl;
    summary << "1 : " << val_in_1 << std::endl;
    summary << "2 : " << val_in_2 << std::endl;
    summary << "3 : " << val_in_3 << std::endl;
    summary << "Network outputs: " << std::endl;
    summary << "Network 1: " << std::endl;
    summary << mlp_1->GetOutputName(0) << " : " << mlp_1->GetOutput(0) << std::endl;
    summary << mlp_1->GetOutputName(1) << " : " << mlp_1->GetOutput(1) << std::endl;
    summary << "Network 2: " << std::endl;
    summary << mlp_2->GetOutputName(0) << " : " << mlp_2->GetOutput(0) << std::endl;
    summary << mlp_2->GetOutputName(1) << " : " << mlp_2->GetOutput(1) << std::endl;
    summary << "Target variables:" << std::endl;
    summary << val_out_1 << std::endl;
    summary << val_out_2 << std::endl;
    
    delete mlp_1;
    delete mlp_2;
    return passed_test;
}


bool InputOutputMapping::SameInputsDifferentOutputs() {
    /*! \brief Different queries with the same network should return the same values. */

    /* Create two identical networks with different output variable names. */
    const std::vector<std::string> network_inputs = {"a", "b", "c"};
    const std::vector<std::string> output_names_1 = {"x", "z"};
    const std::vector<std::string> output_names_2 = {"y", "q"};
    
    MLPToolbox::CNeuralNetwork * mlp_1 = CreateRandomNetwork(network_inputs, output_names_1);

    MLPToolbox::CNeuralNetwork * mlp_2 = new MLPToolbox::CNeuralNetwork(*mlp_1);
    for (auto iOutput=0u; iOutput<mlp_2->GetnOutputs();iOutput++)
        mlp_2->SetOutputName(iOutput, output_names_2[iOutput]);

    /* Add networks to collection */
    MLPToolbox::CLookUp_ANN mlp_collection;
    mlp_collection.AddNetwork(mlp_1);
    mlp_collection.AddNetwork(mlp_2);

    /* Define two queries for the two sets of inputs-outputs that refer to the same variables. */
    double val_in_1, val_in_2, val_in_3, val_out_1, val_out_2;
    MLPToolbox::CIOMap query;
    query.AddQueryInput(network_inputs[0], &val_in_1);
    query.AddQueryInput(network_inputs[1], &val_in_2);
    query.AddQueryInput(network_inputs[2], &val_in_3);
    query.AddQueryOutput(output_names_1[0], &val_out_1);
    query.AddQueryOutput(output_names_2[0], &val_out_2);

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
        val_in_3 = dis(gen);
        inside = mlp_collection.Predict(query);
    }  
    bool passed_test = (val_out_1 == val_out_2);
    summary << "Network inputs: " << std::endl;
    summary << "1 : " << val_in_1 << std::endl;
    summary << "2 : " << val_in_2 << std::endl;
    summary << "3 : " << val_in_3 << std::endl;
    summary << "Network outputs: " << std::endl;
    summary << "Network 1: " << std::endl;
    summary << mlp_1->GetOutputName(0) << " : " << mlp_1->GetOutput(0) << std::endl;
    summary << mlp_1->GetOutputName(1) << " : " << mlp_1->GetOutput(1) << std::endl;
    summary << "Network 2: " << std::endl;
    summary << mlp_2->GetOutputName(0) << " : " << mlp_2->GetOutput(0) << std::endl;
    summary << mlp_2->GetOutputName(1) << " : " << mlp_2->GetOutput(1) << std::endl;
    summary << "Target variables:" << std::endl;
    summary << val_out_1 << std::endl;
    summary << val_out_2 << std::endl;
    
    delete mlp_1;
    delete mlp_2;
    return passed_test;
}

bool InputOutputMapping::DifferentInputsDifferentOutputs2() {
    /*! \brief Single query for networks with different inputs and outputs. */

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
    /* Check if network output is correctly evaluated */
    bool passed_test = (val_out_1 == mlp_1->GetOutput(0)) 
                       && (val_out_2 == mlp_1->GetOutput(1))
                       && (val_out_1 == mlp_1->GetOutput(0))
                       && (val_out_2 == mlp_2->GetOutput(1));
    mlp_1->DisplayNetwork(summary);
    mlp_2->DisplayNetwork(summary);
    delete mlp_1;
    delete mlp_2;
    return passed_test;
}

bool InputOutputMapping::NullOutputs() {
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
    bool passed_test = (val_out_1==0.0) 
                       && (val_out_2 == mlp_1->GetOutput(1))
                       && (val_out_3 == 0.0)
                       && (val_out_4 == 0.0)
                       && (val_out_5 == 0.0);

    delete mlp_1;
    return passed_test;
}

bool InputOutputMapping::VectorInputOutputs() {
    /* Define two randomized MLPs */
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

    bool passed_test{true};
    if (inside_m != inside_v){
        passed_test = false;
        summary << "Member-wise and vector-wise query input returns different inclusion.\n";
    }
    if (val_x_v != val_x_m && val_y_v != val_y_m && val_z_v != val_z_m) {
        passed_test = false;
        summary << "Member-wise and vector-wise query input returns different output:\n";
        summary << "Member-wise: x:" << val_x_m << " y: " << val_y_m << " z: " << val_z_m << std::endl;
        summary << "Vector-wise: x:" << val_x_v << " y: " << val_y_v << " z: " << val_z_v << std::endl;
    }
    if (val_null_m != 0.0) {
        passed_test = false;
        summary << "Member-wise null assignment returns non-zero value: " << val_null_m << std::endl;
    }
    if (val_null_v != 0.0) {
        passed_test = false;
        summary << "Vector-wise null assignment returns non-zero value: " << val_null_v << std::endl;
    }
    delete mlp_1;
    delete mlp_2;
    return passed_test;
}
bool InputOutputMapping::RunTest() {
    bool test_multiple_inputs = DifferentInputsDifferentOutputs();
    bool test_multiple_outputs = SameInputsDifferentOutputs();
    bool test_mimo = DifferentInputsDifferentOutputs2();
    bool test_null = NullOutputs();
    bool test_vectorwise = VectorInputOutputs();

    passed = (test_multiple_inputs && test_multiple_outputs && test_mimo && test_null && test_vectorwise);
    return passed;
};