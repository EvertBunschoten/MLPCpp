#include <string>
#include <vector>
#include "../include/CLookUp_ANN.hpp" 

#pragma once
class UnitTest {
    protected:
    std::string tag;    /*! \brief Display tag */
    bool passed{false}; /*! \brief Test is passed */
    std::stringstream summary;  /*! \brief Message displayed when failed */
    public:
    std::string GetTag() const {return tag;}
    bool did_pass() const {return passed;}
    UnitTest(std::string name_in) : tag{name_in} {}
    virtual bool RunTest() = 0;
    void PrintSummary() const {std::cout << "Unit test: " << tag << std::endl;
    std::cout << summary.str() << std::endl;
    }
};


class OutputCorrectness : public UnitTest {
    private:
    /*! \brief Check whether copy constructor works correctly */
    bool CopyConstructorTest();

    /*! \brief Network read from a written file should be the same */
    bool FileWriterReaderTest();

    /*! \brief Network with same weights and biases should have the same output */
    bool WeightsBiasesTest();

    /*! \brief Passing network input through vector or member-wise should result in the same output. */
    bool VectorInputOutputs();
    
    public:
    OutputCorrectness() : UnitTest("Output correctness") {};
    virtual bool RunTest();
};

class InputOutputMapping : public UnitTest {
    private: 
    /*! \brief Link networks with different inputs and different outputs to query. */
    bool DifferentInputsDifferentOutputs();

    /*! \brief Link networks with same input and different output to query.*/
    bool SameInputsDifferentOutputs();

    /*! \brief Link multiple networks with different inputs and outputs to the same query. */
    bool DifferentInputsDifferentOutputs2();

    /*! \brief Setting query inputs through vector or by reference should result in the same network output. */
    bool VectorInputOutputs();
    
    /*! \brief Query containing null variables. */
    bool NullOutputs();

    public: 
    InputOutputMapping() : UnitTest("Input-output mapping") {};
    virtual bool RunTest();
};

class GradientCorrectness : public UnitTest {
    private:
    const double delta_inp{1e-6}; /* Input step size for finite-differneces. */
    /*! \brief Determine whether Jacobians are correctly evaluated. */
    bool JacobianCorrectness();

    /*! \brief Determine whether Hessians are correctly evaluated. */
    bool HessianCorrectness();
    public:
    GradientCorrectness() : UnitTest("Gradient correctness") {};
    virtual bool RunTest();
};

/*! 
* \brief Create a vector with random values for the network input. 
* \param[in] n_inp - number of network input nodes.
* \returns vector with randomized network inputs.
*/
static std::vector<double> RandomInputs(const size_t n_inp) {
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    std::vector<double> network_inputs;
    network_inputs.resize(n_inp);
    for(auto iInput=0u; iInput<n_inp; iInput++) network_inputs[iInput] = dis(gen);
    return network_inputs;
}

/*!
* \brief Create a multi-layer perceptron with a randomized architecture.
* \param[in] input_names - vector with input variable names.
* \param[out] output_names - vector with output variable names.
* \returns pointer to CNeuralNetwork object.
*/
static MLPToolbox::CNeuralNetwork* CreateRandomNetwork(const std::vector<std::string> &input_names={"x","y","z"}, 
                                                       const std::vector<std::string> &output_names={"a"}) {
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    MLPToolbox::CNeuralNetwork*mlp = new MLPToolbox::CNeuralNetwork();
    
    size_t n_inp = input_names.size(),
           n_outp = output_names.size();

    const size_t N_h_max{20}, N_h_min{5};
    const size_t L_min{1}, L_max{4};

    size_t n_hidden_layers = L_min + (rand() % (L_max));
    size_t N_h = N_h_min + (rand() % (N_h_max - N_h_min));
    std::vector<size_t> NN;
    NN.resize(n_hidden_layers+2);
    std::fill(NN.begin(), NN.end(), N_h);
    NN[0] = n_inp;
    NN[NN.size()-1] = n_outp;
    mlp = new MLPToolbox::CNeuralNetwork(NN);
    mlp->SetInputRegularization(ENUM_SCALING_FUNCTIONS::MINMAX);
    mlp->SetOutputRegularization(ENUM_SCALING_FUNCTIONS::MINMAX);

    for (auto iInput=0u; iInput<n_inp; iInput++) mlp->SetInputName(iInput, input_names[iInput]);
    for (auto iOutput=0u; iOutput<n_outp; iOutput++) mlp->SetOutputName(iOutput, output_names[iOutput]);

    std::vector<std::string> activation_function_options = {"elu","relu","tanh","swish","sigmoid", "gelu", "selu"};
    std::vector<std::string> scaler_functions = {"minmax", "robust", "standard"};
    std::string phi = activation_function_options[rand() % (activation_function_options.size())];
    std::string inp_scaler = scaler_functions[rand() % scaler_functions.size()];
    std::string outp_scaler = scaler_functions[rand() % scaler_functions.size()];
    
    mlp->SetActivationFunction(phi);
    mlp->SetActivationFunction(0, "linear");
    mlp->SetActivationFunction(NN.size()-1, "linear");
    mlp->SetInputRegularization(inp_scaler);
    mlp->SetOutputRegularization(outp_scaler);
    for (auto iInput=0u; iInput < n_inp; iInput++)
        mlp->SetInputNorm(iInput, dis(gen)-2.0, dis(gen)+1.0);
    for (auto iOutput=0u; iOutput < n_outp; iOutput++)
        mlp->SetOutputNorm(iOutput, dis(gen)-1.0, dis(gen)+1.0);
    mlp->RandomWeights();
    return mlp;
}
