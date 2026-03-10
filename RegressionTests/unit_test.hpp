#include <string>
#include <vector>
#include "../include/CLookUp_ANN.hpp" 

#pragma once
class UnitTest {
    protected:
    std::string tag;
    bool passed{false};
    std::stringstream summary;
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
    bool CopyConstructorTest();
    bool FileWriterReaderTest();
    bool WeightsBiasesTest();
    public:
    OutputCorrectness() : UnitTest("Output correctness") {};
    virtual bool RunTest();
};

class InputOutputMapping : public UnitTest {
    private: 
    bool DifferentInputsDifferentOutputs();
    bool SameInputsDifferentOutputs();
    public: 
    InputOutputMapping() : UnitTest("Input-output mapping") {};
    virtual bool RunTest();
};

class ScalerTests : public UnitTest {
    private:
    bool Consistency();
    public:
    ScalerTests() : UnitTest("Scaler functions") {};
    virtual bool RunTest();
};

static std::vector<double> RandomInputs(size_t n_inp) {
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    std::vector<double> network_inputs;
    network_inputs.resize(n_inp);
    for(auto iInput=0u; iInput<n_inp; iInput++) network_inputs[iInput] = dis(gen);
    return network_inputs;
}

static MLPToolbox::CNeuralNetwork* CreateRandomNetwork() {
     std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    MLPToolbox::CNeuralNetwork*mlp = new MLPToolbox::CNeuralNetwork();
    
    std::vector<std::string> input_names = {"x","y", "z"};
    
    size_t N_h_max{20}, N_h_min{5};
    size_t n_hidden_layers = 1 + (rand() % (4)) ;
    size_t N_h = N_h_min + (rand() % (N_h_max - N_h_min));
    std::vector<size_t> NN;
    NN.resize(n_hidden_layers+2);
    std::fill(NN.begin(), NN.end(), N_h);
    NN[0] = 3;
    NN[NN.size()-1] = 1;
    mlp = new MLPToolbox::CNeuralNetwork(NN);
    mlp->SetInputRegularization(ENUM_SCALING_FUNCTIONS::MINMAX);
    mlp->SetOutputRegularization(ENUM_SCALING_FUNCTIONS::MINMAX);

    for (auto iInput=0u; iInput<input_names.size(); iInput++) mlp->SetInputName(iInput, input_names[iInput]);
    mlp->SetOutputName(0, "a");

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
    for (auto iInput=0u; iInput < 3; iInput++)
        mlp->SetInputNorm(iInput, dis(gen)-2.0, dis(gen)+1.0);
    mlp->SetOutputNorm(0, dis(gen)-2.0, dis(gen)+1.0);
    mlp->RandomWeights();
    return mlp;
}
