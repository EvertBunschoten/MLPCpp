#include "../include/CLookUp_ANN.hpp"
#include "unit_test.hpp"
#include <vector>

int main() {

    std::vector<UnitTest*> unit_tests;
    unit_tests.push_back(new OutputCorrectness());
    unit_tests.push_back(new InputOutputMapping());

    std::vector<bool> passed_tests;
    passed_tests.resize(unit_tests.size(), false);
    bool passed{true};
    for (auto iTest=0u; iTest<unit_tests.size(); iTest++) {
        bool passed_test = unit_tests[iTest]->RunTest();
        passed_tests[iTest] = passed_test;
        if (!passed_test) {
            passed = false;
            unit_tests[iTest]->PrintSummary();
        }
    }

    
    size_t n_passed = std::accumulate(passed_tests.begin(), passed_tests.end(),0);
    for (auto test: unit_tests) {
        std::cout << test->GetTag() << " : " << (test->did_pass() ? "passed" : "FAILED") << std::endl;
    }
    std::cout << "Finished unit tests. Passed: " << n_passed << "/" << passed_tests.size() << std::endl;

    for (auto test : unit_tests) delete test;
    if (passed)
        return 0;
    else 
        return 1;
}